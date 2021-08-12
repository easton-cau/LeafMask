from detectron2.structures import ImageList
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs
from detectron2.modeling.meta_arch.semantic_seg import build_sem_seg_head
from detectron2.layers import ShapeSpec
from detectron2.layers import interpolate
from detectron2.structures.boxes import Boxes

from .utils import *
from point_utils.point_head import calculate_uncertainty
from point_utils.point_head import roi_mask_point_loss
from point_utils.point_features import (
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    point_sample_fine_grained_features,
)
from point_utils.point_head import MaskRefine
from .bottom import DAGMask
from .top import MaskAssembly



class LeafMask(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.instance_loss_weight = cfg.MODEL.LEAFMASK.INSTANCE_LOSS_WEIGHT

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.assembly = MaskAssembly(cfg)
        self.mask = DAGMask(cfg, self.backbone.output_shape())
        self.in_features = cfg.MODEL.LEAFMASK.IN_FEATURES

        self.combine_on = cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED
        if self.combine_on:
            self.panoptic_module = build_sem_seg_head(cfg, self.backbone.output_shape())
            self.combine_overlap_threshold = cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH
            self.combine_stuff_area_limit = cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT
            self.combine_instances_confidence_threshold = (
                cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH)

        in_channels = cfg.MODEL.FPN.OUT_CHANNELS
        num_bases = cfg.MODEL.BASIS_MODULE.NUM_BASES
        attn_size = cfg.MODEL.LEAFMASK.ATTN_SIZE
        attn_len = num_bases * attn_size * attn_size
        self.top_layer = nn.Conv2d(
            in_channels, attn_len,
            kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.top_layer.weight, std=0.01)
        torch.nn.init.constant_(self.top_layer.bias, 0)
        self.mask_point_head = MaskRefine(
            cfg, ShapeSpec(channels=256, width=1, height=1)
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if self.combine_on:
            if "sem_seg" in batched_inputs[0]:
                gt_sem = [x["sem_seg"].to(self.device) for x in batched_inputs]
                gt_sem = ImageList.from_tensors(
                    gt_sem, self.backbone.size_divisibility, self.panoptic_module.ignore_value).tensor
            else:
                gt_sem = None
            sem_seg_results, sem_seg_losses = self.panoptic_module(features, gt_sem)
        if "basis_sem" in batched_inputs[0]:
            basis_sem = [x["basis_sem"].to(self.device) for x in batched_inputs]
            basis_sem = ImageList.from_tensors(
                basis_sem, self.backbone.size_divisibility, 0).tensor
        else:
            basis_sem = None

        basis_out, basis_losses = self.mask(features, basis_sem)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, self.top_layer)
        detector_results, detector_losses, pred_mask_logits = self.assembly(basis_out["bases"], proposals, gt_instances)


        if self.training:
            mask_coarse_logits = pred_mask_logits.clone().reshape(-1, 1, 56, 56)
            num_points = 14*14
            oversample_ratio = 5
            importance_sample_ratio = 1
            mask_features_list = [features[k] for k in ["p3"]]
            features_scales = [0.125]

            box_regression = proposals["instances"].reg_targets.view(-1,4)
            box_regression = box_regression * 8
            locations=proposals["instances"].locations.view(-1,2)
            proposal_boxes = torch.stack([
                locations[:, 0] - box_regression[:, 0],
                locations[:, 1] - box_regression[:, 1],
                locations[:, 0] + box_regression[:, 2],
                locations[:, 1] + box_regression[:, 3],], dim=1)

            gt_classes = torch.cat([x.gt_classes for x in gt_instances])
            point_coords = get_uncertain_point_coords_with_randomness(
                mask_coarse_logits,
                lambda logits: calculate_uncertainty(logits, gt_classes),
                num_points=num_points,
                oversample_ratio=oversample_ratio,
                importance_sample_ratio=importance_sample_ratio)

            fine_grained_features, point_coords_wrt_image = point_sample_fine_grained_features(
                mask_features_list, features_scales, proposal_boxes, point_coords, proposals["instances"].im_inds)
            coarse_features = point_sample(mask_coarse_logits, point_coords, align_corners=False)
            point_logits = self.mask_point_head(fine_grained_features, coarse_features)
            point_losses = {
                "loss_mask_point": roi_mask_point_loss(
                    point_logits, gt_instances, point_coords_wrt_image, proposals["instances"].gt_inds)}

            losses = {}
            losses.update(basis_losses)
            losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
            losses.update(proposal_losses)
            losses.update(point_losses)
            if self.combine_on:
                losses.update(sem_seg_losses)
            return losses

        else:
            if pred_mask_logits != None:
                pred_boxes=[ii.pred_boxes for ii in proposals]
                pred_classes = proposals[0].pred_classes
                mask_coarse_logits = pred_mask_logits.clone().reshape(-1, 1, 56, 56)
                mask_features_list = [features[k] for k in ["p3"]]
                features_scales = [0.125]

                if len(pred_classes) == 0:
                    return mask_coarse_logits

                mask_logits = mask_coarse_logits.clone()
                mask_point_subdivision_steps = 4
                mask_point_subdivision_num_points = 14*14
                for subdivions_step in range(mask_point_subdivision_steps):
                    mask_logits = interpolate(mask_logits, scale_factor=2, mode="bilinear", align_corners=False)
                    H, W = mask_logits.shape[-2:]
                    if (
                            mask_point_subdivision_num_points >= 4 * H * W
                            and subdivions_step < mask_point_subdivision_steps - 1):
                        continue
                    uncertainty_map = calculate_uncertainty(mask_logits, pred_classes)
                    point_indices, point_coords = get_uncertain_point_coords_on_grid(
                        uncertainty_map, mask_point_subdivision_num_points)
                    pred_boxes_tmp = Boxes.cat(pred_boxes).tensor
                    fine_grained_features, _ = point_sample_fine_grained_features(
                        mask_features_list, features_scales, pred_boxes_tmp, point_coords)
                    coarse_features = point_sample(
                        mask_coarse_logits, point_coords, align_corners=False)
                    point_logits = self.mask_point_head(fine_grained_features, coarse_features)
                    R, C, H, W = mask_logits.shape
                    point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                    mask_logits = (
                            mask_logits.reshape(R, C, H * W)
                                .scatter_(2, point_indices, point_logits)
                                .view(R, C, H, W))
                detector_results[0].pred_masks = mask_logits.sigmoid()

        processed_results = []
        for i, (detector_result, input_per_image, image_size) in enumerate(zip(
                detector_results, batched_inputs, images.image_sizes)):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            detector_r = detector_postprocess(detector_result, height, width)
            processed_result = {"instances": detector_r}
            if self.combine_on:
                sem_seg_r = sem_seg_postprocess(
                    sem_seg_results[i], image_size, height, width)
                processed_result["sem_seg"] = sem_seg_r
            if "seg_thing_out" in basis_out:
                seg_thing_r = sem_seg_postprocess(
                    basis_out["seg_thing_out"], image_size, height, width)
                processed_result["sem_thing_seg"] = seg_thing_r

            processed_results.append(processed_result)
            if self.combine_on:
                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_threshold,
                    self.combine_stuff_area_limit,
                    self.combine_instances_confidence_threshold)
                processed_results[-1]["panoptic_seg"] = panoptic_r

        return processed_results
