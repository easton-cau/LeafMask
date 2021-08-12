import torch
import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat

from .point_features import point_sample


def calculate_uncertainty(logits,classes):
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone()
    else:
        gt_class_logits = logits[
            torch.arange(logits.shape[0], device=logits.device), classes
        ].unsqueeze(1)

    return -(torch.abs(gt_class_logits))


def roi_mask_point_loss(mask_logits, instances, points_coord, gt_inds):
    with torch.no_grad():
        cls_agnostic_mask = mask_logits.size(1) == 1
        total_num_masks = mask_logits.size(0)
        gt_classes = []
        gt_mask_logits = []
        gt_bit_masks_all = []
        h_list = []
        w_list = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            gt_bit_masks = instances_per_image.gt_masks.tensor
            gt_bit_masks_all += ([_ for _ in gt_bit_masks])
            h, w = instances_per_image.gt_masks.image_size
            h_list += [h for _ in range(len(instances_per_image))]
            w_list += [w for _ in range(len(instances_per_image))]
        gt_bit_masks_all = [gt_bit_masks_all[inds] for inds in gt_inds]
        h_list = torch.tensor(h_list)
        h_list = h_list[gt_inds]
        w_list = torch.tensor(w_list)
        w_list = w_list[gt_inds]
        for coord,h,w,gt_mask in zip(points_coord,h_list,w_list,gt_bit_masks_all):
            scale = torch.tensor([w, h], dtype=torch.float, device=mask_logits.device)
            sample_format = coord / scale
            gt_mask_logits.append(
                point_sample(
                    gt_mask.to(torch.float32).unsqueeze(1).reshape(1,1,w,h),
                    sample_format.reshape(1,-1,2),
                    align_corners=False,).squeeze(1))

    if len(gt_mask_logits) == 0:
        return mask_logits.sum() * 0

    gt_mask_logits = cat(gt_mask_logits, dim=0)
    assert gt_mask_logits.numel() > 0, gt_mask_logits.shape
    if cls_agnostic_mask:
        mask_logits = mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        mask_logits = mask_logits[indices, gt_classes]

    point_loss = F.binary_cross_entropy_with_logits(
        mask_logits, gt_mask_logits.to(dtype=torch.float32), reduction="mean")
    point_loss = point_loss

    return point_loss


class MaskRefine(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(MaskRefine, self).__init__()

        num_classes                 = 80
        fc_dim                      = 16
        num_fc                      = 3
        cls_agnostic_mask           = False
        self.coarse_pred_each_layer = True
        input_channels              = input_shape.channels
        fc_dim_in = input_channels
        self.down_fine_feature_conv = nn.Conv1d(in_channels=256,out_channels=1,kernel_size=1, stride=1, padding=0)
        self.fc_layers = []
        for k in range(num_fc):
            fc = nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = fc_dim
            fc_dim_in += num_classes if self.coarse_pred_each_layer else 0

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = nn.Conv1d(fc_dim_in, num_mask_classes, kernel_size=1, stride=1, padding=0)
        for layer in self.fc_layers:
            weight_init.c2_msra_fill(layer)
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)


    def forward(self, fine_grained_features, coarse_features):
        x = fine_grained_features
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            if self.coarse_pred_each_layer:
                x = cat((x, coarse_features), dim=1)

        return self.predictor(x)
