from typing import Dict
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from adet.layers import conv_with_kaiming_uniform

from .attention import GlobalLocalChannelAttention, GlobalLocalSpatialAttention


class DAGMask(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        mask_dim          = cfg.MODEL.LEAFMASK.NUM_BASES
        planes            = cfg.MODEL.LEAFMASK.CONVS_DIM
        self.in_features  = cfg.MODEL.LEAFMASK.IN_FEATURES
        self.loss_on      = cfg.MODEL.LEAFMASK.LOSS_ON
        norm              = cfg.MODEL.LEAFMASK.NORM
        num_convs         = cfg.MODEL.LEAFMASK.NUM_CONVS
        self.visualize    = True

        feature_channels = {k: v.channels for k, v in input_shape.items()}
        conv_block = conv_with_kaiming_uniform(norm, True)


        ########## DAG-Mask Module ##########
        # 1.Scale Standardization
        self.standard = nn.ModuleList()
        for in_feature in self.in_features:
            self.standard.append(conv_block(feature_channels[in_feature], planes, 3, 1))
        # 2.DeepLab Decoder 1
        tower1 = []
        for i in range(num_convs):
            tower1.append(
                conv_block(planes, planes, 3, 1))
        self.add_module('tower1', nn.Sequential(*tower1))
        # 3.Dual Attention
        self.pre_att = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes * 4),
            nn.ReLU(inplace=True)
        )
        self.glcam = GlobalLocalChannelAttention(planes * 4)
        self.glsam = GlobalLocalSpatialAttention(planes * 4)
        self.conv_down = conv_block(planes * 4, planes, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        # 4.DeepLab Decoder 2
        tower2 = []
        tower2.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        tower2.append(
            conv_block(planes, planes, 3, 1))
        tower2.append(
            nn.Conv2d(planes, mask_dim, 1))
        self.add_module('tower2', nn.Sequential(*tower2))


        ########## Auxiliary Module ##########
        if self.loss_on:
            self.common_stride   = cfg.MODEL.BASIS_MODULE.COMMON_STRIDE
            num_classes          = cfg.MODEL.BASIS_MODULE.NUM_CLASSES + 1
            self.sem_loss_weight = cfg.MODEL.BASIS_MODULE.LOSS_WEIGHT
            inplanes = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(),
                                          nn.Conv2d(planes, planes, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(),
                                          nn.Conv2d(planes, num_classes, kernel_size=1,
                                                    stride=1))


    def forward(self, features, targets=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.standard[i](features[f])
            else:
                x_p = self.standard[i](features[f])
                x_p = F.interpolate(x_p, x.size()[2:], mode="bilinear", align_corners=False)
                x = x + x_p

        residual = x
        x = self.tower1(x)
        x = self.pre_att(x)
        x = self.glsam(x) * x
        x = self.glcam(x) * x
        x = self.conv_down(x)
        x = x + residual
        x = self.relu(x)
        x = self.tower2(x)
        outputs = {"bases": [x]}

        losses = {}
        # auxiliary thing semantic loss
        if self.training and self.loss_on:
            sem_out = self.seg_head(features[self.in_features[0]])
            gt_sem = targets.unsqueeze(1).float()
            gt_sem = F.interpolate(
                gt_sem, scale_factor=1 / self.common_stride)
            seg_loss = F.cross_entropy(
                sem_out, gt_sem.squeeze(1).long())
            losses['loss_basis_sem'] = seg_loss * self.sem_loss_weight
        elif self.visualize and hasattr(self, "seg_head"):
            outputs["seg_thing_out"] = self.seg_head(features[self.in_features[0]])

        return outputs,losses


