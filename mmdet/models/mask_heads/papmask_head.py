import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms, multiclass_nms_with_mask
from mmdet.ops import ModulatedDeformConvPack

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob, build_norm_layer
from IPython import embed
import cv2
import numpy as np
import math
import time

INF = 1e8

@HEADS.register_module
class PAPMask_Head(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 coefficient_channels=16,
                 stacked_convs=4,
                 strides=(8, 16, 32, 64, 128),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 use_dcn=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_polarcontour=dict(type='PolarContourMarginIOULoss'),
                 loss_centerness=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(PAPMask_Head, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.coefficient_channels= coefficient_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_polarcontour = build_loss(loss_polarcontour)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_mask = build_loss(loss_mask)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.use_dcn = use_dcn

        self.angles = torch.range(0, 350, 10).cuda() / 180 * math.pi

        self._init_layers()

    def _init_layers(self):
        self.share_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if not self.use_dcn:
                self.share_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
            else:
                self.share_convs.append(
                    ModulatedDeformConvPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                if self.norm_cfg:
                    self.share_convs.append(build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.share_convs.append(nn.ReLU(inplace=True))

        self.cls_conv = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.centerness_conv = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.polarcontour_conv = nn.Conv2d(self.feat_channels, 36, 3, padding=1)
        self.scales_polar = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.coefficient_conv = nn.Conv2d(
            self.feat_channels, self.coefficient_channels, 3, padding=1)
        self.prototype_conv1 = ConvModule(
                        self.feat_channels,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None)
        self.prototype_conv2 = nn.Conv2d(
            self.feat_channels, self.coefficient_channels, 1, padding=0)

    def init_weights(self):
        if not self.use_dcn:
            for m in self.share_convs:
                normal_init(m.conv, std=0.01)
        else:
            pass

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_conv, std=0.01, bias=bias_cls)
        normal_init(self.centerness_conv, std=0.01)
        normal_init(self.polarcontour_conv, std=0.01)
        normal_init(self.coefficient_conv, std=0.01)
        normal_init(self.prototype_conv1.conv, std=0.01)
        normal_init(self.prototype_conv2, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, 
            feats, self.scales_polar, [i for i in range(len(self.strides))])
    
    def forward_single(self, x, scales_polar, feat_id):
        share_feat = x
        for share_layer in self.share_convs:
            share_feat = share_layer(share_feat)

        cls_score = self.cls_conv(share_feat)
        centerness = self.centerness_conv(share_feat)
        polarcontour = scales_polar(self.polarcontour_conv(share_feat)).float().exp()
        coefficient = self.coefficient_conv(share_feat).float().tanh()

        prototype = None if feat_id != 0 else self.prototype_conv2(
            self.prototype_conv1(
                F.interpolate(share_feat, scale_factor=2, mode='nearest')
            )
        )
        return cls_score, centerness, polarcontour, coefficient, prototype

    @force_fp32(apply_to=('cls_scores', 'centerness','polarcontour', 'coefficient', 'prototype'))
    def loss(self,
             cls_scores,
             centernesses,
             polarcontours,
             coefficients,
             prototypes,
             gt_masks, 
             extra_data, 
             img_metas, 
             cfg):
        assert len(cls_scores) == len(centernesses) == len(polarcontours) == len(coefficients)
        prototypes = prototypes[0]

        # TODO
        # (1024, 1024) 
        # cls_scores[0] = torch.Size([4, 80, 128, 128]) 
        # centernesses[0] = torch.Size([4, 1, 128, 128])
        # polarcontours[0] = torch.Size([4, 36, 128, 128])
        # coefficients[0] = torch.Size([4, 16, 128, 128])
        # prototypes[0] = torch.Size([4, 16, 256, 256])
        # gt_masks = [(num_mask, 1024, 1024)] * 4
        # extra_data['_gt_labels'] = [torch(21824)] * 4 & ids
        # extra_data['_gt_polarcontours'] = [(21824, 36)]*4

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, polarcontours[0].dtype,
                                           polarcontours[0].device)
        















        


    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points



