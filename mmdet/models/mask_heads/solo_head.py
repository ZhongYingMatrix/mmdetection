import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob, build_norm_layer
from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
INF = 1e8

@HEADS.register_module
class SOLO_Head(nn.Module):

    def __init__(self,
                 num_classes=81,
                 in_channels=256,
                 stacked_convs=7,
                 feat_channels=256,
                 strides=[8, 16, 32, 64, 128],
                 grid_number=[40, 36, 24, 16, 12],
                 instance_scale=((-1, 96), (48, 192), (96, 384), (192, 768),
                                          (384, INF)),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_mask=dict(type='DiceLoss'),
                 loss_factor={'loss_cls':1., 'loss_mask':3.}):
        super(SOLO_Head, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.grid_number = grid_number
        self.instance_scale = instance_scale
        self.loss_factor = loss_factor
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()
        self.cls_out_lvls_conv = nn.ModuleList()
        self.mask_out_lvls_conv = nn.ModuleList()
        self.cls_lvls_pooling = nn.ModuleList()
        
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            chn = self.in_channels+2 if i == 0 else self.feat_channels
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))

        for i in range(len(self.strides)):
            self.cls_out_lvls_conv.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.cls_out_channels, 
                    1, 
                    padding=0))
            self.mask_out_lvls_conv.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.grid_number[i] * 2, # Decoupled SOLO 
                    1, 
                    padding=0))
            self.cls_lvls_pooling.append(
                nn.AdaptiveAvgPool2d(self.grid_number[i])
            )

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.mask_convs:
            normal_init(m.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        for m in self.cls_out_lvls_conv:
            normal_init(m, std=0.01, bias=bias_cls)
        for m in self.mask_out_lvls_conv:
            normal_init(m, std=0.01)

    def forward(self, feats):
        return multi_apply(
            self.forward_single, 
            feats, 
            self.cls_lvls_pooling,
            self.cls_out_lvls_conv, 
            self.mask_out_lvls_conv,
            self.grid_number
        )

    def forward_single(self, 
                       x, 
                       cls_pooling, 
                       cls_out_conv, 
                       mask_out_conv, 
                       grid_num):
        cls_feat = cls_pooling(x)
        mask_feat = self.coord_conv_cat(x)

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = cls_out_conv(cls_feat)

        for mask_layer in self.mask_convs:
            mask_feat = mask_layer(mask_feat)
        mask_feat = F.interpolate(mask_feat, scale_factor=2, mode='nearest')
        mask_pred = mask_out_conv(mask_feat)
        
        return cls_score, mask_pred


    def coord_conv_cat(self, feat):
        h, w = feat.shape[-2], feat.shape[-1]
        dtype, device = feat.dtype, feat.device
        x_range = torch.arange(-1, 1, 2/h, dtype=dtype, device=device)
        y_range = torch.arange(-1, 1, 2/w, dtype=dtype, device=device)
        x, y = torch.meshgrid(x_range, y_range)
        coord_feat = torch.stack((x,y)).repeat(feat.shape[0],1,1,1)

        return torch.cat((feat, coord_feat), dim=1)

