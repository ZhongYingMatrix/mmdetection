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
from mmdet.utils import Timer

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

        # TODO prepare target and loss mask take too much time

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, polarcontours[0].dtype,
                                        polarcontours[0].device)
        gt_labels, gt_ids, gt_polarcontours = self.polar_target(all_level_points, extra_data)

        mask_cnt, which_img = 1, {}
        for img_num, _gt_masks in enumerate(gt_masks):
            num = _gt_masks.shape[0]
            for i in range(mask_cnt, mask_cnt+num):
                which_img[i] = img_num
            mask_cnt = mask_cnt + num
        gt_masks = np.concatenate(gt_masks)
        gt_masks = [cv2.resize(gt_mask, (0,0), fx=0.25, fy=0.25)  for gt_mask in gt_masks]
        gt_masks = [torch.from_numpy(gt_mask).to(polarcontours[0].device) 
                    for gt_mask in gt_masks]      

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores]
        flatten_centernesses = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_polarcontours = [
            polarcontour.permute(0, 2, 3, 1).reshape(-1, 36)
            for polarcontour in polarcontours
        ]
        flatten_coefficients = [
            coefficient.permute(0, 2, 3, 1).reshape(-1, self.coefficient_channels)
            for coefficient in coefficients
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)  
        flatten_centernesses = torch.cat(flatten_centernesses)  
        flatten_polarcontours = torch.cat(flatten_polarcontours)  
        flatten_coefficients = torch.cat(flatten_coefficients)  

        flatten_gt_labels = torch.cat(gt_labels).long()  
        flatten_gt_ids = torch.cat(gt_ids)  
        flatten_gt_polarcontours = torch.cat(gt_polarcontours)  
        flatten_points = torch.cat([points.repeat(num_imgs, 1)
                                    for points in all_level_points])  
        pos_inds = flatten_gt_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_gt_labels,
            avg_factor=num_pos + num_imgs)
        pos_centernesses = flatten_centernesses[pos_inds]
        pos_polarcontours = flatten_polarcontours[pos_inds]
        pos_coefficients = flatten_coefficients[pos_inds]

        if num_pos > 0:
            pos_gt_polarcontours = flatten_gt_polarcontours[pos_inds]
            pos_gt_centernesses = self.polar_centerness_target(pos_gt_polarcontours)

            pos_points = flatten_points[pos_inds]

            # centerness weighted iou loss
            loss_polarcontour = self.loss_polarcontour(pos_polarcontours,
                                       pos_gt_polarcontours,
                                       weight=pos_gt_centernesses,
                                       avg_factor=pos_gt_centernesses.sum())

            loss_centerness = self.loss_centerness(pos_centernesses,
                                                   pos_gt_centernesses)

            # prototype mask loss
            pos_gt_ids = flatten_gt_ids[pos_inds]

            pos_contour_x = pos_polarcontours*self.angles.cos() + \
                pos_points[:,0].reshape(-1,1).repeat(1,36)
            pos_contour_y = pos_polarcontours*self.angles.sin() + \
                pos_points[:,1].reshape(-1,1).repeat(1,36)

            loss_mask = []
            for i in range(len(pos_gt_ids)):
                gt_mask = gt_masks[pos_gt_ids[i].int()-1]
                xs, ys = pos_contour_x[i].detach()/4, pos_contour_y[i].detach()/4
                contour = torch.stack((xs,ys),1).cpu().numpy()[None,...].astype(int)
                contour_range = cv2.drawContours(np.zeros(gt_mask.shape), contour, -1,1,-1)

                mask_id = int(pos_gt_ids[i])
                one_img_prototypes = prototypes[which_img[mask_id]]
                one_img_coefficients = pos_coefficients[i][...,None,None].repeat(1, 
                    gt_mask.shape[0], gt_mask.shape[0])
                pred_prototype = (one_img_coefficients*one_img_prototypes).sum(0)
                # outside contour fill with -5, will it work? TODO
                pred_mask = torch.from_numpy(-5*(1-contour_range)).to(pred_prototype.device) \
                    + pred_prototype * torch.from_numpy(contour_range).to(pred_prototype.device)
                pred_mask, gt_mask = self.get_minimal_crop(pred_mask, gt_mask)
                loss_mask.append(
                    self.loss_mask(pred_mask[None,None,...], gt_mask[None,...].float(), [0])
                    )
            loss_mask = sum(loss_mask)/len(loss_mask)

        else:
            loss_polarcontour = pos_polarcontours.sum()
            loss_centerness = pos_centernesses.sum()
            loss_mask = 0
            import pdb
            pdb.set_trace()

        return dict(
            loss_cls=loss_cls,
            loss_polarcontour=loss_polarcontour,
            loss_centerness=loss_centerness,
            loss_mask=loss_mask)



















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

    def polar_target(self, points, extra_data):
        assert len(points) == len(self.regress_ranges)

        num_levels = len(points)

        labels_list, ids_list, polarcontours_list = extra_data.values()
        # accumulate ids
        for img_id in range(1, len(ids_list)):
            ids_list[img_id] += ids_list[img_id-1].max()

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        ids_list = [
            ids.split(num_points, 0)
            for ids in ids_list
        ]
        polarcontours_list = [
            polarcontours.split(num_points, 0)
            for polarcontours in polarcontours_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_ids = []
        concat_lvl_polarcontours = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_ids.append(
                torch.cat(
                    [ids[i] for ids in ids_list]))
            concat_lvl_polarcontours.append(
                torch.cat(
                    [polarcontours[i] for polarcontours in polarcontours_list]))

        return concat_lvl_labels, concat_lvl_ids, concat_lvl_polarcontours

    def polar_centerness_target(self, pos_gt_polarcontours):
        # only calculate pos centerness targets, otherwise there may be nan
        gt_centernesses = (pos_gt_polarcontours.min(dim=-1)[0] / pos_gt_polarcontours.max(dim=-1)[0])
        return torch.sqrt(gt_centernesses)

    def get_minimal_crop(self, pred_mask, gt_mask):
        left, up, right, bottom = 1e10, 1e10, -1, -1
        for mask in (pred_mask, gt_mask):
            if mask.max() <= 0:
                continue
            pos_xs, pos_ys = torch.where(mask>0)
            left = min(pos_xs.min(), left)
            up = min(pos_ys.min(), up)
            right = max(pos_xs.max(), right)
            bottom = max(pos_ys.max(), bottom)
        assert left<=right and up<=bottom
        return pred_mask[left:right+1, up:bottom+1], gt_mask[left:right+1, up:bottom+1]





