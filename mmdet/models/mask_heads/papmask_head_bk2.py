import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms, multiclass_nms_with_mask
from mmdet.core import multiclass_nms_with_contour_coefficient
from mmdet.ops import ModulatedDeformConvPack

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob, build_norm_layer
from IPython import embed
import cv2
import numpy as np
import math
import time
from mmdet.utils import Timer, TimeStamp

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
        self.cls_convs = nn.ModuleList()
        self.polar_convs = nn.ModuleList()
        self.proto_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if not self.use_dcn:
                self.cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
                self.polar_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
                self.proto_convs.append(
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
                self.cls_convs.append(
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
                    self.cls_convs.append(build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.cls_convs.append(nn.ReLU(inplace=True))

                self.polar_convs.append(
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
                    self.polar_convs.append(build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.polar_convs.append(nn.ReLU(inplace=True))
                
                self.proto_convs.append(
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
                    self.proto_convs.append(build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.proto_convs.append(nn.ReLU(inplace=True))

        self.cls_conv = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.coefficient_conv = nn.Conv2d(
            self.feat_channels, self.coefficient_channels, 3, padding=1)
        self.polarcontour_conv = nn.Conv2d(self.feat_channels, 36, 3, padding=1)
        self.centerness_conv = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales_polar = nn.ModuleList([Scale(1.0) for _ in self.strides])
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
            for m in self.cls_convs:
                normal_init(m.conv, std=0.01)
            for m in self.polar_convs:
                normal_init(m.conv, std=0.01)
            for m in self.proto_convs:
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
        cls_feat = x
        polar_feat = x
        
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        for polar_layer in self.polar_convs:
            polar_feat = polar_layer(polar_feat)

        cls_score = self.cls_conv(cls_feat)
        centerness = self.centerness_conv(polar_feat)
        polarcontour = scales_polar(self.polarcontour_conv(polar_feat)).float().exp()
        coefficient = self.coefficient_conv(cls_feat).float().tanh()

        if feat_id != 0:
            prototype = None
        else:
            proto_feat = x
            for proto_layer in self.proto_convs:
                proto_feat = proto_layer(proto_feat)
            prototype = self.prototype_conv2(
                self.prototype_conv1(
                    F.interpolate(proto_feat, scale_factor=2, mode='nearest')
                )
            )

        return cls_score, centerness, polarcontour, coefficient, prototype

    @force_fp32(apply_to=('cls_scores', 'centernesses','polarcontours', 'coefficients', 'prototypes'))
    def loss(self,
             cls_scores,
             centernesses,
             polarcontours,
             coefficients,
             prototypes,
             gt_bboxes,
             gt_masks, 
             extra_data, 
             img_metas, 
             cfg):
        # time stamp
        timestamp = TimeStamp()
        assert len(cls_scores) == len(centernesses) == len(polarcontours) == len(coefficients)
        prototypes = prototypes[0]

        # TODO prepare target and loss mask take too much time

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, polarcontours[0].dtype,
                                        polarcontours[0].device)
        gt_labels, gt_ids, gt_polarcontours = self.polar_target(all_level_points, extra_data)

        which_img = []
        for img_num, _gt_masks in enumerate(gt_masks):
            num = _gt_masks.shape[0]
            which_img += [img_num]*num
        which_img = torch.tensor(which_img).to(polarcontours[0].device)

        # time stamp
        #timestamp('prepare mask assign')
        gt_masks = np.concatenate(gt_masks)
        gt_masks = [cv2.resize(gt_mask, (0,0), fx=0.25, fy=0.25)  for gt_mask in gt_masks]
        gt_masks = [torch.from_numpy(gt_mask).to(polarcontours[0].device)[None,...] 
                    for gt_mask in gt_masks]     
        gt_masks = torch.cat(gt_masks) 

        # time stamp
        #timestamp('prepare mask resize')

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

        # time stamp
        #timestamp('flat all value')

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

            # time stamp
            #timestamp('loss cls & polar & center')

            # prototype mask loss
            pos_gt_ids = flatten_gt_ids[pos_inds]

            # pos_contour_x = pos_polarcontours*self.angles.cos() + \
            #     pos_points[:,0].reshape(-1,1).repeat(1,36)
            # pos_contour_y = pos_polarcontours*self.angles.sin() + \
            #     pos_points[:,1].reshape(-1,1).repeat(1,36)
            # pos_contour_left = pos_contour_x.min(1)[0]
            # pos_contour_right = pos_contour_x.max(1)[0]
            # pos_contour_up = pos_contour_y.min(1)[0]
            # pos_contour_bottom = pos_contour_y.max(1)[0]

            which_img = which_img[pos_gt_ids.long()-1]
            # (Pdb) prototypes.shape
            # torch.Size([4, 16, 256, 256])
            # (Pdb) prototypes[which_img].shape                                                                                                 
            # torch.Size([152, 16, 256, 256])
            # (Pdb) gt_masks.shape
            # torch.Size([16, 256, 256])
            # (Pdb) gt_masks[pos_gt_ids.long()-1].shape
            # torch.Size([152, 256, 256])
            # (Pdb) pos_coefficients.shape
            # torch.Size([152, 16])
            flatten_img_protomasks, flatten_img_gt_masks = [], []
            for img_id in range(prototypes.size(0)):
                flatten_img_prototypes = prototypes[img_id].reshape(self.coefficient_channels,-1)
                flatten_img_protomasks.append(
                    (pos_coefficients[which_img == img_id].mm(flatten_img_prototypes)).reshape(-1)
                )
                img_gt_masks = gt_masks[pos_gt_ids.long()-1][which_img == img_id]
                flatten_img_gt_masks.append(img_gt_masks.reshape(-1))
            flatten_img_protomasks = torch.cat(flatten_img_protomasks)
            flatten_img_gt_masks = torch.cat(flatten_img_gt_masks)

            loss_mask = self.loss_mask(flatten_img_protomasks,
                                                   flatten_img_gt_masks)
            loss_mask_dice = 1 - (flatten_img_protomasks.sigmoid()*flatten_img_gt_masks).sum()*2 \
                /(flatten_img_protomasks.sigmoid().sum()+flatten_img_gt_masks.sum())
            loss_mask += loss_mask_dice


            # time stamp
            #timestamp('compute pos')

            # loss_gt_mask = []
            # loss_proto = []
            # for i in range(len(pos_gt_ids)):
                #gt_mask = gt_masks[pos_gt_ids[i].int()-1]
                # xs, ys = pos_contour_x[i].detach()/4, pos_contour_y[i].detach()/4
                # contour = torch.stack((xs,ys),1).cpu().numpy()[None,...].astype(int)
                # contour_range = cv2.drawContours(np.zeros(gt_mask.shape), contour, -1,1,-1)

                # time stamp
                #timestamp.accumulate('get gt_mask')

                # mask_id = int(pos_gt_ids[i])
                # one_img_prototypes = prototypes[which_img[mask_id]]
                # one_img_coefficients = pos_coefficients[i][...,None,None].repeat(1, 
                #     one_img_prototypes.shape[-2], one_img_prototypes.shape[-1])
                # pred_prototype = (one_img_coefficients*one_img_prototypes).sum(0)

                # time stamp
                #timestamp.accumulate('pred_prototype')

                # # outside contour fill with -5, will it work? TODO
                # pred_mask = torch.from_numpy(-5*(1-contour_range)).to(pred_prototype.device) \
                #     + pred_prototype * torch.from_numpy(contour_range).to(pred_prototype.device)

                # # time stamp
                # timestamp.accumulate('combine proto and contour')                
                
                # left, up, right, bottom = 1e10, 1e10, -1, -1
                # for mask in (pred_mask, gt_mask):
                #     if mask.max() <= 0:
                #         continue
                #     pos_xs, pos_ys = torch.where(mask>0)
                #     left = min(pos_xs.min(), left)
                #     up = min(pos_ys.min(), up)
                #     right = max(pos_xs.max(), right)
                #     bottom = max(pos_ys.max(), bottom)
                # assert left<=right and up<=bottom
 
                # # time stamp
                # timestamp.accumulate('compute crop x y ')

                # pred_mask, gt_mask = pred_mask[left:right+1, up:bottom+1], gt_mask[left:right+1, up:bottom+1]

                # # time stamp
                # timestamp.accumulate('get crop')
            
                # loss_gt_mask.append(gt_masks[pos_gt_ids[i].int()-1])
                # loss_proto.append(pred_prototype)
                # time stamp
                #timestamp.accumulate('append gt mask and proto')
            #timestamp.over()
            
            # loss_gt_mask = torch.cat(loss_gt_mask)
            # loss_proto = torch.cat(loss_proto)
            # loss_mask = self.loss_mask(loss_proto[None,None,...], loss_gt_mask[None,...].float(), [0])
            # loss_proto = loss_proto.sigmoid()
            # inter = (loss_proto*loss_gt_mask).sum()
            # uni = loss_proto.sum() + loss_mask.sum()
            # loss_dice = 1 - 2*inter/uni
            # loss_mask = (loss_mask + loss_dice)/2
            # time stamp
            #timestamp('loss_mask')
            # import pdb
            # pdb.set_trace()
            


        else:
            loss_polarcontour = pos_polarcontours.sum()
            loss_centerness = pos_centernesses.sum()
            loss_mask = pos_coefficients.sum()

        return dict(
            loss_cls=loss_cls,
            loss_polarcontour=loss_polarcontour,
            loss_centerness=loss_centerness,
            loss_mask=loss_mask,
            )

    @force_fp32(apply_to=('cls_scores', 'centernesses','polarcontours', 'coefficients', 'prototypes'))
    def get_masks(self,
                  cls_scores,
                  centernesses,
                  polarcontours,
                  coefficients,
                  prototypes,
                  img_metas,
                  cfg,
                  rescale=False):
        assert len(cls_scores) == len(centernesses) == len(polarcontours) == len(coefficients)
        prototypes = prototypes[0]
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, polarcontours[0].dtype,
                                      polarcontours[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            polarcontour_list = [
                polarcontours[i][img_id].detach() for i in range(num_levels)
            ]
            coefficient_list = [
                coefficients[i][img_id].detach() for i in range(num_levels)
            ]
            prototype = prototypes[img_id].detach()
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_masks = self.get_masks_single(cls_score_list,
                                              centerness_pred_list,
                                              polarcontour_list,
                                              coefficient_list,
                                              prototype,
                                              mlvl_points, img_shape,
                                              scale_factor, cfg, rescale)
            result_list.append(det_masks)
        return result_list

    def get_masks_single(self,
                         cls_scores,
                         centernesses,
                         polarcontours,
                         coefficients,
                         prototype,
                         mlvl_points,
                         img_shape,
                         scale_factor,
                         cfg,
                         rescale=False):
        assert len(cls_scores) == len(centernesses) == len(polarcontours) == len(coefficients)
        mlvl_scores, mlvl_centerness, mlvl_contours, mlvl_expand_contours, mlvl_coefficients \
            = [], [], [], [], []
        for cls_score, centerness, polarcontour, coefficient, points in zip(
                cls_scores, centernesses, polarcontours, coefficients, mlvl_points):
            assert cls_score.size()[-2:] == polarcontour.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            polarcontour = polarcontour.permute(1, 2, 0).reshape(-1, 36)
            coefficient = coefficient.permute(1, 2, 0).reshape(-1, 16)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
                polarcontour = polarcontour[topk_inds,:]
                coefficient = coefficient[topk_inds,:]
                points = points[topk_inds, :]
            contours = self.polarcontour2contour(points,
                polarcontour, self.angles, max_shape=img_shape, expand_factor=1.0)
            expand_contours = self.polarcontour2contour(points,
                polarcontour, self.angles, max_shape=img_shape, expand_factor=1.2)

            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_contours.append(contours)
            mlvl_expand_contours.append(expand_contours)
            mlvl_coefficients.append(coefficient)
       
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1) # ?
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        mlvl_contours = torch.cat(mlvl_contours)
        mlvl_expand_contours = torch.cat(mlvl_expand_contours)
        mlvl_coefficients = torch.cat(mlvl_coefficients)

        if rescale:
            try:
                scale_factor = torch.Tensor(scale_factor)[:2].cuda().unsqueeze(1).repeat(1, 36)
                _mlvl_contours = mlvl_contours / scale_factor
                _mlvl_expand_contours = mlvl_expand_contours / scale_factor
            except:
                _mlvl_contours = mlvl_contours / mlvl_contours.new_tensor(scale_factor)
                _mlvl_expand_contours = mlvl_expand_contours / mlvl_expand_contours.new_tensor(scale_factor)
        else:
            raise NotImplementedError

        # mask centerness is smaller than origin centerness, 
        # so add a constant is important or the score will be too low.
        centerness_factor = 0.5 
        _mlvl_bboxes = torch.stack( [_mlvl_contours[:, 0].min(1)[0],
                                     _mlvl_contours[:, 1].min(1)[0],
                                     _mlvl_contours[:, 0].max(1)[0],
                                     _mlvl_contours[:, 1].max(1)[0]], -1)
 
        det_bboxes, det_labels, det_contours, det_coefficients = multiclass_nms_with_contour_coefficient(
            _mlvl_bboxes,
            mlvl_scores,
            _mlvl_expand_contours,
            mlvl_coefficients,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness + centerness_factor)

        flatten_prototype = prototype.reshape(self.coefficient_channels, -1)
        det_masks = det_coefficients.mm(flatten_prototype).reshape(
            det_labels.size(0), prototype.size(1), prototype.size(2))
        det_masks = det_masks.sigmoid()

        return det_bboxes, det_labels, det_contours, det_masks


















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

    def polarcontour2contour(self, points, distances, angles, max_shape=None, expand_factor=1.2):
        '''Decode distance prediction to 36 mask points
        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 36,from angle 0 to 350.
            angles (Tensor):
            max_shape (tuple): Shape of the image.
        Returns:
            Tensor: Decoded masks.
        '''
        distances *= expand_factor
        num_points = points.shape[0]
        points = points[:, :, None].repeat(1, 1, 36)
        c_x, c_y = points[:, 0], points[:, 1]

        sin = torch.sin(angles)
        cos = torch.cos(angles)
        sin = sin[None, :].repeat(num_points, 1)
        cos = cos[None, :].repeat(num_points, 1)

        x = distances * sin + c_x
        y = distances * cos + c_y

        if max_shape is not None:
            x = x.clamp(min=0, max=max_shape[1] - 1)
            y = y.clamp(min=0, max=max_shape[0] - 1)

        res = torch.cat([x[:, None, :], y[:, None, :]], dim=1)
        return res






