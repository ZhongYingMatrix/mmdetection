import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmdet.ops.nms import nms_wrapper
import mmcv

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob
import torch.nn.functional as F
import numpy as np
import cv2

INF = 1e8


@HEADS.register_module
class FCOS_Proto_Head(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), 
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 use_coord_conv=False,
                 use_reg_feat_in_ctr=False,
                 use_crop_in_loss_mask=False,
                 use_ctr_weight=False,
                 use_edge_weight=False,
                 loss_mask_factor = 1.0,
                 loss_centerness_factor = 1.0,
                 ):
        super(FCOS_Proto_Head, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_mask = build_loss(loss_mask)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.radius = 1.5
        self.use_coord_conv = use_coord_conv
        if self.use_coord_conv:
            self.in_channels += 2
        self.use_reg_feat_in_ctr = use_reg_feat_in_ctr
        self.use_crop_in_loss_mask = use_crop_in_loss_mask
        self.use_ctr_weight = use_ctr_weight
        self.use_edge_weight = use_edge_weight
        self.loss_mask_factor = loss_mask_factor
        self.loss_centerness_factor = loss_centerness_factor

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        # ---------------------protonet---------------------------------------------------------
        self.coefficient_conv = nn.Conv2d(
            self.feat_channels, 16, 3, padding=1)
        self.proto_convs = nn.ModuleList()
        for i in range(2):
            chn = self.in_channels if i == 0 else self.feat_channels
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
            self.feat_channels, 16, 1, padding=0)

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
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
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

        # ---------------------protonet---------------------------------------------------------
        for m in self.proto_convs:
            normal_init(m.conv, std=0.01)
        normal_init(self.prototype_conv1.conv, std=0.01)
        normal_init(self.prototype_conv2, std=0.01)
        normal_init(self.coefficient_conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales, 
            [i for i in range(len(self.strides))]
        )

    def forward_single(self, x, scale, feat_id):
        if self.use_coord_conv:
            x = self.coord_conv_cat(x)
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(self.fcos_reg(reg_feat)).float().exp()

        if self.use_reg_feat_in_ctr:
            centerness = self.fcos_centerness(reg_feat)
        else:
            centerness = self.fcos_centerness(cls_feat)

        # ---------------------protonet---------------------------------------------------------
        coefficient = self.coefficient_conv(cls_feat).float().tanh()
        if feat_id == 0:
            proto_feat = x
            for proto_layer in self.proto_convs:
                proto_feat = proto_layer(proto_feat)
        prototype = None if feat_id != 0 else self.prototype_conv2(
            self.prototype_conv1(
                F.interpolate(proto_feat, scale_factor=2, mode='nearest')
            )
        )

        return cls_score, bbox_pred, centerness, coefficient, prototype

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'coefficients', 'prototypes'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             coefficients,
             prototypes,
             gt_bboxes,
             gt_masks,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        self.num_points_per_level = [i.size()[0] for i in all_level_points]
        labels, bbox_targets, gt_ids = self.fcos_target(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        # ---------------------protonet---------------------------------------------------------
        pred_bigger = (pos_decoded_bbox_preds>=pos_decoded_target_preds).int()
        target_bigger = (pos_decoded_bbox_preds<pos_decoded_target_preds).int()
        mask_range = torch.cat(((target_bigger*pos_decoded_bbox_preds + pred_bigger*pos_decoded_target_preds)[:,:2],
            (pred_bigger*pos_decoded_bbox_preds + target_bigger*pos_decoded_target_preds)[:,2:]), dim=1).int().clamp_min(0)
        mask_range /= 4
        prototypes = prototypes[0]
        which_img = []
        for img_num, _gt_masks in enumerate(gt_masks):
            num = _gt_masks.shape[0]
            which_img += [img_num]*num
        which_img = torch.tensor(which_img).to(cls_scores[0].device)
        new_gt_masks = []
        for _gt_masks in gt_masks:
            _gt_masks = [cv2.resize(gt_mask, (0,0), fx=0.25, fy=0.25) for gt_mask in _gt_masks ]
            _gt_masks = [np.pad(gt_mask, ((0,prototypes.size(2)-gt_mask.shape[0]), (0,prototypes.size(3)-gt_mask.shape[1]))) for gt_mask in _gt_masks]
            _gt_masks = np.stack(_gt_masks)
            _gt_masks = torch.from_numpy(_gt_masks).to(cls_scores[0].device)    
            new_gt_masks.append(_gt_masks)
        new_gt_masks = torch.cat(new_gt_masks)

        flatten_coefficients = [
            coefficient.permute(0, 2, 3, 1).reshape(-1, 16)
            for coefficient in coefficients
        ]
        flatten_coefficients = torch.cat(flatten_coefficients)
        flatten_gt_ids = torch.cat(gt_ids) 
        pos_coefficients = flatten_coefficients[pos_inds]

        pos_gt_ids = flatten_gt_ids[pos_inds]
        which_img = which_img[pos_gt_ids.long()-1]            
        #flatten_img_protomasks, flatten_img_gt_masks, flatten_img_weights = [], [], []
        loss_mask = 0
        for img_id in range(prototypes.size(0)):
            flatten_img_prototypes = prototypes[img_id].reshape(16,-1)
            img_mask_range = mask_range[which_img == img_id]
            img_centerness_targets = pos_centerness_targets[which_img == img_id]
            img_protomasks = (pos_coefficients[which_img == img_id].mm(flatten_img_prototypes)).reshape(-1, prototypes.shape[2], prototypes.shape[3])
            img_gt_masks = new_gt_masks[pos_gt_ids.long()-1][which_img == img_id]
            for mask_id in range(img_protomasks.shape[0]):
                _ctr = img_centerness_targets[mask_id]
                if self.use_crop_in_loss_mask:
                    _range = img_mask_range[mask_id]
                    img_protomask = img_protomasks[mask_id][_range[1]:_range[3]+1, _range[0]:_range[2]+1]
                    img_gt_mask = img_gt_masks[mask_id][_range[1]:_range[3]+1, _range[0]:_range[2]+1]
                else:
                    img_protomask = img_protomasks[mask_id]
                    img_gt_mask = img_gt_masks[mask_id]
                
                #import pdb; pdb.set_trace()
                loss_mask_dice = 1 - (img_protomask.reshape(-1).sigmoid()*img_gt_mask.reshape(-1)).sum()*2 \
                    /(img_protomask.reshape(-1).sigmoid().sum()+img_gt_mask.reshape(-1).sum())
                if self.use_ctr_weight:
                    loss_mask += loss_mask_dice * _ctr
                else:
                    loss_mask += loss_mask_dice
   
        if self.use_ctr_weight:
            loss_mask /= pos_centerness_targets.sum()
        else:
            loss_mask /= pos_centerness_targets.shape[0]
                #flatten_img_protomasks.append(img_protomask.reshape(-1))
                #flatten_img_gt_masks.append(img_gt_mask.reshape(-1))
                #img_weights = torch.ones_like(img_gt_mask)*(_ctr/img_gt_mask.shape[0]/img_gt_mask.shape[1])
                #flatten_img_weights.append(img_weights.reshape(-1))
        # flatten_img_protomasks = torch.cat(flatten_img_protomasks)
        # flatten_img_gt_masks = torch.cat(flatten_img_gt_masks)
        # flatten_img_weights = torch.cat(flatten_img_weights)
        # flatten_img_weights *= (flatten_img_weights.shape[0]/flatten_img_weights.sum())

        # if self.use_ctr_size_weight:
        #     loss_mask = self.loss_mask(flatten_img_protomasks, flatten_img_gt_masks,
        #         weight=flatten_img_weights)
        #     intersection = (flatten_img_protomasks.sigmoid()*flatten_img_gt_masks*flatten_img_weights).sum()
        #     union = (flatten_img_protomasks.sigmoid()*flatten_img_weights).sum() + \
        #         (flatten_img_gt_masks*flatten_img_weights).sum()
        #     loss_mask_dice = 1 - intersection*2/union
        #     loss_mask += loss_mask_dice
        # else:
        #     loss_mask = self.loss_mask(flatten_img_protomasks, flatten_img_gt_masks)
        #     loss_mask_dice = 1 - (flatten_img_protomasks.sigmoid()*flatten_img_gt_masks).sum()*2 \
        #         /(flatten_img_protomasks.sigmoid().sum()+flatten_img_gt_masks.sum())
        #     loss_mask += loss_mask_dice

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness * self.loss_centerness_factor,
            loss_mask=loss_mask * self.loss_mask_factor
            )

    # ---------------------protonet---------------------------------------------------------
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'coefficients', 'prototypes'))
    def get_masks(self,
                  cls_scores,
                  bbox_preds,
                  centernesses,
                  coefficients,
                  prototypes,
                  img_metas,
                  cfg,
                  rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        prototypes = prototypes[0]
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            coefficient_list = [
                coefficients[i][img_id].detach() for i in range(num_levels)
            ]
            prototype = prototypes[img_id].detach()
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']
            det_masks = self.get_masks_single(cls_score_list, bbox_pred_list,
                                                centerness_pred_list,
                                                coefficient_list,
                                                prototype,
                                                mlvl_points, img_shape,
                                                ori_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_masks)
        return result_list

    def get_masks_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          coefficients,
                          prototype,
                          mlvl_points,
                          img_shape,
                          ori_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_coefficients = []
        for cls_score, bbox_pred, centerness, coefficient, points in zip(
                cls_scores, bbox_preds, centernesses, coefficients, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            coefficient = coefficient.permute(1, 2, 0).reshape(-1, 16)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
                coefficient = coefficient[topk_inds,:]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_coefficients.append(coefficient)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        mlvl_coefficients = torch.cat(mlvl_coefficients)
        det_bboxes, det_labels, det_coefficients = self.multiclass_nms_with_coefficient(
            mlvl_bboxes,
            mlvl_scores,
            mlvl_coefficients,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        det_masks = self.coe2mask(det_bboxes, det_coefficients, prototype, img_shape, ori_shape)
        
        return det_bboxes, det_labels, det_masks

    def coe2mask(self, det_bboxes, det_coefficients, prototype, img_shape, ori_shape):
        flatten_prototype = prototype.reshape(16, -1)
        det_masks = det_coefficients.mm(flatten_prototype).reshape(
            det_coefficients.size(0), prototype.size(1), prototype.size(2))
        det_masks = det_masks.sigmoid()
        det_masks = det_masks.data.cpu().numpy().astype(np.float)
        masks_result = []
        for i in range(det_masks.shape[0]):
            #import pdb; pdb.set_trace()
            mask = det_masks[i]
            bbox = det_bboxes[i]
            bbox_range = np.zeros((ori_shape[0], ori_shape[1]), dtype=np.uint8)
            bbox_range[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] += 1
            # mask = mmcv.imresize(mask, (mask.shape[1]*4, mask.shape[0]*4))
            # mask = mask[:img_shape[0],:img_shape[1]]
            mask = mask[:int(img_shape[0]/4),:int(img_shape[1]/4)]
            mask = mmcv.imresize(mask, (ori_shape[1], ori_shape[0]))
            # TODO
            #mask *= bbox_range
            mask = (mask>0.5).astype(np.uint8)
            masks_result.append(mask)
        return masks_result


    def multiclass_nms_with_coefficient(self,
                   multi_bboxes,
                   multi_scores,
                   multi_coefficients,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
        num_classes = multi_scores.shape[1]
        bboxes, labels, coefficients = [], [], []
        nms_cfg_ = nms_cfg.copy()
        nms_type = nms_cfg_.pop('type', 'nms')
        nms_op = getattr(nms_wrapper, nms_type)
        for i in range(1, num_classes):
            cls_inds = multi_scores[:, i] > score_thr
            if not cls_inds.any():
                continue
            # get bboxes and scores of this class
            if multi_bboxes.shape[1] == 4:
                _bboxes = multi_bboxes[cls_inds, :]
                _coefficients = multi_coefficients[cls_inds, :]
            else:
                _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
            _scores = multi_scores[cls_inds, i]
            if score_factors is not None:
                _scores *= score_factors[cls_inds]
            cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
            cls_dets, index = nms_op(cls_dets, **nms_cfg_)
            cls_coefficients = _coefficients[index]
            cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                            i - 1,
                                            dtype=torch.long)
            bboxes.append(cls_dets)
            labels.append(cls_labels)
            coefficients.append(cls_coefficients)
        if bboxes:
            bboxes = torch.cat(bboxes)
            labels = torch.cat(labels)
            coefficients = torch.cat(coefficients)
            if bboxes.shape[0] > max_num:
                _, inds = bboxes[:, -1].sort(descending=True)
                inds = inds[:max_num]
                bboxes = bboxes[inds]
                labels = labels[inds]
                coefficients = coefficients[inds]
        else:
            bboxes = multi_bboxes.new_zeros((0, 5))
            labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
            coefficients = multi_bboxes.new_zeros((0, 16))

        return bboxes, labels, coefficients


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'coefficients', 'prototypes'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   coefficients,
                   prototypes,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                centerness_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels

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

    def fcos_target(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, ids_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges)

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        
        # ---------------------protonet---------------------------------------------------------
        for img_id in range(1, len(ids_list)):
            ids_list[img_id] += ids_list[img_id-1].max()
        ids_list = [
            ids.split(num_points, 0)
            for ids in ids_list
        ]
        concat_lvl_ids = []
        for i in range(num_levels):
            concat_lvl_ids.append(
                torch.cat(
                    [ids[i] for ids in ids_list]))

        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_ids

    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        # ---------------------ctr sampling---------------------------------------------------------
        # inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        inside_gt_bbox_mask = self.get_sample_region(gt_bboxes,
                                                     self.strides,
                                                     self.num_points_per_level,
                                                     xs,
                                                     ys,
                                                     radius=self.radius)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        # ---------------------protonet---------------------------------------------------------
        ids = min_area_inds + 1
        ids[min_area == INF] = 0

        return labels, bbox_targets, ids

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    # ---------------------ctr sampling---------------------------------------------------------
    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1):
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)

        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end

        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  # 上下左右都>0 就是在bbox里面
        return inside_gt_bbox_mask

    def coord_conv_cat(self, feat):
        h, w = feat.shape[-2], feat.shape[-1]
        dtype, device = feat.dtype, feat.device
        x_range = torch.arange(-1, 1, 2/h, dtype=dtype, device=device)
        y_range = torch.arange(-1, 1, 2/w, dtype=dtype, device=device)
        x, y = torch.meshgrid(x_range, y_range)
        coord_feat = torch.stack((x,y)).repeat(feat.shape[0],1,1,1)

        return torch.cat((feat, coord_feat), dim=1)
