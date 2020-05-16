import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.cnn import build_norm_layer

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from mmdet.ops import ModulatedDeformConvPack
from ..builder import HEADS, build_loss

INF = 1e8


@HEADS.register_module()
class OTSSHead(nn.Module):
    """
    TODO
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 background_label=None,
                 IoUtype='CIoU',
                 reg_norm=False,
                 ctr_on_reg=False,
                 use_centerness=False,
                 soft_label=False,
                 use_dcn_in_tower=False,
                 loss_weight={'cls': 1.0, 'ctr': 1.0, 'reg': 1.0},
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 train_cfg=None,
                 test_cfg=None):
        super(OTSSHead, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.IoUtype = IoUtype
        self.reg_norm = reg_norm
        self.ctr_on_reg = ctr_on_reg
        self.use_centerness = use_centerness
        self.soft_label = soft_label
        self.loss_weight = loss_weight
        self.use_dcn_in_tower = use_dcn_in_tower
        self.loss_centerness = build_loss(loss_centerness)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.use_dcn_in_tower and i == self.stacked_convs - 1:
                self.cls_convs.append(
                    ModulatedDeformConvPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1))
                if self.norm_cfg:
                    self.cls_convs.append(
                        build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.cls_convs.append(nn.ReLU(inplace=True))
                self.reg_convs.append(
                    ModulatedDeformConvPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1))
                if self.norm_cfg:
                    self.reg_convs.append(
                        build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.reg_convs.append(nn.ReLU(inplace=True))
            else:
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
            if isinstance(m, ConvModule):
                normal_init(m.conv, std=0.01)
            elif isinstance(m, ModulatedDeformConvPack):
                normal_init(m, std=0.01)
        for m in self.reg_convs:
            if isinstance(m, ConvModule):
                normal_init(m.conv, std=0.01)
            elif isinstance(m, ModulatedDeformConvPack):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        if self.ctr_on_reg:
            centerness = self.fcos_centerness(reg_feat)
        else:
            centerness = self.fcos_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        if self.reg_norm:
            bbox_pred = F.relu(scale(self.fcos_reg(reg_feat)).float()) * stride
        else:
            bbox_pred = scale(self.fcos_reg(reg_feat)).float().exp()
        return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)

        # TODO inside bbox; soft weighted; dynamic threshold
        self.topk = 9
        candidate_idxs = []  # img * [lvl * idxs(9*nums_gt)]
        for gt_b in gt_bboxes:
            candidate_idxs_img = []
            gt_cx = (gt_b[:, 0] + gt_b[:, 2]) / 2.0
            gt_cy = (gt_b[:, 1] + gt_b[:, 3]) / 2.0
            gt_points = torch.stack((gt_cx, gt_cy), dim=1)
            for lvl_points in all_level_points:
                distances = (lvl_points[:, None, :] -
                             gt_points[None, :, :]).pow(2).sum(-1).sqrt()
                _, topk_idxs = distances.topk(self.topk, dim=0, largest=False)
                candidate_idxs_img.append(topk_idxs)
            candidate_idxs_img = torch.cat(
                [idxs[None, ...] for idxs in candidate_idxs_img])
            candidate_idxs.append(candidate_idxs_img)

        loss_bbox = 0
        num_pos = 0
        cls_targets = [torch.zeros_like(cls_score) for cls_score in cls_scores]

        num_imgs = cls_scores[0].size(0)
        if self.use_centerness:
            ctr_target_lst = []
            ctr_pred_lst = []
        for img_id in range(num_imgs):
            cls_lst = []
            de_bbox_lst = []
            centerpoint_lst = []
            ctrness_lst = []
            for lvl_id in range(len(self.strides)):
                flatten_cls_img_lvl = cls_scores[lvl_id][img_id].permute(
                    1, 2, 0).reshape(-1, self.cls_out_channels)
                flatten_bbox_img_lvl = bbox_preds[lvl_id][img_id].permute(
                    1, 2, 0).reshape(-1, 4)
                flatten_ctrness_img_lvl = centernesses[lvl_id][img_id].permute(
                    1, 2, 0).reshape(-1, 1)
                candidate_cls_img_lvl = flatten_cls_img_lvl[
                    candidate_idxs[img_id][lvl_id], :]
                candidate_bbox_img_lvl = flatten_bbox_img_lvl[
                    candidate_idxs[img_id][lvl_id], :]
                candidate_ctrness_img_lvl = flatten_ctrness_img_lvl[
                    candidate_idxs[img_id][lvl_id], :]
                candidate_points_img_lvl = all_level_points[lvl_id][
                    candidate_idxs[img_id][lvl_id], :]
                candidate_decoded_bbox_img_lvl = distance2bbox(
                    candidate_points_img_lvl.reshape(-1, 2),
                    candidate_bbox_img_lvl.reshape(-1, 4)).reshape(
                        self.topk, -1, 4)
                cls_lst.append(candidate_cls_img_lvl)
                de_bbox_lst.append(candidate_decoded_bbox_img_lvl)
                centerpoint_lst.append(candidate_points_img_lvl)
                ctrness_lst.append(candidate_ctrness_img_lvl)
            cls_lst_gt = [
                torch.cat([
                    cls_lst[lvl_id][:, gt_id, gt_labels[img_id][gt_id] - 1]
                    for lvl_id in range(len(self.strides))
                ], dim=0) for gt_id in range(len(gt_bboxes[img_id]))
            ]
            de_bbox_gt = [
                torch.cat([
                    de_bbox_lst[lvl_id][:, gt_id, :]
                    for lvl_id in range(len(self.strides))
                ], dim=0) for gt_id in range(len(gt_bboxes[img_id]))
            ]
            centerpoint_gt = [
                torch.cat([
                    centerpoint_lst[lvl_id][:, gt_id, :]
                    for lvl_id in range(len(self.strides))
                ], dim=0) for gt_id in range(len(gt_bboxes[img_id]))
            ]
            ctrness_gt = [
                torch.cat([
                    ctrness_lst[lvl_id][:, gt_id, :]
                    for lvl_id in range(len(self.strides))
                ], dim=0) for gt_id in range(len(gt_bboxes[img_id]))
            ]
            cls_lst_gt = torch.cat(
                [cls_gt[None, ...] for cls_gt in cls_lst_gt])
            de_bbox_gt = torch.cat([db_gt[None, ...] for db_gt in de_bbox_gt])
            centerpoint_gt = torch.cat(
                [cp_gt[None, ...] for cp_gt in centerpoint_gt])
            ctrness_gt = torch.cat(
                [cn_gt[None, ...] for cn_gt in ctrness_gt]).squeeze(dim=2)
            gt_bboxes_img = gt_bboxes[img_id][:, None, :].repeat(
                1,
                len(self.strides) * self.topk, 1)
            from mmdet.core import bbox_overlaps
            if self.IoUtype == 'IoU':
                de_bbox_gt = bbox_overlaps(
                    de_bbox_gt.reshape(-1, 4),
                    gt_bboxes_img.reshape(-1, 4),
                    is_aligned=True).clamp(min=1e-6).reshape(
                        -1,
                        len(self.strides) * self.topk)
            elif self.IoUtype == 'DIoU':
                _de_bbox_gt = self.DIoU(
                    de_bbox_gt.reshape(-1, 4),
                    gt_bboxes_img.reshape(-1, 4)).reshape(
                        -1,
                        len(self.strides) * self.topk)
                de_bbox_gt = _de_bbox_gt.clamp(min=1e-6)
            elif self.IoUtype == 'GIoU':
                _de_bbox_gt = self.GIoU(
                    de_bbox_gt.reshape(-1, 4),
                    gt_bboxes_img.reshape(-1, 4)).reshape(
                        -1,
                        len(self.strides) * self.topk)
                de_bbox_gt = _de_bbox_gt.clamp(min=1e-6)
            elif self.IoUtype == 'CIoU':
                _de_bbox_gt = self.CIoU(
                    de_bbox_gt.reshape(-1, 4),
                    gt_bboxes_img.reshape(-1, 4)).reshape(
                        -1,
                        len(self.strides) * self.topk)
                de_bbox_gt = _de_bbox_gt.clamp(min=1e-6)
            else:
                raise NotImplementedError

            with torch.no_grad():
                scores = de_bbox_gt * cls_lst_gt.sigmoid()
                threshold = (scores.mean(dim=1) + scores.std(dim=1))
                threshold = threshold[:, None].repeat(
                    1,
                    len(self.strides) * self.topk)
                keep_idxmask = (scores >= threshold)
            if self.use_centerness:
                inside_gt_bbox_mask = (
                    (centerpoint_gt[..., 0] > gt_bboxes_img[..., 0]) *
                    (centerpoint_gt[..., 0] < gt_bboxes_img[..., 2]) *
                    (centerpoint_gt[..., 1] > gt_bboxes_img[..., 1]) *
                    (centerpoint_gt[..., 1] < gt_bboxes_img[..., 3]))
                keep_idxmask *= inside_gt_bbox_mask
                center_p = centerpoint_gt.view(-1, 2)
                gt = gt_bboxes_img.view(-1, 4)
                left = center_p[:, 0] - gt[:, 0]
                right = gt[:, 2] - center_p[:, 0]
                up = center_p[:, 1] - gt[:, 1]
                down = gt[:, 3] - center_p[:, 1]
                l_r = torch.stack((left, right)).clamp(min=1e-6)
                u_d = torch.stack((up, down)).clamp(min=1e-6)
                centerness = ((l_r.min(dim=0)[0] * u_d.min(dim=0)[0]) /
                              (l_r.max(dim=0)[0] * u_d.max(dim=0)[0])).sqrt()
                centerness = centerness.reshape(
                    -1,
                    len(self.strides) * self.topk)[keep_idxmask]
                reweight_factor = centerness
                num_pos += reweight_factor.sum()
                ctrness_pred = ctrness_gt[keep_idxmask]
                ctr_target_lst.append(centerness)
                ctr_pred_lst.append(ctrness_pred)
            else:
                reweight_factor = 1
                num_pos += keep_idxmask.sum()

            if self.IoUtype == 'IoU':
                loss_bbox -= (de_bbox_gt[keep_idxmask].log() *
                              reweight_factor).sum()
            elif self.IoUtype in ['DIoU', 'GIoU', 'CIoU']:
                loss_bbox += ((1-_de_bbox_gt[keep_idxmask]) *
                              reweight_factor).sum()
            else:
                raise NotImplementedError
            # import pdb; pdb.set_trace()

            # cls
            if self.soft_label:
                with torch.no_grad():
                    soft_label = de_bbox_gt/de_bbox_gt.max(
                        dim=1)[0][:, None].repeat(
                            1,
                            len(self.strides) * self.topk).detach()
                    soft_label = soft_label.permute(1, 0).reshape(
                        len(self.strides), self.topk, -1)
            keep_idxmask = keep_idxmask.permute(1, 0).reshape(
                len(self.strides), self.topk, -1)
            for lvl_id in range(len(self.strides)):
                keep_idxmask_lvl = keep_idxmask[lvl_id]
                candidate_idxs_lvl = candidate_idxs[img_id][lvl_id][
                    keep_idxmask_lvl]
                gt_lables_lvl = gt_labels[img_id][None, :].repeat(
                    self.topk, 1)[keep_idxmask_lvl] - 1
                if self.soft_label:
                    soft_label_lvl = soft_label[lvl_id][keep_idxmask_lvl]
                    label_lvl = soft_label_lvl
                else:
                    label_lvl = 1
                cls_targets[lvl_id][img_id].view(
                    self.cls_out_channels,
                    -1)[gt_lables_lvl,
                        candidate_idxs_lvl] = label_lvl
        if self.use_centerness:
            loss_centerness = self.loss_centerness(
                torch.cat(ctr_pred_lst), torch.cat(ctr_target_lst))
        loss_bbox /= num_pos
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_cls_targets = [
            cls_target.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_target in cls_targets
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_cls_targets = torch.cat(flatten_cls_targets)
        # loss_cls = self.loss_cls(
        #     flatten_cls_scores, flatten_cls_targets,
        #     avg_factor=num_pos + num_imgs)  # avoid num_pos is 0
        from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss
        loss_cls = py_sigmoid_focal_loss(
            flatten_cls_scores,
            flatten_cls_targets,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0
        if self.use_centerness:
            return_dict = dict(loss_cls=loss_cls * self.loss_weight['cls'],
                               loss_bbox=loss_bbox * self.loss_weight['reg'],
                               loss_centerness=loss_centerness
                               * self.loss_weight['ctr'])
        else:
            return_dict = dict(loss_cls=loss_cls * self.loss_weight['cls'],
                               loss_bbox=loss_bbox * self.loss_weight['reg'])

        return return_dict

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
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
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 centerness_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
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
                if self.use_centerness:
                    max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                else:
                    max_scores, _ = scores.max(dim=1)
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
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes, mlvl_scores,
            cfg.score_thr, cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness if self.use_centerness else None)
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
                self._get_points_single(featmap_sizes[i], self.strides[i],
                                        dtype, device))
        return mlvl_points

    def _get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        # modify nan to 0
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        centerness_targets = centerness_targets.clamp(min=0)
        return torch.sqrt(centerness_targets)

    def DIoU(self, pred, target, rescale=False, eps=1e-7):
        # overlap
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]

        # union
        ap = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
        ag = (target[:, 2] - target[:, 0] + 1) * (
            target[:, 3] - target[:, 1] + 1)
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union

        # enclose diag
        enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)
        enclose_diag = (enclose_wh[:, 0].pow(2) +
                        enclose_wh[:, 1].pow(2)).sqrt()

        # center distance
        xp = (pred[:, 0] + pred[:, 2]) / 2
        yp = (pred[:, 1] + pred[:, 3]) / 2
        xg = (target[:, 0] + target[:, 2]) / 2
        yg = (target[:, 1] + target[:, 3]) / 2
        center_d = ((xp - xg).pow(2) + (yp - yg).pow(2)).sqrt()

        # DIoU
        dious = ious - center_d / enclose_diag

        if rescale:
            dious = (dious + 1) / 2

        return dious

    def GIoU(self, pred, target, rescale=False, eps=1e-7):
        # overlap
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]

        # union
        ap = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
        ag = (target[:, 2] - target[:, 0] + 1) * (
            target[:, 3] - target[:, 1] + 1)
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union

        # enclose area
        enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps

        # GIoU
        gious = ious - (enclose_area - union) / enclose_area

        if rescale:
            gious = (gious + 1) / 2

        return gious

    def CIoU(self, pred, target, eps=1e-7):
        # overlap
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]

        # union
        ap = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
        ag = (target[:, 2] - target[:, 0] + 1) * (
            target[:, 3] - target[:, 1] + 1)
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union

        # enclose diag
        enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)
        enclose_diag = (enclose_wh[:, 0].pow(2) +
                        enclose_wh[:, 1].pow(2) + eps)

        # center distance
        xp = (pred[:, 0] + pred[:, 2]) / 2
        yp = (pred[:, 1] + pred[:, 3]) / 2
        xg = (target[:, 0] + target[:, 2]) / 2
        yg = (target[:, 1] + target[:, 3]) / 2
        center_d = (xp - xg).pow(2) + (yp - yg).pow(2)

        # DIoU
        dious = ious - center_d / enclose_diag

        # CIoU
        w_gt = target[:, 2] - target[:, 0] + 1
        h_gt = target[:, 3] - target[:, 1] + 1
        w_pred = pred[:, 2] - pred[:, 0] + 1
        h_pred = pred[:, 3] - pred[:, 1] + 1
        v = (4 / math.pi**2) * torch.pow(
            (torch.atan(w_gt/h_gt) - torch.atan(w_pred/h_pred)), 2)
        with torch.no_grad():
            S = 1 - ious
            alpha = v / (S + v)
        cious = dious - alpha * v

        return cious
