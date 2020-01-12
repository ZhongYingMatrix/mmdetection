import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob, build_norm_layer
from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
import cv2
import numpy as np

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
                 scale_factor=0.2,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_mask=dict(type='DiceLoss'),
                 loss_factor={'loss_cls':1., 'loss_mask':3.},
                 debug=False):
        super(SOLO_Head, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.grid_number = grid_number
        self.instance_scale = instance_scale
        self.scale_factor = scale_factor
        self.loss_factor = loss_factor
        self.debug = debug
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.fp16_enabled = False

        self.grid_assign = self.grid2pred(self.grid_number)

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()
        # self.cls_out_lvls_conv = nn.ModuleList()
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
        # self.mask_2x_convs = ConvModule(
        #         self.feat_channels,
        #         self.feat_channels,
        #         3,
        #         stride=1,
        #         padding=1)

        for i in range(len(self.strides)):
            # self.cls_out_lvls_conv.append(
            #     nn.Conv2d(
            #         self.feat_channels,
            #         self.cls_out_channels, 
            #         1, 
            #         padding=0))
            self.mask_out_lvls_conv.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.grid_number[i] * 2, # Decoupled SOLO 
                    1, 
                    padding=0))
            self.cls_lvls_pooling.append(
                nn.AdaptiveAvgPool2d(self.grid_number[i])
            )

        self.cls_out_conv = nn.Conv2d(
                    self.feat_channels,
                    self.cls_out_channels, 
                    3, 
                    padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.mask_convs:
            normal_init(m.conv, std=0.01)
        # normal_init(self.mask_2x_convs.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        # for m in self.cls_out_lvls_conv:
        #     normal_init(m, std=0.01, bias=bias_cls)
        for m in self.mask_out_lvls_conv:
            normal_init(m, std=0.01)

        normal_init(self.cls_out_conv, std=0.01)

    def forward(self, feats):
        return multi_apply(
            self.forward_single, 
            feats, 
            self.cls_lvls_pooling,
            # self.cls_out_lvls_conv, 
            self.mask_out_lvls_conv,
            self.grid_number
        )

    def forward_single(self, 
                       x, 
                       cls_pooling, 
                       # cls_out_conv, 
                       mask_out_conv, 
                       grid_num):
        cls_feat = cls_pooling(x)
        mask_feat = self.coord_conv_cat(x)

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.cls_out_conv(cls_feat)

        for mask_layer in self.mask_convs:
            mask_feat = mask_layer(mask_feat)
        mask_feat = F.interpolate(mask_feat, scale_factor=2, mode='nearest')
        # mask_feat = self.mask_2x_convs(mask_feat)
        mask_pred = mask_out_conv(mask_feat)
        
        return cls_score, mask_pred

    @force_fp32(apply_to=('cls_scores', 'mask_preds'))
    def loss(self,
             cls_scores,
             mask_preds,
             gt_bboxes,
             gt_masks,
             gt_labels,
             img_metas,
             cfg):
        assert len(cls_scores)==len(mask_preds)
        # DEBUG
        self.img_metas = img_metas

        # all_level_points = self.get_points(mask_preds[0].size()[-2:], mask_preds[0].dtype,
        #                                    mask_preds[0].device)
        all_level_grids = self.get_grids(mask_preds[0].size()[-2:], mask_preds[0].dtype,
                                           mask_preds[0].device)
        # labels, gt_ids = self.solo_target(all_level_points, gt_bboxes, 
        #     gt_masks, gt_labels)
        labels, gt_ids = self.solo_target(all_level_grids, gt_bboxes, 
            gt_masks, gt_labels)
        # DEBUG
        # tmp = [all_level_points, labels, gt_ids, img_metas, gt_bboxes, gt_masks, gt_labels]
        # torch.save(tmp, 'demo/tmp/solo_positive.pth')
        # import pdb; pdb.set_trace()

        which_img, new_gt_masks = self.prepare_mask(gt_masks,
                                                    cls_scores[0].device,
                                                    mask_preds[0].shape)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_labels = torch.cat(labels)
        flatten_gt_ids = torch.cat(gt_ids)

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_gt_ids = flatten_gt_ids[pos_inds]
        which_img = which_img[pos_gt_ids.long()-1]
        
        # get pos grid assign 
        flatten_grid_assign = []
        for grid_assign in self.grid_assign:
            flatten_grid_assign += grid_assign*mask_preds[0].size(0)
        flatten_grid_assign = torch.tensor(flatten_grid_assign,
            device=mask_preds[0].device)
        pos_grid_assign = flatten_grid_assign[pos_inds]
        
        loss_mask = 0
        if self.debug: loss_mask_bce = 0
        for grid_assign, gt_ids, img_id in zip(pos_grid_assign, pos_gt_ids, which_img):
            lvl, grid_x, grid_y = grid_assign
            mask_target = new_gt_masks[lvl][gt_ids-1]
            mask_pred = mask_preds[lvl][img_id][grid_x].sigmoid() * \
                mask_preds[lvl][img_id][grid_y].sigmoid()
            if not self.debug:
                loss_mask += self.loss_mask(mask_pred, mask_target)
            else:
                loss_mask += self.loss_mask(mask_pred, mask_target)
                loss_mask_bce += F.binary_cross_entropy(mask_pred, mask_target.float())
        loss_mask /= pos_gt_ids.shape[0]

        
        if self.debug:
            loss_mask_bce /= pos_gt_ids.shape[0]
            return dict(
                loss_cls=loss_cls * self.loss_factor['loss_cls'],
                loss_mask=loss_mask * self.loss_factor['loss_mask'],
                loss_mask_bce=loss_mask_bce * self.loss_factor['loss_mask']
                )
        return dict(
            loss_cls=loss_cls * self.loss_factor['loss_cls'],
            loss_mask=loss_mask * self.loss_factor['loss_mask']
            )

    def grid2pred(self, grid_number):
        mlvl_grid_assign = []
        for lvl, grid_num in enumerate(self.grid_number):
            grid_assign = []
            for i in range(grid_num):
                grid_assign += [[lvl, i,j+grid_num] for j in range(grid_num)]
            mlvl_grid_assign.append(grid_assign)
        return mlvl_grid_assign

    def get_points(self, p2_shape, dtype, device):
        h, w = p2_shape
        h, w = h*4, w*4 # img shape
        mlvl_points = []
        for grid_num in self.grid_number:
            mlvl_points.append(
                self.get_points_single(grid_num, h, w,
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, grid_num, h, w, dtype, device):
        x_range = torch.arange(
            0, w, w/grid_num, dtype=dtype, device=device) + w/grid_num/2
        y_range = torch.arange(
            0, h, h/grid_num, dtype=dtype, device=device) + h/grid_num/2
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1)
        return points

    def get_grids(self, p2_shape, dtype, device):
        h, w = p2_shape
        h, w = h*4, w*4 # img shape
        mlvl_grids = []
        for grid_num in self.grid_number:
            mlvl_grids.append(
                self.get_grids_single(grid_num, h, w,
                                       dtype, device))
        return mlvl_grids

    def get_grids_single(self, grid_num, h, w, dtype, device):
        x_range = torch.arange(
            0, w, w/grid_num, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h, h/grid_num, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        x2_range = torch.arange(
            0, w, w/grid_num, dtype=dtype, device=device) + w/grid_num
        y2_range = torch.arange(
            0, h, h/grid_num, dtype=dtype, device=device) + h/grid_num
        y2, x2 = torch.meshgrid(y2_range, x2_range)
        grids = torch.stack(
            (x.reshape(-1), y.reshape(-1), x2.reshape(-1), y2.reshape(-1)), dim=-1)
        return grids


    def coord_conv_cat(self, feat):
        h, w = feat.shape[-2], feat.shape[-1]
        dtype, device = feat.dtype, feat.device
        x_range = torch.arange(-1, 1, 2/h, dtype=dtype, device=device)
        y_range = torch.arange(-1, 1, 2/w, dtype=dtype, device=device)
        x, y = torch.meshgrid(x_range, y_range)
        coord_feat = torch.stack((x,y)).repeat(feat.shape[0],1,1,1)

        return torch.cat((feat, coord_feat), dim=1)

    def solo_target(self, points, gt_bboxes_list, gt_masks_list, gt_labels_list):
        assert len(points) == len(self.instance_scale)
        num_levels = len(points)
        # expand regress ranges to align with points
        # expanded_scale_ranges = [
        #     points[i].new_tensor(self.instance_scale[i])[None].expand_as(
        #         points[i]) for i in range(num_levels)
        # ]
        expanded_scale_ranges = [
            points[i].new_tensor(self.instance_scale[i])[None].expand_as(
                points[i][:,:2]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_scale_ranges = torch.cat(expanded_scale_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # get labels and bbox_targets of each image
        labels_list, ids_list = multi_apply(
            self.solo_target_single,
            gt_bboxes_list,
            gt_masks_list,
            gt_labels_list,
            points=concat_points,
            scale_ranges=concat_scale_ranges)

        # accumulate img ids for mask assign
        for img_id in range(1, len(ids_list)):
            ids_list[img_id][ids_list[img_id]!=0] += ids_list[img_id-1].max()

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        ids_list = [ids.split(num_points, 0) for ids in ids_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_ids = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_ids.append(
                torch.cat(
                    [ids[i] for ids in ids_list]))

        return concat_lvl_labels, concat_lvl_ids
        
    def solo_target_single(self, gt_bboxes, gt_masks, gt_labels, points, scale_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))
        
        # caculate mass center
        mask_centers = []
        for i, mask in enumerate(gt_masks):
            M = cv2.moments(mask)
            try:
                x, y = M['m10']/M['m00'], M['m01']/M['m00'] 
            except ZeroDivisionError:
                x, y = ((gt_bboxes[i,0]+gt_bboxes[i,2])/2).tolist() , \
                    ((gt_bboxes[i,1]+gt_bboxes[i,3])/2).tolist()
                print('img_metas:', self.img_metas, '\n', 'gt_bboxes:', gt_bboxes[i])
            
            mask_centers.append([x,y])
        mask_centers = torch.Tensor(mask_centers).float().to(gt_bboxes.device)
        mask_centers = mask_centers[None].expand(num_points, num_gts, 2)

        # area for ambiguous sample
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        areas = areas[None].repeat(num_points, 1)
        scale_ranges = scale_ranges[:, None, :].expand(
            num_points, num_gts, 2)    
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)

        # long edge as scale × wrong 
        # sqrt of long and short
        w = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        h = gt_bboxes[..., 3] - gt_bboxes[..., 1]
        scales = (w*h).sqrt()
        inside_scale_range = (
            scales >= scale_ranges[..., 0]) & (
            scales <= scale_ranges[..., 1])


        # center region
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        x2s, y2s = points[:, 2], points[:, 3]
        x2s = x2s[:, None].expand(num_points, num_gts)
        y2s = y2s[:, None].expand(num_points, num_gts)
        left =  mask_centers[..., 0] \
            - (mask_centers[..., 0] - gt_bboxes[..., 0]) * self.scale_factor
        right =  mask_centers[..., 0] \
            + (gt_bboxes[..., 2] - mask_centers[..., 0]) * self.scale_factor
        top =  mask_centers[..., 1] \
            - (mask_centers[..., 1] - gt_bboxes[..., 1]) * self.scale_factor
        bottom =  mask_centers[..., 1] \
            + (gt_bboxes[..., 3] - mask_centers[..., 1]) * self.scale_factor
        #center_region = torch.stack((left, top, right, bottom), -1)
        # inside_center_region = (x2s > left) & (xs < right) & (ys > top) & (ys < bottom)
        inside_center_region = ((right+left-x2s-xs).abs() <= x2s-xs + right-left) & \
            ((bottom+top-y2s-ys).abs() <= y2s-ys + bottom-top)
        

        areas[inside_center_region == 0] = INF
        areas[inside_scale_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0         
        ids = min_area_inds + 1
        ids[min_area == INF] = 0

        return labels, ids

    def prepare_mask(self, gt_masks, device, p2_shape):
        which_img = []
        for img_num, _gt_masks in enumerate(gt_masks):
            num = _gt_masks.shape[0]
            which_img += [img_num]*num
        which_img = torch.tensor(which_img).to(device)
        new_gt_masks = [[] for i in range(5)] # multi level
        for i in range(5):
            for _gt_masks in gt_masks:  
                _gt_masks = [cv2.resize(gt_mask, (0,0), \
                     fx=1./2**(i+2), fy=1./2**(i+2)) for gt_mask in _gt_masks]
                _gt_masks = [np.pad(gt_mask,
                             ((0, int(p2_shape[2]/2**i) - gt_mask.shape[0]), 
                             (0, int(p2_shape[3]/2**i) - gt_mask.shape[1])))
                             for gt_mask in _gt_masks]
                _gt_masks = np.stack(_gt_masks)
                _gt_masks = torch.from_numpy(_gt_masks).to(device)    
                new_gt_masks[i].append(_gt_masks)
            new_gt_masks[i] = torch.cat(new_gt_masks[i])
        return which_img, new_gt_masks

    @force_fp32(apply_to=('cls_scores', 'mask_preds'))
    def get_masks(self,
                  cls_scores,
                  mask_preds,
                  img_metas,
                  cfg,
                  rescale=None):
        assert len(cls_scores) == len(mask_preds)
        # reshape to same shape
        _, _, b_h, b_w = mask_preds[0].shape
        for i in range(len(mask_preds) ):
            mask_preds[i] = F.upsample_bilinear(mask_preds[i],(b_h,b_w))

        num_levels = len(cls_scores)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            mask_pred_list = [
                mask_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']
            det_results = self.get_masks_single(cls_score_list, mask_pred_list,
                                              img_shape,
                                              ori_shape,
                                              scale_factor, cfg, rescale)
            result_list.append(det_results)
        return result_list

    def get_masks_single(self,
                         cls_scores,
                         mask_preds,
                         img_shape,
                         ori_shape,
                         scale_factor,
                         cfg,
                         rescale=False):
        assert len(cls_scores) == len(mask_preds) == len(self.grid_assign)
        mlvl_scores = []
        mlvl_masks = []
        for cls_score, grid_assign_lvl, mask_pred in zip(
            cls_scores, self.grid_assign, mask_preds):
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            grid_assign_lvl = torch.tensor(grid_assign_lvl)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                scores = scores[topk_inds, :]
                grid_assign_lvl = grid_assign_lvl[topk_inds, :]
            # score threshold
            score_thr = cfg.get('score_thr', -1)
            if score_thr > 0:
                thr_mask = scores.max(dim=1)[0]>score_thr
                scores = scores[thr_mask,:]
                grid_assign_lvl = grid_assign_lvl[thr_mask]
            mask_pred = mask_pred.sigmoid()
            mask_lvl = mask_pred[grid_assign_lvl[:,1]] * \
                mask_pred[grid_assign_lvl[:,2]]
            mlvl_scores.append(scores)
            mlvl_masks.append(mask_lvl)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_masks = torch.cat(mlvl_masks)
        # reshape to original shape
        mlvl_masks = F.upsample_bilinear(
            mlvl_masks[None], (mlvl_masks.size(-2)*4, mlvl_masks.size(-1)*4))[0]
        mlvl_masks = mlvl_masks[:, :img_shape[0], :img_shape[1]]
        mlvl_masks = F.upsample_bilinear(
            mlvl_masks[None], (ori_shape[0], ori_shape[1]))[0]
        mlvl_masks = (mlvl_masks>0.5).int()
        # remv empty masks 
        mlvl_scores = mlvl_scores[mlvl_masks.sum(-1).sum(-1)>0]
        mlvl_masks = mlvl_masks[mlvl_masks.sum(-1).sum(-1)>0]

        det_labels, det_bboxes, det_masks = self.nms_test(
            mlvl_scores, mlvl_masks, cfg.nms.iou_thr)
        return det_labels, det_bboxes, det_masks

    def nms_test(self, scores, masks, iou_thr):
        scores, labels = scores.max(dim=1)
        scores = scores.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        masks = masks.cpu().detach().numpy().astype(np.uint8)
        det_labels, det_bboxes, det_masks = [], [], []
        n = len(labels)
        if n > 0:
            masks_dict = {}
            for i in range(n):
                if labels[i] in masks_dict:
                    masks_dict[labels[i]].append([masks[i],labels[i],scores[i]])
                else:
                    masks_dict[labels[i]] = [[masks[i],labels[i],scores[i]]]
            
            for masks in masks_dict.values():
                if len(masks) == 1:
                    det_masks.append(masks[0][0])
                    det_labels.append(masks[0][1])
                    det_bboxes.append(self.mask2bbox(masks[0][0], masks[0][2]))
                else:
                    while(len(masks)):
                        best_mask = masks.pop(0)
                        det_masks.append(best_mask[0])
                        det_labels.append(best_mask[1])
                        det_bboxes.append(self.mask2bbox(
                            best_mask[0], best_mask[2]))
                        j = 0
                        for i in range(len(masks)):
                            i -= j
                            if self.iou_calc(best_mask[0], masks[i][0]) > iou_thr:
                                masks.pop(i)
                                j += 1
        return det_labels, det_bboxes, det_masks

    def mask2bbox(self, mask, scores):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        x1, y1, x2, y2 = float('inf'), float('inf'), float('-inf'), float('-inf')
        for contour in contours:
            _x1, _y1, w, h = cv2.boundingRect(contour)
            _x2, _y2 = _x1+w, _y1+h
            x1, y1 = min(x1, _x1), min(y1, _y1)
            x2, y2 = max(x2, _x2), max(y2, _y2)
        return [x1, y1, x2, y2, scores]

    def iou_calc(self,mask1,mask2):
        m1 = mask1.astype(bool)
        m2 = mask2.astype(bool)
        overlap = m1*m2
        union = m1+m2
        iou = float(overlap.sum())+1/(float(union.sum())+1)
        return iou
            
        