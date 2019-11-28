from ..registry import PIPELINES
from mmcv.parallel import DataContainer as DC
import torch
import cv2
import numpy as np
from math import ceil

INF = 1e8

@PIPELINES.register_module
class PAPTargetOffline(object):
    """offline ray label generation
    """
    def __init__(self):
        self.radius = 1.5
        self.strides = [8, 16, 32, 64, 128]
        self.regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF))
    
    def __call__(self, results):
        # first remove small objects
        gt_masks = results['gt_masks'][:len(results['gt_bboxes'])]
        gt_bboxes = torch.Tensor(results['gt_bboxes'])
        gt_labels = torch.Tensor(results['gt_labels'])
        small_objects_mask = (gt_bboxes[:,2]-gt_bboxes[:,0]>10) & \
                             (gt_bboxes[:,3]-gt_bboxes[:,1]>10) & \
                             torch.from_numpy(gt_masks.sum(-1).sum(-1)>15)
        gt_bboxes = gt_bboxes[small_objects_mask]
        gt_labels = gt_labels[small_objects_mask]
        gt_masks = gt_masks[small_objects_mask]
        results['gt_bboxes'] = gt_bboxes.numpy()
        results['gt_labels'] = gt_labels.numpy().astype(np.int64)
        results['gt_masks'] = gt_masks

        featmap_sizes = self.get_featmap_size(results['pad_shape'])
        self.featmap_sizes = featmap_sizes
        num_levels = len(self.strides)
        all_level_points = self.get_points(featmap_sizes)
        self.num_points_per_level = [i.size()[0] for i in all_level_points]
        expanded_regress_ranges = [
            all_level_points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                all_level_points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(all_level_points, 0)

        _labels, _ids, _polarcontours = self.polar_target_single(
            gt_bboxes,gt_masks,gt_labels,concat_points, concat_regress_ranges)

        results['_gt_labels'] = DC(_labels)
        results['_gt_ids'] = DC(_ids)
        results['_gt_polarcontours'] = DC(_polarcontours)
        return results

    def get_featmap_size(self, shape):
        h,w = shape[:2]
        featmap_sizes = []
        for i in self.strides:
            featmap_sizes.append([ceil(h / i), ceil(w / i)])
        return featmap_sizes

    def get_points(self, featmap_sizes):
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i]))
        return mlvl_points

    def get_points_single(self, featmap_size, stride):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride)
        y_range = torch.arange(
            0, h * stride, stride)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points.float()

    def polar_target_single(self, gt_bboxes, gt_masks, gt_labels, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), gt_labels.new_zeros(num_points), \
                gt_labels.new_zeros((num_points, 36))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
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

        mask_centers = []
        mask_contours = []
        for mask in gt_masks:
            '''debug IndexError: list index out of range'''
            cnt, contour = self.get_single_centerpoint(mask)

            contour = contour[0]
            contour = torch.Tensor(contour).float()
            y, x = cnt
            mask_centers.append([x,y])
            mask_contours.append(contour)
        mask_centers = torch.Tensor(mask_centers).float()
        mask_centers = mask_centers[None].expand(num_points, num_gts, 2)

        inside_gt_bbox_mask = self.get_mask_sample_region(gt_bboxes,
                                                             mask_centers,
                                                             self.strides,
                                                             self.num_points_per_level,
                                                             xs,
                                                             ys,
                                                             radius=self.radius)
                                                            
        max_regress_distance = bbox_targets.max(-1)[0]

        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
            max_regress_distance <= regress_ranges[..., 1])

        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0

        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        pos_inds = labels.nonzero().reshape(-1)

        polarcontours = torch.zeros(num_points, 36).float()

        pos_mask_ids = min_area_inds[pos_inds]
        ids=labels.new_zeros(labels.shape)
        for p,id in zip(pos_inds, pos_mask_ids):
            x, y = points[p]
            pos_mask_contour = mask_contours[id]

            dists, coords = self.get_36_coordinates(x, y, pos_mask_contour)
            polarcontours[p] = dists
            ids[p] = id + 1

        return labels, ids, polarcontours

    def get_mask_sample_region(self, gt_bb, mask_center, strides, num_points_per, gt_xs, gt_ys, radius=1):
        center_y = mask_center[..., 0]
        center_x = mask_center[..., 1]
        center_gt = gt_bb.new_zeros(gt_bb.shape)
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)

        beg = 0
        for level,n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            center_gt[beg:end, :, 0] = torch.where(xmin > gt_bb[beg:end, :, 0], xmin, gt_bb[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt_bb[beg:end, :, 1], ymin, gt_bb[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt_bb[beg:end, :, 2], gt_bb[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt_bb[beg:end, :, 3], gt_bb[beg:end, :, 3], ymax)
            beg = end

        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  
        return inside_gt_bbox_mask

    def get_centerpoint(self, lis):
        area = 0.0
        x, y = 0.0, 0.0
        a = len(lis)
        for i in range(a):
            lat = lis[i][0]
            lng = lis[i][1]
            if i == 0:
                lat1 = lis[-1][0]
                lng1 = lis[-1][1]
            else:
                lat1 = lis[i - 1][0]
                lng1 = lis[i - 1][1]
            fg = (lat * lng1 - lng * lat1) / 2.0
            area += fg
            x += fg * (lat + lat1) / 3.0
            y += fg * (lng + lng1) / 3.0
        x = x / area
        y = y / area

        return [int(x), int(y)]

    def get_single_centerpoint(self, mask):
        contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour.sort(key=lambda x: cv2.contourArea(x), reverse=True) 
        '''debug IndexError: list index out of range'''
        count = contour[0][:, 0, :]
        try:
            center = self.get_centerpoint(count)
        except:
            x,y = count.mean(axis=0)
            center=[int(x), int(y)]

        return center, contour

    def get_36_coordinates(self, c_x, c_y, pos_mask_contour):
        ct = pos_mask_contour[:, 0, :]
        x = ct[:, 0] - c_x
        y = ct[:, 1] - c_y
        angle = torch.atan2(x, y) * 180 / np.pi
        angle[angle < 0] += 360
        angle = angle.int()
        dist = torch.sqrt(x ** 2 + y ** 2)
        angle, idx = torch.sort(angle)
        dist = dist[idx]

        new_coordinate = {}
        for i in range(0, 360, 10):
            if i in angle:
                d = dist[angle==i].max()
                new_coordinate[i] = d
            elif i + 1 in angle:
                d = dist[angle == i+1].max()
                new_coordinate[i] = d
            elif i - 1 in angle:
                d = dist[angle == i-1].max()
                new_coordinate[i] = d
            elif i + 2 in angle:
                d = dist[angle == i+2].max()
                new_coordinate[i] = d
            elif i - 2 in angle:
                d = dist[angle == i-2].max()
                new_coordinate[i] = d
            elif i + 3 in angle:
                d = dist[angle == i+3].max()
                new_coordinate[i] = d
            elif i - 3 in angle:
                d = dist[angle == i-3].max()
                new_coordinate[i] = d


        distances = torch.zeros(36)

        for a in range(0, 360, 10):
            if not a in new_coordinate.keys():
                new_coordinate[a] = torch.tensor(1e-6)
                distances[a//10] = 1e-6
            else:
                distances[a//10] = new_coordinate[a]

        return distances, new_coordinate


    def __repr__(self):
        return self.__class__.__name__
