from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import bbox2result
import pycocotools.mask as mask_util
import numpy as np


@DETECTORS.register_module
class FCOS_PROTO(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOS_PROTO, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_masks,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_masks, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        mask_inputs = outs + (img_meta, self.test_cfg, rescale)
        mask_list = self.bbox_head.get_masks(*mask_inputs)
        results = [
            self.bbox_mask2result(det_bboxes, det_labels, det_masks, self.bbox_head.num_classes)
            for det_bboxes, det_labels, det_masks in mask_list
        ]
        bbox_results = results[0][0]
        mask_results = results[0][1]

        return bbox_results, mask_results

    def bbox_mask2result(self, bboxes, labels, masks, num_classes):
        mask_results = [[] for _ in range(num_classes - 1)]

        for i in range(bboxes.shape[0]):
            rle = mask_util.encode(
                np.array(masks[i][:, :, np.newaxis], order='F'))[0]

            label = labels[i]

            mask_results[label].append(rle)


        if bboxes.shape[0] == 0:
            bbox_results = [
                np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
            ]
            return bbox_results, mask_results
        else:
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            bbox_results = [bboxes[labels == i, :] for i in range(num_classes - 1)]
            return bbox_results, mask_results
