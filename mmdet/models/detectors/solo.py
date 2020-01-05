from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from .base import BaseDetector
from .. import builder
import torch.nn as nn
import numpy as np
import pycocotools.mask as mask_util


@DETECTORS.register_module
class SOLO(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SOLO, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.mask_head = builder.build_head(mask_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SOLO, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.mask_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_masks,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.mask_head(x)
        loss_inputs = outs + (gt_bboxes, gt_masks, gt_labels, img_metas, self.train_cfg)
        losses = self.mask_head.loss(*loss_inputs)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.mask_head(x)
        mask_inputs = outs + (img_meta, self.test_cfg, rescale)
        mask_list = self.mask_head.get_masks(*mask_inputs)
        results = [
            self.bbox_mask2result(det_bboxes, det_labels, det_masks, self.mask_head.num_classes)
            for det_labels, det_bboxes, det_masks in mask_list
        ]
        bbox_results = results[0][0]
        mask_results = results[0][1]

        return bbox_results, mask_results

    def bbox_mask2result(self, bboxes, labels, masks, num_classes):
        mask_results = [[] for _ in range(num_classes - 1)]
        bboxes = np.array(bboxes)
        labels = np.array(labels)

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
            bbox_results = [bboxes[labels == i, :] for i in range(num_classes - 1)]
            return bbox_results, mask_results

