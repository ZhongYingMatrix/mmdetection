from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn as nn
from mmdet.core import bbox_mask2result
from ..registry import DETECTORS
from .. import builder

@DETECTORS.register_module
class PAPMask(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 mask_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.mask_head = builder.build_head(mask_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.mask_head.init_weights()
        
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks=None,
                      _gt_labels=None,
                      _gt_ids=None,
                      _gt_polarcontours=None
                      ):

        if _gt_labels is not None:
            extra_data = dict(_gt_labels=_gt_labels,
                              _gt_ids=_gt_ids,
                              _gt_polarcontours=_gt_polarcontours)
        else:
            extra_data = None


        x = self.extract_feat(img)
        outs = self.mask_head(x)
        loss_inputs = outs + (gt_masks, extra_data, img_metas, self.train_cfg)

        import pdb
        pdb.set_trace()

        losses = self.mask_head.loss(*loss_inputs)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.mask_head(x)

        mask_inputs = outs + (img_meta, self.test_cfg, rescale)
        mask_list = self.mask_head.get_masks(*mask_inputs)

        results = [
            mask2result(det_masks, det_labels, self.mask_head.num_classes, img_meta[0])
            for det_labels, det_masks in mask_list]

        bbox_results = results[0][0]
        mask_results = results[0][1]

        return bbox_results, mask_results