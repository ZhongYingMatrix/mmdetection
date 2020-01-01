import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss

@LOSSES.register_module
class DiceLoss(nn.Module):
    """pred should have sigmoid
    """

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.

    def forward(self,
                pred,
                target,
                weight=None):
        flatten_pred = pred.reshape(-1)
        flatten_target = target.reshape(-1)
        if weight is not None:
            flatten_pred *= weight.reshape(-1)
            flatten_target *= weight.reshape(-1)
        loss_dice = 1 - ((flatten_pred*flatten_target).sum()*2+self.smooth)/ \
            (flatten_target.sum()+flatten_pred.sum()+self.smooth)
        return loss_dice