import numpy as np 
import cv2
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from pycocotools.coco import COCO
import random
import pycocotools.mask as maskUtils
import torch
from mmdet.utils.timer import Timer
import torch.nn.functional as F

tmp = torch.load('demo/tmp/solo_positive.pth')
all_level_points, labels, gt_ids, img_metas, gt_bboxes, gt_masks, gt_labels = tmp
import pdb; pdb.set_trace()

img1 = cv2.imread(img_metas[0]['filename'])
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
if img_metas[0]['flip']:
    img1 = cv2.flip(img1, 1)
ppp = []
for i in range(5):
    p = all_level_points[i]
    l = labels[i]
    pp = p[l.chunk(4, 0)[0]>0]
    ppp.append(pp)
ppp = torch.cat(ppp)
ppp /= img_metas[0]['scale_factor']
for p in ppp:
    cv2.circle(img1, tuple(p.int().tolist()), 2, (0,255,0), 2)
gb = gt_bboxes[0]/img_metas[0]['scale_factor']
for b in gb:
    x0, y0, x1, y1 = b.int().tolist()
    img1 = cv2.rectangle(img1, (x0, y0), (x1, y1), (255, 0, 0), 1)
plt.imshow(img1)

plt.show()