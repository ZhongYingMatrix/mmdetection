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
img_num = 2
img_idx = 0
#import pdb; pdb.set_trace()

img = cv2.imread(img_metas[img_idx]['filename'])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
if img_metas[img_idx]['flip']:
    img = cv2.flip(img, 1)
ids_dict = {}
for i in range(5):
    p = all_level_points[i]
    l = labels[i]
    ids = gt_ids[i].chunk(img_num)[img_idx]
    ids = ids[l.chunk(img_num, 0)[img_idx]>0]
    pp = p[l.chunk(img_num, 0)[img_idx]>0]
    for _id, _p in zip(ids, pp):
        _id = _id.tolist()
        _p /= img_metas[img_idx]['scale_factor']
        _p = _p.tolist() 
        if _id not in ids_dict:
            ids_dict[_id] = [_p]
        else:
            ids_dict[_id].append(_p)

for _id in ids_dict:
    color = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
    color = tuple(color.tolist()[0])
    for _p in ids_dict[_id]: 
        cv2.circle(img, tuple(map(int, _p)), 1, color, 2)
    x0, y0, x1, y1 = (gt_bboxes[img_idx]/img_metas[img_idx]['scale_factor'])[_id-1].int().tolist()
    img = cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

plt.imshow(img)

plt.show()