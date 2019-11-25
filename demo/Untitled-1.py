
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
#from mmdet.core import multiclass_nms
from mmdet.ops.nms.nms_wrapper import nms
import cv2

np.set_printoptions(suppress=True)

config_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_11_13/test_htc_dconv_c5_r101_fpn_sbn_20e_total.py'
checkpoint_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_11_13/epoch_100.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

img = '/home/zhongying/dataset/WalkValidationData/validation/22_019456_005120.tif'
img = mmcv.imread(img)
#img = np.ones((1024, 1024, 3), dtype=np.uint8)
#img[:,:,2] += 255
#img *=255
result = inference_detector(model, img)


import pdb
pdb.set_trace()



mmcv.imwrite(show_result(img, result, [''], score_thr=0.001, show=False), '/home/zhongying/other/walkgis/mmdetection/demo/tmp.jpg')
#mmcv.imwrite(show_result(img, crop_result, [''], score_thr=0.001, show=False), '/home/zhongying/other/walkgis/mmdetection/demo/tmp1.jpg')



