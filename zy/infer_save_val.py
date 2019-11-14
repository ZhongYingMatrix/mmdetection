import os
# try:
# 	os.chdir(os.path.join(os.getcwd(), 'demo'))
# 	print(os.getcwd())
# except:
# 	pass

from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result
import mmcv
from matplotlib import pyplot as plt
import numpy as np
from pycocotools.coco import COCO

dir_name = 'val_11_13'
#config_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_11_06/test_htc_dconv_c5_r101_fpn_sbn_10e_retrain_renew_set_with_total.py'
config_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_11_13/test_htc_dconv_c5_r101_fpn_sbn_20e_total.py'
checkpoint_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_11_13/epoch_100.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

save_all_val = True
if save_all_val:
	coco = COCO('/home/zhongying/dataset/WalkValidationData/correct_validation_instance.json')
	for id in coco.getImgIds():
		path = '/home/zhongying/dataset/' + coco.loadImgs(id)[0]['file_name']
		img_name = coco.loadImgs(id)[0]['file_name'].split('/')[-1]
		img = mmcv.imread(path)
		result = inference_detector(model, img)
		mmcv.imwrite(show_result(img, result, [''], score_thr=0.001, show=False), './zy/img/'+dir_name+'_all/'+ img_name + '.jpg')
else:
	img_dir = '/home/zhongying/dataset/WalkValidationData/ValidDataCuoJianLouJian'
	for img_name in os.listdir(img_dir):
		img = '/home/zhongying/dataset/WalkValidationData/ValidDataCuoJianLouJian/' + img_name
		img = mmcv.imread(img)

		result = inference_detector(model, img)
		#import pdb
		#pdb.set_trace()

		mmcv.imwrite(show_result(img, result, [''], score_thr=0.001, show=False), './zy/img/'+dir_name+'/'+ img_name + '.jpg')

