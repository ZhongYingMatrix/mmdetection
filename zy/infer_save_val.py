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

dir_name = 'val_11_7'
#config_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_11_06/test_htc_dconv_c5_r101_fpn_sbn_10e_retrain_renew_set_with_total.py'
config_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_10_25/htc_r101_fpn_20e_new_set_retrain_with_total.py'
checkpoint_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_10_25/epoch_20.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

img_dir = '/home/zhongying/dataset/WalkValidationData/ValidDataCuoJianLouJian'
for img_name in os.listdir(img_dir):
	img = '/home/zhongying/dataset/WalkValidationData/ValidDataCuoJianLouJian/' + img_name
	img = mmcv.imread(img)

	result = inference_detector(model, img)
	#import pdb
	#pdb.set_trace()

	mmcv.imwrite(show_result(img, result, [''], score_thr=0.001, show=False), './zy/img/'+dir_name+'/'+ img_name + '.jpg')
