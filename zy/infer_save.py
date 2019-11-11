# import os
# try:
# 	os.chdir(os.path.join(os.getcwd(), 'demo'))
# 	print(os.getcwd())
# except:
# 	pass

from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result
import mmcv
from matplotlib import pyplot as plt
import numpy as np

dir_name = 'retrain_with_total'
#config_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_10_24/htc_r101_fpn_20e_new_set_retrain.py'
#config_file = '/home/zhongying/other/walkgis/mmdetection/zy/walkgis/htc_r101_fpn_20e.py'
config_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_10_25/htc_r101_fpn_20e_new_set_retrain_with_total.py'

#checkpoint_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_10_24/epoch_20.pth'
#checkpoint_file = '/home/zhongying/other/walkgis/mmdetection/zy/walkgis/epoch_110.pth'
checkpoint_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_10_25/epoch_20.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# param = []
# def hook(module, input, output):
#     param.append(input)
#     param.append(output)
# handle = model.rpn_head.register_forward_hook(hook)


# test a single image
#img = 'demo.jpg'
img = '/home/zhongying/other/walkgis/mmdetection/zy/walkgis/ValidDataCuoJianLouJian/22_014336_019456.tif'
#img = '/home/zhongying/other/walkgis/mmdetection/zy/walkgis/ValidDataCuoJianLouJian/22_014336_018432.tif'
img = mmcv.imread(img)
#img = mmcv.imcrop(img, np.array([512,512,1024,1024]))
#img = mmcv.imresize(img, (1024, 1024))
result = inference_detector(model, img)

#bbox_result, segm_result = result
#bboxes = np.vstack(bbox_result)

#import pdb
#pdb.set_trace()

# show the results
#print(model.CLASSES)
#show_result_pyplot(img, result, model.CLASSES, score_thr=0.0001)
mmcv.imwrite(show_result(img, result, [''], score_thr=0.0001, show=False), './zy/img/'+dir_name+'/whole.jpg')
#img=show_result(img, result, [' '], score_thr=0.0001, show=False)
# import pdb
# pdb.set_trace()
for i, j in [(0,0), (256, 0),(512,0), (0,256), (256, 256),(512,256),(0,512), (256,512), (512,512)]:
	_img = mmcv.imcrop(img, np.array([i, j, i+512, j+512]))
	_result = inference_detector(model, _img)
	mmcv.imwrite(show_result(_img, _result, [''], score_thr=0.0001, show=False), './zy/img/'+dir_name+'/%d_%d.jpg'%(i,j) )

#plt.show()

#handle.remove()
