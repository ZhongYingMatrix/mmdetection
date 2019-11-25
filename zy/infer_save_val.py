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
import pycocotools.mask as maskUtils
from mmdet.ops.nms.nms_wrapper import nms

dir_name = 'val_11_24_crop_aug'
#config_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_11_06/test_htc_dconv_c5_r101_fpn_sbn_10e_retrain_renew_set_with_total.py'
config_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_11_24/test_htc_dconv_c5_r101_fpn_sbn_20e_total.py'
checkpoint_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_11_24/epoch_100.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

def detect_img(model, file_path):
	img = mmcv.imread(file_path)
	result = inference_detector(model, img)
	#import pdb
	#pdb.set_trace()
	crop_bbox, crop_segm = np.zeros((0,5),dtype=np.float32), []
	for left, up in [(0, 0), (256, 0), (512, 0),
					(0, 256), (256, 256), (512, 256),
					(0, 512), (256, 512), (512, 512)]:
		img_crop = mmcv.imcrop(img, np.array([left, up, left+512, up+512]))
		img_crop = mmcv.imresize(img_crop, (1024, 1024))
		result_crop = inference_detector(model, img_crop)

		bbox, segm = result_crop
		bbox, segm = bbox[0], segm[0]

		edge_mask = ((bbox[:,0]>10)|(left==0)) & ((bbox[:,1]>10)|(up==0)) & ((bbox[:,2]<1014)|(left==512)) & ((bbox[:,3]<1014)|(up==512))
		bbox, segm = bbox[edge_mask,:], [segm[i] for i in np.where(edge_mask)[0]]

		segm_new = []
		for mask in segm:
			mask = maskUtils.decode(mask)
			mask_resize = mmcv.imresize(mask,(512,512))
			canvas = np.zeros((1024, 1024))
			canvas[up:(up+512), left:(left+512)] += mask_resize
			mask_encode = maskUtils.encode(np.asfortranarray(canvas.astype(np.uint8)))
			segm_new.append(mask_encode)
		bbox_new = bbox/2
		bbox_new[:,4] *= 2
		bbox_new[:,[0,2]]+=left
		bbox_new[:,[1,3]]+=up

		crop_bbox = np.concatenate((crop_bbox, bbox_new))
		crop_segm += segm_new

	crop_bbox = np.concatenate((crop_bbox, result[0][0]))
	crop_segm += result[1][0]

	crop_bbox, nms_ids = nms(crop_bbox,0.5)
	crop_segm = [crop_segm[i] for i in nms_ids]

	# contours=[]
	# for mask in crop_segm:
	#     mask = maskUtils.decode(mask)
	#     contour,_ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#     contours.append(list(np.array(contour[0]).reshape(-1))) 

	# scores = list(crop_bbox[:,-1].astype(float))
	# bboxes = list(crop_bbox[:,:-1].astype(int))

	# content = [{'segmentation':contours[i], 'bbox':list(bboxes[i]), 'scores': scores[i]} for i in range(len(bboxes))]
	content = ([crop_bbox],[crop_segm])
	return content

save_all_val = False
if save_all_val:
	coco = COCO('/home/zhongying/dataset/WalkValidationData_old/correct_validation_instance.json')
	for id in coco.getImgIds():
		path = '/home/zhongying/dataset/' + coco.loadImgs(id)[0]['file_name']
		img_name = coco.loadImgs(id)[0]['file_name'].split('/')[-1]
		img = mmcv.imread(path)
		result = inference_detector(model, img)
		mmcv.imwrite(show_result(img, result, [''], score_thr=0.001, show=False), './zy/img/'+dir_name+'_all/'+ img_name + '.jpg')
else:
	img_dir = '/home/zhongying/dataset/WalkValidationData_old/ValidDataCuoJianLouJian'
	for img_name in os.listdir(img_dir):
		img = '/home/zhongying/dataset/WalkValidationData_old/ValidDataCuoJianLouJian/' + img_name
		#img = mmcv.imread(img)

		result = detect_img(model, img)
		#import pdb
		#pdb.set_trace()

		mmcv.imwrite(show_result(img, result, [''], score_thr=0.001, show=False), './zy/img/'+dir_name+'/'+ img_name + '.jpg')

