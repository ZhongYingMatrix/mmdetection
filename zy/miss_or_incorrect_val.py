import mmcv
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


root = '/home/zhongying/dataset/'
ann_file = '/home/zhongying/dataset/WalkValidationData/correct_validation_instance.json'
result_file = '/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_10_25/val_result.pkl.segm.json'

coco = COCO(ann_file)
coco_dets = coco.loadRes(result_file)
assert coco.getImgIds()==coco_dets.getImgIds()
ImgIds = coco.getImgIds()
for imgid in ImgIds:
    img = mmcv.imread(root+coco.loadImgs(ImgIds)[0]['file_name'])
    anns = coco.loadAnns(coco.getAnnIds(imgid))
    dets = coco_dets.loadAnns(coco_dets.getAnnIds(imgid))
    maskUtils.iou([dets[1]['segmentation']],maskUtils.frPyObjects(anns[0]['segmentation'],1024,1024
),[0])





    import pdb
    pdb.set_trace()

#cocoEval = COCOeval(coco, coco_dets, 'segm')

