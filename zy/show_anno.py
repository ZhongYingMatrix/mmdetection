from pycocotools.coco import COCO
import numpy as np
from matplotlib import pyplot as plt
import mmcv

root = 'data/coco/'
img_id = 152019
#126#236#123
coco = COCO('/home/zhongying/other/walkgis/mmdetection/data/coco/annotations/instances_train2017.json')
#coco = COCO('/home/zhongying/dataset/WalkTrainData/xihu_instance.json')
annIds=coco.getAnnIds(img_id)
anns=coco.loadAnns(annIds)

img = coco.loadImgs(img_id)
#import pdb
#pdb.set_trace()
img = mmcv.imread(root + 'train2017/' +img[0]['file_name'])
plt.imshow(img)
coco.showAnns(anns)
plt.show()
