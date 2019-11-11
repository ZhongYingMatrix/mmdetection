from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from matplotlib import pyplot as plt

#config_file = '../configs/faster_rcnn_r50_fpn_1x.py'
config_file = '../configs/htc/htc_x101_64x4d_fpn_20e_16gpu.py'
#checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth'
checkpoint_file = '../checkpoints/htc_x101_64x4d_fpn_20e_20190408-497f2561.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

img = 'demo.jpg'
result = inference_detector(model, img)

show_result_pyplot(img, result, model.CLASSES)
plt.show()