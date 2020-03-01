from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import matplotlib.pyplot as plt

config_file = '/home/zhongying/research/repo/mmdetection/configs/solo/solo_r50_caffe_fpn_gn_1x_4gpu.py'
checkpoint_file = '/home/zhongying/research/repo/mmdetection/work_dirs/solo_r50_caffe_fpn_gn_1x_4gpu/latest.pth'
# config_file = '/home/zhongying/research/repo/mmdetection/configs/fcos_proto/fcos_proto_r50_caffe_fpn_gn_1x_4gpu.py'
# checkpoint_file = '/home/zhongying/research/repo/mmdetection/work_dirs/fcos_proto_r50_caffe_fpn_gn_1x_4gpu_weight/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

img = '/home/zhongying/research/repo/mmdetection/demo/img/000000000552.jpg'
result = inference_detector(model, img)
#import pdb; pdb.set_trace()
#show_result_pyplot(img, result, model.CLASSES)
show_result_pyplot(img, result, model.CLASSES, score_thr=0.0001)
plt.show()