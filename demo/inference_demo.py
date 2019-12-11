from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

config_file = '/home/zhongying/research/repo/mmdetection/configs/fcos_proto/fcos_proto_r50_caffe_fpn_gn_1x_4gpu.py'
checkpoint_file = '/home/zhongying/research/repo/mmdetection/work_dirs/fcos_proto_r50_caffe_fpn_gn_1x_4gpu/latest.pth'
# config_file = '/home/zhongying/research/repo/mmdetection/configs/polarmask/polar_768_1x_r50.py'
# checkpoint_file = '/home/zhongying/research/repo/mmdetection/work_dirs/polar_768_1x_r50_old/r50_1x.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

img = '/home/zhongying/research/repo/mmdetection/demo/demo.jpg'
result = inference_detector(model, img)
import pdb; pdb.set_trace()
#show_result_pyplot(img, result, model.CLASSES)