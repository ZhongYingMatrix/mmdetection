from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

config_file = '/home/zhongying/research/repo/mmdetection/configs/papmask/pap_1024_1x_r50.py'
checkpoint_file = '/home/zhongying/research/repo/mmdetection/work_dirs/pap_1024_1x_r50/epoch_12.pth'
# config_file = '/home/zhongying/research/repo/mmdetection/configs/polarmask/polar_768_1x_r50.py'
# checkpoint_file = '/home/zhongying/research/repo/mmdetection/work_dirs/polar_768_1x_r50_old/r50_1x.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

img = '/home/zhongying/research/repo/mmdetection/demo/demo.jpg'
result = inference_detector(model, img)
import pdb; pdb.set_trace()
#show_result_pyplot(img, result, model.CLASSES)