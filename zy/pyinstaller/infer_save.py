from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmdet.ops.nms.nms_wrapper import nms
import argparse
from pycocotools.coco import COCO
import cv2
import json

def detect_img(model, file_path):
    img = mmcv.imread(file_path)
    result = inference_detector(model, img)

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

    contours=[]
    for mask in crop_segm:
        mask = maskUtils.decode(mask)
        contour,_ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ## sort for biggest contour
        contour.sort(key=lambda x: cv2.contourArea(x), reverse=True) 

        contours.append(list(np.array(contour[0]).reshape(-1))) 

    scores = list(crop_bbox[:,-1].astype(float))
    bboxes = list(crop_bbox[:,:-1].astype(int))

    content = [{'segmentation':contours[i], 'bbox':list(bboxes[i]), 'scores': scores[i]} for i in range(len(bboxes))]
    return content

def parse_args():
    parser = argparse.ArgumentParser(description='infer and save for pyinstaller')
    # parser.add_argument('--config', help='test config file path',
    #                     default='/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_11_13/test_htc_dconv_c5_r101_fpn_sbn_20e_total.py')
    # parser.add_argument('--checkpoint', help='checkpoint file',
    #                     default='/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/2019_11_13/epoch_100.pth')
    # parser.add_argument('--root', help='output root', default='/home/zhongying/other/walkgis/mmdetection/zy/crop_res/val_result_1113/')
    # parser.add_argument('--ann', help='annotation file', default='/home/zhongying/dataset/WalkValidationData/validation_instance.json')
    parser.add_argument('--cuda', help='use gpu or not', default=False)
    parser.add_argument('--img', help='img path', default='./test.tif')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model = init_detector('./test_htc_dconv_c5_r101_fpn_sbn_20e_total.py', './epoch_100.pth', device="cuda:0" if args.cuda else "cpu")
    content = detect_img(model, args.img)

    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    main()