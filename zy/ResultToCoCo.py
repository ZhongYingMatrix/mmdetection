import pickle
import cv2
import os
import numpy as np
import json
from tqdm import tqdm
import pycocotools.mask as maskUtils

class Transform_PKL():
    def __init__(self, pkl_file, img_file,output_root,data_root, single_img=True, show=False):
        self.pkl_file = pkl_file
        self.show = show
        self.root = data_root
        self.output_root = output_root

        with open(self.pkl_file, 'rb') as f:
            self.result = pickle.load(f)
        with open(img_file,'r') as f:
           self.image_infos = json.load(f)
        self.image_names = [image['file_name'] for image in self.image_infos['images']]
        self.dataset = {
            "info": self.image_infos["info"],
            "license": self.image_infos["license"],
            "images": self.image_infos["images"],
            "annotations": [],
            "categories": self.image_infos["categories"],
        }

        self.run()

        with open(output_root +"/result.json", 'w') as f:
            json.dump(self.dataset, f, indent = 4)
        print("Output result to "+ output_root +"/result.json")

    def run(self):
        bboxes_output = []
        cnt = 0

        for index,image in enumerate(tqdm(self.image_infos["images"])):
            self.data_single = []

            img_name = image["file_name"]
            id = image["id"]
            img_path = os.path.join(self.root,img_name)
            img = cv2.imread(img_path)
            try:
                img_show, bboxes, labels, contours = self.transform_pkl(img, self.result[index])
            except:
                pass
                #import pdb
                #pdb.set_trace()
            # name, class, bboxes
            for i in range(len(bboxes)):
                box = bboxes[i]
                label = labels[i]
                contour = contours[i]
                confidence = box[-1]

                self.dataset['annotations'].append({
                    'area': float((box[2]-box[0])*(box[3]-box[1])),
                    'bbox': [int(box[0]),int(box[1]),int(box[2]),int(box[3])],
                    'category_id': 0,
                    'id': cnt,
                    'image_id': int(id),
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [int(c) for c in contour]
                })

                self.data_single.append({
                    # 'area': float((box[2] - box[0]) * (box[3] - box[1])),
                    'segmentation': [int(c) for c in contour],
                    'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    'score':str(box[-1])
                    # 'category_id': 0,
                    # 'id':i,
                    # 'image_id': int(id),
                    # 'iscrowd': 0
                    # mask, 矩形是从左上角点按顺时针的四个顶点

                })

                cnt += 1

            single_json = os.path.join(self.output_root,img_name.split("/")[-1].split(".")[0]+".json")
            with open(single_json, 'w') as f:
                json.dump(self.data_single, f, indent=4)
            print("Output single result to " + single_json)

            if self.show:
                #                 self.show_minRects(img,minRects,labels)
                self.show_bboxes(img, bboxes)


    def transform_pkl(self, img, result):
        bbox_result, segm_result = result
        h, w, _ = img.shape
        img_show = img[:h, :w, :]

        # get labels
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        bboxes = np.vstack(bbox_result)
        #Set threshold for cofindence
        inds = np.where(bboxes[:, -1] > 0.1)[0]
        #         print(bboxes[:,-1])

        # draw segmentation masks
        Contours = []
        if segm_result is not None:
            segms = []
            for seg in segm_result:
                for a in seg:
                    segms.append(a)

            for i in inds:
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                mask = maskUtils.decode(segms[i]).astype(np.bool)

                #                 binary = mask.astype(np.int)
                binary = np.ascontiguousarray(mask, dtype=np.uint8) * 250
                #                 cv2.imwrite(str(i)+'.png',binary)
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # TODO zy debug
                try:
                    Contours.append(contours[0].reshape(-1))
                except:
                    Contours.append(np.array([],dtype=np.int32))
                    #import pdb
                    #pdb.set_trace()
                #Contours.append(contours[0].reshape(-1))

        #                 img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
        return img_show, bboxes[inds], labels[inds], Contours

    def show_bboxes(self, img, bboxes):
        for bbox in bboxes:
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(img, left_top, right_bottom, color=(0, 0, 255), thickness=2)
            cv2.putText(img, str(round(bbox[-1],4)), left_top, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
        cv2.imshow('image', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_minRects(self, img, bboxs, labels=None):
        w, h, _ = img.shape
        #         w, h = img.shape

        bboxs = np.array(bboxs)

        for bbox, label in zip(bboxs, labels):
            x0, y0 = bbox[0], bbox[1]
            x1, y1 = bbox[2], bbox[3]
            x2, y2 = bbox[4], bbox[5]
            x3, y3 = bbox[6], bbox[7]

            red = (0, 0, 255)
            cv2.line(img, (x0, y0), (x1, y1), red, 2)
            cv2.line(img, (x1, y1), (x2, y2), red, 2)
            cv2.line(img, (x2, y2), (x3, y3), red, 2)
            cv2.line(img, (x0, y0), (x3, y3), red, 2)

            # cv2.putText(img, CLASSES[label], (x0, y0), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":

    root = "/home/zhongying/other/walkgis/mmdetection/zy/htc_new_set/"
    # pkl_file = os.path.join(root,"2019_10_23/result.pkl")
    # output_root = os.path.join(root,"2019_10_23/result")
    pkl_file = os.path.join(root,"2019_11_11/cn_result.pkl")
    output_root = os.path.join(root,"2019_11_11/cn_result")

    data_root =  "/home/zhongying/dataset/"
    json_file = os.path.join(data_root,'WalkTrainData/correct_cn_instance.json')
    #json_file = os.path.join(data_root,'WalkValidationData/correct_validation_instance.json')

    # img_root = "E:/work/Walk/try/super_mini"
    transform = Transform_PKL(pkl_file, json_file,output_root, data_root,single_img=False, show=False)

