from __future__ import division
from ps_detect import PsDetect
from vps_classify import vpsClassify
import torch
import os
import cv2
import numpy as np
import copy
import glob
import tqdm
from PIL import Image
from utils.utils import compute_four_points

if __name__ == "__main__":


    input_folder = 'Parking_lots'
    output_folder = 'Parking_spots'
    model_def = 'config/ps-4.cfg'
    weights_path_yolo = 'weights/yolov3_4.pth'
    weights_path_vps = 'weights/Customized.pth'
    conf_thres = 0.9
    nms_thres = 0.5
    img_size = 416
    save_files = 1

    os.makedirs(output_folder, exist_ok=True)
    device = torch.device("cpu")


    ps_detect =PsDetect(model_def, weights_path_yolo, img_size, device)
    vps_classify = vpsClassify(weights_path_vps, device)

    with torch.no_grad():
        imgs_list = glob.glob(input_folder + '/*.jpg')
        print(input_folder)
        print(len(imgs_list))
        for img_path in tqdm.tqdm(imgs_list):
            if save_files:
                img_name = img_path.split('/')[-1]
                filename = img_name.split('.')[0] + '.txt'
                file_path = os.path.join(output_folder, filename)
                file = open(file_path, 'w')
            img = np.array(Image.open(img_path))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            if img.shape[0] != 600:
                img = cv2.resize(img, (600, 600))
            detections = ps_detect.detect_ps(img, conf_thres, nms_thres)
            if len(detections) !=0:
                for detection in detections:
                    point1 = detection[0]
                    point2 = detection[1]
                    angle = detection[2]
                    pts = compute_four_points(angle, point1, point2)
                    point3_org = copy.copy(pts[2])
                    point4_org = copy.copy(pts[3])
                    label_vacant = vps_classify.vps_classify(img, pts)
                    if label_vacant == 0:
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)
                    pts_show = np.array([pts[0], pts[1], point3_org, point4_org], np.int32)
                    if save_files:
                        file.write(str(angle))
                        file.write(' ')
                        points = list((pts[0][0], pts[0][1], pts[1][0], pts[1][1]))
                        for value in points:
                            file.write(str(value.item()))
                            file.write(' ')
                        file.write('\n')
                    cv2.polylines(img, [pts_show], True, color, 2)
            cv2.imshow('Detect PS', img[:,:,::-1])
            #cv2.imwrite("result.png", img[:,:,::-1])
            cv2.waitKey(1)
            if save_files:
                file.close()
                cv2.imwrite(os.path.join(output_folder, img_name), img[:,:,::-1])
        cv2.destroyAllWindows()






