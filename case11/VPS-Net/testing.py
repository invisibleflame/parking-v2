#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from __future__ import division
from ps_detect import PsDetect
from vps_classify import vpsClassify
import torch
import copy
import glob
import tqdm
from PIL import Image
from utils.utils import compute_four_points

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def VPS_net(IPM_image):

	with torch.no_grad():
			if save_files:
				file = open("bounding_box.txt",'w')
			bridge = CvBridge()
			img = bridge.imgmsg_to_cv2(IPM_image, desired_encoding='passthrough')
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
					cv2.polylines(img,[pts_show], True, color,2)
			ros_msg = bridge.cv2_to_imgmsg(img, "8UC3")
			pub.publish(ros_msg)	
			if save_files:
				file.close()

def Read_image_from_IPM():

    # Node cycle rate (in Hz).
    loop_rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        rospy.Subscriber("IPM", Image, VPS_net)
    
    rospy.spin()

if __name__ == "__main__":

    
    input_folder = 'VPS-Net/Parking_lots'
    output_folder = 'VPS-Net/Parking_spots'
    model_def = 'VPS-Net/config/ps-4.cfg'
    weights_path_yolo = 'VPS-Net/weights/yolov3_4.pth'
    weights_path_vps = 'VPS-Net/weights/Customized.pth'
    conf_thres = 0.9
    nms_thres = 0.5
    img_size = 416
    save_files = 0

    os.makedirs(output_folder, exist_ok=True)
    device = torch.device("cpu")

    ps_detect =PsDetect(model_def, weights_path_yolo, img_size, device)
    vps_classify = vpsClassify(weights_path_vps, device)

    rospy.init_node("VPS", anonymous = True)
    pub = rospy.Publisher("VPS",Image,queue_size = 10)
    Read_image_from_IPM()





