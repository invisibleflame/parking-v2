#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def IPM(Parking_lot):


	s = np.array([[300, 560], [852, 560], [0, 864], [1152, 864]], dtype=np.float32)
	# Vertices coordinates in the destination image
	t = np.array([[0, 0], [1152, 0], [250, 864], [902, 864]], dtype=np.float32)
	TARGET_H, TARGET_W = 500, 500 # resizing of image msg (can be changed)
	br = CvBridge()
	image1 = br.imgmsg_to_cv2(Parking_lot , "bgr8")

	# Compute projection matrix
	M = cv2.getPerspectiveTransform(s, t)
	# Warp the image
	warped = cv2.warpPerspective(image1, M, (image1.shape[1], image1.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

	warped2=cv2.resize(warped, (600,600), interpolation = cv2.INTER_AREA)
	
	ros_msg = br.cv2_to_imgmsg(warped2 , "8UC3")
	pub.publish(ros_msg)	

def Read_image_for_IPM():
    
    # Node cycle rate (in Hz).
	loop_rate = rospy.Rate(1)
	while not rospy.is_shutdown():
		rospy.Subscriber("Read_Image", Image, IPM)
	
	rospy.spin()

if __name__ == '__main__':

	rospy.init_node("IPM", anonymous = True)
	pub = rospy.Publisher('IPM', Image,queue_size=10)
	Read_image_for_IPM()
