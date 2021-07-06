#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def Read_image():

	Parking_lot = cv2.imread("Images/1.jpeg")


	Parking_lot = np.array(Parking_lot)
    
    # Node cycle rate (in Hz).
	loop_rate = rospy.Rate(30)

    # Publishers
	br = CvBridge()
	ros_msg = br.cv2_to_imgmsg(Parking_lot , "bgr8")
	while not rospy.is_shutdown():
		pub.publish(ros_msg)
	rospy.spin()

if __name__ == '__main__':

	rospy.init_node("Read_Image", anonymous = True)
	pub = rospy.Publisher('Read_Image', Image,queue_size=10)
	
	Read_image()
