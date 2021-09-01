import random
import cv2
import os
import numpy as np
import rospy

import pyrealsense2 as rs

from glob import glob
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import darknet

# yolo image detection threshold vlaue
THRESHOLD = 0.5

def image_detection(color_image, darknet_network, thresh, is_realsense):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    network, class_names, class_colors = darknet_network
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = color_image
    # change RGB values for realsense
    if is_realsense:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def image_callback(img_msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    # cv_image == pepper img convert -> cv2
    image, detections = image_detection(
        cv_image, network, THRESHOLD, True
    )
    print(detections) # ('label', 'probability', (x, y, width, height))

    cv2.imshow('img', image)
    cv2.waitKey(3)

network = darknet.load_network(
        "data/obj.cfg",
        "data/obj.data",
        "data/obj_10000.weights",
        batch_size=1
    )
def listener():
    rospy.init_node('yolo_detector', anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    rospy.spin()



if __name__ == "__main__":
    listener()



