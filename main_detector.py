"""
detector mainly used with yolo, naoqi and trained CNN model(torch)
"""

from __future__ import print_function
from PIL import Image
import cv2
import os
import sys

import torch
import torch.cuda.random

# import ROS modules
import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as Img

# import naoqi modules
import qi
from naoqi import ALProxy

# import custom yolo modules
import yolo.darknet as dn
import yolo.custom_yolo as yl
from yolo.yolo_class import Yolo  # use yolo as a class

# import custom modules
from connect_pepper.pepper import Pepper
from detector.model_cnn import Net
from detector.dataset_image import ImageTransform

YOLO_PATH = "./yolo/yolo_dataset/"
# MODEL_PATH = "main_ckpt/best_1.pth"
MODEL_PATH = "additional_ckpt/saved_second.pth"


USE_CUDA = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(USE_CUDA)
torch.manual_seed(1051325718700006030)  # 1051325718700006030: initial seed
if USE_CUDA == 'cuda':
    torch.cuda.random.manual_seed_all(2826009950518439)  # 2826009950518439: initial seed of cuda


class FaceDetector:
    def __init__(self, session):
        self.pepper = Pepper(session)
        self.ip = self.pepper.ip
        self.port = self.pepper.port
        self.threshold = self.pepper.threshold  # for finding a face part
        self.criterion = 0.5  # criterion of classification

        self.detector = Detector()

        # for using yolo
        self.options = {"model": "obj.cfg", "load": "obj_10000.weights",
                        "data": "obj.data", "threshold": self.threshold,
                        "gpu": 0.2, "summary": None}
        # darknet
        dn.set_gpu(0)
        self.net = dn.load_net(self.options['model'], self.options['load'], self.options['data'], batch_size=1)
        self.meta = dn.load_meta(self.options['data'])
        # declaration of yolo with a custom module
        self.yolo = Yolo(model="obj.cfg", load="obj_10000.weights", datapath=YOLO_PATH)
        self.cv_net = self.yolo.net
        self.classes = self.yolo.classes

        self.cam_topic = "pepper_robot/camera/front/image_raw"
        self.cam_sub = rospy.Subscriber(self.cam_topic, Img, self.image_callback, queue_size=1)

    def image_callback(self, msg):
        self.detect(msg)

    def detect(self, msg):
        detector = self.detector
        image, coordinates = yl.image_callback(msg)
        x1, y1, w, h = coordinates
        x2, y2 = x1 + w, y1 + h
        cropped = Image.fromarray(image[y1:y2, x1:x2])  # crop the image and convert for detection
        calculated = detector.calculate(cropped)
        _, classified = detector.classify(calculated, self.criterion)
        self.pepper.reaction(classified)

    def main(self):
        rospy.init_node('yolo_detector', anonymous=True)
        rospy.Subscriber(self.cam_topic, Img, self.image_callback)
        rospy.spin()


class Detector:
    def __init__(self):
        self.model = self.load_model(MODEL_PATH, USE_CUDA)
        self.transform = ImageTransform(size=(128, 128))
        self.classifier = {
            0: ('no_mask', False),
            1: ('mask', True)
        }

    def load_model(self, filepath, cuda):
        assert os.path.isfile(filepath)
        available = cuda == 'cuda'
        net = Net([3, 4, 6, 3]).to(DEVICE)  # declare the model
        cnn_data = torch.load(filepath)
        if available:
            net = torch.nn.DataParallel(net)
        net.load_state_dict(cnn_data['model'], strict=False)
        if available:
            net.eval()
            net.cuda()
        return net

    # calculate the model
    def calculate(self, image):
        model, transform = self.model, self.transform
        image = torch.unsqueeze(transform(img=image), 0)  # transform an image for detecting
        result = model(image)
        return result

    # classify the image with the result value
    def classify(self, value, criteria):
        if value < criteria:
            return self.classifier[0]
        else:
            return self.classifier[1]
