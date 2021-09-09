"""
detector mainly used with yolo, naoqi and trained CNN model(torch)
"""

from __future__ import print_function
import numpy as np
from PIL import Image
import cv2
import os
import sys

import torch
import torch.cuda.random

# import naoqi modules
import qi
from naoqi import ALProxy

# import custom yolo modules
import yolo.darknet as dn
from yolo.custom_yolo import image_detection, network
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
        self.width, self.height = 640, 480

        self.detector = Detector()

    def detect(self, pil_img):
        detector = self.detector
        # convert pillow image to cv2 format
        cv_image = np.array(pil_img)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        image, detections = image_detection(cv_image, network, self.threshold, True)
        cv2.imshow('img', image)
        cv2.waitKey(3)
        if len(detections) == 0:
            return
        for item in detections:
            _, prob, coordinates = item
            x1, y1, w, h = coordinates
            x1, y1 = x1 - w/2, y1 - h/2
            x2, y2 = x1 + w, y1 + h
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1, x2, y2 = self.edit_size(x1, y1, x2, y2)
            croped_cv_image = cv_image[y1:y2, x1:x2]
            cropped = Image.fromarray(croped_cv_image)  # crop the image and convert for detection
            calculated = detector.calculate(cropped)
            _, classified = detector.classify(calculated, self.criterion)
            self.pepper.reaction(classified)

    def edit_size(self, x1, y1, x2, y2):
        if int(y2) >= self.height:
            y2 -= 1
        if int(y1) <= 0:
            y1 = 0
        if int(x2) >= self.width:
            x2 -= 1
        if int(x1) <= 0:
            x1 = 0

        return x1, y1, x2, y2

    def main(self):
        while True:
            pil_img = self.pepper.capture_image()
            self.detect(pil_img)


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
        print(value.item())
        if value.item() < criteria:
            return self.classifier[0]
        else:
            return self.classifier[1]
