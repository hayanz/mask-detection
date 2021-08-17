from collections import OrderedDict
from PIL import Image
import numpy as np
import cv2
import dlib
import os
import random
import shutil

# marks for detecting the face with the predictor
NOSE_MARK = 30
JAW_MARK = 8

# colors of the mask
MASKS = ["white", "black", "black_grey", "white_grey", "grey"]
# number of images to fix
COUNT = 1  # 1000 exactly
# size of the images  # NEED TO FIX THIS
SIZE = {"face": 500, "mask": 100}

# path of the images
target_dir = "faces_dataset/picked_mask/"
pic_type = ".jpg"

detector = dlib.get_frontal_face_detector()
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

for i in range(COUNT):
    # target = target_dir + str(i) + pic_type
    target = "faces_dataset/sample_2.jpg"
    # load an image to detect
    img = cv2.imread(target)
    # convert the image into grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # x1, x2 = face.left(), face.right()
        # y1, y2 = face.top(), face.bottom()

        facemarks = predictor(image=gray, box=face)
        mask_nose = [facemarks.part(NOSE_MARK).x, facemarks.part(NOSE_MARK).y]
        mask_jaw = [facemarks.part(JAW_MARK).x, facemarks.part(JAW_MARK).y]
        # print(mask_nose, mask_jaw)  # for debugging

        mask_file = "mask_images/" + random.choice(MASKS) + ".png"
        mask = Image.open(mask_file)

        # NEED TO FIX THIS --> not implemented yet
        # resize images with pillow -> put the mask on the target image
        # pick 1000 images to use and change the value of 'COUNT' to 1000
