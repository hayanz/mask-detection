from PIL import Image
import numpy as np
import cv2
import dlib
import os
import random
import shutil

img_dir = "faces_dataset/"
img_type = ".jpg"


class Person(object):
    def __init__(self, filename):
        self.filename = img_dir + filename + img_type
        self.img = Image.open(self.filename).convert("RGBA")
        self.width, self.height = self.img.size
        self.detector = dlib.get_frontal_face_detector()
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def find_lowerface(self):
        img = cv2.imread(self.filename)
        # convert the image into grayscale
        grey = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        faces = self.detector(grey)

        for face in faces:
            x1, x2 = face.left(), face.right()
            y1, y2 = face.top(), face.bottom()

            facemarks = predictor(image=grey, box=face)


# for debugging
if __name__ == "__main___":
    pass
