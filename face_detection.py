from collections import OrderedDict
import imutils.face_utils as face_utils
import imutils
import numpy as np
import dlib
import cv2


FACEMARKS = OrderedDict([("mouth", (48, 68)), ("r_eyebrow", (17, 22)), ("l_eyebrow", (22, 27)),
                         ("r_eye", (36, 42)), ("l_eye", (42, 48)), ("nose", (27, 35)),
                         ("jaw", (0, 17))])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load an image to detect
img = cv2.imread("faces_dataset/sample.jpg")
# convert the image into grayscale
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

faces = detector(gray)



