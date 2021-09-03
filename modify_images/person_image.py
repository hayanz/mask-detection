from PIL import Image
import numpy as np
import math
import cv2
import dlib

# marks for detecting the face with the predictor
NOSE_MARK = 29
JAW_MARK = 8


# class of the person image
class Person(object):
    def __init__(self, filename):
        self.filename = filename
        self.img = Image.open(self.filename).convert("RGBA")
        self.width, self.height = self.img.size
        self.detector = dlib.get_frontal_face_detector()
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # calculate the length and coordinates of the lower face
        self.lower_length, self.nose, self.jaw = self.find_lowerface()

    # find coordinates of the lower face at the image
    def find_lowerface(self):
        img = cv2.imread(self.filename)
        # convert the image into grayscale
        grey = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        faces = self.detector(grey)

        for face in faces:
            x1, x2 = face.left(), face.right()
            y1, y2 = face.top(), face.bottom()

            facemarks = self.predictor(image=grey, box=face)
            # calculate the point to paste the mask image
            nose = [facemarks.part(NOSE_MARK).x, facemarks.part(NOSE_MARK).y]
            jaw = [facemarks.part(JAW_MARK).x, facemarks.part(JAW_MARK).y]

            lowerlength = abs(jaw[1] - nose[1])

        return lowerlength, nose, jaw

    # find the coordinate of the center of the lower face
    def find_center(self):
        # check if the length and coordinates of the lower face is not calculated yet
        if self.lower_length is None or self.nose is None or self.jaw is None:
            self.lower_length, self.nose, self.jaw = self.find_lowerface()
        nose, jaw = self.nose, self.jaw
        center = tuple(int(np.mean(n)) for n in zip(nose, jaw))
        return center

    # calculate the degree to rotate the mask image for combining images
    def get_degree(self):
        # check if the length and coordinates of the lower face is not calculated yet
        if self.lower_length is None or self.nose is None or self.jaw is None:
            self.lower_length, self.nose, self.jaw = self.find_lowerface()
        horizontal = -(self.nose[0] - self.jaw[0])
        vertical = float(self.lower_length)
        angle = math.atan(horizontal / vertical)  # radian value
        # convert radian to degree
        degree = math.degrees(angle)
        return degree

    # paste the mask image on the original person image
    def put_mask(self, mask_img, coordinate):
        img = self.img.copy().convert("RGBA")
        rotated = mask_img.rotate(self.get_de_0.egree())
        img.paste(rotated, coordinate, rotated.convert("RGBA"))

        return img


# for testing the code (debugging)
if __name__ == "__main__":
    person = Person("../face_images/mask_sample_0.jpg")
    print(person.find_lowerface())
