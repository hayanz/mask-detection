import random
import cv2
import dlib

# import the custom module to pick an image to use
from pick_samples import *


# class to detect the face part and crop
class Face(object):
    def __init__(self, filename):
        self.filename = filename
        self.img = cv2.imread(filename)
        self.grey = self.make_grayscale()
        self.detector = dlib.get_frontal_face_detector()
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def make_grayscale(self):
        temp = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    # detect the face
    def find_face(self):
        img, grey = self.img, self.grey
        faces = self.detector(self.grey)

        # marks for detecting the face with the predictor
        nose_mark, jaw_mark = 29, 8

        for face in faces:
            x1, x2 = face.left(), face.right()
            y1, y2 = face.top(), face.bottom()
            # raise exception if all coordinates are not positive
            if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                raise UnboundLocalError

            facemarks = self.predictor(image=grey, box=face)
            # try calculating the length of the lower face
            # the UnboundLocalError would be raised if calculation fails
            lowerlength = abs(facemarks.part(nose_mark).y - facemarks.part(jaw_mark).y)

        # cv2.imshow(winname="found faces", mat=img)  # for debugging
        # cv2.waitKey(3000)  # wait for 3 seconds
        # cv2.destroyAllWindows()  # close all windows

        return [x1, x2, y1, y2]

    # crop the face image
    def crop(self):
        # find coordinates of the border of the face
        faces = self.find_face()
        if faces is None or len(faces) == 0:
            raise UnboundLocalError

        # crop the image
        x1, x2, y1, y2 = faces
        cropped = self.img[y1:y2, x1:x2]

        # cv2.imshow(winname="cropped faces", mat=cropped)  # for debugging
        # cv2.waitKey(3000)  # wait for 3 seconds
        # cv2.destroyAllWindows()  # close all windows

        return cropped


# to crop the face part and save as a different file
if __name__ == "__main__":
    # path of images
    before_dir = "faces_images/without_cropped/picked/"
    after_dir = "faces_images/cropped/picked/"
    # index of starting point and endpoint
    start, end = 500, 1000
    for i in range(start, end):
        while True:
            try:
                target = Face(before_dir + str(i) + ".jpg")
                cv2.imwrite(after_dir + str(i) + ".jpg", target.crop())
                print("[%d] Successfully saved!" % i)
                break  # the loop ends if an image is successfully saved
            except (UnboundLocalError, ValueError, cv2.error):
                dir_chosen = random.choice([dir1, dir2])
                move_files(dir_chosen, end=1, add_num=i)
