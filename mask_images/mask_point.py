"""
find points of the mask to combine images
"""

from PIL import Image
import cv2
import dlib
import shutil


# class of the mask
class Mask(object):
    def __init__(self, filename):
        self.filename = filename
        self.img = Image.open(filename)

    # resize the image
    def resize(self, width, height, returned=True):
        # initial size of the mask is 1200 * 1200
        resized = self.img.resize((width, height))
        if returned:  # return as the new object
            return resized
        self.img = resized  # save the resized image to the current file

    # save the image
    def save(self):
        self.img.save(self.filename)

    # save the image as the new one
    def save_as(self, filename, filetype, filepath=None):
        valid_type = {"PNG": ".png", "JPEG": ".jpeg", "JPG": ".jpg"}

        try:
            if filetype not in valid_type:
                raise TypeError
        except TypeError:
            print("Invalid type.")  # print the error message
            return  # make the program ends

        # save the image
        filename = filename + valid_type[filetype]
        if filepath is not None:
            filename = filepath + filename
        self.img.save(filename)


# for debugging
if __name__ == "__main__":
    test = Mask("mask_images/white.png")
    test.resize(300, 300, returned=True).save("mask_images/test.png")
