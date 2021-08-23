from PIL import Image
import numpy as np
import cv2
import dlib
import random

# import the custom modules
from mask_images.mask_image import Mask
from person_image import Person
from pick_samples import *  # to pick again if the face cannot be found at the picture
from face_crop import Face  # to detect the face part and crop the image

# colors of the mask
MASKS = ["white.png", "black.png", "black_grey.png", "white_grey.png", "grey.png"]
# number of images to fix
COUNT = 1000

# path of the images
target_dir = "faces_dataset/picked_mask_cropped/"
mask_dir = "mask_images/"

start_idx = 0

for i in range(start_idx, COUNT):
    while True:
        try:
            # define the class to access the image
            person = Person(target_dir + str(i) + ".jpg")  # class of the person image
            # get the value of the length of the lower face to resize the mask image
            length = int(person.lower_length)

            # crop the face part
            # cropped = Face(person.img).crop()  # TODO erase '#'

            # pick the color of the mask and define the class to access the image
            img = person.img.convert("RGBA")
            # img = cropped.convert("RGBA")  # TODO erase '#'
            # maskfile = mask_dir + random.choice(MASKS)
            maskfile = mask_dir + "white.png"
            mask = Mask(maskfile)  # class of the mask image

            # calculate the size with the ratio of the mask image
            size = int(round(length * mask.get_ratio()))
            mask.resize(size, size)  # resized

            # calculate the coordinate of the center again with the size of the mask image
            mask_center = mask.find_center()
            coordinate = tuple(int(a - b) for a, b in zip(person.find_center(), mask_center))
            # paste the mask image to the original face image
            combined = person.put_mask(mask.img, coordinate)
            combined = combined.convert("RGB")
            # combined.show()  # for debugging
            combined.save(target_dir + str(i) + ".jpg")
            print("[%d] Successfully Done!" % i)  # for debugging
            break  # the infinite loop ends if the mask image is pasted

        except (UnboundLocalError, ValueError):
            # UnboundLocalError: when the face cannot be found
            # ValueError: when the pixel of top and bottom are selected in duplicate
            pick_dir = random.choice([dir1, dir2])  # choose a directory randomly
            move_files(dir2, end=1, add_num=i)  # pick one picture randomly again

# TODO crop the face part of the image and paste a mask image, then save as the training data
