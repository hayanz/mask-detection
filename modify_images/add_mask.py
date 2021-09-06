from PIL import Image
import cv2
import random

# import the custom modules
from mask_images.mask_image import Mask
from person_image import Person
from pick_samples import *  # to pick again if the face cannot be found at the picture
from face_crop import Face  # to detect the face part and crop the image

# colors of the mask
MASKS = ["white.png", "black.png", "skyblue.png", "pink.png"]
# number of images to fix (index of the endpoint)
COUNT = 645

# path of images
target_dir = "../face_images/cropped/picked/"
mask_dir = "../mask_images/"

# index of the starting point
start_idx = 644


# to crop the image and save
def save_cropped(filename):
    try:
        target = Face(filename)
        cropped = target.crop()
        cv2.imwrite(filename, cropped)
        return True
    except (UnboundLocalError, ValueError, cv2.error):
        return False


if __name__ == "__main__":
    for i in range(start_idx, COUNT):
        while True:
            try:
                filename = target_dir + str(i) + ".jpg"
                # define the class to access the image
                person = Person(filename)  # class of the person image
                # get the value of the length of the lower face to resize the mask image
                length = int(person.lower_length)

                # crop the face part and save the result
                success_crop = save_cropped(filename)
                if not success_crop:
                    raise UnboundLocalError
                # reload the image
                person = Person(filename)

                # pick the color of the mask and define the class to access the image
                img = person.img.convert("RGBA")
                maskfile = mask_dir + random.choice(MASKS)
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
                combined.save(filename)
                print("[%d] Successfully Done!" % i)  # for debugging
                break  # the infinite loop ends if the mask image is pasted

            except (UnboundLocalError, ValueError):
                # UnboundLocalError: when the face cannot be found
                # ValueError: when the pixel of top and bottom are selected in duplicate
                dir_chosen = random.choice([dir1, dir2])  # choose a directory randomly
                move_files(dir_chosen, end=1, add_num=i)  # pick one picture randomly again
