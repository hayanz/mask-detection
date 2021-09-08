# from naoqi import ALProxy
from PIL import Image
import os

# string of the directory to save photos
DIRECTORY = "./cameras/photos/"
# the name of the file to save
FILENAME = "captured"

# settings of the device to take a picture
RESOLUTION = 2
COLORSPACE = 11
FPS = 20


class Photo:
    def __init__(self, srv):
        self.srv = srv
        self.directory = DIRECTORY
        self.filename = FILENAME
        self.photo_name = None
        self.count = 30  # TODO modify this

    # TODO implement this
    def capture(self):
        video_service = self.srv['video_device']
        id = video_service.subscribe("rgb_t", RESOLUTION, COLORSPACE, FPS)

        for i in range(self.count):
            pepper_img = video_service.getImageRemote(id)
            width, height = pepper_img[0], pepper_img[1]
            array = pepper_img[6]
            img_str = str(bytearray(array))
            im = Image.frombytes("RGB", (width, height), img_str)
            self.photo_name = self.filename + str(i) + '.jpg'
            im.save(self.directory + self.photo_name, "JPEG")

        video_service.unsubscribe(id)
