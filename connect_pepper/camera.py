from naoqi import ALProxy
from PIL import Image
import os
import time

# string of the directory to save photos
DIRECTORY = "./cameras/photos/"
# the name of the file to save
FILENAME = "captured"

# settings of the device to take a picture
RESOLUTION = 2
COLORSPACE = 11
FPS = 20


class Camera:
    def __init__(self, srv):
        self.srv = srv
        self.directory = DIRECTORY
        self.filename = FILENAME
        # 0: camera on the top | 1: camera on the bottom | 2: 3D camera in the eyes
        self.camera_idx = 0

    def set_video(self):
        video = self.srv['video_device']

        video_subs_list = ['detector_rgb_t_0', 'detector_pc_0', 'detector_dep_0']
        print(video.getSubscribers())  # for debugging
        for sub_name in video_subs_list:  # print for debugging
            print(sub_name, video.unsubscribe(sub_name))

        # only a camera on the top would be used in this project
        # parameters: name, idx, resolution, colorspace, fps
        top_camera = video.subscribeCamera('detector_rgb_t', self.camera_idx,
                                           RESOLUTION, COLORSPACE, FPS)

        return top_camera

    def capture_video(self):
        video_service = self.srv['video_device']
        id = video_service.subscribe("rgb_t", RESOLUTION, COLORSPACE, FPS)
        img_count = 30  # TODO change this value

        for img_idx in range(img_count):
            pepper_img = video_service.getImageRemote(id)
            width, height = pepper_img[0], pepper_img[1]
            array = pepper_img[6]
            img_str = str(bytearray(array))
            im = Image.frombytes("RGB", (width, height), img_str)
            captured = os.path.sep.join([self.directory, self.filename + str(img_idx) + '.jpg'])
            im.save(captured, "JPEG")

        video_service.unsubscribe(id)

    def record_video(self):
        record_time = 5

        recorder = self.srv['video_recorder']
        recorder.setFrameRate(10.0)
        recorder.setResolution(2)  # 640 * 480
        # [/home/nao/recordings/cameras/]: pepper internal storage
        recorder.startRecording("/home/nao/recordings/cameras", "test")
        print("Video record started.")  # for debugging

        time.sleep(record_time)

        video_info = recorder.stopRecording()
        print("Video was saved on the robot: ", video_info[1])  # for debugging
        print("Total number of frames: ", video_info[0])  # for debugging

    def capture_image(self):
        img_count = 1  # TODO change this value
        for img_idx in range(img_count):
            video_device = self.srv['video_device']
            video_client = video_device.subscribe("python_client", RESOLUTION, COLORSPACE, FPS)
            nao_image = video_device.getImageRemote(video_client)
            video_device.unsubscribe(video_client)
            width, height = nao_image[0], nao_image[1]
            array = nao_image[6]
            image_string = str(bytearray(array))
            im = Image.frombytes("RGB", (width, height), image_string)

            # img_path = os.path.sep.join([self.directory, "cpatured.jpg"])
            # im.save(img_path, "JPEG")
            # im.show()  # for debugging
            # time.sleep(0.01)

        return im

    def open(self):
        video = self.srv['video_device']
        return video.openCamera(self.camera_idx)

    def close(self):
        video = self.srv['video_device']
        return video.closeCamera(self.camera_idx)
