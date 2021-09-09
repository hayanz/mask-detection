import qi
# from naoqi import ALProxy

import cv2
import time

# import the custom module to access cameras in pepper
from camera import Camera

# settings of Pepper robot
PEPPER_IP = "192.168.1.123"  # IP address of Pepper robot
PORT = 9559  # number of the port to connect with Pepper robot (default value)
VALID = 0.35
VOLUME = 0.8
DEFAULT_VOLUME = 70


# class of manipulating Pepper robot
class Pepper:
    def __init__(self, session):
        self.ip = PEPPER_IP
        self.port = PORT
        self.srv = self.create_srv(session)
        self.camera = Camera(self.srv)
        self.threshold = 0.5

    def create_srv(self, session):
        srv = dict()
        srv['tablet'] = session.service("ALTabletService")
        srv['memory'] = session.service("ALMemory")
        srv['motion'] = session.service("ALMotion")
        srv['asr'] = session.service("ALSpeechRecognition")
        srv['tts'] = session.service("ALTextToSpeech")
        srv['aas'] = session.service("ALAnimatedSpeech")
        srv['audio_device'] = session.service("ALAudioDevice")
        srv['video_device'] = session.service("ALVideoDevice")
        srv['photo_capture'] = session.service("ALPhotoCapture")

        # for detecting the face part
        srv['face_detection'] = session.service("ALFaceDetection")
        # for recording a video
        srv['video_recorder'] = session.service("ALVideoRecorder")

        # set the audio
        srv['tts'].setVolume(VOLUME)
        srv['tts'].setParameter("defaultVoiceSpeed", 85)
        srv['audio_player'] = session.service("ALAudioPlayer")

        return srv

    @staticmethod
    def compare_hist(img1, img2):
        """
        comparing method for upper function
        Args:
            img1, img2 : numpy array of cropped image
        """
        img1 = img1.astype('uint8')
        img2 = img2.astype('uint8')
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

        hist0 = cv2.calcHist([img1], [0, 1], None, [10, 16], [0, 180, 0, 256])
        hist1 = cv2.calcHist([img2], [0, 1], None, [10, 16], [0, 180, 0, 256])
        score = cv2.compareHist(hist0, hist1, 0)  # method 0~6
        # print(score)
        return score

    # to make Pepper react if it found a person who does not wear mask
    def reaction(self, truth):
        tts = self.srv['tts']
        assert type(truth) == bool
        if not truth:
            # print("Cannot find a mask!")  # for debugging
            tts.say("Put on your mask.")
        else:
            return
            # print("Found a mask")  # for debugging

    def set_video(self):
        camera = self.camera
        return camera.set_video()

    def capture_video(self):
        camera = self.camera
        camera.capture_video()

    def record_video(self):
        camera = self.camera
        camera.record_video()

    def capture_image(self):
        camera = self.camera
        return camera.capture_image()

    def access_camera(self, start=True):
        camera = self.camera
        if start:
            print("Open the camera")  # for debugging
            result = camera.open()
        else:
            print("Close the camera")  # for debugging
            result = camera.close()
        return result
