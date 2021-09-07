import time
import qi
from naoqi import ALProxy

from capture import Photo

# settings of Pepper robot
PEPPER_IP = "192.168.1.188"  # IP address of Pepper robot
PORT = 9559  # number of the port to connect with Pepper robot (default value)
VALID = 0.35
VOLUME = 0.8
DEFAULT_VOLUME = 70


# class of manipulating Pepper robot
class Pepper:
    def __init__(self, session):
        self.srv = self.create_srv(session)
        self.photo = Photo(self.srv)

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
        srv['video_record'] = session.service("ALVideoRecorder")

        # set the audio
        srv['tts'].setVolume(VOLUME)
        srv['tts'].setParameter("defaultVoiceSpeed", 85)
        srv['audio_player'] = session.service("ALAudioPlayer")

        return srv

    # to make Pepper react if
    def reaction(self, detected):
        tts = self.srv['tts']
        assert type(detected) == bool
        if detected:
            tts.say("Put on your mask!")

    def video_capture(self):
        pass # TODO implement this


if __name__ == "__main__":
    pass
