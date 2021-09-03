import torch
import torch.cuda.random

import cv2
from PIL import Image

# import the custom module
from detector.detection_model import Net
from modify_images.face_crop import Face
from modify_images.person_image import Person

detector_path = "checkpoint/best.pth"

USE_CUDA = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(USE_CUDA)
torch.manual_seed(2390783316385493207)  # 2390783316385493207: initial random seed
if torch.cuda.is_available():
    torch.cuda.random.manual_seed_all(6399406970264898)  # 6399406970264898: initial random seed of cuda


# to load a trained model to use as a detector
def load_detector():
    checkpoint = torch.load(detector_path)
    model = Net([3, 4, 6, 3]).to(DEVICE)  # based on ResNet34
    model.load_state_dict(checkpoint['model'])
    return model


# to convert an image for detecting
def convert_image(imgpath):
    image = Face(imgpath).crop()
    # convert from 'BGR' to 'RGB'
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # convert OpenCV to PIL
    image_pil = Image.fromarray(image_rgb)
    return image_pil


if __name__ == "__main__":
    test_image = "face_images/mask_sample_0.jpg"
    test_cropped = Face(test_image).crop()
    # convert from 'BGR' to 'RGB'
    test_converted = cv2.cvtColor(test_cropped, cv2.COLOR_BGR2RGB)
    # convert OpenCV to PIL
    test_pil = Image.fromarray(test_converted)



