import torch
import torch.cuda.random
import torch.nn as nn

import cv2
from PIL import Image

# import the custom module
from detector.detection_model import Net
from modify_images.face_crop import Face
from detector.dataset_image import ImageTransform

detector_path = "checkpoint/saved_1.pth"
classes = {
    0: 'no_mask',
    1: 'mask'
}

USE_CUDA = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(USE_CUDA)
torch.manual_seed(2390783316385493207)  # 2390783316385493207: initial random seed
if torch.cuda.is_available():
    torch.cuda.random.manual_seed_all(6399406970264898)  # 6399406970264898: initial random seed of cuda


# to load a trained model to use as a detector
def load_detector(cuda):
    checkpoint = torch.load(detector_path)
    model = Net([3, 4, 6, 3]).to(DEVICE)  # based on ResNet34
    if cuda == 'cuda':
        model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'], strict=False)
    if cuda == 'cuda':
        model.eval()
        model.cuda()
    return model


# to convert an image for detecting
def convert_image(imgpath, crop=True):
    if crop:
        image = Face(imgpath).crop()
    else:
        image = cv2.imread(imgpath)
    # convert from 'BGR' to 'RGB'
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # convert OpenCV to PIL
    image_pil = Image.fromarray(image_rgb)
    return image_pil


def detection(image, detector):
    transform = ImageTransform(size=(128, 128))
    image = torch.unsqueeze(transform(img=image), 0)  # transform an image for detecting
    result = detector(image)
    return result


if __name__ == "__main__":
    print("without mask...")
    for i in range(4):
        test_path = "face_images/sample_%d.jpg" % i
        test_image = convert_image(test_path, crop=True)
        detector = load_detector(USE_CUDA)
        print(detection(image=test_image, detector=detector).item())
    print("with mask...")
    for i in range(2):
        test_path_1 = "face_images/mask_cropped_%d.jpg" % i
        test_image_1 = convert_image(test_path_1, crop=False)
        detector = load_detector(USE_CUDA)
        print(detection(image=test_image_1, detector=detector).item())