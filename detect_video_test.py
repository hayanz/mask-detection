import torch
import torch.cuda.random
import os
import cv2
import numpy as np

# import custom modules
from detector.model_cnn import Net
from detector.dataset_image import ImageTransform

# best model of main_ckpt: best_1.pth
# best model of additional_ckpt: saved_seocnd.pth
# detector_path = "main_ckpt/best_1.pth"
detector_path = "main_ckpt/best_1.pth"
classifier = {
    0: 'no_mask',
    1: 'mask'
}


USE_CUDA = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(USE_CUDA)
torch.manual_seed(1051325718700006030)  # 1051325718700006030: initial seed
if USE_CUDA == 'cuda':
    torch.cuda.random.manual_seed_all(2826009950518439)  # 2826009950518439: initial seed of cuda


# to load the trained model for using as a detector
def load_detector(filepath, cuda):
    assert os.path.isfile(filepath)
    model = Net([3, 4, 6, 3]).to(DEVICE)  # declare the model

    loaded = torch.load(filepath)
    if cuda == 'cuda':
        model = torch.nn.DataParallel(model)
    model.load_state_dict(loaded['model'], strict=False)
    if cuda == 'cuda':
        model.eval()
        model.cuda()
    return model


def extract_faces(image, coordinates):
    pass  # TODO implement this


# to detect the image with a trained model
def detection(image, detector):
    transform = ImageTransform(size=(128, 128))
    image = torch.unsqueeze(transform(img=image), 0)  # transform an image for detecting
    result = detector(image)
    return result


# to classify the image with the result
def classify(prob):
    if prob <= 0:
        return classifier[0], 100
    elif prob >= 1:
        return classifier[1], 100
    elif prob < 0.5:
        accuracy = 100 * (1 - prob)
        return classifier[0], accuracy
    else:
        accuracy = 100 * prob
        return classifier[1], accuracy


if __name__ == "__main__":
    pass  # TODO implement this
