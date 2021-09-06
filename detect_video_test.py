import torch
import torch.cuda.random
import os
import pyrealsense2 as rs

# import custom modules
from detector.model_cnn import Net
from detector.dataset_image import ImageTransform


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


def classify():
    pass  # TODO implement this


if __name__ == "__main__":
    pass  # TODO implement this
