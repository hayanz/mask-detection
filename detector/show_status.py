import os
import torch
import pandas as pd

# import a custom module
from model_cnn import Net

USE_CUDA = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(USE_CUDA)


# to print out key-value pairs of the dictionary
def print_status(path, csv_path):
    assert os.path.isfile(path), os.path.isfile(csv_path)
    checkpoint = torch.load(path)
    accuracy = checkpoint['accuracy']
    epoch = checkpoint['epoch']

    csvfile = pd.read_csv(csv_path)
    train_loss = csvfile['train_cost'][epoch - 1]
    test_loss = csvfile['test_cost'][epoch - 1]

    status = [('accuracy', accuracy), ('epoch', epoch), ('train loss', train_loss), ('test_loss', test_loss)]
    for k, v in status:
        if type(v) == int:
            print("%s: %d" % (k, v))
            continue
        print("%s: %f" % (k, v))


if __name__ == "__main__":
    print("train with simple dataset")
    print_status("main_ckpt/best_1.pth", "main_ckpt/result_1.csv")
    print("train with more complex dataset")
    print_status("additional_ckpt/saved_second.pth", "additional_ckpt/result_second.csv")
