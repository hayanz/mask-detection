import torch
import torch.cuda.random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

import csv
import pandas as pd

master_path = "datasets/"
torch.manual_seed(1051325718700006030)  # 1051325718700006030: initial random seed
if torch.cuda.is_available() == 'cuda':
    torch.cuda.random.manual_seed_all(6763088558125263)  # 6763088558125263: initial random seed of cuda


class ImageTransform:
    def __init__(self, size, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        assert type(size) == tuple
        self.data_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


class ImgDataset(Dataset):
    def __init__(self, transform, files, labels=None):
        self.transform = transform
        self.X = files
        if labels is not None:
            self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = Image.open(self.X[idx])
        transformed = self.transform(img)
        if self.y is None:
            return transformed
        else:
            return transformed, self.y[idx]


# to make the list of images and labels and save to csv file
def make_csv(filename):
    positive_path = "positive/"  # faces with mask
    negative_path = "negative/"  # faces without mask
    filetype = ".jpg"
    num_all = 1000

    positive = [positive_path + str(i) + filetype for i in range(num_all)]
    negative = [negative_path + str(i) + filetype for i in range(num_all)]

    filepath = master_path + filename + ".csv"
    with open(filepath, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "label"])
        for i in range(num_all):
            writer.writerow([positive[i], 1])
            writer.writerow([negative[i], 0])

    return filepath


# to divide the columns in csv file
def separate_csv(filename):
    datalist = pd.read_csv(filename)
    img = [master_path + path for path in datalist['filename']]
    labels = [label for label in datalist['label']]
    return img, labels


# to divide data to train set and test set
def divide_sets(dataset, ratio):
    X, y = dataset.X, dataset.y
    transform = dataset.transform
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, shuffle=True, random_state=1004)
    trainset = ImgDataset(transform, X_train, y_train)
    testset = ImgDataset(transform, X_test, y_test)
    return trainset, testset


# for debugging
if __name__ == "__main__":
    # datalist = pd.read_csv(make_csv("dataset"))
    # print(datalist.head())  # for debugging
    # print(datalist['label'].value_counts())  # for debugging

    images, labels = separate_csv(make_csv("dataset"))

    size = (100, 100)
    custom_transform = ImageTransform(size)
    data = ImgDataset(custom_transform, images, labels=labels)
    train, test = divide_sets(data, 0.3)
    print(train[0])  # for debugging
    print(test[0])  # for debugging
