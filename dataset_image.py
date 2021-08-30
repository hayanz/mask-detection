import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd


class ImageTransform:
    def __init__(self, size, mean, std):
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

    positive = [positive_path + str(i) + filetype for i in range(1000)]
    negative = [negative_path + str(i) + filetype for i in range(1000)]

    filepath = "face_images/" + filename + ".csv"
    with open(filepath, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "label"])
        for i in range(1000):
            writer.writerow([positive[i], 1])
            writer.writerow([negative[i], 0])

    return filepath


# for debugging
if __name__ == "__main__":
    master_path = "face_images/"
    datalist = pd.read_csv(make_csv("dataset"))
    print(datalist.head())  # for debugging
    print(datalist['label'].value_counts())  # for debugging

    train_img = [master_path + path for path in datalist['filename']]
    train_labels = datalist['label']

    size, one_half = (100, 100), (0.5, 0.5, 0.5)
    custom_transform = ImageTransform(size, mean=one_half, std=one_half)
    trains = ImgDataset(custom_transform, train_img, labels=train_labels)
