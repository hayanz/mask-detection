import torch
import torch.cuda.random
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import argparse

# import custom modules
from dataset_image import *
from detection_model import Net

parser = argparse.ArgumentParser(description="Custom model training:")
parser.add_argument("--resume", '-r', action='store_true', help="resume from checkpoint")
args = parser.parse_args()

USE_CUDA = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(USE_CUDA)
torch.manual_seed(1051325718700006030)  # 1051325718700006030: initial random seed
if USE_CUDA == 'cuda':
    torch.cuda.random.manual_seed_all(6763088558125263)  # 6763088558125263: initial random seed of cuda

batch_size = 20  # number of batch size
num_iter = 5000  # total number of iterations
best_accuracy = 0  # accuracy of the testset
start_epoch = 0  # initial epoch (changes if last checkpoint exists)
learning_rate = 0.1  # learning rate to train the model
ratio = 0.3  # (number of images in test set) / (number of all images)

# set the dataset with image files and labels
images, labels = separate_csv(make_csv("dataset"))
img_dataset = ImgDataset(ImageTransform(size=(128, 128)), files=images, labels=labels)  # full dataset
train_set, test_set = divide_sets(img_dataset, ratio=ratio)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# calculate the total number of epochs
total_epochs = int(num_iter / (len(train_set) / batch_size))

# set the model to train
model = Net([3, 4, 6, 3]).to(DEVICE)  # based on ResNet34
if USE_CUDA == 'cuda':
    model = nn.DataParallel(model)
    cudnn.benchmark = True

criterion = nn.MSELoss()  # measures the mean squared error (squared L2 norm)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)


# to save the state of the model as the checkpoint
def save_state(acc, epoch):
    state = {'model': model.state_dict(), 'accuracy': acc, 'epoch': epoch}
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/point.pth')
    return acc


# to load the state
def load_state(resume=False):
    global best_accuracy, start_epoch
    if resume:
        # load the checkpoint
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint')
        checkpoint = torch.load('./checkpoint/point.pth')
        model.load_state_dict(checkpoint['model'])
        best_accuracy = checkpoint['acc']
        start_epoch = checkpoint['epoch']


# to calculate the accuracy
def calculate_acc(output, answer):
    total = answer.size(0)
    return output.eq(answer).sum().item() / float(total)


# to train the model
def train_model(epoch):
    train_cost = 0  # initialization
    model.train()  # train the model
    train_best = 0
    for batch_idx, (inputs, answers) in enumerate(train_loader):
        answers = answers.unsqueeze(-1)
        inputs, answers = inputs.to(DEVICE), answers.to(DEVICE)
        predictions = model(inputs)
        # calculate the value of cost
        cost = criterion(predictions, answers)
        # calculate the hypothesis with the value of cost
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        train_cost += cost.item()
        accuracy = calculate_acc(predictions, answers)
        if accuracy > train_best:
            train_best = accuracy
        # print the result
        print("[%d] batch: cost = %d, accuracy = %d" % (batch_idx, cost.item(), accuracy))

    return train_cost, train_best


# to test the model
def test_model(epoch):
    global best_accuracy
    test_cost = 0
    test_best = 0

    with torch.no_grad():
        for batch_idx, (inputs, answers) in enumerate(test_loader):
            answers = answers.unsqueeze(-1)
            inputs, answers = inputs.to(DEVICE), answers.to(DEVICE)
            predicted = model(inputs)
            # calculate the value of cost
            cost = criterion(predicted, answers)

            test_cost += cost.item()
            accuracy = calculate_acc(inputs, answers)
            if accuracy > test_best:
                test_best = accuracy
            # print the result
            print("[%d] batch: cost = %d, accuracy = %d" % (batch_idx, cost.item(), accuracy))

    # save if it has the highest accuracy
    if test_best > best_accuracy:
        save_state(test_best, epoch)
        best_accuracy = test_best  # update

    return test_cost, test_best


if __name__ == "__main__":
    # load_state(args.resume)
    for epoch in range(start_epoch, total_epochs):
        print("[%4d/%4d] Epoch" % (epoch + 1, total_epochs))
        break  # TODO implement this
