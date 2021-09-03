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
num_iter = 1000  # total number of iterations
best_accuracy = 0  # accuracy of the test set
start_epoch = 0  # initial epoch (changes if last checkpoint exists)
learning_rate = 0.1  # learning rate to train the model
ratio = 0.2  # (number of images in test set) / (number of all images)

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
def save_state(acc, epoch, best=False):
    global model, optimizer
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'accuracy': acc, 'epoch': epoch}
    if not os.path.isdir('../checkpoint'):
        os.mkdir('../checkpoint')
    if best:
        torch.save(state, '../checkpoint/best_1.pth')
        print("The value of the best accuracy is updated.")
    else:
        torch.save(state, '../checkpoint/saved_1.pth')
        print("The state is saved.")


# to load the state
def load_state(resume=False):
    global best_accuracy, start_epoch, model, optimizer
    if resume:
        # load the checkpoint
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint')
        checkpoint = torch.load('../checkpoint/saved_1.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_accuracy = checkpoint['accuracy']
        start_epoch = checkpoint['epoch']


# to calculate the accuracy
def calculate_acc(output, answer):
    output[output > 0.5] = 1
    output[output <= 0.5] = 0
    acc = output.eq(answer).sum().float() / float(answer.size(0))
    acc *= 100
    return acc


# to train the model
def train_model(epoch):
    global batch_size
    global model, optimizer, criterion
    train_cost = 0  # initialization
    train_best = 0  # maximum value of the accuracy (initialization)
    acc_total = 0  # sum of whole accuracy to get the average (initialization)
    model.train()  # train the model

    for batch_idx, (inputs, answers) in enumerate(train_loader):
        answers = answers.unsqueeze(-1)
        inputs, answers = inputs.to(DEVICE), answers.to(DEVICE)
        answers = answers.type(torch.FloatTensor).cuda()
        predictions = model(inputs)
        # calculate the value of cost
        cost = criterion(predictions, answers)
        # calculate the hypothesis with the value of cost
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()  # update

        train_cost += cost.item()  # update
        accuracy = calculate_acc(predictions, answers)
        acc_total += accuracy  # update
        if accuracy > train_best:
            train_best = accuracy  # update
        # print the result
        # print("[%d] batch: cost = %.5f, accuracy = %.5f" % (batch_idx, cost.item(), accuracy))

    # calculate the average of the accuracy for one epoch
    acc_mean = acc_total / float(len(train_loader))
    print("total cost: %.5f | average accuracy: %.5f" % (train_cost, acc_mean))

    return train_cost, train_best.item(), acc_mean.item()


# to test the model
def test_model(epoch):
    global best_accuracy, batch_size
    global model, optimizer, criterion
    test_cost = 0  # initialization
    test_best = 0  # maximum value of the accuracy (initialization)
    acc_total = 0  # sum of whole accuracy to get the average (initialization)

    with torch.no_grad():
        for batch_idx, (inputs, answers) in enumerate(test_loader):
            answers = answers.unsqueeze(-1)
            inputs, answers = inputs.to(DEVICE), answers.to(DEVICE)
            predicted = model(inputs)
            # calculate the value of cost
            cost = criterion(predicted, answers)

            test_cost += cost.item()  # update
            accuracy = calculate_acc(predicted, answers)
            acc_total += accuracy  # update
            if accuracy > test_best:
                test_best = accuracy  # update
            # print the result
            # print("[%d] batch: cost = %.5f, accuracy = %.5f" % (batch_idx, cost.item(), accuracy))

    # calculate the average of the accuracy for one epoch
    acc_mean = acc_total / float(len(test_loader))
    print("total cost: %.5f | average accuracy: %.5f" % (test_cost, acc_mean))
    # save if it has the highest accuracy
    if acc_mean > best_accuracy:
        save_state(acc_mean, epoch, best=True)
        best_accuracy = acc_mean  # update

    return test_cost, test_best.item(), acc_mean.item()


# save the result to a csv file
def save_csv(epoch, train_result, test_result):
    assert type(train_result) == list and type(test_result) == list
    assert len(train_result) == 3 and len(test_result) == 3

    if not os.path.isfile('../checkpoint/result_1.csv'):  # check if csv file exists
        with open('../checkpoint/result_1.csv', 'w') as f:  # create a file
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_cost', 'train_max_acc', 'train_acc',
                             'test_cost', 'test_max_acc', 'test_acc'])
    result = [epoch + 1] + train_result + test_result  # list of the result to save

    with open('../checkpoint/result_1.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(result)


if __name__ == "__main__":
    # load the state
    load_state(args.resume)
    # train the model
    for epoch in range(start_epoch, total_epochs):
        print("[%4d/%4d] Epoch" % (epoch + 1, total_epochs))
        train_result = list(train_model(epoch))
        test_result = list(test_model(epoch))
        save_csv(epoch, train_result, test_result)
        save_state(test_result[2], epoch)  # test_result[2] = the average accuracy of the test set
