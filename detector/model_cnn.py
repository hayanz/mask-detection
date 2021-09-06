"""
The following custom class is based on ResNet;
Reference:
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun,
    Deep Residual Learning for Image Recognition
"""

# base code: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# class of the classifier based on softmax regression
class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


# class of the basic block based on residual learning
class ResBlock(nn.Module):
    def __init__(self, input_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # batch normalization
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # batch normalization

        self.shortcut = nn.Sequential()
        # deeper residual function
        if stride != 1 or input_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)  # batch normalization
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # for residual learning
        out = F.relu(out)
        return out


# class of the custom CNN based on ResNet (used in 18/34)
# the number of blocks is not fixed (to modify the model more easily)
class Net(nn.Module):
    def __init__(self, num_blocks, num_classes=1):
        super(Net, self).__init__()
        self.input_planes = 16
        assert type(num_blocks) in (list, tuple) and len(num_blocks) == 4
        self.first, self.second, self.third, self.fourth = num_blocks

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)  # batch normalization
        self.layer1 = self.__custom_layer(16, self.first, stride=1)
        self.layer2 = self.__custom_layer(32, self.second, stride=2)
        self.layer3 = self.__custom_layer(64, self.third, stride=2)
        self.layer4 = self.__custom_layer(128, self.fourth, stride=2)
        self.classifier = Classifier(128, num_classes)

    def __custom_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.input_planes, planes, stride=stride))
            self.input_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 16)  # average pool
        out = out.view(out.size(0), -1)  # flatten
        out = self.classifier(out)
        return out


# for debugging
if __name__ == "__main__":
    net = Net([3, 4, 6, 3])
    print(net)  # for debugging
    test = net(torch.rand([1, 3, 128, 128]))
    print(test.size())
