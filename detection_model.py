"""
The following custom class is based on ResNet;
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun,
    Deep Residual Learning for Image Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPANSION = 4


class SoftmaxClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


# class of the basic block based on deeper bottleneck block
class BaseBlock(nn.Module):
    def __init__(self, input_planes, planes, stride=1):
        super(BaseBlock, self).__init__()
        self.expansion = EXPANSION
        self.conv1 = nn.Conv2d(input_planes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # batch normalization
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # batch normalization
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(planes)  # batch normalization

        self.shortcut = nn.Sequential()
        # deeper residual function
        if stride != 1 or input_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(planes, self.expansion * planes, kernel_size=(1, 1), stride=1, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)  # for residual learning
        out = F.relu(out)
        return out


# class of the custom CNN based on ResNet (50/101/152 ResNet)
class Net(nn.Module):
    def __init__(self, num_blocks, num_classes=2):
        super(Net, self).__init__()
        self.input_planes = 64
        self.expansion = EXPANSION
        assert type(num_blocks) == list
        self.first, self.second, self.third, self.fourth = num_blocks

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # batch normalization
        self.layer1 = self.__custom_layer(64, self.first, stride=1)
        self.layer2 = self.__custom_layer(128, self.second, stride=2)
        self.layer3 = self.__custom_layer(256, self.third, stride=2)
        self.layer4 = self.__custom_layer(512, self.fourth, stride=2)
        self.softmax = SoftmaxClassifier(512 * self.expansion, num_classes)

    def __custom_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BaseBlock(self.input_planes, planes, stride=stride))
            self.input_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # flatten
        out = self.softmax(out)
        return out
