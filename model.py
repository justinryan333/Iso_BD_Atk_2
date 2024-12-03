# model.py
# Description: This file contains the model for the resnet-18 model
# this model will be brought into the main file to be used for training


# imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2

# ResNet-18 model
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, 10)

    def _make_layer(self, out_planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        in_planes = self.in_planes
        for stride in strides:
            layers.append(BasicBlock(in_planes, out_planes, stride))
            in_planes = out_planes  # Update input planes for the next block
        self.in_planes = out_planes  # Update the class's state
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Basic Block
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # Downsample the input if the dimensions don't match
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Add skip connection
        return F.relu(out)

# get the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



