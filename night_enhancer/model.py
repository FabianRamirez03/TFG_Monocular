import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet152_Weights


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class DecomNet(nn.Module):
    def __init__(self):
        super(DecomNet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            # You can add more residual blocks here
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Assuming the output is an image
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.residual_blocks(x)
        x = self.final(x)
        return x


class IllumNet(nn.Module):
    def __init__(self):
        super(IllumNet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 64),
            # Fewer residual blocks compared to DecomNet
        )
        self.final = nn.Sequential(
            nn.Conv2d(
                64, 1, kernel_size=3, padding=1
            ),  # Assuming we want a 1-channel illumination map
            nn.Sigmoid(),  # The illumination map values should be between [0,1]
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.residual_blocks(x)
        x = self.final(x)
        return x


class DarkEnhancementNet(nn.Module):
    def __init__(self):
        super(DarkEnhancementNet, self).__init__()
        self.decom_net = DecomNet()
        self.illum_net = IllumNet()  # Define the illumination network

    def forward(self, x):
        reflectance = self.decom_net(x)
        illumination = self.illum_net(x)
        return reflectance, illumination
