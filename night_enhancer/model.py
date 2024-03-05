import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet152_Weights
import torch.nn.functional as F

import math
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, debug=False
    ):
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

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

        self.debug = debug

    def forward(self, x):
        debug_print("ResidualBlock initial:", x.size(), self.debug)
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        debug_print("ResidualBlock before conv1:", x.size(), self.debug)
        out = self.conv1(x)
        debug_print("ResidualBlock after conv1:", x.size(), self.debug)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        debug_print("ResidualBlock final:", x.size(), self.debug)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_in_channels=None):
        super(UpsampleBlock, self).__init__()
        self.skip_in_channels = skip_in_channels
        total_in_channels = in_channels + (
            skip_in_channels if skip_in_channels is not None else 0
        )

        self.conv = nn.Conv2d(total_in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip_connection=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if self.skip_in_channels is not None:
            x = torch.cat((x, skip_connection), dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecomNet(nn.Module):
    def __init__(self, debug):
        super(DecomNet, self).__init__()
        self.debug = debug
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 128),
            ResidualBlock(128, 64),
            ResidualBlock(64, 64),
        )

        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(64, 64),
            UpsampleBlock(64, 32),
        )

        self.final = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Assuming the output is an image
        )

    def forward(self, x):
        debug_print("DecomNet initial size", x.size(), self.debug)
        x = self.initial(x)
        debug_print("DecomNet after initial size", x.size(), self.debug)
        x = self.residual_blocks(x)
        debug_print("DecomNet residual blocks size", x.size(), self.debug)
        x = self.upsample_blocks(x)
        debug_print("Upsample blocks size", x.size(), self.debug)
        x = self.final(x)
        debug_print("Final size", x.size(), self.debug)
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
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 512),
            ResidualBlock(512, 256),
            ResidualBlock(256, 128),
            ResidualBlock(128, 64),
            ResidualBlock(64, 64),
        )
        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(64, 64),
            UpsampleBlock(64, 32),
        )
        self.final = nn.Sequential(
            nn.Conv2d(
                32, 1, kernel_size=3, padding=1
            ),  # Assuming we want a 1-channel illumination map
            nn.Sigmoid(),  # The illumination map values should be between [0,1]
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.residual_blocks(x)
        x = self.upsample_blocks(x)
        x = self.final(x)
        return x


class DnCnnBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, padding=1, batch_norm=True
    ):
        super(DnCnnBlock, self).__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=not batch_norm,
            )
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.dncnn_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.dncnn_block(x)


class DarkEnhancementNet(nn.Module):
    def __init__(self):
        super(DarkEnhancementNet, self).__init__()
        self.debug = False
        self.decom_net = DecomNet(self.debug)
        self.illum_net = IllumNet()

    def forward(self, x):
        debug_print("Inital size", x.size(), self.debug)
        reflectance = self.decom_net(x)
        illumination = self.illum_net(x)
        enhanced_image = reflectance * illumination
        enhanced_image = F.interpolate(
            enhanced_image, size=(224, 224), mode="bilinear", align_corners=False
        )
        return enhanced_image


def debug_print(key, value, debug):

    if debug:
        print(f"{key}: {value}")
