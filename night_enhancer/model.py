import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet152_Weights
import torch.nn.functional as F

import math
import matplotlib.pyplot as plt


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


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


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
            ResidualBlock(64, 64),
            # You can add more residual blocks here
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
        x = self.initial(x)
        x = self.residual_blocks(x)
        x = self.upsample_blocks(x)
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
            ResidualBlock(64, 64),
            # Fewer residual blocks compared to DecomNet
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


class DenoiseNet(nn.Module):
    def __init__(self, num_layers=17, num_features=64):
        super(DenoiseNet, self).__init__()
        layers = [
            DnCnnBlock(3, num_features, batch_norm=False)
        ]  # First layer without batch normalization
        for _ in range(num_layers - 2):
            layers.append(DnCnnBlock(num_features, num_features))
        layers.append(
            nn.Conv2d(num_features, 3, kernel_size=3, padding=1)
        )  # Final layer without ReLU
        self.denoise_net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.denoise_net(x)
        return x + out  # Residual learning


class DarkEnhancementNet(nn.Module):
    def __init__(self):
        super(DarkEnhancementNet, self).__init__()
        self.decom_net = DecomNet()
        self.illum_net = IllumNet()
        self.denoise_net = DenoiseNet()

    def forward(self, x, debug=False):
        reflectance = self.decom_net(x)
        illumination = self.illum_net(x)
        enhanced_image = reflectance * illumination
        enhanced_image = F.interpolate(
            enhanced_image, size=(224, 224), mode="bilinear", align_corners=False
        )

        if debug:
            # Asegúrate de que el tensor esté en el rango de 0 a 1 y en el CPU para visualización
            illum_map = (
                illumination.detach().cpu().squeeze(0)
            )  # Asumiendo que el batch size es 1
            plt.imshow(
                illum_map[0], cmap="gray"
            )  # [0] para seleccionar el primer canal si es necesario
            plt.title("Illumination Map")
            plt.colorbar()
            plt.show()

        # denoised_image = self.denoise_net(enhanced_image)  # Aplicar la red de denoising
        return enhanced_image
