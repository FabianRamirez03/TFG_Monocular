import torch
import torch.nn as nn
import torch.nn.functional as F
from util import save_feature_map_combined


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),  # Instance Normalization
            nn.LeakyReLU(inplace=True),  # LeakyReLU
            nn.Dropout(0.1),  # Adding Dropout
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),  # Adding Dropout
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),  # Adding Dropout
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Adding Dropout
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Adding Dropout
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(
            in_channels, out_channels
        )  # in_channels + out_channels for concatenation

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Alineación del tamaño de las imágenes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Dehazing_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dehazing_UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        save_feature_map_combined(x, "temp_images\\dehazing\\input.png")
        x1 = self.inc(x)
        save_feature_map_combined(x1, "temp_images\\dehazing\\x1.png")
        x2 = self.down1(x1)
        save_feature_map_combined(x2, "temp_images\\dehazing\\x2.png")
        x3 = self.down2(x2)
        save_feature_map_combined(x3, "temp_images\\dehazing\\x3.png")
        x4 = self.down3(x3)
        save_feature_map_combined(x4, "temp_images\\dehazing\\x4.png")
        x5 = self.down4(x4)
        save_feature_map_combined(x5, "temp_images\\dehazing\\x5.png")
        x = self.up1(x5, x4)
        save_feature_map_combined(x, "temp_images\\dehazing\\up1.png")
        x = self.up2(x, x3)
        save_feature_map_combined(x, "temp_images\\dehazing\\up2.png")
        x = self.up3(x, x2)
        save_feature_map_combined(x, "temp_images\\dehazing\\up3.png")
        x = self.up4(x, x1)
        save_feature_map_combined(x, "temp_images\\dehazing\\up4.png")
        x = self.outc(x)
        save_feature_map_combined(x, "temp_images\\dehazing\\outc.png")

        save_feature_map_combined(
            torch.sigmoid(x), "temp_images\\dehazing\\sigmoid.png"
        )
        return torch.sigmoid(x)
