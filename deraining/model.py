import torch
from torch import nn
import torch.nn.functional as F
from util import save_feature_map_combined


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class ConvolutionalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels // 8),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class Deraining_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Define the architecture here
        self.down1 = ConvBlock(in_channels, 64)
        self.att_d_1 = ConvolutionalAttention(64)

        self.down2 = ConvBlock(64, 128)
        self.att_d_2 = ConvolutionalAttention(128)

        self.down3 = ConvBlock(128, 256)
        self.att_d_3 = ConvolutionalAttention(256)

        self.down4 = ConvBlock(256, 512)
        self.att_d_4 = ConvolutionalAttention(512)

        self.middle = ConvBlock(512, 1024)
        self.att_mid = ConvolutionalAttention(1024)

        self.up1 = UpConv(1024, 512)
        self.upconv1 = ConvBlock(1024, 512)
        self.att_up_m1 = ConvolutionalAttention(512)

        self.up2 = UpConv(512, 256)
        self.upconv2 = ConvBlock(512, 256)
        self.att_up_m2 = ConvolutionalAttention(256)

        self.up3 = UpConv(256, 128)
        self.upconv3 = ConvBlock(256, 128)
        self.att_up_m3 = ConvolutionalAttention(128)

        self.up4 = UpConv(128, 64)
        self.upconv4 = ConvBlock(128, 64)
        self.att4 = ConvolutionalAttention(64)

        self.final_conv = nn.Conv2d(64, out_channels, 1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        # Downward path with attention
        save_feature_map_combined(x, "temp_images\\deraining\\input.png")

        d1 = self.down1(x)
        d1 = self.att_d_1(d1)

        save_feature_map_combined(d1, "temp_images\\deraining\\d1.png")

        d2 = self.down2(F.max_pool2d(d1, 2))
        d2 = self.att_d_2(d2)

        save_feature_map_combined(d2, "temp_images\\deraining\\d2.png")

        d3 = self.down3(F.max_pool2d(d2, 2))
        d3 = self.att_d_3(d3)
        save_feature_map_combined(d3, "temp_images\\deraining\\d3.png")

        d4 = self.down4(F.max_pool2d(d3, 2))
        d4 = self.att_d_4(d4)

        save_feature_map_combined(d4, "temp_images\\deraining\\d4.png")

        # Middle with attention
        middle = self.middle(F.max_pool2d(d4, 2))
        middle = self.att_mid(middle)

        save_feature_map_combined(middle, "temp_images\\deraining\\mid.png")

        # Upward path with attention
        u1 = self.up1(middle)
        u1 = torch.cat((u1, d4), 1)
        u1 = self.upconv1(u1)
        u1 = self.att_up_m1(u1)
        save_feature_map_combined(u1, "temp_images\\deraining\\u1.png")

        u2 = self.up2(u1)
        u2 = torch.cat((u2, d3), 1)
        u2 = self.upconv2(u2)
        u2 = self.att_up_m2(u2)
        save_feature_map_combined(u2, "temp_images\\deraining\\u2.png")

        u3 = self.up3(u2)
        u3 = torch.cat((u3, d2), 1)
        u3 = self.upconv3(u3)
        u3 = self.att_up_m3(u3)
        save_feature_map_combined(u3, "temp_images\\deraining\\u3.png")

        u4 = self.up4(u3)
        u4 = torch.cat((u4, d1), 1)
        u4 = self.upconv4(u4)
        u4 = self.att4(u4)
        save_feature_map_combined(u4, "temp_images\\deraining\\u4.png")

        output = self.final_conv(u4)
        save_feature_map_combined(
            self.final_activation(output), "temp_images\\deraining\\output.png"
        )
        return self.final_activation(output)
