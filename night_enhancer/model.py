import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet152_Weights


class DarkEnhancementNet(nn.Module):
    def __init__(self, num_classes=3):
        super(DarkEnhancementNet, self).__init__()
        # Utilizar ResNet preentrenado como base
        resnet = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)

        # Afinar algunas de las últimas capas convolucionales
        self.base_layers = nn.Sequential(*list(resnet.children())[:-2])

        # Módulo de atención
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, kernel_size=1),
            nn.Sigmoid(),
        )

        # Capas de realce con normalización por lotes
        self.enhancement_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Upsampling a 14x14
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Upsampling a 28x28
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Upsampling a 56x56
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Upsampling a 112x112
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Upsampling a 224x224
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Pasar la entrada a través de las capas base de ResNet
        # print(f"x initial size: {x.size()}")
        x = self.base_layers(x)
        # print(f"x after base layers size: {x.size()}")

        # Módulo de atención
        attention = self.attention(x)
        x = x * attention
        # print(f"x after attention size: {x.size()}")

        # Pasar a través de las capas de realce
        x = self.enhancement_layers(x)
        # print(f"x after enhancement size: {x.size()}")

        return x
