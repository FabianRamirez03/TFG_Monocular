import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet152_Weights


class DarkEnhancementNet(nn.Module):
    def __init__(self, num_classes=3):
        super(DarkEnhancementNet, self).__init__()
        # Utilizar ResNet preentrenado como base
        resnet = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)

        # Extraer las capas de ResNet excepto la última capa fully connected
        self.base_layers = nn.Sequential(*list(resnet.children())[:-1])

        # Congelar los parámetros (pesos)
        for param in self.base_layers.parameters():
            param.requires_grad = False

        # Capas adicionales para la tarea específica
        self.enhancement_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Upsample(size=(56, 56)),  # Upsampling a un tamaño intermedio
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(112, 112)),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(224, 224)),  # Upsampling al tamaño de entrada
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Pasar la entrada a través de las capas base de ResNet
        x = self.base_layers(x)
        # Aplanar la salida para la secuencia de capas de realce
        x = x.view(x.size(0), -1, 1, 1)

        # Pasar a través de las capas de realce
        x = self.enhancement_layers(x)
        return x
