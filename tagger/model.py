import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DualInputCNN(nn.Module):
    def __init__(self):
        super(DualInputCNN, self).__init__()

        # Utilizamos un modelo preentrenado y reutilizamos las capas convolucionales para ambas ramas.
        base_model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.base_layers = nn.Sequential(
            *list(base_model.children())[:-2]
        )  # Excluir las capas de clasificación finales.

        # Agregar una capa adaptativa de pooling promedio para manejar diferentes tamaños de imagen
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Capas de clasificación después de la fusión
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                base_model.fc.in_features * 2, 512
            ),  # base_model.fc.in_features es la cantidad de características antes de las capas finales en ResNet18
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 6),  # 6 etiquetas finales
        )

    def forward(self, x_upper, x_lower):
        # Procesar la parte superior e inferior con las mismas capas base
        x_upper = self.base_layers(x_upper)
        x_lower = self.base_layers(x_lower)

        # Aplicar pooling para convertir las características a un tamaño fijo
        x_upper = self.avgpool(x_upper)
        x_lower = self.avgpool(x_lower)

        # Aplanar las características y concatenarlas
        x_upper = torch.flatten(x_upper, 1)
        x_lower = torch.flatten(x_lower, 1)
        x_fused = torch.cat((x_upper, x_lower), dim=1)

        # Clasificación final
        out = self.classifier(x_fused)

        return out


class DecomNet(nn.Module):
    def __init__(self):
        super(DecomNet, self).__init__()
        # Capas convolucionales para la descomposición inicial
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Capas para la reflectancia
        self.reflec1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.reflec2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        # Capas para la luminancia
        self.lum1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.lum2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Propagación común inicial
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Rama de la reflectancia
        reflectance = F.relu(self.reflec1(x))
        reflectance = torch.sigmoid(
            self.reflec2(reflectance)
        )  # Usar sigmoid para mantener los valores entre 0 y 1

        # Rama de la luminancia
        luminance = F.relu(self.lum1(x))
        luminance = torch.sigmoid(
            self.lum2(luminance)
        )  # Usar sigmoid para mantener los valores entre 0 y 1

        return reflectance, luminance


class EnhanceNet(nn.Module):
    def __init__(self):
        super(EnhanceNet, self).__init__()
        # La entrada ahora es de 1 canal, proveniente de la luminancia de DenoisingNet
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Agregamos más capas para profundizar la red
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # Incluimos conexiones residuales más complejas
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_final = nn.Conv2d(
            64, 1, kernel_size=3, padding=1
        )  # Devuelve a 1 canal para la luminancia mejorada

    def forward(self, x):
        # Conexión residual a nivel de entrada
        residual = x
        # Propagación a través de las capas
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Reducción y conexión residual interna
        x = F.relu(self.conv5(x))

        x = x + residual  # Primera conexión residual

        # Capa final para la luminancia mejorada
        x = self.conv_final(x)

        # Podríamos aplicar otra conexión residual aquí si se desea, dependiendo de la profundidad deseada
        return x
