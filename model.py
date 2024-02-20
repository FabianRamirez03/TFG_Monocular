import torch
import torch.nn as nn
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
