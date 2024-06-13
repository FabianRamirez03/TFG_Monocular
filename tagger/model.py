import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from util import save_feature_map_combined, save_random_feature_map


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
        save_feature_map_combined(x_upper, "temp_images\\tagger\\x_upper.png")
        save_feature_map_combined(x_lower, "temp_images\\tagger\\x_lower.png")

        x_upper = self.base_layers(x_upper)
        x_lower = self.base_layers(x_lower)

        save_feature_map_combined(x_upper, "temp_images\\tagger\\x_upper_resnet.png")
        save_feature_map_combined(x_lower, "temp_images\\tagger\\x_lower_resnet.png")

        save_random_feature_map(x_upper, "temp_images\\tagger\\u_r1.png")
        save_random_feature_map(x_upper, "temp_images\\tagger\\u_r2.png")
        save_random_feature_map(x_upper, "temp_images\\tagger\\u_r3.png")

        save_random_feature_map(x_lower, "temp_images\\tagger\\d_r1.png")
        save_random_feature_map(x_lower, "temp_images\\tagger\\d_r2.png")
        save_random_feature_map(x_lower, "temp_images\\tagger\\d_r3.png")

        # Aplicar pooling para convertir las características a un tamaño fijo
        x_upper = self.avgpool(x_upper)
        x_lower = self.avgpool(x_lower)

        save_feature_map_combined(x_upper, "temp_images\\tagger\\avgpool_upper.png")
        save_feature_map_combined(x_lower, "temp_images\\tagger\\avgpool_lower.png")

        # Aplanar las características y concatenarlas
        x_upper = torch.flatten(x_upper, 1)
        x_lower = torch.flatten(x_lower, 1)

        x_fused = torch.cat((x_upper, x_lower), dim=1)

        print(x_fused.shape)

        # Clasificación final
        out = self.classifier(x_fused)

        return out
