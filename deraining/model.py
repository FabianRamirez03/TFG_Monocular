import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class AttentionGeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64, num_layers=2):
        super(AttentionGeneratorBlock, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bloques convolucionales para procesar la entrada
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )

        # Bloques residuales
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_size, hidden_size),
            ResidualBlock(hidden_size, hidden_size),
        )

        # LSTM para procesar las características secuenciales
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Upsample usando convolución transpuesta
        self.upsample_conv = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_size,
                hidden_size // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(hidden_size // 2, out_channels, kernel_size=1)

    def forward(self, x, prev_attention=None):
        # Pasar la entrada a través de las capas convolucionales
        conv_out = self.conv_blocks(x)

        # Pasar la salida de las convoluciones a través de los bloques residuales
        res_out = self.res_blocks(conv_out)

        # Global average pooling para obtener una representación secuencial
        pooled_out = res_out.mean(dim=[2, 3])  # Tamaño: [batch_size, hidden_size]
        pooled_out = pooled_out.unsqueeze(1)  # Tamaño: [batch_size, 1, hidden_size]

        # Preparar el estado inicial para LSTM si es necesario
        batch_size = x.size(0)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        h_prev, c_prev = (h_0, c_0) if prev_attention is None else prev_attention

        # Pasar las características agrupadas a través de LSTM
        lstm_out, (h, c) = self.lstm(pooled_out, (h_prev, c_prev))

        # Realizar upsampling en la salida de LSTM
        lstm_out = lstm_out[:, -1, :].view(batch_size, self.hidden_size, 1, 1)

        upsample_out = self.upsample_conv(lstm_out)
        upsample_out = F.interpolate(
            upsample_out, size=(224, 224), mode="bilinear", align_corners=False
        )

        attention_map = self.conv_out(upsample_out)

        return attention_map, (h, c)


class SkipConnection(nn.Module):
    def __init__(self, input_channels):
        super(SkipConnection, self).__init__()

        # Incrementar el número de canales
        self.expand = nn.Conv2d(
            input_channels, input_channels * 2, kernel_size=3, padding=1
        )
        self.relu1 = nn.ReLU(inplace=True)

        # Reducir el tamaño (downsampling)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # Procesamiento adicional en la resolución reducida
        self.process = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels * 4, input_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Incrementar el tamaño (upsampling)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Reducir nuevamente el número de canales y ajustar la salida
        self.refine = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1),
        )

    def forward(self, img, attention_map):
        expanded_attention = attention_map.repeat(1, img.size(1), 1, 1)

        # Aplica el mapa de atención a la imagen original
        attention_applied = img * expanded_attention

        # Procesar con capas convolucionales
        x = self.expand(attention_applied)
        x = self.relu1(x)
        x = self.downsample(x)
        x = self.process(x)
        x = self.upsample(x)
        x = self.refine(x)

        return x


class Generator(nn.Module):
    def __init__(self, num_attention_blocks, image_shape):
        super(Generator, self).__init__()

        self.num_attention_blocks = num_attention_blocks
        self.image_shape = image_shape
        channels, height, width = image_shape

        # AttentionGeneratorBlocks
        self.attention_blocks = nn.ModuleList(
            [
                (
                    AttentionGeneratorBlock(3, 1)
                    if i == 0
                    else AttentionGeneratorBlock(1, 1)
                )
                for i in range(self.num_attention_blocks)
            ]
        )

        self.skip_connection = SkipConnection(channels)

    def forward(self, x):
        attention_maps = []
        prev_attention = None
        original_x = x
        for i in range(self.num_attention_blocks):
            x, prev_attention = self.attention_blocks[i](x, prev_attention)
            attention_maps.append(x)

        show_attention_maps(attention_maps)
        # Agregar función de mostar attention maps acá
        last_attention_map = x
        # Skip connections
        output_image = self.skip_connection(original_x, last_attention_map)

        return output_image


def show_attention_maps(attention_maps):
    # Visualizar los mapas de atención uno por uno
    fig, axes = plt.subplots(1, len(attention_maps), figsize=(20, 10))
    for i, attention_map in enumerate(attention_maps):
        ax = axes[i] if len(attention_maps) > 1 else axes
        # Convertir el mapa de atención de PyTorch a una imagen de PIL para visualizar
        attention_map_pil = TF.to_pil_image(attention_map.squeeze(0).cpu().detach())
        ax.imshow(attention_map_pil, cmap="hot")
        ax.axis("off")
        ax.set_title(f"Attention Map {i+1}")
    plt.show()


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        # Define las capas convolucionales con una cantidad constante de canales
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                input_channels, input_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                input_channels, input_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                input_channels, input_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                input_channels, input_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # Suponiendo que hay una capa de downsampling aquí
            nn.Conv2d(
                input_channels, input_channels, kernel_size=3, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                input_channels, input_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Capa final antes de las capas completamente conectadas
        self.final_conv = nn.Conv2d(
            input_channels, input_channels, kernel_size=3, padding=1
        )

        # La capa completamente conectada que clasifica como real o falso
        self.fc_layer = nn.Sequential(
            # Si todas las capas anteriores mantienen el tamaño constante excepto una con stride de 2
            nn.Linear(112 * 112 * input_channels, 1),
        )

    def forward(self, x):
        # Pasar la entrada a través de las capas convolucionales
        conv_out = self.conv_layers(x)
        conv_out = self.final_conv(conv_out)

        # Aplanamos la salida para pasarla a la capa completamente conectada
        conv_out_flat = conv_out.view(conv_out.size(0), -1)

        # Pasar la salida aplanada a través de la capa completamente conectada
        fc_out = self.fc_layer(conv_out_flat)

        return fc_out
