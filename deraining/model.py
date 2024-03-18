import torch
from torch import nn
import torch.nn.functional as F
from data_loader import RaindropDataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
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
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AttentionMapGenerator(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(AttentionMapGenerator, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, hidden_channels, kernel_size=3, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(
                hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_channels * 2),
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_channels * 2, hidden_channels * 2),
            ResidualBlock(hidden_channels * 2, hidden_channels * 2),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels * 2,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_channels),
            nn.ConvTranspose2d(
                hidden_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.res_blocks(x)
        attention_map = self.upsample(x)
        return attention_map


class ImageGeneratorFromAttention(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super(ImageGeneratorFromAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Additional convolutional layers
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            hidden_channels, hidden_channels // 2, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(hidden_channels // 2)
        self.relu3 = nn.ReLU(inplace=True)

        # Final convolution to generate the output image
        self.final_conv = nn.Conv2d(
            hidden_channels // 2, out_channels, kernel_size=3, padding=1
        )
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x, attention_map):
        # Apply the attention map to the input image
        x = x * attention_map

        # Pass through the convolutional layers
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        # Generate the final image
        x = self.final_relu(self.final_bn(self.final_conv(x)))
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Generator, self).__init__()
        self.attention_map_generator = AttentionMapGenerator(
            in_channels, hidden_channels
        )
        self.image_generator = ImageGeneratorFromAttention(in_channels, out_channels)
        self.training_mode = True

    def forward(self, x, y=None):
        if self.training_mode and y is not None:
            # Modo de entrenamiento: usa tanto x como y
            input_attention_map_generator = torch.cat([x, y], dim=1)
        else:
            # Modo de inferencia: solo usa x, duplicando la entrada para simular la presencia de y
            input_attention_map_generator = torch.cat([x, x], dim=1)

        attention_map = self.attention_map_generator(input_attention_map_generator)
        out = self.image_generator(x, attention_map)
        return out, attention_map

    def set_training_mode(self, mode=True):
        """Establece el modo de entrenamiento del generador."""
        self.training_mode = mode


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


def test_full_generator():
    data_dir = "..\datasets\Raindrop_dataset\\train"
    model_path = "models\\best_generator.pth"
    # model_path = "generator_epoch_20.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raindrop_dataset = RaindropDataset(data_dir)
    raindrop_dataloader = DataLoader(raindrop_dataset, batch_size=1, shuffle=True)

    # Suponiendo que ya has definido e inicializado tu Generador Completo
    full_generator = Generator(3, 128, 3).to(device)
    full_generator.load_state_dict(torch.load(model_path, map_location=device))

    # Obtener un batch de imágenes
    for batch in raindrop_dataloader:
        rain_image, clear_image = batch["rain"].to(device), batch["clear"].to(device)

        # Generar imagen sin gotas usando el Generador Completo
        # Asegúrate de que full_generator está en modo evaluación y en el dispositivo correcto
        full_generator.eval()
        with torch.no_grad():
            generated_image, attention_map = full_generator(
                rain_image.to(device), clear_image.to(device)
            )

        # Mostrar las imágenes: lluviosa, limpia y la generada por el generador completo
        show_images_with_generated(
            rain_image.squeeze(0),  # Imagen con lluvia
            clear_image.squeeze(0),  # Imagen limpia
            generated_image.squeeze(0),  # Imagen generada
            attention_map.squeeze(0),
        )

        a = input("Press 'e' to exit, any other key to continue \n")
        if a.lower() == "e":
            break


def show_images_with_generated(
    rain_image, clear_image, generated_image, attention_map, title="Image Comparison"
):
    """
    Muestra tres imágenes: una con lluvia, su versión limpia y la generada por el modelo.
    """
    # Convertir tensores a imágenes PIL
    rain_image_pil = TF.to_pil_image(rain_image)
    clear_image_pil = TF.to_pil_image(clear_image)
    attention_map = TF.to_pil_image(attention_map)
    generated_image_pil = TF.to_pil_image(
        generated_image.clamp(0, 1)
    )  # Asegurar que los valores estén en [0, 1]

    # Visualizar las imágenes usando matplotlib
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    axs[0].imshow(rain_image_pil)
    axs[0].set_title("Rainy Image")
    axs[0].axis("off")

    axs[1].imshow(clear_image_pil)
    axs[1].set_title("Clear Image")
    axs[1].axis("off")

    axs[2].imshow(generated_image_pil)
    axs[2].set_title("Generated Image")
    axs[2].axis("off")

    axs[3].imshow(attention_map)
    axs[3].set_title("attention_map")
    axs[3].axis("off")

    plt.suptitle(title)
    plt.show()


# test_full_generator()
