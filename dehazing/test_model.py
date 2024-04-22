import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from data_loader import OHazeDataset
from model import Dehazing_UNet
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models\\best_model.pth"
# model_path = "models\\generator_epoch_40.pth"

# Cargar el modelo
generator = Dehazing_UNet(3, 3).to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()


# Cargar el dataset
data_dir = "..\datasets\O-Haze-Cityscapes"
oHaze_dataset = OHazeDataset(data_dir)
data_loader = DataLoader(oHaze_dataset, batch_size=1, shuffle=True)

mean = [
    0.5909,
    0.6072,
    0.5899,
]
std = [0.5909, 0.6072, 0.5899]


def denormalize(tensor, mean, std):
    """Reverses the normalization on a tensor."""
    tensor = tensor.clone()  # make a copy of the tensor
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # apply denormalization
    return tensor


# Visualizar resultados
def visualize_results_raindropDataset(data_loader, generator, device):
    global mean, std
    with torch.no_grad():
        for i, batch in enumerate(data_loader, start=1):
            rain_image = batch["rain"].to(device)
            generated_image = generator(rain_image)

            # Mostrar imagen real
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 0)
            plt.imshow(rain_image.cpu().squeeze(0).permute(1, 2, 0))
            plt.title("Original Image")
            plt.axis("off")

            # Mostrar imagen generada
            plt.subplot(1, 3, 1)
            plt.imshow(generated_image.cpu().squeeze(0).permute(1, 2, 0))
            plt.title("Generated Image")
            plt.axis("off")

            plt.show()

            if i >= 10:  # Muestra solo las primeras 5 imágenes para el ejemplo
                break


# Visualizar resultados
def visualize_results(data_loader, generator, device):
    global mean, std

    with torch.no_grad():
        for batch in data_loader:
            haze_image, real_image = batch["hazy"].to(device), batch["clear"].to(device)

            generated_image = generator(haze_image)

            # Mostrar imagen real
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(real_image.cpu().squeeze(0).permute(1, 2, 0))
            plt.title("Original Image")
            plt.axis("off")

            # Mostrar imagen generada
            plt.subplot(1, 3, 2)
            plt.imshow(generated_image.cpu().squeeze(0).permute(1, 2, 0))
            plt.title("Generated Image")
            plt.axis("off")

            # Mostrar imagen generada
            plt.subplot(1, 3, 3)
            plt.imshow(haze_image.cpu().squeeze(0).permute(1, 2, 0))
            plt.title("Haze Image")
            plt.axis("off")

            plt.show()


# Supongamos que esta es la función que carga y transforma la imagen
def load_transform_image(image_path):
    # Transformaciones comunes podrían incluir
    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),  # Suponiendo que el modelo trabaja con imágenes de 256x256
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Añade una dimensión de lote


# Visualizar resultados
def visualize_results_from_path(data_loader, generator, device):

    image_path = (
        "..\datasets\custom_dataset\Processed\BlueFalls-parqueo-neblina\\000000114.png"
    )
    with torch.no_grad():
        haze_image = load_transform_image(image_path).to(device)
        generated_image = generator(haze_image)

    # Mostrar imagen generada
    plt.subplot(1, 2, 1)
    plt.imshow(generated_image.cpu().squeeze(0).permute(1, 2, 0))
    plt.title("Generated Image")
    plt.axis("off")

    # Mostrar imagen generada
    plt.subplot(1, 2, 2)
    plt.imshow(haze_image.cpu().squeeze(0).permute(1, 2, 0))
    plt.title("Haze Image")
    plt.axis("off")

    plt.show()


visualize_results(data_loader, generator, device)
# visualize_results_from_path(data_loader, generator, device)
