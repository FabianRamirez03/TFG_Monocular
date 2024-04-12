import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from data_loader import RainCustomDataset, RaindropDataset
from model import Deraining_UNet
from torch.utils.data import DataLoader
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models\\best_generator.pth"
# model_path = "models\\generator_epoch_10.pth"

# Cargar el modelo
generator = Deraining_UNet(in_channels=3, out_channels=3).to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

# Definir transformaciones
transformations = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Cargar el dataset
dataset = RainCustomDataset(
    csv_file="..\\frames_labels.csv",  # Reemplaza con la ruta de tu archivo CSV
    root_dir="..\\datasets\\custom_dataset\\Processed",  # Reemplaza con la ruta de tu directorio de imágenes
    transform=transformations,
)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

val_dataset = RaindropDataset("../datasets/Raindrop_dataset/test_b")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


# Visualizar resultados
def visualize_results_raindropDataset(data_loader, generator, device):
    with torch.no_grad():
        for i, batch in enumerate(data_loader, start=1):
            rain_image = batch["rain"].to(device)
            generated_image = generator(rain_image)

            # Mostrar imagen real
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(rain_image.cpu().squeeze(0).permute(1, 2, 0))
            plt.title("Original Image")
            plt.axis("off")

            # Mostrar imagen generada
            plt.subplot(1, 3, 2)
            plt.imshow(generated_image.cpu().squeeze(0).permute(1, 2, 0))
            plt.title("Generated Image")
            plt.axis("off")

            plt.show()

            if i >= 10:  # Muestra solo las primeras 5 imágenes para el ejemplo
                break


# Visualizar resultados
def visualize_results(data_loader, generator, device):
    with torch.no_grad():
        for i, real_image in enumerate(data_loader, start=1):
            real_image = real_image.to(device)
            generated_image = generator(real_image)

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

            plt.show()

            if i >= 10:  # Muestra solo las primeras 5 imágenes para el ejemplo
                break


# visualize_results(data_loader, generator, device)
visualize_results_raindropDataset(val_loader, generator, device)
