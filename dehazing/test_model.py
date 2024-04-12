import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from data_loader import OHazeDataset
from model import Dehazing_UNet
from torch.utils.data import DataLoader
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models\\best_model.pth"
# model_path = "models\\generator_epoch_101.pth"

# Cargar el modelo
generator = Dehazing_UNet(3, 3).to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()


# Cargar el dataset
data_dir = "..\datasets\O-Haze"
oHaze_dataset = OHazeDataset(data_dir)
data_loader = DataLoader(oHaze_dataset, batch_size=1, shuffle=True)


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


visualize_results(data_loader, generator, device)
