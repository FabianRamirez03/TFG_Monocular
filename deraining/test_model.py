import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from data_loader import RainCustomDataset
from model import Generator, Discriminator
from torch.utils.data import DataLoader
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models\\best_generator.pth"

# Cargar el modelo
generator = Generator(3, 128, 3).to(device)
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


# Visualizar resultados
def visualize_results(data_loader, generator, device):
    with torch.no_grad():
        for i, real_image in enumerate(data_loader, start=1):
            real_image = real_image.to(device)
            generated_image, attention_map = generator(real_image)

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

            # Mostrar mapa de atención
            plt.subplot(1, 3, 3)
            plt.imshow(attention_map.cpu().squeeze(0).permute(1, 2, 0), cmap="jet")
            plt.title("Attention Map")
            plt.axis("off")

            plt.show()

            if i >= 10:  # Muestra solo las primeras 5 imágenes para el ejemplo
                break


visualize_results(data_loader, generator, device)
