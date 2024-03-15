import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from data_loader import RaindropDataset
from model import Generator, Discriminator
from torch.utils.data import DataLoader


def load_model(model_path, model):
    # Carga los pesos del modelo entrenado
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


def show_images(val_loader, generator):
    # Obtener un batch del conjunto de validación
    for batch in val_loader:
        rain_images, clear_images = batch["rain"].to(device), batch["clear"].to(device)
        break  # Solo necesitamos un batch para la demostración

    # Generar la imagen sin gotas de lluvia
    with torch.no_grad():
        fake_images = generator(rain_images)

    # Convertir los tensores a imágenes para visualizarlas
    # No necesitamos desnormalizar si tus imágenes ya están en el rango [0, 1]
    rain_image = T.ToPILImage()(rain_images[0].cpu())
    generated_image = T.ToPILImage()(fake_images[0].cpu())

    # Visualizar las imágenes usando matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(rain_image)
    axs[0].set_title("Rainy Image")
    axs[0].axis("off")

    axs[1].imshow(generated_image)
    axs[1].set_title("Generated Image Without Raindrops")
    axs[1].axis("off")

    plt.show()


# Definiciones previas
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Asegúrate de definir y compilar tu modelo generador aquí con la arquitectura correcta antes de cargar los pesos
generator = Generator(10, (3, 224, 224)).to(
    device
)  # Asumiendo que tienes las variables attention_blocks e image_shape definidas

# Cargar el modelo
model_path = "best_generator.pth"  # Actualiza la ruta según corresponda
load_model(model_path, generator)

val_dataset = RaindropDataset("..\datasets\Raindrop_dataset\\test_b")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# Mostrar imágenes
# Asegúrate de tener val_loader definido como en tu código original
show_images(val_loader, generator)
