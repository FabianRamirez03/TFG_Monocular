import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader
from data_loader import (
    LOLDataset,
    CustomDataset,
)
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from model import LowLightEnhancer

# Configura el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo
model = LowLightEnhancer().to(device)
model.load_state_dict(
    torch.load(
        "models\\first_version_light_enhancer\\best_low_light_enhancer_model.pth"
    )
)
model.eval()

# Transformaciones para las imágenes de entrada
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # Agrega aquí cualquier otra transformación que hayas usado durante el entrenamiento
    ]
)


# Función para visualizar la imagen
def show_images(low_light_img, enhanced_img):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(
        np.array(low_light_img)
    )  # Convierte la imagen PIL a un array de NumPy
    axs[0].set_title("Low Light Image")
    axs[1].imshow(np.array(enhanced_img))  # Convierte la imagen PIL a un array de NumPy
    axs[1].set_title("Enhanced Image")
    for ax in axs:
        ax.axis("off")
    plt.show()


def LOL_main():
    # Directorio del conjunto de datos
    dataset_dir = "LOLdataset/train"  # Asegúrate de cambiar esto por la ruta correcta de tu conjunto de datos

    # Crear un DataLoader para el conjunto de datos
    dataset = LOLDataset(directory=dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Bucle para procesar y mostrar imágenes aleatorias
    for low_light_img, _ in dataloader:
        low_light_img = low_light_img.to(device)
        with torch.no_grad():
            enhanced_img = model(low_light_img)

        # Mostrar las imágenes
        show_images(low_light_img[0].cpu(), enhanced_img[0].cpu())

        # Preguntar al usuario si quiere continuar
        cont = input("¿Quieres ver la siguiente imagen aleatoria? (s/n): ")
        if cont.lower() != "s":
            break


def custom_main():
    csv_file = "frames_labels.csv"
    root_dir = "custom_dataset\Processed"

    dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Procesar y mostrar imágenes aleatorias mejoradas
    for images, labels in dataloader:
        # Comprueba si la imagen tiene la etiqueta "noche"
        if labels[0, 0] != 1:  # Asumiendo que la primera etiqueta corresponde a "noche"
            continue  # Si no es "noche", pasa a la siguiente imagen

        low_light_image = images.to(device)
        with torch.no_grad():
            enhanced_image = model(low_light_image)

        enhanced_image = (enhanced_image - enhanced_image.min()) / (
            enhanced_image.max() - enhanced_image.min()
        )

        low_light_image = low_light_image.cpu().squeeze(0)
        enhanced_image = enhanced_image.cpu().squeeze(0)

        print("Antes de la conversión a PIL:")

        # Verificar el tipo de objeto
        print(f"Tipo de low_light_image: {type(low_light_image)}")
        print(f"Tipo de enhanced_image: {type(enhanced_image)}")

        # Verificar el tamaño de los tensores
        print(f"Tamaño de low_light_image: {low_light_image.size()}")
        print(f"Tamaño de enhanced_image: {enhanced_image.size()}")

        # Verificar el rango de valores de los tensores
        print(
            f"Valores min y max en low_light_image: {low_light_image.min()}, {low_light_image.max()}"
        )
        print(
            f"Valores min y max en enhanced_image: {enhanced_image.min()}, {enhanced_image.max()}"
        )

        # Convertir tensores a imágenes PIL para visualización
        original_img_pil = transforms.ToPILImage()(low_light_image)
        enhanced_img_pil = transforms.ToPILImage()(enhanced_image)

        show_images(original_img_pil, enhanced_img_pil)

        if input("Mostrar otra imagen? (s/n): ").lower() != "s":
            break


if __name__ == "__main__":
    LOL_main()
    # custom_main()
