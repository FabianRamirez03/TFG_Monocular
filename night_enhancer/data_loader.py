import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode
import random


# Function to perform gamma correction using PIL
def adjust_gamma_pil(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    lut = [pow(x / 255.0, inv_gamma) * 255 for x in range(256)] * 3
    lut = np.array(lut, dtype=np.uint8)
    return Image.fromarray(np.array(image)).point(lut)

    # Convierte la imagen a array y asegúrate de que es de tipo uint8
    image_array = np.array(image).astype(np.uint8)

    # Aplica la LUT utilizando el método point de PIL
    return Image.fromarray(image_array).point(lut)


# Function to perform contrast adjustment using PIL
def adjust_contrast_pil(image, alpha=1.0):
    return TF.adjust_contrast(image, alpha)


# Function to perform the darkening process using PIL and PyTorch
def darken_transform(image, gamma_val=3, alpha_val=0.5):
    image = adjust_gamma_pil(image, gamma_val)
    image = adjust_contrast_pil(image, alpha_val)
    return image


def add_artificial_lights_to_dark_image(darkened_array):
    num_lights = random.randint(0, 5)  # Número aleatorio de fuentes de luz
    # Crear una imagen PIL a partir del array para poder dibujar sobre ella
    darkened_image = Image.fromarray(darkened_array)
    draw = ImageDraw.Draw(
        darkened_image, "RGBA"
    )  # Utilizar modo RGBA para transparencia

    for _ in range(num_lights):
        # Posición y propiedades de la luz artificial
        light_position = (
            np.random.randint(0, darkened_array.shape[1]),
            np.random.randint(0, darkened_array.shape[0] / 3),
        )
        light_intensity = np.random.randint(180, 255)
        light_radius = np.random.randint(20, 200)

        # Crear un gradiente radial para la luz
        for radius in range(light_radius, 0, -5):
            alpha = int(
                (255 * (light_radius - radius) ** 3) / light_radius**3
            )  # Transparencia más tenue hacia los bordes
            draw.ellipse(
                (
                    light_position[0] - radius,
                    light_position[1] - radius,
                    light_position[0] + radius,
                    light_position[1] + radius,
                ),
                fill=(light_intensity, light_intensity, light_intensity, alpha),
            )

    return np.array(darkened_image)


# Función para oscurecer manualmente la imagen
def darken_image_manual(image, darken_factor=0.08):
    """
    Oscurece la imagen de entrada y simula variabilidad en la iluminación y fuentes de luz artificiales.

    Args:
    image (PIL.Image): La imagen de entrada.
    darken_factor (float): Factor de oscurecimiento base de la imagen.
    Returns:
    PIL.Image: Imagen oscurecida con modificaciones.
    """
    # Convertir la imagen a un array de NumPy y oscurecerla
    image_array = np.array(image).astype(float)
    height = image_array.shape[0]

    # Crear un gradiente vertical para el factor de oscurecimiento
    gradient = np.linspace(0.01, 2, height).reshape(height, 1, 1)

    # Modificar el darken_factor a lo largo de la imagen usando el gradiente
    dynamic_darken_factor = darken_factor * gradient

    # Aplicar el gradiente de oscurecimiento de manera diferente en la mitad superior e inferior
    darkened_array = image_array * dynamic_darken_factor

    # Asegurar que los valores estén dentro de los límites aceptables
    darkened_array = np.clip(darkened_array, 0, 255).astype(np.uint8)

    # Simulación de variabilidad en la iluminación
    brightness_variation = random.uniform(0.5, 1.5)
    darkened_array = np.clip(darkened_array * brightness_variation, 0, 255).astype(
        np.uint8
    )

    # Función para añadir luces artificiales aquí, si es necesario
    darkened_array = add_artificial_lights_to_dark_image(darkened_array)

    # Convertir el array oscurecido de vuelta a una imagen PIL y retornar
    return Image.fromarray(darkened_array)


dark_transforms = transforms.Compose([transforms.Lambda(darken_image_manual)])


class DarkenerDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
        darkening_transform=dark_transforms,
    ):
        """
        Args:
            csv_file (string): Ruta al archivo CSV con anotaciones.
            root_dir (string): Directorio con todas las imágenes.
            transform (callable, optional): Transformaciones opcionales a aplicar a las imágenes originales.
            darkening_transform (callable, optional): Transformaciones opcionales para oscurecer las imágenes.
        """
        self.annotations = pd.read_csv(csv_file)
        # Filtrar por imágenes etiquetadas como soleado o nublado
        self.annotations = self.annotations[
            (self.annotations["soleado"] == 1) | (self.annotations["nublado"] == 1)
        ]
        self.root_dir = root_dir
        self.transform = transform
        self.darkening_transform = darkening_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        # Aplicar la transformación de oscurecimiento si existe
        if self.darkening_transform:
            darkened_image = self.darkening_transform(image)
        else:
            darkened_image = image

        # Aplicar otras transformaciones si existen
        if self.transform:
            original_image = self.transform(image)
            darkened_image = self.transform(darkened_image)
        else:
            original_image = image

        return original_image, darkened_image


def test_dataset():

    dataset = DarkenerDataset(
        csv_file="..\\frames_labels.csv",
        root_dir="..\\datasets\\custom_dataset\\Processed",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # Asumiendo que tu clase DarkenerDataset necesita esta transformación
            ]
        ),
    )

    # Seleccionar una imagen para probar (por ejemplo, la primera imagen del dataset)
    i = random.randint(0, len(dataset) - 1)  # Obtiene un índice aleatorio
    original_image, darkened_image = dataset[i]

    if torch.is_tensor(original_image):
        original_image = original_image.permute(1, 2, 0).numpy()
    if torch.is_tensor(darkened_image):
        darkened_image = darkened_image.permute(1, 2, 0).numpy()

    # Visualizar las imágenes
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(darkened_image)
    plt.title("Darkened Image")
    plt.axis("off")

    plt.show()


while True:
    test_dataset()
    a = input("E para salir, cualquier tecla continuar \n")
    if a.lower() == "e":
        break
