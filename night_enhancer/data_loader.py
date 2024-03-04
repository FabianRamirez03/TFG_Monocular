import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode
import random


def add_artificial_lights_to_dark_image(
    darkened_array, light_count=7, max_light_radius=180
):
    # Crear una imagen PIL a partir del array para poder dibujar sobre ella
    darkened_image = Image.fromarray(darkened_array)
    draw = ImageDraw.Draw(
        darkened_image, "RGBA"
    )  # Utilizar modo RGBA para transparencia

    for _ in range(random.randint(0, light_count)):
        # Posición y propiedades de la luz artificial
        light_position = (
            np.random.randint(0, darkened_array.shape[1]),
            np.random.randint(0, darkened_array.shape[0]),
        )
        light_color = (
            np.random.randint(180, 256),  # R
            np.random.randint(180, 256),  # G
            np.random.randint(180, 256),  # B
        )
        light_radius = np.random.randint(20, max_light_radius)

        # Crear un gradiente radial para la luz
        for radius in range(light_radius, 0, -5):
            alpha = int(
                (255 * (light_radius - radius) ** 2) / light_radius**2
            )  # Atenuación más realista
            draw.ellipse(
                (
                    light_position[0] - radius,
                    light_position[1] - radius,
                    light_position[0] + radius,
                    light_position[1] + radius,
                ),
                fill=light_color + (alpha,),
            )

    # Aplicar un desenfoque gaussiano para simular el desenfoque de la luz en la noche
    darkened_image = darkened_image.filter(ImageFilter.GaussianBlur(radius=2))

    return np.array(darkened_image)


def darken_sky(image_array, max_darken_factor, min_darken_factor):
    height = image_array.shape[0]
    # Crear un gradiente que vaya de max_darken_factor a min_darken_factor
    gradient = np.linspace(max_darken_factor, min_darken_factor, height).reshape(
        height, 1, 1
    )
    # Aplicar el gradiente al cielo, asumiendo que el cielo está en la parte superior de la imagen
    darkened_sky = image_array * gradient
    return darkened_sky


def reduce_saturation(image_array, saturation_factor):
    # Convertir a PIL Image para reducir la saturación
    image = Image.fromarray(image_array.astype(np.uint8))
    converter = ImageEnhance.Color(image)
    # Reducir la saturación usando el factor proporcionado
    image = converter.enhance(
        saturation_factor
    )  # Un valor menor que 1 reduce la saturación
    return np.array(image)


# Función para oscurecer manualmente la imagen
def darken_image_manual(
    image,
    darken_factor=0.1,
    brightness_variation_range=(0.001, 0.01),
    saturation_factor=0.4,
):
    # Convertir la imagen a un array de NumPy y oscurecerla
    image_array = np.array(image).astype(float)
    height, width = image_array.shape[:2]

    # Aplicar un oscurecimiento base
    darkened_array = image_array * darken_factor

    # Simulación de variabilidad en la iluminación
    brightness_variation = random.uniform(*brightness_variation_range)
    darkened_array *= brightness_variation

    # Aplicar HDR y tone mapping simulado
    darkened_array = np.clip(darkened_array, 0, 255).astype(np.uint8)
    darkened_array = (
        np.log(1 + darkened_array) / np.log(256)
    ) * 255  # Simulación simple de tone mapping

    # Oscurecer el cielo de manera más agresiva
    darkened_array = darken_sky(
        image_array, max_darken_factor=0.01, min_darken_factor=0.5
    )

    # Reducir la saturación para evitar colores extraños en el cielo
    darkened_array = reduce_saturation(darkened_array, saturation_factor)

    # Añadir luces artificiales
    darkened_array = np.clip(darkened_array, 0, 255).astype(np.uint8)
    darkened_array = add_artificial_lights_to_dark_image(darkened_array)

    # Convertir el array oscurecido de vuelta a una imagen PIL y retornar
    return Image.fromarray(darkened_array)


dark_transforms = transforms.Compose([transforms.Lambda(darken_image_manual)])


class NightDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
    ):

        self.annotations = pd.read_csv(csv_file)
        # Filtrar por imágenes etiquetadas como soleado o nublado
        self.annotations = self.annotations[self.annotations["noche"] == 1]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        # Aplicar otras transformaciones si existen
        if self.transform:
            original_image = self.transform(image)
        else:
            original_image = image

        return original_image


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


def test_dataset_loop():
    while True:
        test_dataset()
        a = input("E para salir, cualquier tecla continuar \n")
        if a.lower() == "e":
            break


test_dataset_loop()
