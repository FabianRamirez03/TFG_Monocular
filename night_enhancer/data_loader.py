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
    darkened_array, light_count=10, max_light_radius=150
):
    # Crear una imagen PIL a partir del array para poder dibujar sobre ella
    darkened_image = Image.fromarray(darkened_array)
    draw = ImageDraw.Draw(
        darkened_image, "RGBA"
    )  # Utilizar modo RGBA para transparencia

    for _ in range(random.randint(1, light_count)):
        # Posición de la luz artificial
        light_position = (
            np.random.randint(0, darkened_array.shape[1]),
            np.random.randint(0, darkened_array.shape[0]),
        )

        # Seleccionar un color de luz aleatoriamente para el borde del círculo
        outer_light_color = random.choice(
            [
                (255, 240, 200),
                (255, 100, 100),
                (191, 0, 15),
                (135, 149, 175),
                (134, 169, 215),
            ]
        )  # Amarillento o Rojizo
        light_radius = np.random.randint(30, max_light_radius)

        # Crear un gradiente radial para la luz
        for radius in range(light_radius, 0, -2):
            # Interpolar entre blanco en el centro y el color seleccionado hacia el borde
            ratio = radius / light_radius
            r = int((255 * (1 - ratio)) + outer_light_color[0] * ratio)
            g = int((255 * (1 - ratio)) + outer_light_color[1] * ratio)
            b = int((255 * (1 - ratio)) + outer_light_color[2] * ratio)
            alpha = int(
                (1 - ratio**0.05) * 255
            )  # Aumenta el exponente para una caída más rápida de la transparencia

            draw.ellipse(
                (
                    light_position[0] - radius,
                    light_position[1] - radius,
                    light_position[0] + radius,
                    light_position[1] + radius,
                ),
                fill=(r, g, b, alpha),
            )

    # Aplicar un desenfoque gaussiano para simular el desenfoque de la luz en la noche
    darkened_image = darkened_image.filter(ImageFilter.GaussianBlur(radius=5))

    return np.array(darkened_image).astype(np.uint8)


def darken_sky(image_array, sky_darken_factor=0.1, transition_height=0.5):
    """
    Oscurece la parte superior de la imagen para simular un cielo nocturno.

    Args:
    - image_array: numpy array de la imagen.
    - sky_darken_factor: factor de oscurecimiento aplicado al cielo.
    - transition_height: altura de la transición del oscurecimiento expresada como
      una fracción de la altura total de la imagen.

    Returns:
    - numpy array de la imagen con el cielo oscurecido.
    """
    height, width, _ = image_array.shape
    sky_limit = int(height * transition_height)  # Donde comienza la transición
    transition_zone_height = height - sky_limit  # Altura de la zona de transición

    # Oscurecer la mitad superior de la imagen
    sky_darkened = np.ones((sky_limit, width, 3), dtype=np.uint8) * sky_darken_factor
    sky_darkened = image_array[:sky_limit] * sky_darkened

    # Crear gradiente para la mitad inferior de la imagen
    transition_gradient = np.linspace(
        sky_darken_factor, 1, transition_zone_height
    ).reshape(transition_zone_height, 1)
    transition_gradient = np.tile(transition_gradient, (1, width))
    transition_gradient = np.repeat(transition_gradient[:, :, np.newaxis], 3, axis=2)

    # Aplicar el gradiente a la zona de transición
    transition_darkened = image_array[sky_limit:height] * transition_gradient

    # Combinar el cielo oscurecido con la zona de transición
    combined_image = np.concatenate((sky_darkened, transition_darkened), axis=0)

    return combined_image.astype(np.uint8)


def reduce_saturation(image_array, saturation_factor):
    # Convertir a PIL Image para reducir la saturación
    image = Image.fromarray(image_array.astype(np.uint8))
    converter = ImageEnhance.Color(image)
    # Reducir la saturación usando el factor proporcionado
    image = converter.enhance(
        saturation_factor
    )  # Un valor menor que 1 reduce la saturación
    return np.array(image).astype(np.uint8)


def simulate_headlights(image_array, spread=0.1, min_intensity=0.3, max_intensity=0.7):
    height, width = image_array.shape[:2]
    headlights = np.zeros((height, width), dtype=np.float32)

    # La intensidad de los faros es un valor aleatorio dentro del rango dado
    intensity = np.random.uniform(min_intensity, max_intensity)

    # Simular la iluminación central de los faros
    center_x = width // 2
    horizon_y = int(
        height * 0.8
    )  # Asumimos que el horizonte está al 80% de la altura de la imagen

    # El spread determina qué tan rápido se reduce la intensidad de la luz desde el centro
    spread = 1 / (spread * width)

    for y in range(horizon_y, height):
        for x in range(width):
            dx = center_x - x
            dy = horizon_y - y
            distance = np.sqrt(dx * dx + dy * dy)
            headlights[y, x] = np.exp(-distance * spread) * intensity

    # Asegurarse de que la máscara de faros no exceda la intensidad de 1
    headlights = np.clip(headlights, 0, 1)

    # Aplicamos la máscara de faros sobre la imagen original
    result = image_array.copy()
    for i in range(3):  # Aplicamos la máscara a cada canal de color
        result[horizon_y:, :, i] = image_array[horizon_y:, :, i] * headlights[
            horizon_y:, :
        ] + image_array[horizon_y:, :, i] * (1 - headlights[horizon_y:, :])

    return result.astype(np.uint8)


def darken_image_manual(image, darken_factor=1, saturation_factor=0.5):
    # Convertir la imagen a un array de NumPy y oscurecerla
    image_array = np.array(image).astype(np.uint8)
    darkened_array = (image_array * darken_factor).astype(np.uint8)

    # Oscurecer el cielo de manera más agresiva
    darkened_sky = darken_sky(darkened_array)

    # Simular la iluminación de los faros
    darkened_with_headlights = simulate_headlights(darkened_sky).astype(np.uint8)

    # Añadir luces artificiales
    darkened_with_lights = add_artificial_lights_to_dark_image(darkened_with_headlights)

    # Reducir la saturación
    darkened_with_lights_reduced_saturation = reduce_saturation(
        darkened_with_lights, saturation_factor
    )

    # Convertir el array oscurecido de vuelta a una imagen PIL y retornar
    return Image.fromarray(darkened_with_lights_reduced_saturation.astype(np.uint8))


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


# test_dataset_loop()
