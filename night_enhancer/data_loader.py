import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import matplotlib.pyplot as plt


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


# Función para oscurecer manualmente la imagen
def darken_image_manual(image, darken_factor=0.15):
    # Convertir la imagen a un array de NumPy
    image_array = np.array(image)

    # Asegurarse de que la imagen esté en el rango correcto y tenga tres dimensiones para RGB
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("La imagen debe tener tres dimensiones para RGB")

    # Multiplicar los valores de los píxeles por el factor de oscurecimiento
    # Convertir a float para la operación y luego a uint8 para evitar problemas de desbordamiento
    darkened_array = np.clip(image_array.astype(float) * darken_factor, 0, 255).astype(
        np.uint8
    )

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
    # Ruta al archivo CSV con anotaciones y al directorio de imágenes
    csv_file = "..\\frames_labels.csv"
    root_dir = "..\\datasets\\custom_dataset\\Processed"

    # Crear una instancia del dataset
    dataset = DarkenerDataset(
        csv_file=csv_file,
        root_dir=root_dir,
    )

    # Seleccionar una imagen para probar (por ejemplo, la primera imagen del dataset)
    original_image, darkened_image = dataset[0]

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


# test_dataset()
