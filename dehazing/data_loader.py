import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import pandas as pd

mean = [
    0.5909,
    0.6072,
    0.5899,
]
std = [0.5909, 0.6072, 0.5899]


class OHazeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Inicializa el dataset.

        Parámetros:
            data_dir (str): Ruta al directorio donde están las carpetas 'data' y 'gt'.
            transform (callable, opcional): Opcional transform para ser aplicado a las parejas de imágenes.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data_images = sorted(
            [
                os.path.join(data_dir, "hazy", img)
                for img in os.listdir(os.path.join(data_dir, "hazy"))
                if img.endswith("_hazy.png") or img.endswith("_hazy.jpg")
            ]
        )
        self.gt_images = sorted(
            [
                os.path.join(data_dir, "GT", img)
                for img in os.listdir(os.path.join(data_dir, "GT"))
                if img.endswith("_GT.png")
                or img.endswith("_GT.jpg")
                or img.endswith("_clean.png")
                or img.endswith("_GT.jpg")
            ]
        )

        if transform is None:
            self.transform = Compose([Resize((224, 224)), ToTensor()])

    def __len__(self):
        """Devuelve el número total de pares de imágenes en el dataset."""
        return len(self.data_images)

    def __getitem__(self, idx):
        """
        Obtiene un par de imágenes (con lluvia y sin lluvia) por índice.

        Parámetros:
            idx (int): Índice del par de imágenes a obtener.

        Retorna:
            Un dict con las imágenes 'rain' y 'clear' como tensores.
        """
        hazy_image_path = self.data_images[idx]
        clear_image_path = self.gt_images[idx]

        hazy_image = Image.open(hazy_image_path)
        clear_image = Image.open(clear_image_path)

        if self.transform:
            hazy_image = self.transform(hazy_image)
            clear_image = self.transform(clear_image)

        return {"hazy": hazy_image, "clear": clear_image}

    def denormalize(self, tensor):
        """Reverses the normalization on a tensor."""
        tensor = tensor.clone()  # make a copy of the tensor
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)  # apply denormalization
        return tensor


def show_images(rain_image, clear_image, title="Image Pair"):
    """
    Muestra un par de imágenes: una con lluvia y su correspondiente versión limpia.

    Parámetros:
        rain_image (Tensor): Imagen con lluvia.
        clear_image (Tensor): Imagen limpia.
        title (str): Título para la figura.
    """
    # Convertir tensores a imágenes PIL
    rain_image_pil = TF.to_pil_image(rain_image)
    clear_image_pil = TF.to_pil_image(clear_image)

    # Visualizar las imágenes usando matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(rain_image_pil)
    axs[0].set_title("Rainy Image")
    axs[0].axis("off")

    axs[1].imshow(clear_image_pil)
    axs[1].set_title("Clear Image")
    axs[1].axis("off")

    plt.suptitle(title)
    plt.show()


def test_dataset():
    data_dir = "..\datasets\O-Haze"
    raindrop_dataset = OHazeDataset(data_dir)
    raindrop_dataloader = DataLoader(raindrop_dataset, batch_size=1, shuffle=True)

    # Obtener un batch de imágenes
    for batch in raindrop_dataloader:
        rain_image, clear_image = batch["hazy"], batch["clear"]
        # Mostrar las imágenes
        show_images(
            rain_image.squeeze(0), clear_image.squeeze(0)
        )  # squeeze(0) para remover el batch dimension"
        a = input("E para salir, cualquier tecla continuar \n")
        if a.lower() == "e":
            break


def calculate_mean_std_aux(dataset):
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    mean = 0.0
    std = 0.0
    nb_samples = 0.0
    for data in loader:
        # Suponiendo que la salida del dataset es un diccionario con las claves 'hazy' y 'clear'
        batch_samples = data["hazy"].size(0)
        data = data["hazy"].view(batch_samples, data["hazy"].size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


def calculate_mean_std():
    dataset_path = "..\datasets\O-Haze-Cityscapes"
    full_dataset = OHazeDataset(dataset_path)

    # Calcular la media y desviación estándar
    mean, std = calculate_mean_std_aux(full_dataset)

    print(f"Media del conjunto de datos: {mean}")
    print(f"Desviación estándar del conjunto de datos: {std}")


# calculate_mean_std()
# test_dataset()
