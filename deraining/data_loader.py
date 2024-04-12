import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import pandas as pd


class RaindropDataset(Dataset):
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
                os.path.join(data_dir, "data", img)
                for img in os.listdir(os.path.join(data_dir, "data"))
                if img.endswith("_rain.png") or img.endswith("_rain.jpg")
            ]
        )
        self.gt_images = sorted(
            [
                os.path.join(data_dir, "gt", img)
                for img in os.listdir(os.path.join(data_dir, "gt"))
                if img.endswith("_clean.png") or img.endswith("_clean.jpg")
            ]
        )

        if transform is None:
            self.transform = Compose(
                [
                    Resize((224, 224)),
                    ToTensor(),
                ]
            )

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
        rain_image_path = self.data_images[idx]
        clear_image_path = self.gt_images[idx]

        rain_image = Image.open(rain_image_path)
        clear_image = Image.open(clear_image_path)

        if self.transform:
            rain_image = self.transform(rain_image)
            clear_image = self.transform(clear_image)

        return {"rain": rain_image, "clear": clear_image}


class RainCustomDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
    ):

        self.annotations = pd.read_csv(csv_file)
        # Filtrar por imágenes etiquetadas como soleado o nublado
        self.annotations = self.annotations[self.annotations["lluvia"] == 1]
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
    data_dir = "..\datasets\Raindrop_dataset\\train"
    raindrop_dataset = RaindropDataset(data_dir)
    raindrop_dataloader = DataLoader(raindrop_dataset, batch_size=1, shuffle=True)

    # Obtener un batch de imágenes
    for batch in raindrop_dataloader:
        rain_image, clear_image = batch["rain"], batch["clear"]
        # Mostrar las imágenes
        show_images(
            rain_image.squeeze(0), clear_image.squeeze(0)
        )  # squeeze(0) para remover el batch dimension"
        a = input("E para salir, cualquier tecla continuar \n")
        if a.lower() == "e":
            break


# test_dataset()
