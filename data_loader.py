import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Ruta al archivo CSV con anotaciones.
            root_dir (string): Directorio con todas las imágenes.
            transform (callable, optional): Transformaciones opcionales a aplicar a las imágenes.
        """
        self.frame_annotations = pd.read_csv(csv_file)
        # Filtrar filas que no tienen etiquetas
        self.frame_annotations = self.frame_annotations[
            self.frame_annotations.iloc[:, 1:].sum(axis=1) != 0
        ]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame_annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.frame_annotations.iloc[idx, 0])
        image = Image.open(img_name)
        labels = self.frame_annotations.iloc[idx, 1:].values.astype(
            np.float32
        )  # Cambio aquí
        labels = torch.from_numpy(labels)  # Convertir a tensor de PyTorch

        if self.transform:
            image = self.transform(image)

        return image, labels


def load_split_data(csv_file, root_dir, batch_size=32, transform=None):
    dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

    # Dividir los datos en entrenamiento y validación
    train_size = int(0.7 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader


def imshow(img):
    """
    Función para mostrar una imagen
    """
    img = img / 2 + 0.5  # desnormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def test_data_loaders():
    """
    Prueba los DataLoaders mostrando el tamaño de cada conjunto y una imagen aleatoria de cada uno.
    """
    # Definir las transformaciones de las imágenes (ajuste según sea necesario)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalización para la visualización
        ]
    )

    # Uso del DataLoader
    csv_file = "frames_labels.csv"  # Asegúrate de ajustar la ruta
    root_dir = "custom_dataset\Processed"
    batch_size = 4

    train_loader, valid_loader = load_split_data(
        csv_file, root_dir, batch_size, transform
    )

    print(f"Largo del conjunto de entrenamiento: {len(train_loader.dataset)}")
    print(f"Largo del conjunto de validación: {len(valid_loader.dataset)}")

    # Obtener un batch de datos de entrenamiento
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Mostrar imagen aleatoria del conjunto de entrenamiento
    print("Imagen aleatoria del conjunto de entrenamiento:")
    print("Etiquetas:", labels[0].numpy())

    imshow(torchvision.utils.make_grid(images[0]))

    # Obtener un batch de datos de validación
    dataiter = iter(valid_loader)
    images, labels = next(dataiter)

    # Mostrar imagen aleatoria del conjunto de validación
    print("Imagen aleatoria del conjunto de validación:")
    print("Etiquetas:", labels[0].numpy())
    imshow(torchvision.utils.make_grid(images[0]))


test_data_loaders()
