import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from model import DualInputCNN
from data_loader import (
    CustomDataset,
)  # Asegúrate de que data_loader.py esté en el mismo directorio que train.py
import os
import time
import copy


# Configuración de parámetros
resize_size = [232]
crop_size = [224]
batch_size = 50  # Puede ser ajustado según se requiera
workers = 16
prefetch_factor = 5
learning_rate = 0.0001
num_epochs = 150
csv_file = "..\\frames_labels.csv"
root_dir = "..\\datasets\\custom_dataset\\Processed"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformaciones como se especificó para ResNet50
transform = transforms.Compose(
    [
        transforms.Resize(
            resize_size, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def prepare_dataloaders(csv_file, root_dir, batch_size, transform):
    dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    train_size = int(0.7 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        prefetch_factor=prefetch_factor,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, valid_loader


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device="cpu"):
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Establecer el modelo en modo de entrenamiento
            else:
                model.eval()  # Establecer el modelo en modo de evaluación

            running_loss = 0.0
            start_time = time.time()

            # Iterar sobre los datos.
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                # Dividir las imágenes en secciones superior e inferior
                height = inputs.size(2)
                upper_section = inputs[:, :, : int(height * 0.25), :]
                lower_section = inputs[:, :, int(height * 0.25) :, :]

                upper_section, lower_section = upper_section.to(
                    device
                ), lower_section.to(device)
                labels = labels.to(device).type(torch.float32)

                optimizer.zero_grad()  # Poner los gradientes a cero

                # Adelante
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(upper_section, lower_section)
                    loss = criterion(outputs, labels)

                    # Retroceso + optimizar solo si estamos en fase de entrenamiento
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Estadísticas
                running_loss += loss.item() * inputs.size(0)
                # Calcular y mostrar el ETA.
                if phase == "train":
                    elapsed_time = time.time() - start_time
                    remaining_time = (
                        elapsed_time
                        / (batch_idx + 1)
                        * (len(dataloaders[phase]) - batch_idx - 1)
                    )
                    print(
                        f"\rBatch {batch_idx+1}/{len(dataloaders[phase])} - Loss: {loss.item():.4f} - ETA: {remaining_time:.0f}s",
                        end="",
                    )

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(f"\n{phase} Loss: {epoch_loss:.4f}")

            # Copia profunda del modelo
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "best_model.pth")
                print("Best model updated")

        if epoch % 5 == 4:  # Cada 5 épocas
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")

    # Cargar los mejores pesos del modelo
    model.load_state_dict(best_model_wts)
    return model


def main():
    dataloaders = {
        "train": prepare_dataloaders(csv_file, root_dir, batch_size, transform)[0],
        "val": prepare_dataloaders(csv_file, root_dir, batch_size, transform)[1],
    }
    model = DualInputCNN()
    criterion = (
        nn.BCEWithLogitsLoss()
    )  # Función de pérdida adecuada para clasificación multi-etiqueta
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    trained_model = train_model(
        model, dataloaders, criterion, optimizer, num_epochs, device
    )


if __name__ == "__main__":
    main()
