import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datetime import datetime, timedelta
from torchvision.transforms.functional import InterpolationMode
import time

# Suponiendo que CustomDataset es el nombre de tu clase de dataset
from data_loader import DarkenerDataset
from model import DarkEnhancementNet

# Parámetros
batch_size = 128
num_epochs = 200
patience = 5  # Número de épocas para esperar después de una mejora antes de detener el entrenamiento
epochs_no_improve = 0
learning_rate = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
workers = 5
prefecth = 5

# Transformaciones
transform = transforms.Compose(
    [
        # Redimensionar la imagen a 232x232. Ajustamos ligeramente el tamaño según tu descripción.
        transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),
        # Recortar el centro de la imagen a 224x224.
        transforms.CenterCrop(224),
        # Convertir la imagen a un tensor de PyTorch.
        transforms.ToTensor(),
        # Normalizar con los valores medios y desviaciones estándar de ImageNet.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Carga del dataset
dataset = DarkenerDataset(
    csv_file="..\\frames_labels.csv",
    root_dir="..\\datasets\\custom_dataset\\Processed",
    transform=transform,
)

# Dividir en conjuntos de entrenamiento y validación
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    prefetch_factor=prefecth,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    prefetch_factor=prefecth,
)


def main():
    # Modelo
    print("Begin train")
    model = DarkEnhancementNet()
    model = model.to(device)
    print("Model loaded")

    # Función de pérdida y optimizador
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Entrenamiento
    best_loss = float("inf")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculamos el ETA para la finalización de la época
            current_time = time.time()
            elapsed_time = current_time - start_time
            images_processed = (i + 1) * batch_size
            total_images = len(train_loader.dataset)
            eta = elapsed_time / images_processed * (total_images - images_processed)

            if (i + 1) % 10 == 0:
                print(
                    f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, ETA for epoch: {timedelta(seconds=int(eta))}"
                )

        epoch_loss = running_loss / len(train_loader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time elapsed: {time.time() - start_time}"
        )

        # Validación
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Guardar el mejor modelo
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0  # Restablecer el contador de paciencia
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved")
        else:
            epochs_no_improve += 1
            print(
                f"No se observaron mejoras en {epochs_no_improve} época(s). Mejor val_loss: {best_loss}"
            )
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping ejecutado después de {patience} épocas sin mejora."
                )
                break  # Detener el entrenamiento

        # ETA para la siguiente época
        time_elapsed = time.time() - start_time
        time_remaining = (num_epochs - epoch - 1) * time_elapsed
        print(f"ETA for next epoch: {time_elapsed}")
        print(f"ETA for total: {time_remaining}")

        # Guardar el modelo cada 10 épocas
    if (epoch + 1) % 10 == 0:
        save_path = f"model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Modelo guardado: {save_path}")

    print("Training complete")


if __name__ == "__main__":
    main()
