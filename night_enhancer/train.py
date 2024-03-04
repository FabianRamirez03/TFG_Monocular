import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datetime import datetime, timedelta
from torchvision.transforms.functional import InterpolationMode
import time
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from data_loader import DarkenerDataset
from model import DarkEnhancementNet


def show_images(bright, dark):
    # Suponiendo que 'bright' y 'dark' son tensores de PyTorch con shape [B, C, H, W]
    # Donde B es el tamaño del batch, C es el número de canales, y H, W son la altura y anchura
    global mean, std

    # Desnormalizar las imágenes
    bright_denorm = denormalize(bright[0].cpu(), mean, std)
    dark_denorm = denormalize(dark[0].cpu(), mean, std)

    # Convertir los tensores a imágenes de PIL para visualizarlas
    bright_img_norm = TF.to_pil_image(bright[0].cpu())
    dark_img_norm = TF.to_pil_image(dark[0].cpu())
    bright_img_denorm = TF.to_pil_image(bright_denorm)
    dark_img_denorm = TF.to_pil_image(dark_denorm)

    # Visualizar las imágenes usando matplotlib
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Mostrar imágenes normalizadas
    axs[0, 0].imshow(bright_img_norm)
    axs[0, 0].set_title("Bright Image Normalized")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(dark_img_norm)
    axs[0, 1].set_title("Dark Image Normalized")
    axs[0, 1].axis("off")

    # Mostrar imágenes desnormalizadas
    axs[1, 0].imshow(bright_img_denorm)
    axs[1, 0].set_title("Bright Image Denormalized")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(dark_img_denorm)
    axs[1, 1].set_title("Dark Image Denormalized")
    axs[1, 1].axis("off")

    plt.show()


def denormalize(tensor, mean, std):
    # Clonamos el tensor para no hacer cambios in-place
    tensor = tensor.clone()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(
        mean[:, None, None]
    )  # Multiplicar y sumar para desnormalizar
    return tensor


# Parámetros
batch_size = 16
num_epochs = 200
patience = 10  # Número de épocas para esperar después de una mejora antes de detener el entrenamiento
epochs_no_improve = 0
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
workers = 20
prefecth = 20
std = [0.1714, 0.1724, 0.1898]
mean = [0.2534, 0.2483, 0.2497]
print_interval = 25

# Transformaciones
transform = transforms.Compose(
    [
        # Redimensionar la imagen a 232x232. Ajustamos ligeramente el tamaño según tu descripción.
        # transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),
        # Recortar el centro de la imagen a 224x224.
        # transforms.CenterCrop(224),
        # Convertir la imagen a un tensor de PyTorch.
        transforms.ToTensor(),
        # Normalizar con los valores medios y desviaciones estándar de ImageNet
        # transforms.Normalize(std=std, mean=mean),
    ]
)

# Carga del dataset
dataset = DarkenerDataset(
    csv_file="..\\frames_labels.csv",
    root_dir="..\\datasets\\custom_dataset\\Processed_cropped",
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
    model = DarkEnhancementNet().to(device)
    print("Model loaded")

    # Función de pérdida y optimizador
    criterion = MS_SSIM(data_range=1.0, size_average=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    # Entrenamiento
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Initializing epoch {epoch+1}/{num_epochs}")

        start_time = time.time()
        model.train()
        running_loss = 0.0
        print(f"Enumerating train_loader")
        for i, (bright, dark) in enumerate(train_loader):
            bright, dark = bright.to(device), dark.to(device)

            """
            if i == 0:  # Verificar que es el primer batch
                show_images(bright, dark)
                input(
                    "Presiona Enter para continuar con el entrenamiento..."
                )  # Espera a que el usuario presione Enter
                raise Exception("Exit")
            """
            optimizer.zero_grad()
            enhanced_image = model(dark)
            loss = 1 - criterion(enhanced_image, bright)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculamos el ETA para la finalización de la época
            current_time = time.time()
            elapsed_time = current_time - start_time
            images_processed = (i + 1) * batch_size
            total_images = len(train_loader.dataset)
            eta = elapsed_time / images_processed * (total_images - images_processed)

            if (i + 1) % print_interval == 0:
                print(
                    f"\rBatch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, ETA for epoch: {timedelta(seconds=int(eta))}",
                    end="",
                )
        print("")

        epoch_loss = running_loss / len(train_loader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time elapsed: {time.time() - start_time}"
        )

        # Validación
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            val_loss = 0.0
            for i, (bright, dark) in enumerate(val_loader):
                bright, dark = bright.to(device), dark.to(device)

                enhanced_image = model(dark)
                loss = 1 - criterion(enhanced_image, bright)
                val_loss += loss.item()

                current_time = time.time()
                elapsed_time = current_time - start_time
                images_processed = (i + 1) * batch_size
                total_images = len(val_loader.dataset)
                eta = (
                    elapsed_time / images_processed * (total_images - images_processed)
                )
                if (i + 1) % print_interval == 0:
                    print(
                        f"\rValidation Batch {i+1}/{len(val_loader)}, Loss: {loss.item():.4f}, ETA for epoch: {timedelta(seconds=int(eta))}",
                        end="",
                    )

        print("")
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

        # Convertir segundos en formato de horas, minutos y segundos
        time_elapsed_formatted = str(timedelta(seconds=int(time_elapsed)))
        time_remaining_formatted = str(timedelta(seconds=int(time_remaining)))

        print(f"ETA for next epoch: {time_elapsed_formatted}")
        print(f"ETA for total: {time_remaining_formatted}")

        # Guardar el modelo cada 10 épocas
        if (epoch + 1) % 5 == 0:
            save_path = f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Modelo guardado: {save_path}")

        # Actualizar el learning rate
        scheduler.step()

    print("Training complete")


if __name__ == "__main__":
    main()
