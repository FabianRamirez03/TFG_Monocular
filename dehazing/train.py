import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import vgg19, VGG19_Weights
import time
from datetime import timedelta
import torch.nn.functional as F
from data_loader import OHazeDataset
from model import Dehazing_UNet

# Definimos el dispositivo como GPU si está disponible, de lo contrario CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_interval = 20
batch_size = 12
learning_rate = 0.001


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    criterion_loss,
    device,
    epoch,
    num_epochs,
    print_interval,
):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for i, batch in enumerate(train_loader):
        hazy_images = batch["hazy"].to(device)
        clear_images = batch["clear"].to(device)

        # Entrenar el modelo
        optimizer.zero_grad()
        generated_images = model(hazy_images)  # El modelo produce la imagen limpia
        loss = criterion_loss(generated_images, clear_images)
        loss.backward()
        optimizer.step()

        # Acumular las pérdidas
        running_loss += loss.item()

        # Logs
        if (i + 1) % print_interval == 0:
            elapsed_time = time.time() - start_time
            eta_epoch = elapsed_time / (i + 1) * (len(train_loader) - (i + 1))
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / (i + 1):.6f}, "
                f"ETA for epoch: {timedelta(seconds=int(eta_epoch))}"
            )

    avg_loss = running_loss / len(train_loader)
    elapsed_time_epoch = time.time() - start_time

    # Log de final de época
    print(
        f"End of Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, Epoch Time: {timedelta(seconds=int(elapsed_time_epoch))}"
    )

    return avg_loss, elapsed_time_epoch


def validate_model(val_loader, model, criterion_loss, device):
    model.eval()  # Poner el modelo en modo evaluación
    total_loss = 0.0
    total_batches = len(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            hazy_images = batch["hazy"].to(device)
            clear_images = batch["clear"].to(device)

            generated_images = model(hazy_images)
            loss = criterion_loss(generated_images, clear_images)

            total_loss += loss.item()

    avg_loss = total_loss / total_batches
    return avg_loss


def train_model(
    num_epochs,
    train_loader,
    val_loader,
    model,
    optimizer,
    criterion_loss,
    device,
    print_interval=20,
    save_interval=10,
    patience=20,
):
    best_loss = float("inf")
    best_val_loss = float("inf")
    epochs_no_improvement = 0
    losses = []
    val_losses = []
    start_time_total = time.time()

    for epoch in range(num_epochs):
        # Entrenamiento
        avg_loss, elapsed_time_epoch = train_one_epoch(
            train_loader,
            model,
            optimizer,
            criterion_loss,
            device,
            epoch,
            num_epochs,
            print_interval,
        )

        losses.append(avg_loss)

        # Validación
        avg_val_loss = validate_model(val_loader, model, criterion_loss, device)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}")

        # Guardar el mejor modelo según la pérdida de validación
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print("Best model saved")
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= patience:
            print(f"Training stopped due to no improvement for {patience} epochs.")
            break

        if epoch % save_interval == 0:  # Guardar modelo cada 'save_interval' épocas
            torch.save(model.state_dict(), f"models/generator_epoch_{epoch}.pth")
            print(f"Model saved at epoch {epoch+1}")

        # ETA for total training
        time_so_far = time.time() - start_time_total
        eta_total = time_so_far / (epoch + 1) * (num_epochs - epoch - 1)
        print(f"ETA for total training: {timedelta(seconds=int(eta_total))}")

    total_training_time = time.time() - start_time_total
    print(f"Total Training Time: {timedelta(seconds=int(total_training_time))}")

    return losses, val_losses


def color_balance_loss(generated_image, clear_image):
    # Compute the mean color channels of the generated and clear images
    mean_gen = torch.mean(generated_image, dim=(2, 3))
    mean_clear = torch.mean(clear_image, dim=(2, 3))

    # Calculate the color balance loss as the MSE between the mean color channels
    r_loss = F.mse_loss(mean_gen[:, 0], mean_clear[:, 0])
    g_loss = F.mse_loss(mean_gen[:, 1], mean_clear[:, 1])
    b_loss = F.mse_loss(mean_gen[:, 2], mean_clear[:, 2])

    # We can put more weight on the green channel if the model is biased towards green color
    color_balance_loss = r_loss + 2 * g_loss + b_loss
    return color_balance_loss


def color_loss(generated_image, clear_image):
    # Calculates the color difference between the generated and clear images
    color_diff_loss = F.mse_loss(
        torch.mean(generated_image, dim=(2, 3)), torch.mean(clear_image, dim=(2, 3))
    )
    return color_diff_loss


def loss_function(generated_image, clear_image):
    # Direct reconstruction loss (L1 loss)
    reconstruction_loss = F.l1_loss(generated_image, clear_image)

    # Perceptual loss (MSE loss)
    perceptual_loss_value = F.mse_loss(generated_image, clear_image)

    # Pixel-wise dispersion loss (L2 loss)
    dispersion_loss = F.mse_loss(
        torch.std(generated_image, dim=(2, 3)), torch.std(clear_image, dim=(2, 3))
    )

    # Color difference loss
    color_loss_value = color_loss(generated_image, clear_image)

    # Color balance loss (giving more weight to the green channel if it's biased)
    color_balance_loss_value = color_balance_loss(generated_image, clear_image)

    # Combine losses with appropriate weights
    total_loss = (
        0.6 * reconstruction_loss
        + 0.2 * perceptual_loss_value
        + 0.05 * dispersion_loss
        + 0.15 * (color_loss_value + color_balance_loss_value)
    )

    return total_loss


def main():

    print("Begin train")

    # Cargar los datos
    dataset_path = "..\datasets\O-Haze-Cityscapes"

    train_split = 0.8  # 80% para entrenamiento
    val_split = 0.2  # 20% para validación

    full_dataset = OHazeDataset(dataset_path)

    # Calcula las longitudes para cada conjunto de datos
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Dividir el conjunto de datos
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=5,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=5,
    )

    # Inicializar modelo
    print("Loading U-Net model")
    model = Dehazing_UNet(in_channels=3, out_channels=3).to(device)

    model_path = "models\\generator_epoch_140.pth"
    # model.load_state_dict(torch.load(model_path, map_location=device))

    print("Model loaded")

    # Definir la función de pérdida y el optimizador para el modelo
    criterion_loss = loss_function
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),  # Ajusta los hiperparámetros según sea necesario
    )

    num_epochs = 150

    train_model(
        num_epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        criterion_loss=criterion_loss,
        device=device,
        print_interval=print_interval,
    )


if __name__ == "__main__":
    main()
