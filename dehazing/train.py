import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vgg19, VGG19_Weights
import time
from datetime import timedelta
import torch.nn.functional as F
from data_loader import OHazeDataset
from model import Dehazing_UNet

# Definimos el dispositivo como GPU si está disponible, de lo contrario CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_interval = 20
batch_size = 8
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
            eta = elapsed_time / (i + 1) * (len(train_loader) - (i + 1))
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / (i + 1):.6f}, ETA: {timedelta(seconds=int(eta))}"
            )

    avg_loss = running_loss / len(train_loader)

    # Log de final de época
    print(f"End of Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    return avg_loss


def train_model(
    num_epochs,
    train_loader,
    model,
    optimizer,
    criterion_loss,
    device,
    print_interval=20,
    save_interval=10,
    patience=20,
):
    best_loss = float("inf")
    epochs_no_improvement = 0
    losses = []

    for epoch in range(num_epochs):
        # Entrenamiento
        avg_loss = train_one_epoch(
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

        if avg_loss < best_loss:
            best_loss = avg_loss
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

    return losses


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
    train_dataset = OHazeDataset(
        "..\datasets\O-Haze"
    )  # Ajusta la ruta según sea necesario
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # Ajusta según tus necesidades
        shuffle=True,
        num_workers=4,  # Ajusta según tu sistema
        pin_memory=True,
    )

    # Inicializar modelo
    print("Loading U-Net model")
    model = Dehazing_UNet(in_channels=3, out_channels=3).to(device)

    model_path = "models\\best_model.pth"
    # model.load_state_dict(torch.load(model_path, map_location=device))

    print("Model loaded")

    # Definir la función de pérdida y el optimizador para el modelo
    criterion_loss = loss_function
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),  # Ajusta los hiperparámetros según sea necesario
    )

    num_epochs = 400

    train_model(
        num_epochs=num_epochs,
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
        criterion_loss=criterion_loss,
        device=device,
        print_interval=print_interval,
    )


if __name__ == "__main__":
    main()
