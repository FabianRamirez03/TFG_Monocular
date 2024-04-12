import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import time
from datetime import datetime, timedelta
from torch.cuda.amp import GradScaler, autocast
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


from data_loader import RaindropDataset
from model import Deraining_UNet

# Definimos el dispositivo como GPU si está disponible, de lo contrario CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_interval = 20
validation_interval = 1
batch_size = 4
patience = 20
epochs_no_improve = 0
learning_rate = 0.001


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg19 = (
            vgg19(weights=VGG19_Weights.DEFAULT).features[:36].eval().to(device)
        )
        for param in self.vgg19.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        perception_loss = F.mse_loss(self.vgg19(input), self.vgg19(target))
        return perception_loss


perceptual_loss = PerceptualLoss()


def train_one_epoch(
    train_loader,
    generator,
    optimizer_G,
    criterion_loss,
    device,
    epoch,
    num_epochs,
    print_interval,
):
    generator.train()
    running_loss_G = 0.0
    start_time = time.time()

    for i, batch in enumerate(train_loader):
        real_images = batch["clear"].to(device)
        rain_images = batch["rain"].to(device)

        # Entrenar el generador
        optimizer_G.zero_grad()
        generated_images = generator(
            rain_images
        )  # El generador produce la imagen limpia
        loss_G = criterion_loss(
            generated_image=generated_images,
            clear_image=real_images,
            perceptual_loss_module=perceptual_loss,
        )
        loss_G.backward()
        optimizer_G.step()

        # Acumular las pérdidas
        running_loss_G += loss_G.item()

        # Logs
        if (i + 1) % print_interval == 0:
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (i + 1) * (len(train_loader) - (i + 1))
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Generator Loss: {running_loss_G / (i + 1):.6f}, ETA: {timedelta(seconds=int(eta))}"
            )

    avg_loss_G = running_loss_G / len(train_loader)

    # Log de final de época
    print(
        f"End of Epoch [{epoch+1}/{num_epochs}], Avg Generator Loss: {avg_loss_G:.4f}"
    )

    return avg_loss_G


def validate(
    val_loader,
    generator,
    criterion_loss,
    device,
):
    generator.eval()
    running_loss_G = 0.0
    with torch.no_grad():
        for batch in val_loader:
            real_images = batch["clear"].to(device)
            rain_images = batch["rain"].to(device)

            # Generar imágenes limpias a partir de las imágenes con lluvia
            generated_images = generator(rain_images)

            # Calcular la pérdida entre las imágenes generadas y las imágenes reales sin lluvia
            loss_G = criterion_loss(
                generated_image=generated_images,
                clear_image=real_images,
                perceptual_loss_module=perceptual_loss,
            )
            running_loss_G += loss_G.item()

    avg_loss_G = running_loss_G / len(val_loader)

    print(f"Validation - Avg Generator Loss: {avg_loss_G:.4f}")

    return avg_loss_G


def train_model(
    num_epochs,
    train_loader,
    val_loader,
    generator,
    optimizer_G,
    criterion_loss,
    device,
    print_interval=20,
    validation_interval=1,
):
    best_Generator_loss = float("inf")
    epochs_no_improve = 0
    patience = 20

    for epoch in range(num_epochs):
        # Entrenamiento
        avg_loss_G = train_one_epoch(
            train_loader,
            generator,
            optimizer_G,
            criterion_loss,
            device,
            epoch,
            num_epochs,
            print_interval,
        )

        # Validación cada 'validation_interval' épocas
        if (epoch + 1) % validation_interval == 0:
            val_loss_G = validate(
                val_loader,
                generator,
                criterion_loss,
                device,
            )
            print(
                f"Validation Results - Epoch [{epoch+1}/{num_epochs}]: Avg Generator Loss: {val_loss_G:.6f}"
            )

        # Comparar con la mejor pérdida del generador y hacer checkpointing si hay mejora
        if val_loss_G <= best_Generator_loss:
            best_Generator_loss = val_loss_G
            torch.save(generator.state_dict(), "models/best_generator.pth")
            epochs_no_improve = 0
            print(f"Mejor modelo guardado con pérdida: {best_Generator_loss:.6f}")
        else:
            epochs_no_improve += 1
            print(f"Sin mejora por {epochs_no_improve} épocas.")
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping ejecutado después de {patience} épocas sin mejora."
                )
                break

        if epoch % 5 == 4:  # Cada 5 épocas
            torch.save(generator.state_dict(), f"models/generator_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")


def loss_function(generated_image, clear_image, perceptual_loss_module):
    # Direct reconstruction loss
    reconstruction_loss = F.l1_loss(generated_image, clear_image)

    # Perceptual loss
    perceptual_loss_value = perceptual_loss_module(generated_image, clear_image)

    # Combine losses
    total_loss = reconstruction_loss + 0.1 * perceptual_loss_value

    return total_loss


def main():

    print("Begin train")

    # Cargar los datos
    train_dataset = RaindropDataset(
        "../datasets/Raindrop_dataset/train"
    )  # Ajusta la ruta según sea necesario
    val_dataset = RaindropDataset(
        "../datasets/Raindrop_dataset/test_b"
    )  # Ajusta la ruta según sea necesario
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Ajusta según tus necesidades
        shuffle=True,
        num_workers=4,  # Ajusta según tu sistema
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,  # Ajusta según tus necesidades
        shuffle=False,
        num_workers=4,  # Ajusta según tu sistema
        pin_memory=True,
    )

    # Inicializar modelo
    print("Loading U-Net model")
    generator = Deraining_UNet(in_channels=3, out_channels=3).to(device)

    model_path = "models\\best_generator_original.pth"
    generator.load_state_dict(torch.load(model_path, map_location=device))

    print("Model loaded")

    # Definir la función de pérdida y el optimizador para el generador
    criterion_loss = loss_function
    optimizer_G = optim.Adam(
        generator.parameters(),
        lr=0.001,
        betas=(0.5, 0.999),  # Ajusta los hiperparámetros según sea necesario
    )

    # Definir el scheduler para el optimizador del generador
    g_scheduler = StepLR(optimizer_G, step_size=50, gamma=0.1)

    num_epochs = 150

    train_model(
        num_epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        generator=generator,
        optimizer_G=optimizer_G,
        criterion_loss=criterion_loss,
        device=device,
        print_interval=20,
        validation_interval=1,
    )


if __name__ == "__main__":
    main()
