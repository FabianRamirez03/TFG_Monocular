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
from model import Generator, Discriminator

# Definimos el dispositivo como GPU si está disponible, de lo contrario CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_interval = 20
validation_interval = 1
batch_size = 4
patience = 20
epochs_no_improve = 0
learning_rate = 0.001


def train_one_epoch(
    train_loader,
    generator,
    discriminator,
    optimizer_G,
    optimizer_D,
    criterion_GAN,
    criterion_loss,
    device,
    epoch,
    num_epochs,
    print_interval,
):
    generator.train()
    discriminator.train()

    running_loss_G = 0.0
    running_loss_D = 0.0
    start_time = time.time()

    for i, batch in enumerate(train_loader):
        real_images = batch["clear"].to(device)
        rain_images = batch["rain"].to(device)

        # Entrenar el generador
        optimizer_G.zero_grad()
        generated_images, attention_maps = generator(rain_images, real_images)
        loss_G = criterion_loss(
            generated_images, attention_maps, rain_images, real_images
        )
        loss_G.backward()
        optimizer_G.step()

        # Entrenar el discriminador
        optimizer_D.zero_grad()
        real_preds = discriminator(real_images)
        fake_preds = discriminator(generated_images.detach())
        loss_D_real = criterion_GAN(real_preds, torch.ones_like(real_preds))
        loss_D_fake = criterion_GAN(fake_preds, torch.zeros_like(fake_preds))
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # Acumular las pérdidas para imprimir logs
        running_loss_G += loss_G.item()
        running_loss_D += loss_D.item()

        # Logs
        if (i + 1) % print_interval == 0:
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (i + 1) * (len(train_loader) - (i + 1))
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Generator Loss: {running_loss_G / (i + 1):.6f}, Discriminator Loss: {running_loss_D / (i + 1):.6f}, ETA: {timedelta(seconds=int(eta))}"
            )

    avg_loss_G = running_loss_G / len(train_loader)
    avg_loss_D = running_loss_D / len(train_loader)

    # Log de final de época
    print(
        f"End of Epoch [{epoch+1}/{num_epochs}], Avg Generator Loss: {avg_loss_G:.4f}, Avg Discriminator Loss: {avg_loss_D:.4f}"
    )

    return avg_loss_G, avg_loss_D


def validate(
    val_loader, generator, discriminator, criterion_loss, criterion_GAN, device
):
    generator.eval()
    discriminator.eval()
    running_loss_G = 0.0
    running_loss_D = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            real_images = batch["clear"].to(device)
            rain_images = batch["rain"].to(device)
            generated_images, attention_maps = generator(rain_images, real_images)

            # Pérdida del generador usando la función de pérdida personalizada
            loss_G = criterion_loss(
                generated_images, attention_maps, rain_images, real_images
            )
            running_loss_G += loss_G.item()

            # Evaluación del discriminador
            real_preds = discriminator(real_images)
            fake_preds = discriminator(generated_images)
            loss_D_real = criterion_GAN(real_preds, torch.ones_like(real_preds))
            loss_D_fake = criterion_GAN(fake_preds, torch.zeros_like(fake_preds))
            loss_D = (loss_D_real + loss_D_fake) / 2
            running_loss_D += loss_D.item()

    avg_loss_G = running_loss_G / len(val_loader)
    avg_loss_D = running_loss_D / len(val_loader)

    return avg_loss_G, avg_loss_D


def train_model(
    num_epochs,
    train_loader,
    val_loader,
    generator,
    discriminator,
    optimizer_G,
    optimizer_D,
    criterion_GAN,
    criterion_loss,
    device,
    print_interval,
    validation_interval,
):
    best_Discriminator_loss = float("inf")
    best_Generator_loss = float("inf")

    for epoch in range(num_epochs):
        # Entrenamiento
        avg_loss_G, avg_loss_D = train_one_epoch(
            train_loader,
            generator,
            discriminator,
            optimizer_G,
            optimizer_D,
            criterion_GAN,
            criterion_loss,
            device,
            epoch,
            num_epochs,
            print_interval,
        )

        # Validación cada 'validation_interval' épocas
        if (epoch + 1) % validation_interval == 0:
            val_loss_G, val_loss_D = validate(
                val_loader,
                generator,
                discriminator,
                criterion_loss,
                criterion_GAN,
                device,
            )
            print(
                f"Validation Results - Epoch [{epoch+1}/{num_epochs}]: Avg Generator Loss: {val_loss_G:.6f}, Avg Discriminator Loss: {val_loss_D:.6f}"
            )

        # Comparar con las mejores pérdidas y hacer checkpointing si hay mejora
        if val_loss_D < best_Discriminator_loss:
            best_Discriminator_loss = val_loss_D
            torch.save(discriminator.state_dict(), f"models\\best_discriminator.pth")
            epochs_no_improve = 0
        if val_loss_G <= best_Generator_loss:
            best_Generator_loss = val_loss_G
            torch.save(generator.state_dict(), f"models\\best_generator.pth")
            epochs_no_improve = 0

        # Incrementar el contador de épocas sin mejora y comprobar si se alcanzó la paciencia
        if val_loss_G > best_Generator_loss:
            epochs_no_improve += 1
            print(f"Sin mejora por {epochs_no_improve} épocas")
            print(f"Mejor pérdida del generador: {best_Generator_loss}")
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping ejecutado después de {patience} épocas sin mejora."
                )
                break

        if epoch % 5 == 4:  # Cada 5 épocas
            torch.save(
                discriminator.state_dict(), f"models\\discriminator_epoch_{epoch+1}.pth"
            )
            torch.save(generator.state_dict(), f"models\\generator_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")


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


def loss_function(
    generated_image,
    attention_map,
    rain_image,
    clear_image,
    perceptual_loss=perceptual_loss,
):
    # Pérdida de reconstrucción
    reconstruction_loss = F.l1_loss(generated_image, clear_image)

    # Pérdida perceptual
    perceptual_loss_value = perceptual_loss(generated_image, clear_image)

    # Pérdida de consistencia de contenido
    content_consistency_loss = F.l1_loss(generated_image, rain_image)

    # Pérdida de esparcimiento de los mapas de atención
    sparsity_loss = torch.mean(attention_map)

    # Combinar las pérdidas
    total_loss = (
        reconstruction_loss
        + 0.1 * perceptual_loss_value
        + 0.05 * content_consistency_loss
        + 0.01 * sparsity_loss
    )

    return total_loss


def main():

    print("Begin train")

    image_shape = (3, 224, 224)
    # Cargar los datos
    train_dataset = RaindropDataset(
        "..\datasets\Raindrop_dataset\\train"
    )  # Aquí deberías cargar tu dataset de entrenamiento
    val_dataset = RaindropDataset(
        "..\datasets\Raindrop_dataset\\test_b"
    )  # Aquí deberías cargar tu dataset de validación
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        prefetch_factor=3,
        pin_memory=True,
    )  # Definir el batch_size
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        prefetch_factor=3,
        pin_memory=True,
    )  # Definir el batch_size

    # Inicializar modelos
    print("Loading generator model")

    # Definir funciones de pérdida
    generator = Generator(in_channels=3, hidden_channels=128, out_channels=3).to(device)
    discriminator = Discriminator(input_channels=3).to(device)

    print("Models loaded")

    # Definir las funciones de pérdida y optimizadores para ambos modelos
    criterion_loss = loss_function
    gan_loss = nn.BCEWithLogitsLoss().to(device)

    # Definir optimizadores
    optimizer_G = optim.Adam(
        generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )

    # Definir los schedulers para los optimizadores
    g_scheduler = StepLR(optimizer_G, step_size=50, gamma=0.1)
    d_scheduler = StepLR(optimizer_D, step_size=50, gamma=0.1)

    num_epochs = 150

    train_model(
        num_epochs,
        train_loader,
        val_loader,
        generator,
        discriminator,
        optimizer_G,
        optimizer_D,
        gan_loss,
        criterion_loss,
        device,
        print_interval,
        validation_interval,
    )


if __name__ == "__main__":
    main()
