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

from data_loader import RaindropDataset
from model import Generator, Discriminator

# Definimos el dispositivo como GPU si está disponible, de lo contrario CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_interval = 10
validation_interval = 1
batch_size = 8
patience = 10
epochs_no_improve = 0


def train_one_epoch(
    train_loader,
    criterion,
    generator,
    discriminator,
    optimizer_G,
    optimizer_D,
    g_scheduler,
    d_scheduler,
    epoch,
    num_epochs,
    scaler,
):
    print(f"Initializing epoch {epoch+1}/{num_epochs}")
    start_time = time.time()

    generator.train()  # Establecer el modo de entrenamiento para el generador
    discriminator.train()  # Establecer el modo de entrenamiento para el discriminador

    # Variables para almacenar las pérdidas
    g_loss_sum = 0.0
    d_loss_sum = 0.0

    # Entrenamiento de un lote de datos
    print(f"Enumerating train_loader")

    for batch_idx, batch in enumerate(train_loader):
        # Obtener el lote de datos y enviarlo al dispositivo
        rain_images = batch["rain"].to(device)
        clear_images = batch["clear"].to(device)

        with autocast():  # Usar autocast para la generación de imágenes falsas también
            fake_images = generator(rain_images)

        # Entrenamiento del discriminador con imágenes reales
        optimizer_D.zero_grad()
        with autocast():
            real_preds = discriminator(clear_images)
            real_loss = criterion(real_preds, torch.ones_like(real_preds))
        scaler.scale(real_loss).backward()

        # Entrenamiento del discriminador con imágenes falsas
        with autocast():
            fake_preds = discriminator(fake_images.detach())  # Nota el detach() aquí
            fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))

        d_loss = real_loss + fake_loss
        scaler.scale(fake_loss).backward()
        scaler.step(optimizer_D)
        scaler.update()

        # Entrenamiento del generador
        optimizer_G.zero_grad()
        with autocast():
            # Aquí reutilizamos fake_images sin detach() para que los gradientes fluyan hacia atrás hasta el generador
            fake_preds_for_gen = discriminator(fake_images)
            g_loss = criterion(fake_preds_for_gen, torch.ones_like(fake_preds_for_gen))
        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

        # Acumular las pérdidas
        g_loss_sum += g_loss.item()
        d_loss_sum += d_loss.item()

        current_time = time.time()
        elapsed_time = current_time - start_time
        images_processed = (batch_idx + 1) * batch_size
        total_images = len(train_loader.dataset)
        eta = elapsed_time / images_processed * (total_images - images_processed)

        # Número de batches procesados hasta ahora
        batches_processed = batch_idx + 1

        if batches_processed % print_interval == 0:
            # Calcular la pérdida promedio por batch hasta el momento
            avg_g_loss = g_loss_sum / batches_processed
            avg_d_loss = d_loss_sum / batches_processed

            print(
                f"\r Train: Batch {batches_processed}/{len(train_loader)} "
                f"Generator Loss: {avg_g_loss:.4f} "
                f"Discriminator Loss: {avg_d_loss:.4f} "
                f"ETA for epoch: {timedelta(seconds=int(eta))}",
                end="",
            )


def val_one_epoch(val_loader, generator, discriminator, criterion, epoch, num_epochs):
    generator.eval()  # Poner el generador en modo de evaluación
    discriminator.eval()  # Poner el discriminador en modo de evaluación

    best_Discriminator_loss = float("inf")
    best_Generator_loss = float("inf")

    with torch.no_grad():  # No es necesario calcular gradientes para la validación
        val_g_loss_sum = 0.0
        val_d_loss_sum = 0.0

        for batch in val_loader:
            # Cargar el lote de datos y enviarlo al dispositivo
            val_rain_images = batch["rain"].to(device)
            val_clear_images = batch["clear"].to(device)

            # Generar imágenes sin lluvia con el generador
            val_fake_images = generator(val_rain_images)

            # Calcular la pérdida del discriminador con imágenes reales y falsas
            val_real_preds = discriminator(val_clear_images)
            val_real_loss = criterion(
                val_real_preds, torch.ones_like(val_real_preds).to(device)
            )

            val_fake_preds = discriminator(val_fake_images)
            val_fake_loss = criterion(
                val_fake_preds, torch.zeros_like(val_fake_preds).to(device)
            )

            val_d_loss = val_real_loss + val_fake_loss

            # Calcular la pérdida del generador
            val_g_loss = criterion(
                val_fake_preds, torch.ones_like(val_fake_preds).to(device)
            )

            # Acumular las pérdidas para calcular la media después
            val_g_loss_sum += val_g_loss.item()
            val_d_loss_sum += val_d_loss.item()

            # Calcular la pérdida media en el conjunto de validación
            val_g_loss_avg = val_g_loss_sum / len(val_loader)
            val_d_loss_avg = val_d_loss_sum / len(val_loader)

        print(
            f"Validation - Epoch [{epoch+1}/{num_epochs}] G_loss_avg: {val_g_loss_avg:.4f} D_loss_avg: {val_d_loss_avg:.4f}"
        )

        return val_g_loss_avg, val_d_loss_avg


def main():

    print("Begin train")

    attention_blocks = 10
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

    generator = Generator(attention_blocks, image_shape).to(device)

    print("Loading discriminator model")
    channels, _, _ = image_shape
    discriminator = Discriminator(channels).to(device)

    print("Models loaded")

    # Definir las funciones de pérdida y optimizadores para ambos modelos
    criterion = BCEWithLogitsLoss().to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Definir los schedulers para los optimizadores
    g_scheduler = StepLR(optimizer_G, step_size=50, gamma=0.1)
    d_scheduler = StepLR(optimizer_D, step_size=50, gamma=0.1)

    num_epochs = 10
    best_Discriminator_loss = float("inf")
    best_Generator_loss = float("inf")

    scaler = GradScaler()

    # Ciclo de entrenamiento

    for epoch in range(num_epochs):

        train_one_epoch(
            train_loader,
            criterion,
            generator,
            discriminator,
            optimizer_G,
            optimizer_D,
            g_scheduler,
            d_scheduler,
            epoch,
            num_epochs,
            scaler,
        )

        if (epoch + 1) % validation_interval == 0:
            val_g_loss_avg, val_d_loss_avg = val_one_epoch(
                val_loader, generator, discriminator, criterion, epoch, num_epochs
            )

        # Comparar con las mejores pérdidas y hacer checkpointing si hay mejora
        if val_d_loss_avg < best_Discriminator_loss:
            best_Discriminator_loss = val_d_loss_avg
            torch.save(discriminator.state_dict(), f"best_discriminator.pth")
            epochs_no_improve = 0
        if val_g_loss_avg < best_Generator_loss:
            best_Generator_loss = val_g_loss_avg
            torch.save(generator.state_dict(), f"best_generator.pth")
            epochs_no_improve = 0

        # Incrementar el contador de épocas sin mejora y comprobar si se alcanzó la paciencia
        if (
            val_d_loss_avg >= best_Discriminator_loss
            or val_g_loss_avg >= best_Generator_loss
        ):
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping ejecutado después de {patience} épocas sin mejora."
                )
                break


if __name__ == "__main__":
    main()
