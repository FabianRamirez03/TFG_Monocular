import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from model import LowLightEnhancer
from data_loader import LOLDataset
import copy
import time

# Configuración del entorno de entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100  # Número de épocas para entrenar
learning_rate = 1e-4
batch_size = 8
workers = 5
prefetch_factor = 5
resize_size = [232]
crop_size = [224]

transformations = transforms.Compose(
    [
        transforms.Resize(
            resize_size, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Carga del dataloader para entrenamiento y validación
train_directory = "LOLdataset/train"
train_dataset = LOLDataset(directory=train_directory, transform=transformations)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    prefetch_factor=prefetch_factor,
)

eval_directory = "LOLdataset/eval"
eval_dataset = LOLDataset(directory=eval_directory, transform=transformations)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    prefetch_factor=prefetch_factor,
)


def main():
    # Cargar el modelo
    model = LowLightEnhancer().to(device)

    # Definir la función de pérdida y el optimizador
    criterion = nn.MSELoss()  # Por ejemplo, MSE para una tarea de regresión
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler para decaimiento del learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Bucle de entrenamiento con validación y checkpointing
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        start_time_epoch = time.time()
        print(f"Iniciando con época: {epoch}.")

        for i, data in enumerate(train_dataloader, 0):
            start_time_batch = time.time()

            low_light_images, high_light_images = data
            low_light_images = low_light_images.to(device)
            high_light_images = high_light_images.to(device)

            optimizer.zero_grad()
            outputs = model(low_light_images)
            loss = criterion(outputs, high_light_images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Cálculo del tiempo por lote y ETA
            time_per_batch = time.time() - start_time_batch
            batches_left = len(train_dataloader) - (i + 1)
            eta = batches_left * time_per_batch

            # Imprime el feedback del progreso incluyendo el ETA
            if i % 10 == 9:
                print(
                    f"Epoch: {epoch+1}, Batch: {i+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}, ETA: {eta:.2f}s"
                )

        # Información al final de la época
        epoch_duration = time.time() - start_time_epoch
        print(f"Epoch {epoch+1} completada en {epoch_duration:.2f} s.")

        # Validación al final de cada época
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for data in eval_dataloader:
                low_light_images, high_light_images = data
                low_light_images = low_light_images.to(device)
                high_light_images = high_light_images.to(device)

                outputs = model(low_light_images)
                loss = criterion(outputs, high_light_images)
                valid_loss += loss.item()

        # Calculando la pérdida promedio sobre el conjunto de entrenamiento y validación
        train_loss = train_loss / len(eval_dataloader)
        valid_loss = valid_loss / len(eval_dataloader)

        # Actualizar el scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
        )

        # Guardar el mejor modelo
        if valid_loss < best_loss:
            model_save_path = "best_low_light_enhancer_model.pth"
            print(
                f"Validation loss decreased ({best_loss:.4f} --> {valid_loss:.4f}). Saving model..."
            )
            best_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_save_path)
            print(f"Model saved to {model_save_path}")
        # Guardar checkpoints cada 10 épocas
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": valid_loss,
                "best_loss": best_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Cargar los mejores pesos del modelo y guardar el modelo entrenado
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "best_low_light_enhancer_model.pth")

    print("Entrenamiento finalizado y modelo guardado.")


if __name__ == "__main__":
    main()
