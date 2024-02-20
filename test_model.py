import torch
import torchvision.transforms as transforms
from torchvision import models
import os
import numpy as np
from PIL import Image
import random
from model import DualInputCNN
import matplotlib.pyplot as plt

# Configuración
model_path = "models\\first_version_dual_input\\best_model.pth"
test_folder = "custom_dataset\\Unprocessed"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
labels = ["Noche", "Soleado", "Nublado", "Lluvia", "Neblina", "Sombras"]

# Transformaciones necesarias para la entrada del modelo
transform = transforms.Compose(
    [
        transforms.Resize([232], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop([224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Cargar el modelo
def load_model(model_path, device):
    model = DualInputCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# Realizar predicción sobre una imagen
def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image_transformed = transform(image).unsqueeze(0)

    # Calcular las secciones superior e inferior de la imagen transformada
    height = image_transformed.size(2)
    upper_section = image_transformed[:, :, : int(height * 0.25), :]
    lower_section = image_transformed[:, :, int(height * 0.25) :, :]

    upper_section, lower_section = upper_section.to(device), lower_section.to(device)

    with torch.no_grad():
        outputs = model(upper_section, lower_section)
        preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
    return preds


# Mostrar etiquetas de forma amigable
def display_labels(preds):
    pred_labels = [labels[i] for i, pred in enumerate(preds[0]) if pred]
    return ", ".join(pred_labels) if pred_labels else "Ninguna"


# Seleccionar una imagen aleatoria de los folders
def select_random_image(test_folder):
    folders = [
        os.path.join(test_folder, f)
        for f in os.listdir(test_folder)
        if os.path.isdir(os.path.join(test_folder, f))
    ]
    random_folder = random.choice(folders)
    images = [
        os.path.join(random_folder, f)
        for f in os.listdir(random_folder)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    random_image = random.choice(images)
    return random_image


def show_image(image_path, labels):
    # Cargar la imagen
    image = Image.open(image_path)

    # Mostrar la imagen
    plt.imshow(image)
    plt.title(f"Etiquetas: {labels}")
    plt.axis("off")  # No mostrar ejes
    plt.show()


# Proceso principal
def main():
    model = load_model(model_path, device=device)
    while True:
        image_path = select_random_image(test_folder)
        preds = predict_image(image_path, model)
        labels = display_labels(preds)
        print(f"Imagen: {image_path}")
        print(f"Etiquetas: {labels}")

        # Mostrar la imagen con las etiquetas predichas
        show_image(image_path, labels)

        input("Presiona Enter para continuar con otra imagen o Ctrl+C para salir...")


if __name__ == "__main__":
    main()
