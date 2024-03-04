import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from model import DarkEnhancementNet
import torchvision.transforms.functional as TF
from data_loader import DarkenerDataset, NightDataset
from torch.utils.data import DataLoader
import random
import cv2
import numpy as np


def denormalize(tensor, mean, std):
    # Clonamos el tensor para no hacer cambios in-place
    tensor = tensor.clone()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(
        mean[:, None, None]
    )  # Multiplicar y sumar para desnormalizar
    return tensor


def show_images(normalized_tensor, output, title="Image"):
    # Asumiendo que normalized_tensor es un tensor PyTorch con shape [C, H, W] y ya está en el dispositivo 'cpu'
    # Convertir tensor a imagen PIL para visualización
    image_normalized_pil = TF.to_pil_image(normalized_tensor)
    output_normalized_pil = TF.to_pil_image(output)

    # Visualizar las imágenes usando matplotlib
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Mostrar imágenes normalizadas
    axs[0, 0].imshow(image_normalized_pil)
    axs[0, 0].set_title("Input")
    axs[0, 0].axis("off")

    # Mostrar imágenes desnormalizadas
    axs[1, 0].imshow(output_normalized_pil)
    axs[1, 0].set_title("Output")
    axs[1, 0].axis("off")

    plt.show()


def show_images_dataloader(original, output, original_processed):
    # Suponiendo que 'bright' y 'dark' son tensores de PyTorch con shape [B, C, H, W]
    # Donde B es el tamaño del batch, C es el número de canales, y H, W son la altura y anchura
    global mean, std

    # Desnormalizar las imágenes
    # original_denorm = denormalize(original[0].cpu(), mean, std)
    # output_denorm = denormalize(output[0].cpu(), mean, std)

    # Convertir los tensores a imágenes de PIL para visualizarlas
    original_img_norm = TF.to_pil_image(original[0].cpu())
    output_img_norm = TF.to_pil_image(output[0].cpu())
    original_processed_img_norm = TF.to_pil_image(original_processed[0].cpu())

    post_processed_image = post_process_image(output_img_norm)

    # Visualizar las imágenes usando matplotlib
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Mostrar imágenes normalizadas
    axs[0, 0].imshow(original_img_norm)
    axs[0, 0].set_title("Original Image ")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(output_img_norm)
    axs[0, 1].set_title("Output Image ")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(post_processed_image)
    axs[1, 0].set_title("Post Process Image")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(original_processed_img_norm)
    axs[1, 1].set_title("Post Process input Image")
    axs[1, 1].axis("off")

    plt.show()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Carga del modelo
model = DarkEnhancementNet()
model = model.to(device)
model.load_state_dict(torch.load("model_epoch_70.pth"))  # Carga los pesos entrenados
# model.load_state_dict(torch.load("model_epoch_10.pth"))
model.eval()  # Modo de evaluación

# Transformación para la imagen de entrada
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Carga y transformación de la imagen
image_path = (
    "..\\datasets\\custom_dataset\\Unprocessed\\Garage-Dani-noche-1\\000000110.png"
)

image_path_2 = "..\\datasets\\custom_dataset\\Processed_cropped\\Barrael-Garage-noche\\000000048.png"

image_path_3 = (
    "..\\datasets\\custom_dataset\\Unprocessed\\Guti-Garage-noche\\000000075.png"
)

input_image = Image.open(image_path_3).convert("RGB")
transformed_image = transform(input_image).to(device)

# Añadir una dimensión de batch y pasar la imagen a través del modelo
input_batch = transformed_image.unsqueeze(0)  # Añade una dimensión de batch
with torch.no_grad():
    output = model(input_batch)

# Mostrar las imágenes de entrada y salida
# show_images(transformed_image, output.squeeze(0), title="Input")


##########################################################################################################################################
def show_dataloader_training():

    # Carga del dataset
    dataset = DarkenerDataset(
        csv_file="..\\frames_labels.csv",
        root_dir="..\\datasets\\custom_dataset\\Processed_cropped",
        transform=transform,
    )

    # Data loaders
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )

    for i, (bright, dark) in enumerate(data_loader):
        bright, dark = bright.to(device), dark.to(device)
        outputs = model(dark)
        show_images_dataloader(bright, outputs)
        break


def preprocess_night_images(image_tensor):
    # Asegurarse de que estamos trabajando con un tensor de una sola imagen, no un lote
    if image_tensor.ndim == 4:
        image_tensor = image_tensor.squeeze(0)  # Eliminar la dimensión del lote

    # Convertir tensor a imagen de numpy para procesamiento de OpenCV
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(
        np.uint8
    )  # Escalar a [0, 255] y convertir a uint8
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convertir a LAB para aplicar CLAHE solo al canal L
    lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Aplicar CLAHE al canal de luminancia
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Volver a combinar los canales y convertir a RGB
    lab_clahe = cv2.merge((l_clahe, a, b))
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    image_clahe = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2RGB)

    # Reducir el ruido
    image_denoised = cv2.fastNlMeansDenoisingColored(image_clahe, None, 10, 10, 7, 21)

    # Ajuste de gamma
    gamma = 2
    inv_gamma = 1.0 / gamma
    table = (
        np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
    ).astype("uint8")
    image_gamma_corrected = cv2.LUT(image_denoised, table)

    # Convertir la imagen de OpenCV de vuelta a tensor de PyTorch
    image_gamma_corrected = cv2.cvtColor(image_gamma_corrected, cv2.COLOR_BGR2RGB)
    processed_tensor = transforms.ToTensor()(image_gamma_corrected).unsqueeze(0)

    return processed_tensor


def post_process_image(image_pil):

    # Convertir la imagen PIL a un array de NumPy en formato BGR para OpenCV
    image_np = np.array(image_pil)[:, :, ::-1]  # Convertir RGB a BGR
    # Convertir a un formato adecuado para OpenCV si es necesario
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Aplicar un filtro bilateral para suavizar la imagen manteniendo los bordes
    image_bilateral = cv2.bilateralFilter(image_cv, d=9, sigmaColor=75, sigmaSpace=75)

    # Ajustar gamma
    gamma = 1  # Cambiar según sea necesario para aclarar la imagen
    look_up_table = np.empty((1, 256), np.uint8)
    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    image_adjusted = cv2.LUT(image_bilateral, look_up_table)

    # Convertir de vuelta a RGB si se va a mostrar con herramientas que esperan ese formato
    image_adjusted = cv2.cvtColor(image_adjusted, cv2.COLOR_BGR2RGB)

    # No olvides convertir de vuelta a RGB si después vas a trabajar con la imagen en PIL o matplotlib
    image_np = image_adjusted[:, :, ::-1]  # Convertir BGR a RGB
    return image_np


# show_dataloader_training()


def show_dataloader_night():

    # Carga del dataset
    dataset = NightDataset(
        csv_file="..\\frames_labels.csv",
        root_dir="..\\datasets\\custom_dataset\\Processed_cropped",
        transform=transform,
    )

    # Data loaders
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )
    while True:
        i = random.randint(0, len(dataset) - 1)  # Obtiene un índice aleatorio
        night_image = dataset[i]
        night_image = night_image.to(device).unsqueeze(0)
        night_image_enhanced = preprocess_night_images(night_image).to(device)
        outputs = model(night_image_enhanced)
        show_images_dataloader(night_image, outputs, night_image_enhanced)
        a = input("E para salir, cualquier tecla continuar \n")
        if a.lower() == "e":
            break


show_dataloader_night()
