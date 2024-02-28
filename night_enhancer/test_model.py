import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from model import DarkEnhancementNet
import torchvision.transforms.functional as TF
from data_loader import DarkenerDataset
from torch.utils.data import DataLoader


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

    # Desnormalizar tensor
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    denormalized_tensor = denormalize(normalized_tensor, mean, std)
    denormalized_output = denormalize(output, mean, std)

    # Convertir tensor desnormalizado a imagen PIL para visualización
    image_denormalized_pil = TF.to_pil_image(denormalized_tensor)
    output_denormalized_pil = TF.to_pil_image(denormalized_output)

    # Visualizar las imágenes usando matplotlib
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Mostrar imágenes normalizadas
    axs[0, 0].imshow(image_normalized_pil)
    axs[0, 0].set_title("Input normalized")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(image_denormalized_pil)
    axs[0, 1].set_title("Input denormalized")
    axs[0, 1].axis("off")

    # Mostrar imágenes desnormalizadas
    axs[1, 0].imshow(output_normalized_pil)
    axs[1, 0].set_title("Output normalized")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(output_denormalized_pil)
    axs[1, 1].set_title("Output denormalized")
    axs[1, 1].axis("off")

    plt.show()


def show_images_dataloader(bright, dark):
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
    axs[0, 1].set_title("Output Image Normalized")
    axs[0, 1].axis("off")

    # Mostrar imágenes desnormalizadas
    axs[1, 0].imshow(bright_img_denorm)
    axs[1, 0].set_title("Bright Image Denormalized")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(dark_img_denorm)
    axs[1, 1].set_title("Output Image Denormalized")
    axs[1, 1].axis("off")

    plt.show()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Carga del modelo
model = DarkEnhancementNet()
model = model.to(device)
# model.load_state_dict(torch.load("best_model.pth"))  # Carga los pesos entrenados
model.load_state_dict(torch.load("best_model.pth"))
model.eval()  # Modo de evaluación

# Transformación para la imagen de entrada
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose(
    [
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

# Carga y transformación de la imagen
image_path = (
    "..\\datasets\\custom_dataset\\Unprocessed\\Garage-Dani-noche-1\\000000110.png"
)

image_path_2 = (
    "..\\datasets\\custom_dataset\\Processed\\Barrael-Garage-noche\\000000048.png"
)

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

# Carga del dataset
dataset = DarkenerDataset(
    csv_file="..\\frames_labels.csv",
    root_dir="..\\datasets\\custom_dataset\\Processed",
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
