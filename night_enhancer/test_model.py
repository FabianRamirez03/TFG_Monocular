import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from model import DarkEnhancementNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Carga del modelo
model = DarkEnhancementNet()
model = model.to(device)
model.load_state_dict(torch.load("best_model.pth"))  # Carga los pesos entrenados
model.eval()  # Modo de evaluación

# Preparar la imagen de entrada
transform = transforms.Compose(
    [
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Carga y transformación de la imagen
image_path = (
    "..\\datasets\\custom_dataset\\Unprocessed\\Garage-Dani-noche-1\\000000110.png"
)

image_path_2 = (
    "..\\datasets\\custom_dataset\\Processed\\Barrael-Garage-noche\\000000048.png"
)


input_image = Image.open(image_path_2).convert("RGB")
transformed_image = transform(input_image).to(device)

# Añadir una dimensión de batch y pasar la imagen a través del modelo
input_batch = transformed_image.unsqueeze(0)  # Añade una dimensión de batch
with torch.no_grad():
    output = model(input_batch)

# Preparar y mostrar la imagen de salida
# Asumiendo que la salida es una imagen en el mismo dominio que la entrada
output_image = output.squeeze().detach().cpu()
output_image = transforms.ToPILImage()(output_image)

# input_image = input_image.cpu()

# Mostrar las imágenes
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output_image)
plt.title("Output Image")
plt.axis("off")

plt.show()
