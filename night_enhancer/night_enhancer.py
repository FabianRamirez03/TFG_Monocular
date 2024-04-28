import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


def enhance_night_image(image_tensor):
    # Asegurarse de que estamos trabajando con un tensor de una sola imagen, no un lote
    if not isinstance(image_tensor, torch.Tensor):
        transform = transforms.ToTensor()
        image_tensor = transform(image_tensor)
    if image_tensor.ndim == 4:
        image_tensor = image_tensor.squeeze(0)

    # Convertir tensor a imagen de numpy para procesamiento de OpenCV
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convertir a LAB y aplicar CLAHE al canal L
    lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))

    # Convertir de vuelta a BGR y aplicar denoising
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    image_denoised = cv2.fastNlMeansDenoisingColored(image_clahe, None, 5, 10, 7, 21)

    # Aplicar ajuste de gamma
    gamma = 10.0
    image_gamma_corrected = adjust_gamma(image_denoised, gamma)

    # Simular HDR
    image_hdr = simulate_hdr(image_gamma_corrected)

    reduce_highlights = reduce_highlights_array(image_hdr)

    # Convertir la imagen de OpenCV de vuelta a tensor de PyTorch
    final_tensor = cv2.cvtColor(image_denoised, cv2.COLOR_BGR2RGB)
    processed_tensor = transforms.ToTensor()(final_tensor)

    image_pil_final = TF.to_pil_image(processed_tensor)

    return image_pil_final


def adjust_gamma(image, gamma=1.0):
    # Asegúrate de que la imagen está en el rango correcto
    min_val = np.min(image.ravel())
    max_val = np.max(image.ravel())

    # Escala los valores solo si están en [0,1]
    if min_val >= 0 and max_val <= 1:
        image = np.clip(image * 255, 0, 255).astype("uint8")

    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def simulate_hdr(image):
    # Suponiendo que 'image' ya está en el rango [0, 255] y es de tipo 'uint8'
    # Simular diferentes exposiciones
    exposures = [0.5, 1.0, 2.0]
    images = [adjust_gamma(image, gamma) for gamma in exposures]

    # Asegúrate de que las imágenes sean continuas y estén en el rango correcto
    images = [
        img.astype("float32") / 255.0
        for img in images
        if img.dtype == np.uint8 and img.flags["C_CONTIGUOUS"]
    ]

    # Combina las exposiciones en una imagen HDR
    merge_mertens = cv2.createMergeMertens()
    hdr_image = merge_mertens.process(images)

    # Convierte la imagen HDR a valores de 8 bits
    hdr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype("uint8")
    return hdr_image_8bit


def reduce_highlights_array(image_np):
    # Convertir a escala de grises para detectar áreas brillantes
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral para identificar fuentes de luz intensas
    _, bright_regions = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)

    # Expandir las regiones brillantes para suavizar los bordes
    bright_regions = cv2.dilate(bright_regions, None, iterations=2)

    # Invertir la máscara para tener el fondo oscuro
    mask_inv = cv2.bitwise_not(bright_regions)

    # Crear una imagen que solo contenga las reducciones de brillo en las regiones identificadas
    image_bright_reduced = cv2.addWeighted(image_np, 1, image_np, 0, -30)

    # Aplicar la máscara a la imagen de brillo reducido
    image_bright_reduced = cv2.bitwise_and(
        image_bright_reduced, image_bright_reduced, mask=bright_regions
    )

    # Combinar con la imagen original donde no había regiones brillantes
    final_image = cv2.bitwise_and(image_np, image_np, mask=mask_inv)
    final_image += image_bright_reduced

    return final_image


def post_process_image(image_pil):
    # Asegurarse de que tenemos una imagen de PIL
    if not isinstance(image_pil, Image.Image):
        raise TypeError("El objeto proporcionado no es una imagen de PIL (Pillow).")

    # Convertir la imagen PIL a un array de NumPy en formato RGB para OpenCV
    image_np = np.array(image_pil)

    # OpenCV espera el formato BGR, así que convertimos
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

    # Convertir el array de NumPy de vuelta a una imagen de PIL
    image_pil_final = Image.fromarray(image_adjusted)

    return image_pil_final
