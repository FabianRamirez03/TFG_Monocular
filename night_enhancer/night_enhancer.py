import cv2
import numpy as np


def ajuste_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
        "uint8"
    )
    return cv2.LUT(image, table)


def mejorar_imagen_nocturna_color(imagen_path):
    # Cargar la imagen
    img = cv2.imread(imagen_path)

    # Separar los canales de color
    channels = cv2.split(img)

    # Lista para guardar los canales procesados
    channels_clahe = []

    # Aplicar CLAHE a cada canal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for chan in channels:
        channels_clahe.append(clahe.apply(chan))

    # Unir los canales después de aplicar CLAHE
    img_clahe = cv2.merge(channels_clahe)

    # Ajuste de gamma
    gamma = 1.5
    img_gamma = ajuste_gamma(img_clahe, gamma)

    # Aplicar fastNlMeansDenoising a cada canal
    channels_denoised = []
    for chan in cv2.split(img_gamma):
        chan_denoised = cv2.fastNlMeansDenoising(chan, None, 10, 7, 21)
        channels_denoised.append(chan_denoised)

    # Unir los canales después de la reducción de ruido
    img_denoised = cv2.merge(channels_denoised)

    # Mostrar la imagen original y la imagen mejorada
    cv2.imshow("Imagen Original", img)
    cv2.imshow("Imagen Mejorada", img_gamma)
    cv2.imshow("Imagen Ruido reducido", img_denoised)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Opcionalmente, guardar la imagen mejorada
    # cv2.imwrite("imagen_mejorada.jpg", img_final)


# Llamar a la función con el path de tu imagen
mejorar_imagen_nocturna_color(
    "..\\datasets\\custom_dataset\Processed\Cartago-Paraiso-Noche\\000000029.png"
)
