import cv2
import numpy as np


# Function to perform gamma correction
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


# Function to perform contrast adjustment
def adjust_contrast(image, alpha=1.0):
    return cv2.convertScaleAbs(image, alpha=alpha)


def darken_image(image_path):

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError("Image not found")

    # Parameters (randomly chosen for demonstration purposes)
    gamma_value = 6
    alpha_value = 0.07

    # Apply gamma correction
    gamma_corrected = adjust_gamma(original_image, gamma_value)

    # Apply contrast adjustment
    darkened_image = adjust_contrast(gamma_corrected, alpha_value)

    return original_image, darkened_image


image_path = "..\\datasets\\custom_dataset\\Unprocessed\\Dota-dia\\000000032.png"  # replace with the path to your image

original_image, darkened_image = darken_image(image_path)

cv2.imshow("Original Image", original_image)
cv2.imshow("Darkened Image", darkened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
