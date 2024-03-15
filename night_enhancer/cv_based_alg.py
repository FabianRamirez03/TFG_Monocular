import cv2
from cv2.ximgproc import guidedFilter
import numpy as np
import random
from torch.utils.data import DataLoader
from data_loader import NightDataset
from torchvision import transforms
import matplotlib.pyplot as plt


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def get_atmosphere(I, brightch, p=0.1):
    M, N = brightch.shape
    flatI = I.reshape(M * N, 3)  # reshaping image array
    flatbright = brightch.ravel()  # flattening image array

    searchidx = (-flatbright).argsort()[: int(M * N * p)]  # sorting and slicing
    A = np.mean(flatI.take(searchidx, axis=0), dtype=np.float64, axis=0)
    return A


def get_initial_transmission(A, brightch):
    A_c = np.max(A)
    init_t = (brightch - A_c) / (1.0 - A_c)  # finding initial transmission map
    return (init_t - np.min(init_t)) / (
        np.max(init_t) - np.min(init_t)
    )  # normalized initial transmission map


def get_corrected_transmission(I, A, darkch, brightch, init_t, alpha, omega, w):
    im = np.empty(I.shape, I.dtype)
    for ind in range(0, 3):
        im[:, :, ind] = (
            I[:, :, ind] / A[ind]
        )  # divide pixel values by atmospheric light
    dark_c, _ = get_illumination_channel(im, w)  # dark channel transmission map
    dark_t = 1 - omega * dark_c  # corrected dark transmission map
    corrected_t = (
        init_t  # initializing corrected transmission map with initial transmission map
    )
    diffch = brightch - darkch  # difference between transmission maps

    for i in range(diffch.shape[0]):
        for j in range(diffch.shape[1]):
            if diffch[i, j] < alpha:
                corrected_t[i, j] = dark_t[i, j] * init_t[i, j]

    return np.abs(corrected_t)


def get_final_image(I, A, refined_t, tmin):
    refined_t_broadcasted = np.broadcast_to(
        refined_t[:, :, None], (refined_t.shape[0], refined_t.shape[1], 3)
    )  # duplicating the channel of 2D refined map to 3 channels
    J = (I - A) / (
        np.where(refined_t_broadcasted < tmin, tmin, refined_t_broadcasted)
    ) + A  # finding result

    return (J - np.min(J)) / (np.max(J) - np.min(J))  # normalized image


def get_illumination_channel(I, w):
    M, N, _ = I.shape
    # padding for channels
    padded = np.pad(
        I, ((int(w / 2), int(w / 2)), (int(w / 2), int(w / 2)), (0, 0)), "edge"
    )
    darkch = np.zeros((M, N))
    brightch = np.zeros((M, N))

    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i : i + w, j : j + w, :])  # dark channel
        brightch[i, j] = np.max(padded[i : i + w, j : j + w, :])  # bright channel

    return darkch, brightch


def reduce_init_t(init_t):
    init_t = (init_t * 255).astype(np.uint8)
    xp = [0, 32, 255]
    fp = [0, 32, 48]
    x = np.arange(256)  # creating array [0,...,255]
    table = np.interp(x, xp, fp).astype(
        "uint8"
    )  # interpreting fp according to xp in range of x
    init_t = cv2.LUT(init_t, table)  # lookup table
    init_t = init_t.astype(np.float64) / 255  # normalizing the transmission map
    return init_t


def show_four_images(image_0, image_1, image_2, image_3):

    # Visualizar las imágenes usando matplotlib
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    image, label = image_0
    axs[0, 0].imshow(image)
    axs[0, 0].set_title(label)
    axs[0, 0].axis("off")

    image, label = image_1
    axs[0, 1].imshow(image)
    axs[0, 1].set_title(label)
    axs[0, 1].axis("off")

    image, label = image_2
    axs[1, 0].imshow(image)
    axs[1, 0].set_title(label)
    axs[1, 0].axis("off")

    image, label = image_3
    axs[1, 1].imshow(image)
    axs[1, 1].set_title(label)
    axs[1, 1].axis("off")

    plt.show()


def main(tmin=0.1, w=5, alpha=0.4, omega=0.75, p=0.1, eps=1e-3, reduce=False):

    # Asumiendo que NightDataset y transform están definidos correctamente
    dataset = NightDataset(
        csv_file="..\\frames_labels.csv",
        root_dir="..\\datasets\\custom_dataset\\Processed_cropped",
        transform=transform,
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for night_image in data_loader:

        night_image_np = (
            night_image.squeeze().permute(1, 2, 0).numpy().astype(np.float32)
        )

        darkch, brightch = get_illumination_channel(night_image_np, w)

        atmosphere = get_atmosphere(night_image_np, brightch, p)

        initial_transmission = get_initial_transmission(atmosphere, brightch)

        normI = (night_image_np - night_image_np.min()) / (
            night_image_np.max() - night_image_np.min()
        ).astype(np.float32)

        ##############################################################################
        corrected_transmission = get_corrected_transmission(
            night_image_np,
            atmosphere,
            darkch,
            brightch,
            initial_transmission,
            alpha,
            omega,
            w,
        )

        refined_transmission = guidedFilter(
            guide=normI,
            src=corrected_transmission.astype(np.float32),
            radius=w,
            eps=eps,
            dDepth=-1,
        )

        final_image = get_final_image(
            night_image_np, atmosphere, refined_transmission, tmin
        )
        enhanced_final_image = (final_image * 255).astype(np.uint8)
        enhanced_final_image = cv2.detailEnhance(
            enhanced_final_image, sigma_s=10, sigma_r=0.15
        )
        enhanced_final_image = cv2.edgePreservingFilter(
            enhanced_final_image, flags=1, sigma_s=64, sigma_r=0.2
        )
        #####################################################################################

        initial_transmission_reduced = reduce_init_t(initial_transmission)
        corrected_transmission_reduced = get_corrected_transmission(
            normI,
            atmosphere,
            darkch,
            brightch,
            initial_transmission_reduced,
            alpha,
            omega,
            w,
        )
        refined_transmission_reduced = guidedFilter(
            guide=night_image_np.astype(np.float32),
            src=corrected_transmission_reduced.astype(np.float32),
            radius=w,
            eps=eps,
            dDepth=-1,
        )

        final_image_reduced = get_final_image(
            night_image_np, atmosphere, refined_transmission_reduced, tmin
        )
        enhanced_final_image_reduced = (final_image_reduced * 255).astype(np.uint8)
        enhanced_final_image_reduced = cv2.detailEnhance(
            enhanced_final_image_reduced, sigma_s=10, sigma_r=0.15
        )
        enhanced_final_image_reduced = cv2.edgePreservingFilter(
            enhanced_final_image_reduced, flags=1, sigma_s=64, sigma_r=0.2
        )

        show_four_images(
            (night_image_np, "Input image"),
            (final_image, "Input image processed stock"),
            (final_image_reduced, "Final imagen light sources reduced"),
            (
                enhanced_final_image_reduced,
                "Final imagen light sources reduced and enhanced",
            ),
        )

        a = input("E para salir, cualquier tecla para continuar \n")
        if a.lower() == "e":
            break


if __name__ == "__main__":
    main()
