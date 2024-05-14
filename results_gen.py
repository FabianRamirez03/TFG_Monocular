from tagger.model import DualInputCNN
from tagger.data_loader import CustomDataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main_tagger():
    global device
    model_tagger_path = "models\\tagger.pth"
    model_tagger = DualInputCNN()
    model_tagger.load_state_dict(torch.load(model_tagger_path, map_location=device))
    model_tagger = model_tagger.to(device)
    model_tagger.eval()

    root_dir = "datasets\\Results_datasets\\tagger\\frames"
    csv_file = "datasets\\Results_datasets\\tagger\\frames\\frames_labels.csv"

    resize_size = [232]
    crop_size = [224]

    transform = transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

    dataloader = DataLoader(dataset, batch_size=1)

    bit_accuracies = tagger_validate_model(model_tagger, dataloader, device)

    tagger_error_statistics(bit_accuracies, csv_file)


def tagger_validate_model(model, dataloader, device="cpu"):
    model.to(device)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    results = []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        height = inputs.size(2)
        upper_section = inputs[:, :, : int(height * 0.25), :]
        lower_section = inputs[:, :, int(height * 0.25) :, :]

        upper_section, lower_section = upper_section.to(device), lower_section.to(
            device
        )
        labels = labels.to(device).type(torch.float32)

        outputs = model(upper_section, lower_section)
        preds = torch.sigmoid(outputs).cpu().detach().numpy() > 0.5
        labels_bool = np.array(labels.cpu().detach().numpy()).astype(bool)

        results.append(preds == labels_bool)

    return results


def tagger_error_statistics(results, csv_path):
    # Leer el CSV para obtener las rutas de las imágenes
    data = pd.read_csv(csv_path)
    image_paths = data["Path"].values

    # Asumimos que los resultados están en una lista de listas, convertimos a un array de numpy
    results_array = np.vstack(results)

    # Calculamos la cantidad de False por columna, que representan los errores
    errors_per_category = np.sum(~results_array, axis=0)

    # Calculamos el total de predicciones evaluadas por categoría
    total_predictions = results_array.shape[0]

    # Nombres de las categorías para hacer más entendible el resultado
    categories = ["noche", "soleado", "nublado", "lluvia", "neblina", "sombras"]

    # Imprimir de manera amigable
    print("Estadísticas de errores por categoría:")
    for category, errors in zip(categories, errors_per_category):
        error_percentage = (
            errors / total_predictions
        ) * 100  # Calculamos el porcentaje de errores
        print(f"{category.capitalize()}: {errors} errores ({error_percentage:.2f}%)")

    # Guardar los resultados originales en un nuevo CSV
    save_results_to_csv(results, image_paths, csv_path)


def save_results_to_csv(results, image_paths, original_csv_path):
    # Crear un DataFrame para guardar los nuevos resultados
    updated_data = pd.read_csv(original_csv_path)

    for idx, result in enumerate(results):
        updated_data.loc[idx, updated_data.columns[1:]] = result[0].astype(
            int
        )  # Actualizamos las predicciones

    # Guardamos el nuevo CSV con las predicciones actualizadas
    updated_data.to_csv("updated_results.csv", index=False)


def print_image_names_no_rain(csv_path):
    """
    Imprime los nombres de las imágenes donde la columna de lluvia sea 0.

    :param csv_path: Ruta al archivo CSV.
    """
    # Leer el archivo CSV usando pandas
    data = pd.read_csv(csv_path)

    # Filtrar las filas donde la columna 'lluvia' es igual a 0
    no_rain_images = data[data["neblina"] == 0]

    # Imprimir los nombres de las imágenes sin lluvia
    for image_path in no_rain_images["Path"]:
        print(image_path)


if __name__ == "__main__":
    results_csv = "updated_results.csv"
    print_image_names_no_rain(results_csv)
    # main_tagger()
