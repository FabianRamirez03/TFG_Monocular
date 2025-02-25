import tkinter as tk
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tkinter import filedialog, scrolledtext
from datetime import datetime
from PIL import Image, ImageTk
from tagger.model import DualInputCNN
from deraining.model import Deraining_UNet
from night_enhancer.night_enhancer import enhance_night_image
from dehazing.model import Dehazing_UNet
from skimage.feature import hog
import csv
import matplotlib.pyplot as plt

# Processing Globals

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

current_pil_image = None

model_tagger_path = "models\\tagger.pth"
model_tagger = DualInputCNN()
model_tagger.load_state_dict(torch.load(model_tagger_path, map_location=device))
model_tagger = model_tagger.to(device)
model_tagger.eval()

transform_tagger = transforms.Compose(
    [
        transforms.Resize([232], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop([224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

derain_model_path = "models\\deraining.pth"
derain_model = Deraining_UNet(in_channels=3, out_channels=3).to(device)
derain_model.load_state_dict(torch.load(derain_model_path, map_location=device))
derain_model.eval()

dehazing_model_path = "models\\dehazing.pth"
dehazing_model = Dehazing_UNet(3, 3).to(device)
dehazing_model.load_state_dict(torch.load(dehazing_model_path, map_location=device))
dehazing_model.eval()

# Logic


def button_clicked():

    print("El botón fue presionado")


def upload_action(event=None):
    filename = filedialog.askopenfilename()
    print(
        "Selected:", filename
    )  # Esto es solo para comprobar que se ha seleccionado un archivo


def save_label_image(label, filepath):
    # Asegúrate de que el directorio existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Obtén el nombre de la imagen asociada al label
    image_name = label.cget("image")
    if not image_name:
        raise ValueError("El label no tiene una imagen asociada")

    # Obtén la referencia real de PhotoImage
    photo_image = label.image

    # Convierte PhotoImage a una imagen de PIL
    pil_image = ImageTk.getimage(photo_image)

    # Guarda la imagen en el path especificado
    pil_image.save(filepath)


def save_gui_images():
    global input_image_label, processed_input_image_label, default_image_label, processed_image_label
    save_label_image(input_image_label, "temp_images\\gui\\input_image.png")
    save_label_image(
        processed_input_image_label, "temp_images\\gui\\processed_input_image.png"
    )
    save_label_image(default_image_label, "temp_images\\gui\\default_map.png")
    save_label_image(processed_image_label, "temp_images\\gui\\processed_map.png")


def hog_score(depth_map_path):

    depth_map = cv2.imread(depth_map_path)
    if depth_map is None:
        raise ValueError(f"No se pudo cargar la imagen de la ruta: {depth_map_path}")

    # Convertir la imagen de mapa de profundidad a escala de grises
    gray = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

    # Calcular características HOG
    features, hog_image = hog(
        gray,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
    )

    # Calcular la puntuación como la media de las características HOG
    hog_score = np.mean(features)

    image_name = depth_map_path.split("\\")[-1]

    ConsolePrint(f"HOG score for {image_name}: {hog_score}")

    return hog_score


def edge_detection_score(depth_map_path):
    # Cargar la imagen de mapa de profundidad
    depth_map = cv2.imread(depth_map_path)
    if depth_map is None:
        raise ValueError(f"No se pudo cargar la imagen de la ruta: {depth_map_path}")

    # Convertir la imagen de mapa de profundidad a escala de grises
    gray = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro Gaussiano para suavizar la imagen y reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar detección de bordes usando el algoritmo de Canny
    edges = cv2.Canny(blurred, 50, 150, apertureSize=5, L2gradient=True)

    # Contar el número de píxeles de borde detectados
    edge_count = np.sum(edges > 0)

    # Normalizar según el tamaño de la imagen
    height, width = gray.shape
    total_pixels = height * width
    edge_score = edge_count / total_pixels

    # Mostrar la imagen de bordes para depuración
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title(f"Edges Detected (Score: {edge_score:.2f})")
    plt.imshow(edges, cmap="gray")

    # plt.show()

    return edge_score


def update_csv(default_score, processed_score, csv_path):
    ConsolePrint(f"Guardando puntajes en {csv_path}")
    # Abrir el archivo en modo de añadido (append)
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        # Escribir la nueva fila con los puntajes
        writer.writerow([default_score, processed_score])


def ConsolePrint(message):
    global console
    current_time = datetime.now().strftime("%H:%M:%S")
    timestamped_message = f"{current_time} -> {message}\n"

    console.config(state=tk.NORMAL)
    console.insert(tk.END, timestamped_message)
    console.config(state=tk.DISABLED)
    console.see(tk.END)


def save_logs():
    global console

    filename = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    if filename:  # Si el usuario no cancela el diálogo
        # Abrir el archivo para escritura y guardar el contenido de la consola
        with open(filename, "w") as file:
            # Extraer el texto del widget de texto
            log_text = console.get("1.0", tk.END)
            file.write(log_text)
            ConsolePrint(f"Logs guardados en {filename}")


def reset_images_results():
    global processed_input_image_label, default_image_label, processed_image_label, no_image_PI

    processed_input_image_label.config(image=no_image_PI)
    processed_input_image_label.image = no_image_PI

    default_image_label.config(image=no_image_PI)
    default_image_label.image = no_image_PI

    processed_image_label.config(image=no_image_PI)
    processed_image_label.image = no_image_PI


def open_and_resize_image():
    global input_image_label, current_pil_image
    file_path = filedialog.askopenfilename(
        title="Open Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")],
    )
    ConsolePrint("Image selected: " + file_path)
    if file_path:
        with Image.open(file_path) as img:
            current_pil_image = img
            resized_img = img.resize((426, 240), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(resized_img)

            # Actualizar la imagen en el label
            input_image_label.config(image=tk_image)
            input_image_label.image = tk_image  # Mantener una referencia

            inference_tagger()

            reset_images_results()


def update_single_led(flag, label):
    global led_off_PI, led_on_PI
    if flag:
        label.config(image=led_on_PI)
    else:
        label.config(image=led_off_PI)


def update_leds():
    global nublado, noche, soleado, lluvia, neblina
    global nublado_image, noche_image, soleado_image, lluvia_image, neblina_image

    update_single_led(nublado, nublado_image)
    update_single_led(noche, noche_image)
    update_single_led(soleado, soleado_image)
    update_single_led(lluvia, lluvia_image)
    update_single_led(neblina, neblina_image)


def toggle_nublado():
    global nublado
    nublado = not nublado
    update_leds()


def toggle_noche():
    global noche
    noche = not noche
    update_leds()


def toggle_soleado():
    global soleado
    soleado = not soleado
    update_leds()


def toggle_lluvia():
    global lluvia
    lluvia = not lluvia
    update_leds()


def toggle_neblina():
    global neblina
    neblina = not neblina
    update_leds()


def normalize_depth_array(depth_array):
    """Normaliza un mapa de profundidad para estar en el rango 0-255"""
    depth_min = depth_array.min()
    depth_max = depth_array.max()
    if depth_max - depth_min > 0:
        depth_array = (depth_array - depth_min) / (depth_max - depth_min) * 255
    else:
        depth_array = np.zeros(depth_array.shape)
    return depth_array.astype(np.uint8)


def apply_color_map(depth_array):
    """Aplica un mapa de colores a un mapa de profundidad y devuelve una imagen de PIL"""
    colored_depth = cv2.applyColorMap(depth_array, cv2.COLORMAP_MAGMA)
    colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored_depth)


def change_working_directory(path):
    """Cambia el directorio de trabajo y devuelve el original para restaurarlo más tarde"""
    original_cwd = os.getcwd()
    os.chdir(path)
    return original_cwd


def AdaBins_infer():
    global neblina, noche, lluvia
    # Infer the default image
    if current_pil_image is not None:
        AdaBins_infer_processed()

        AdaBins_infer_default()

        save_gui_images()

        edge_score_default = edge_detection_score("temp_images\\gui\\default_map.png")
        edge_score_processed = edge_detection_score(
            "temp_images\\gui\\processed_map.png"
        )

        if neblina:
            update_csv(edge_score_default, edge_score_processed, "haze_results.csv")
        elif noche:
            update_csv(edge_score_default, edge_score_processed, "night_results.csv")
        elif lluvia:
            update_csv(edge_score_default, edge_score_processed, "rain_results.csv")

    else:
        ConsolePrint("No image or video selected.")


def AdaBins_infer_default():
    global current_pil_image, default_image_label

    # Cambiar al directorio del submódulo y guardar el directorio actual
    original_cwd = change_working_directory("AdaBins")

    try:
        from AdaBins.infer import InferenceHelper

        infer_helper = InferenceHelper(dataset="kitti")

        bin_centers, predicted_depth = infer_helper.predict_pil(
            current_pil_image.resize((852, 480), Image.Resampling.LANCZOS)
        )
        predicted_depth = np.squeeze(predicted_depth)

        normalized_depth = normalize_depth_array(predicted_depth)
        inference_image = apply_color_map(normalized_depth)

        inference_image = inference_image.resize((426, 240), Image.Resampling.LANCZOS)
        photo_image = ImageTk.PhotoImage(inference_image)
        default_image_label.config(image=photo_image)
        default_image_label.image = photo_image  # Mantener una referencia

    finally:
        # Asegurarse de volver al directorio original
        os.chdir(original_cwd)


def AdaBins_infer_processed():
    global current_pil_image, processed_image_label, processed_input_image_label
    global nublado, noche, soleado, lluvia, neblina

    original_cwd = os.getcwd()

    try:
        from AdaBins.infer import InferenceHelper

        image_to_process = current_pil_image.resize(
            (224, 224), Image.Resampling.LANCZOS
        )
        if noche:
            image_to_process = enhance_night_image(image_to_process)
        if lluvia:
            image_to_process = derain_image(image_to_process)
        if neblina:
            image_to_process = dehaze_image(image_to_process)

        change_working_directory("AdaBins")

        infer_helper = InferenceHelper(dataset="nyu")

        bin_centers, predicted_depth = infer_helper.predict_pil(
            image_to_process.resize((852, 480), Image.Resampling.LANCZOS)
        )
        predicted_depth = np.squeeze(predicted_depth)

        normalized_depth = normalize_depth_array(predicted_depth)
        inference_image = apply_color_map(normalized_depth)

        inference_image = inference_image.resize((426, 240), Image.Resampling.LANCZOS)
        photo_image = ImageTk.PhotoImage(inference_image)

        processed_image_label.config(image=photo_image)
        processed_image_label.image = photo_image  # Mantener una referencia

        image_to_process_pi = ImageTk.PhotoImage(
            image_to_process.resize((426, 240), Image.Resampling.LANCZOS)
        )
        processed_input_image_label.config(image=image_to_process_pi)
        processed_input_image_label.image = image_to_process_pi

    finally:
        # Asegurarse de volver al directorio original
        os.chdir(original_cwd)


def derain_image(image):
    ConsolePrint("Deraining image")
    global device, derain_model

    transform = transforms.ToTensor()
    tensor_image = transform(image).to(device).unsqueeze(0)

    generated_image = derain_model(tensor_image).cpu().squeeze(0)

    pil_image = TF.to_pil_image(generated_image)

    return pil_image


def dehaze_image(image):
    ConsolePrint("Dehazing image")
    global device, dehazing_model

    transform = transforms.ToTensor()
    tensor_image = transform(image).to(device).unsqueeze(0)

    generated_image = dehazing_model(tensor_image).cpu().squeeze(0)

    pil_image = TF.to_pil_image(generated_image)

    return pil_image


def update_boolean_leds(preds):
    global nublado, noche, soleado, lluvia, neblina
    noche = preds[0][0]
    soleado = preds[0][1]
    nublado = preds[0][2]
    lluvia = preds[0][3]
    neblina = preds[0][4]

    update_leds()


def inference_tagger():
    global current_pil_image, model_tagger, transform_tagger

    image_transformed = transform_tagger(current_pil_image).unsqueeze(0)

    height = image_transformed.size(2)
    upper_section = image_transformed[:, :, : int(height * 0.25), :]
    lower_section = image_transformed[:, :, int(height * 0.25) :, :]

    upper_section, lower_section = upper_section.to(device), lower_section.to(device)

    with torch.no_grad():
        outputs = model_tagger(upper_section, lower_section)
        preds = torch.sigmoid(outputs).cpu().numpy() > 0.5

    update_boolean_leds(preds)


# GUI Globals

title_Lato = ("Lato", 22)
bg_color = "#FDFDFD"

# Configuración inicial de la ventana principal
root = tk.Tk()
root.title("Interfaz de procesamiento de imagen")
root.geometry("1056x882")
root.resizable(False, False)


# Frames
main_frame = tk.Frame(root, background=bg_color)
main_frame.place(x=0, y=0, width=1056, height=882)

input_frame = tk.Frame(
    main_frame,
    background=bg_color,
    bd=5,
    highlightbackground="black",
    highlightthickness=2,
)
input_frame.place(x=36, y=36, width=984, height=353)


botton_frame = tk.Frame(
    main_frame,
    background=bg_color,
    bd=5,
    highlightbackground="black",
    highlightthickness=2,
)
botton_frame.place(x=36, y=408, width=984, height=304)

console_frame = tk.Frame(
    main_frame,
    background=bg_color,
    bd=5,
    highlightbackground="black",
    highlightthickness=2,
)
console_frame.place(x=36, y=726, width=984, height=140)

# Titles

input_label = tk.Label(input_frame, text="Input", bg=bg_color, font=title_Lato)
input_label.place(x=43, y=6)

input_label = tk.Label(
    input_frame, text="Processed input image", bg=bg_color, font=title_Lato
)
input_label.place(x=513, y=6)


console_label = tk.Label(console_frame, text="Console", bg=bg_color, font=title_Lato)
console_label.place(x=43, y=6)

processed_label = tk.Label(
    botton_frame, text="Default depth map", bg=bg_color, font=title_Lato
)
processed_label.place(x=43, y=6)

default_label = tk.Label(
    botton_frame, text="Depth map with pre processed", bg=bg_color, font=title_Lato
)
default_label.place(x=515, y=6)

# Images

no_image_path = "gui_images\\no_image.png"
no_image_PI = tk.PhotoImage(file=no_image_path)

input_image_label = tk.Label(input_frame, image=no_image_PI, width=426, height=240)
input_image_label.place(x=43, y=43)

processed_input_image_label = tk.Label(
    input_frame, image=no_image_PI, width=426, height=240
)
processed_input_image_label.place(x=513, y=43)

default_image_label = tk.Label(botton_frame, image=no_image_PI, width=426, height=240)
default_image_label.place(x=43, y=43)

processed_image_label = tk.Label(botton_frame, image=no_image_PI, width=426, height=240)
processed_image_label.place(x=513, y=43)


# Buttons

Upload_bt_path = "gui_images\\Upload_bt.png"
Upload_bt_PI = tk.PhotoImage(file=Upload_bt_path)
Upload_button = tk.Button(
    input_frame,
    image=Upload_bt_PI,
    command=open_and_resize_image,
    borderwidth=0,
    highlightthickness=0,
    padx=0,
    pady=0,
)
Upload_button.place(x=513, y=300)

process_bt_path = "gui_images\\process_bt.png"
process_bt_PI = tk.PhotoImage(file=process_bt_path)
process_button = tk.Button(
    input_frame,
    image=process_bt_PI,
    command=AdaBins_infer,
    borderwidth=0,
    highlightthickness=0,
    padx=0,
    pady=0,
)
process_button.place(x=673, y=300)

save_logs_bt_path = "gui_images\\save_logs_bt.png"
save_logs_bt_PI = tk.PhotoImage(file=save_logs_bt_path)
save_logs_button = tk.Button(
    console_frame,
    image=save_logs_bt_PI,
    command=save_logs,
    borderwidth=0,
    highlightthickness=0,
    padx=0,
    pady=0,
)
save_logs_button.place(x=844, y=4)


# Console

# Configurar el widget de texto como una consola de logs
console = scrolledtext.ScrolledText(
    console_frame, height=4, width=127, font=("Lato", 9), state=tk.DISABLED
)
console.place(x=43, y=50)


# State leds images

led_off_image = Image.open("gui_images\\led_off_trans.png")
led_off_image = led_off_image.resize((14, 14), Image.Resampling.LANCZOS)
led_off_PI = ImageTk.PhotoImage(led_off_image)

led_on_image = Image.open("gui_images\\led_on_trans.png")
led_on_image = led_on_image.resize((14, 14), Image.Resampling.LANCZOS)
led_on_PI = ImageTk.PhotoImage(led_on_image)

nublado = False
noche = False
soleado = False
lluvia = False
neblina = False

nublado_image = tk.Label(input_frame, bg=bg_color)
nublado_image.place(x=43, y=313)
nublado_image.bind("<Button-1>", lambda e: toggle_nublado())

noche_image = tk.Label(input_frame, bg=bg_color)
noche_image.place(x=139, y=313)
noche_image.bind("<Button-1>", lambda e: toggle_noche())

soleado_image = tk.Label(input_frame, bg=bg_color)
soleado_image.place(x=223, y=313)
soleado_image.bind("<Button-1>", lambda e: toggle_soleado())

lluvia_image = tk.Label(input_frame, bg=bg_color)
lluvia_image.place(x=316, y=313)
lluvia_image.bind("<Button-1>", lambda e: toggle_lluvia())

neblina_image = tk.Label(input_frame, bg=bg_color)
neblina_image.place(x=395, y=313)
neblina_image.bind("<Button-1>", lambda e: toggle_neblina())

update_leds()

# State leds labels

nublado_label = tk.Label(
    input_frame,
    text="Nublado",
    bg=bg_color,
    font=("Lato", 14),
    borderwidth=0,
    highlightthickness=0,
    padx=0,
    pady=0,
)
nublado_label.place(x=64, y=312)

noche_label = tk.Label(
    input_frame,
    text="Noche",
    bg=bg_color,
    font=("Lato", 14),
    borderwidth=0,
    highlightthickness=0,
    padx=0,
    pady=0,
)
noche_label.place(x=157, y=312)

soleado_label = tk.Label(
    input_frame,
    text="Soleado",
    bg=bg_color,
    font=("Lato", 14),
    borderwidth=0,
    highlightthickness=0,
    padx=0,
    pady=0,
)
soleado_label.place(x=242, y=312)

lluvia_label = tk.Label(
    input_frame,
    text="Lluvia",
    bg=bg_color,
    font=("Lato", 14),
    borderwidth=0,
    highlightthickness=0,
    padx=0,
    pady=0,
)
lluvia_label.place(x=337, y=312)

neblina_label = tk.Label(
    input_frame,
    text="Neblina",
    bg=bg_color,
    font=("Lato", 14),
    borderwidth=0,
    highlightthickness=0,
    padx=0,
    pady=0,
)
neblina_label.place(x=414, y=312)

# Ejecuta la ventana principal
root.mainloop()
