import cv2
import os
import threading
import csv

###############################################################################################################################

# Esta función recibe como entrada un directorio con videos y obtiene los frames de cada uno de esos videos,
# Lo guarda en un directorio con el nombre del video, los redimensiona a 480p
# Además, solamente guarda un frame por segundo para alivianar el proceso.


def process_video(video_path, output_directory):
    video_name = os.path.basename(video_path)
    frame_directory = os.path.join(output_directory, video_name[:-4])

    # Validación para evitar procesar si ya fue procesado previamente
    if os.path.exists(frame_directory):
        return

    if not os.path.exists(frame_directory):
        os.makedirs(frame_directory)

    print(f"Processing {video_name}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Obtener el número de FPS del video
    frame_count = 0
    saved_frame_count = 0  # Contador para los frames guardados

    interval = int(fps * 2)  # Calcular el número de frames que equivalen a 2 segundos

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Guardar un frame cada intervalo calculado (cada 2 segundos)
        if frame_count % interval == 0:
            frame = cv2.resize(frame, (852, 480))  # Redimensionar el frame a 480p
            frame_file = os.path.join(frame_directory, f"{saved_frame_count:09d}.png")
            cv2.imwrite(frame_file, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Frames extracted for {video_name}: {saved_frame_count} frames")


def video_to_frames(video_directory, output_directory):
    print(f"Processing {video_directory} directory.")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    threads = []
    for video_name in os.listdir(video_directory):
        if video_name.lower().endswith(".mov") or video_name.lower().endswith(".mp4"):
            video_path = os.path.join(video_directory, video_name)
            t = threading.Thread(
                target=process_video, args=(video_path, output_directory)
            )
            t.start()
            threads.append(t)

    # Esperar a que todos los hilos terminen
    for t in threads:
        t.join()


def convert_videos_dir_to_frame():
    # Asegúrate de ajustar las rutas de video_directory y output_directory según sea necesario
    output_directory = "custom_dataset"

    video_directory = "E:\DashCam_videos\\common_rides"
    video_to_frames(video_directory, output_directory)

    video_directory = "E:\DashCam_videos\\uncommon_rides"
    video_to_frames(video_directory, output_directory)

    video_directory = "E:\DashCam_videos\\rest"
    video_to_frames(video_directory, output_directory)


###############################################################################################################################


def create_csv_from_directory(base_dir, output_csv):
    """
    Recorre todos los subdirectorios en base_dir, lista los frames y crea un CSV.

    :param base_dir: Directorio base con subdirectorios que contienen imágenes.
    :param output_csv: Path del archivo CSV de salida.
    """
    header = ["Path", "noche", "soleado", "nublado", "lluvia", "neblina", "sombras"]

    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Escribir el encabezado en el CSV

        for root, dirs, files in os.walk(base_dir):
            for file in files:
                # Asumiendo que todos los archivos en los directorios son imágenes relevantes
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    # Construir el path relativo para cada imagen
                    rel_path = os.path.relpath(os.path.join(root, file), start=base_dir)
                    # Inicializar todas las condiciones ambientales en 0
                    row = [rel_path] + [""] * (len(header) - 1)
                    writer.writerow(row)


def create_csv():
    # Reemplaza 'directorioA' con tu path relativo o absoluto al directorio base
    base_dir = "custom_dataset\Processed"
    output_csv = "frames_labels.csv"  # Nombre del archivo CSV de salida

    create_csv_from_directory(base_dir, output_csv)


###############################################################################################################################


def count_rows_with_data(csv_path):
    """
    Cuenta el número de filas en el archivo CSV que tienen datos en los campos de tags.

    :param csv_path: Ruta al archivo CSV.
    :return: Número de filas con datos.
    """
    count = 0
    with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Saltar el encabezado
        for row in reader:
            # Verificar si alguna de las columnas de tags tiene datos
            if any(tag for tag in row[1:] if tag.strip()):
                count += 1

    return count


def print_data_rows_counter():
    csv_path = "frames_labels.csv"  # Reemplaza esto con la ruta real a tu archivo CSV
    num_rows_with_data = count_rows_with_data(csv_path)
    print(f"Número de filas con datos: {num_rows_with_data}")


###############################################################################################################################


def update_csv_tags(csv_path, image_prefix, tags):
    """
    Actualiza los tags de las imágenes en un archivo CSV.

    :param csv_path: Ruta al archivo CSV.
    :param image_prefix: Prefijo del path de las imágenes a actualizar.
    :param tags: Diccionario con los nombres de los tags como claves y booleanos como valores.
    """
    updated_rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["Path"].startswith(image_prefix):
                for tag, value in tags.items():
                    row[tag] = "1" if value else "0"
            updated_rows.append(row)

    # Escribir los datos actualizados de nuevo al archivo CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = updated_rows[
            0
        ].keys()  # Tomar los nombres de las columnas del primer row actualizado
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)


def update_specific_csv_tags(csv_path, image_prefix, frames, tags):
    """
    Actualiza los tags de imágenes específicas en un archivo CSV.

    :param csv_path: Ruta al archivo CSV.
    :param image_prefix: Prefijo del path de las imágenes a actualizar.
    :param frames: Lista de números de frame específicos para actualizar.
    :param tags: Diccionario con los nombres de los tags como claves y booleanos como valores.
    """
    updated_rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Extraer el número de frame del nombre del archivo
            frame_number = int(row["Path"].split("\\")[-1].split(".")[0])
            # Verificar si el path de la imagen coincide con el prefijo y si el frame está en la lista
            if row["Path"].startswith(image_prefix) and frame_number in frames:
                for tag, value in tags.items():
                    row[tag] = "1" if value else "0"
            updated_rows.append(row)

    # Escribir los datos actualizados de nuevo al archivo CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = updated_rows[
            0
        ].keys()  # Tomar los nombres de las columnas del primer row actualizado
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)


def update_csv():
    # Ejemplo de uso
    csv_path = "frames_labels.csv"  # Asegúrate de reemplazar 'tu_archivo.csv' con la ruta real de tu archivo CSV
    image_prefix = "BlueFalls-RioCuarto-nublado\\0"  # Prefijo del path de las imágenes a actualizar
    frames = []
    tags = {
        "noche": False,
        "soleado": False,
        "nublado": True,
        "lluvia": True,
        "neblina": False,
        "sombras": False,
    }
    if frames == []:
        update_csv_tags(csv_path, image_prefix, tags)
    else:
        update_specific_csv_tags(csv_path, image_prefix, frames, tags)


###############################################################################################################################


def find_and_remove_empty_directories_aux(root_dir):
    """
    Busca y imprime los nombres de los directorios vacíos dentro de root_dir.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Un directorio se considera vacío si no contiene subdirectorios ni archivos
        if not dirnames and not filenames:
            print(f"Directorio vacío encontrado: {dirpath}")
            os.rmdir(dirpath)


def find_and_remove_empty_directories():
    root_dir = "custom_dataset"
    find_and_remove_empty_directories_aux(root_dir)


###############################################################################################################################


def rename_directories(root_dir):
    """
    Renombra los directorios bajo root_dir para eliminar tildes y cambiar 'ñ' por 'n'.
    """
    for dirpath, dirnames, _ in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            print(dirname)
            new_dirname = dirname.translate(
                str.maketrans("áéíóúñÁÉÍÓÚÑ", "aeiounAEIOUN")
            )
            if dirname != new_dirname:
                original_path = os.path.join(dirpath, dirname)
                new_path = os.path.join(dirpath, new_dirname)
                print(f"Renombrando '{original_path}' a '{new_path}'")
                os.rename(original_path, new_path)


def remove_accents_and_rename_directories():

    root_dir = "E:\DashCam_videos\\common_rides"
    rename_directories(root_dir)

    root_dir = "E:\DashCam_videos\\uncommon_rides"
    rename_directories(root_dir)

    root_dir = "E:\DashCam_videos\\rest"
    rename_directories(root_dir)


#########################################################################################################################################


def cleaning_wrong_directories_pipeline():
    find_and_remove_empty_directories()
    remove_accents_and_rename_directories()
    convert_videos_dir_to_frame()


# cleaning_wrong_directories_pipeline()
# create_csv()
update_csv()
print_data_rows_counter()
# find_and_remove_empty_directories()
#  remove_accents_and_rename_directories()
# convert_videos_dir_to_frame()
