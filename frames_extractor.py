import cv2
import os
import threading

# Este archivo recibe como entrada un directorio con videos y obtiene los frames de cada uno de esos videos,
# Lo guarda en un directorio con el nombre del video, los redimensiona a 480p
# Además, solamente guarda un frame por segundo para alivianar el proceso.


def process_video(video_path, output_directory):
    video_name = os.path.basename(video_path)
    frame_directory = os.path.join(output_directory, video_name[:-4])

    # Validación para evitar procesar si ya fue procesado previamente
    if os.path.exists(frame_directory):
        print(f"El video {video_name} ya fue procesado anteriormente. Omitiendo...")
        return

    if not os.path.exists(frame_directory):
        os.makedirs(frame_directory)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Obtener el número de FPS del video
    frame_count = 0
    saved_frame_count = 0  # Contador para los frames guardados

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Guardar solo un frame por segundo
        if frame_count % int(fps) == 0:
            frame = cv2.resize(frame, (852, 480))  # Redimensionar el frame a 480p
            frame_file = os.path.join(frame_directory, f"{saved_frame_count:09d}.png")
            cv2.imwrite(frame_file, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Frames extracted for {video_name}: {saved_frame_count} frames")


def video_to_frames(video_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    threads = []
    for video_name in os.listdir(video_directory):
        print(f"Processing video: {video_name}")
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


# Asegúrate de ajustar las rutas de video_directory y output_directory según sea necesario
video_directory = "E:\DashCam_videos\common_rides"
output_directory = "D:\Fabian\Datasets_Backups\Personal\common_rides_frames"
video_to_frames(video_directory, output_directory)
