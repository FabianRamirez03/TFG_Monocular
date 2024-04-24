import tkinter as tk
from tkinter import filedialog, scrolledtext
from datetime import datetime
from PIL import Image, ImageTk

# Logic


def button_clicked():

    print("El botón fue presionado")


def upload_action(event=None):
    filename = filedialog.askopenfilename()
    print(
        "Selected:", filename
    )  # Esto es solo para comprobar que se ha seleccionado un archivo


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


def open_and_resize_image():
    global input_image_label
    file_path = filedialog.askopenfilename(
        title="Open Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")],
    )
    ConsolePrint("Image selected: " + file_path)
    if file_path:
        with Image.open(file_path) as img:
            resized_img = img.resize((426, 240), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(resized_img)

            # Actualizar la imagen en el label
            input_image_label.config(image=tk_image)
            input_image_label.image = tk_image  # Mantener una referencia


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


# Globals

title_Lato = ("Lato", 22)

# Configuración inicial de la ventana principal
root = tk.Tk()
root.title("Interfaz de procesamiento de imagen")
root.geometry("1280x720")
root.resizable(False, False)


# Frames
main_frame = tk.Frame(root, background="white")
main_frame.place(x=0, y=0, width=1280, height=720)

input_frame = tk.Frame(
    main_frame,
    background="white",
    bd=5,
    highlightbackground="black",
    highlightthickness=4,
)
input_frame.place(x=36, y=33, width=728, height=353)

console_frame = tk.Frame(
    main_frame,
    background="white",
    bd=5,
    highlightbackground="black",
    highlightthickness=4,
)
console_frame.place(x=792, y=33, width=454, height=353)

botton_frame = tk.Frame(
    main_frame,
    background="white",
    bd=5,
    highlightbackground="black",
    highlightthickness=4,
)
botton_frame.place(x=36, y=400, width=1210, height=307)


# Titles

input_label = tk.Label(input_frame, text="Input", bg="white", font=title_Lato)
input_label.place(x=43, y=6)

console_label = tk.Label(console_frame, text="Console", bg="white", font=title_Lato)
console_label.place(x=43, y=6)

processed_label = tk.Label(
    botton_frame, text="Default depth map", bg="white", font=title_Lato
)
processed_label.place(x=43, y=6)

default_label = tk.Label(
    botton_frame, text="Depth map with pre processed", bg="white", font=title_Lato
)
default_label.place(x=497, y=6)

# Images

no_image_path = "gui_images\\no_image.png"
no_image_PI = tk.PhotoImage(file=no_image_path)

input_image_label = tk.Label(input_frame, image=no_image_PI, width=426, height=240)
input_image_label.place(x=43, y=43)

default_image_label = tk.Label(botton_frame, image=no_image_PI, width=426, height=240)
default_image_label.place(x=43, y=43)

processed_image_label = tk.Label(botton_frame, image=no_image_PI, width=426, height=240)
processed_image_label.place(x=497, y=43)


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
Upload_button.place(x=497, y=43)

process_bt_path = "gui_images\\process_bt.png"
process_bt_PI = tk.PhotoImage(file=process_bt_path)
process_button = tk.Button(
    input_frame,
    image=process_bt_PI,
    command=lambda: ConsolePrint("Mensaje de prueba a la consola"),
    borderwidth=0,
    highlightthickness=0,
    padx=0,
    pady=0,
)
process_button.place(x=497, y=88)

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
save_logs_button.place(x=311, y=287)

rewind_bt_path = "gui_images\\rewind_bt.png"
rewind_bt_PI = tk.PhotoImage(file=rewind_bt_path)
rewind_button = tk.Button(
    botton_frame,
    image=rewind_bt_PI,
    command=button_clicked,
    borderwidth=0,
    highlightthickness=0,
    padx=0,
    pady=0,
)
rewind_button.place(x=1002, y=43)

play_bt_path = "gui_images\\play_bt.png"
play_bt_PI = tk.PhotoImage(file=play_bt_path)
play_button = tk.Button(
    botton_frame,
    image=play_bt_PI,
    command=button_clicked,
    borderwidth=0,
    highlightthickness=0,
    padx=0,
    pady=0,
)
play_button.place(x=1002, y=88)


# Console

# Configurar el widget de texto como una consola de logs
console = scrolledtext.ScrolledText(
    console_frame, height=15, width=57, font=("Lato", 9), state=tk.DISABLED
)
console.place(x=11, y=53)


# State leds images

led_off_PI = ImageTk.PhotoImage(Image.open("gui_images\\led_off.png"))
led_on_PI = ImageTk.PhotoImage(Image.open("gui_images\\led_on.png"))

nublado = False
noche = False
soleado = False
lluvia = False
neblina = False

nublado_image = tk.Label(input_frame)
nublado_image.place(x=43, y=313)

noche_image = tk.Label(input_frame)
noche_image.place(x=139, y=313)

soleado_image = tk.Label(input_frame)
soleado_image.place(x=223, y=313)

lluvia_image = tk.Label(input_frame)
lluvia_image.place(x=316, y=313)

neblina_image = tk.Label(input_frame)
neblina_image.place(x=395, y=313)

update_leds()

# State leds labels

nublado_label = tk.Label(
    input_frame,
    text="Nublado",
    bg="white",
    font=("Lato", 14),
    borderwidth=0,
    highlightthickness=0,
    padx=0,
    pady=0,
)
nublado_label.place(x=61, y=312)

noche_label = tk.Label(
    input_frame,
    text="Noche",
    bg="white",
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
    bg="white",
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
    bg="white",
    font=("Lato", 14),
    borderwidth=0,
    highlightthickness=0,
    padx=0,
    pady=0,
)
lluvia_label.place(x=334, y=312)

neblina_label = tk.Label(
    input_frame,
    text="Neblina",
    bg="white",
    font=("Lato", 14),
    borderwidth=0,
    highlightthickness=0,
    padx=0,
    pady=0,
)
neblina_label.place(x=414, y=312)

# Ejecuta la ventana principal
root.mainloop()
