import tkinter as tk
from tkinter import filedialog


# Función para abrir el diálogo de selección de archivos
def upload_action(event=None):
    filename = filedialog.askopenfilename()
    print(
        "Selected:", filename
    )  # Esto es solo para comprobar que se ha seleccionado un archivo


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
    background="red",
    bd=5,
    highlightbackground="black",
    highlightthickness=4,
)
input_frame.place(x=36, y=33, width=728, height=353)

console_frame = tk.Frame(
    main_frame,
    background="blue",
    bd=5,
    highlightbackground="black",
    highlightthickness=4,
)
console_frame.place(x=792, y=33, width=454, height=353)

botton_frame = tk.Frame(
    main_frame,
    background="yellow",
    bd=5,
    highlightbackground="black",
    highlightthickness=4,
)
botton_frame.place(x=36, y=400, width=1210, height=307)


# Titles

input_label = tk.Label(input_frame, text="Input", bg="white", font=title_Lato)
input_label.place(x=43, y=11)

console_label = tk.Label(console_frame, text="Console", bg="white", font=title_Lato)
console_label.place(x=43, y=11)

processed_label = tk.Label(
    botton_frame, text="Default depth map", bg="white", font=title_Lato
)
processed_label.place(x=43, y=11)

default_label = tk.Label(
    botton_frame, text="Depth map with pre processed", bg="white", font=title_Lato
)
default_label.place(x=497, y=11)

# Images

no_image_path = "gui_images\\no_image.png"
no_image_PI = tk.PhotoImage(file=no_image_path)

input_image_label = tk.Label(input_frame, image=no_image_PI)
input_image_label.place(x=43, y=43)

default_image_label = tk.Label(botton_frame, image=no_image_PI)
default_image_label.place(x=43, y=43)

processed_image_label = tk.Label(botton_frame, image=no_image_PI)
processed_image_label.place(x=497, y=43)


# Ejecuta la ventana principal
root.mainloop()
