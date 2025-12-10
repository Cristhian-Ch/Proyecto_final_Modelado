import os
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import subprocess

# ==================================================
# FUNCIONES
# ==================================================
def ejecutar_script(script_name):
    """Ejecuta un script .py si existe."""
    # Usar la carpeta del propio archivo Menu.py como base para scripts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    if os.path.exists(script_path):
        print(f"Ejecutando {script_name}...\n")
        try:
            subprocess.Popen(["python", script_path])
        except Exception as e:
            print(f"Error al ejecutar {script_name}: {e}")
    else:
        print(f"❌ El archivo '{script_name}' no se encontró en {script_dir}")

#Asocia las opciones del menú principal.
"""def opcion_seleccionada(opcion):

    scripts = {
        1: "MatrizInversa.py",
        2: "jacobi.py",
        3: "biseccion.py",
        4: "NewtonRaphson.py",
        5: "secante.py",
        6: "PuntoFijo.py",             
        7: "NewtonModificado.py"
    }
    if opcion in scripts:
        ejecutar_script(scripts[opcion])
    else:
        print("Opción inválida.")"""


def abrir_submenu_sistema(parent):
    """Abre una ventana secundaria con los métodos de Gauss."""
    submenu = tk.Toplevel(parent)
    submenu.title("Métodos de resolucion de Sistemas lineales")
    submenu.geometry("450x600")
    submenu.config(bg="#ecf0f1")

    ttk_style = {
        "font": ("Arial", 11, "bold"),
        "bg": "#ecf0f1",
        "fg": "black",
        "relief": "raised",
        "width": 28,
        "height": 2,
        "bd": 2
    }

    tk.Label(submenu, text="Seleccione un Método",
             font=("Arial", 14, "bold"), bg="#ecf0f1", fg="#2c3e50").pack(pady=15)

    botones_gauss = [
        ("🔹 Eliminación de Gauss", "gausss.py"),
        ("🔹 Método de Gauss-Jordan", "gauss_jordan.py"),
        ("🔢  Método Iterativo de Jacobi", "jacobi.py"),
        ("🔹 Método Iterativo de Gauss-Seidel", "gauss_seidel.py"),
        ("📐 Matriz Inversa", "MatrizInversa.py"),
        ("📌 Punto Fijo", "PuntoFijo.py"),        
        ("🔁 Newton Modificado","NewtonModificado.py"),

    ]

    for texto, script in botones_gauss:
        btn = tk.Button(submenu, text=texto, **ttk_style,
                        command=lambda s=script: ejecutar_script(s))
        btn.pack(pady=8)

    tk.Button(submenu, text="⬅️ Volver al Menú Principal",
              font=("Arial", 10, "bold"),
              bg="#e74c3c", fg="white", width=20,
              command=submenu.destroy).pack(pady=20)
def abrir_submenu_raiz(parent):
    """Abre una ventana secundaria con los métodos de Gauss."""
    submenu = tk.Toplevel(parent)
    submenu.title("Métodos de obtencion de Raiz de funciones no lineales")
    submenu.geometry("450x600")
    submenu.config(bg="#ecf0f1")

    ttk_style = {
        "font": ("Arial", 11, "bold"),
        "bg": "#ecf0f1",
        "fg": "black",
        "relief": "raised",
        "width": 28,
        "height": 2,
        "bd": 2
    }

    tk.Label(submenu, text="Seleccione un Método ",
             font=("Arial", 14, "bold"), bg="#ecf0f1", fg="#2c3e50").pack(pady=15)

    botones_gauss = [
        
        ("✂️ Método de Bisección","biseccion.py"),
        ("⚡ Newton-Raphson", "NewtonRaphson.py"),
        ("📏 Método de Secante","secante.py" ),
        
       
    ]

    for texto, script in botones_gauss:
        btn = tk.Button(submenu, text=texto, **ttk_style,
                        command=lambda s=script: ejecutar_script(s))
        btn.pack(pady=8)

    tk.Button(submenu, text="⬅️ Volver al Menú Principal",
              font=("Arial", 10, "bold"),
              bg="#e74c3c", fg="white", width=20,
              command=submenu.destroy).pack(pady=20)


def abrir_submenu_interpolacion(parent):
    """Abre una ventana secundaria con los métodos de interpolación."""
    submenu = tk.Toplevel(parent)
    submenu.title("Métodos de Interpolación")
    submenu.geometry("450x400")
    submenu.config(bg="#ecf0f1")

    ttk_style = {
        "font": ("Arial", 11, "bold"),
        "bg": "#ecf0f1",
        "fg": "black",
        "relief": "raised",
        "width": 28,
        "height": 2,
        "bd": 2
    }

    tk.Label(submenu, text="Seleccione un Método",
             font=("Arial", 14, "bold"), bg="#ecf0f1", fg="#2c3e50").pack(pady=15)

    botones_interpolacion = [
        ("📐 Interpolación de Lagrange", "lagrange.py"),
        ("📐 Interpolación de Newton", "Inewton.py"),
        ("📊 Mínimos Cuadrados", "MinimosCuadrados.py"),
    ]

    for texto, script in botones_interpolacion:
        btn = tk.Button(submenu, text=texto, **ttk_style,
                        command=lambda s=script: ejecutar_script(s))
        btn.pack(pady=8)

    tk.Button(submenu, text="⬅️ Volver al Menú Principal",
              font=("Arial", 10, "bold"),
              bg="#e74c3c", fg="white", width=20,
              command=submenu.destroy).pack(pady=20)

def abrir_submenu_integracion(parent):
    """Submenú para métodos de integración numérica."""
    submenu = tk.Toplevel(parent)
    submenu.title("Métodos de Integración Numérica")
    submenu.geometry("450x400")
    submenu.config(bg="#ecf0f1")

    ttk_style = {
        "font": ("Arial", 11, "bold"),
        "bg": "#ecf0f1",
        "fg": "black",
        "relief": "raised",
        "width": 28,
        "height": 2,
        "bd": 2
    }

    tk.Label(
        submenu,
        text="Seleccione un Método",
        font=("Arial", 14, "bold"),
        bg="#ecf0f1",
        fg="#2c3e50"
    ).pack(pady=15)

    botones_integracion = [
        ("📏 Método del Trapecio", "trapecio.py"),
        ("📐 Simpson 1/3 (Simple y Compuesto)", "simpson13.py"),
        ("📘 Método Taylor", "taylor.py"),
        ("📊 Euler", "euler.py"),
        ("🚀 Runge_Kutta (RK2 y RK4)", "Runge_kutta.py"),

    ]

    for texto, script in botones_integracion:
        btn = tk.Button(
            submenu,
            text=texto,
            **ttk_style,
            command=lambda s=script: ejecutar_script(s)
        )
        btn.pack(pady=8)

    tk.Button(
        submenu,
        text="⬅️ Volver al Menú Principal",
        font=("Arial", 10, "bold"),
        bg="#e74c3c",
        fg="white",
        width=20,
        command=submenu.destroy
    ).pack(pady=20)

def salir():
    """Cierra la aplicación."""
    if cap.isOpened():
        cap.release()
    root.destroy()


def actualizar_video(video_label):
    """Actualiza continuamente el video."""
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 360))  
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
    video_label.after(10, actualizar_video, video_label)


# ==================================================
# INTERFAZ PRINCIPAL
# ==================================================
def mostrar_interfaz():
    global root, cap
    root = tk.Tk()
    root.title("🎓 Métodos Numéricos - Proyecto Final")
    root.geometry("1000x800")
    root.resizable(False, False)

    # Video de fondo
    video_path = "videomat.mp4"
    cap = cv2.VideoCapture(video_path)

    # Título
    titulo_frame = tk.Frame(root, bg="#2c3e50", height=60)
    titulo_frame.pack(fill="x", side="top")

    titulo_label = tk.Label(
        titulo_frame,
        text="MÉTODOS DE RESOLUCIÓN LINEALES Y NO LINEALES",
        font=("Arial", 16, "bold"),
        fg="white", bg="#2c3e50"
    )
    titulo_label.pack(expand=True)
  
    # Video
    video_label = tk.Label(root)
    video_label.pack(fill="both", expand=True)
    video_label.place(relx=0.5, rely=0.4, anchor="center")
    actualizar_video(video_label)

    # Menú inferior
    menu_frame = tk.Frame(root, bg="#ecf0f1", bd=5, relief="ridge")
    menu_frame.place(relx=0.5, rely=0.85, anchor="center")

    botones = [
    ("🧮 Solucion de sistemas", lambda: abrir_submenu_sistema(root)),
    ("🧮 Metodos de Raiz", lambda: abrir_submenu_raiz(root)),
    ("📐 Interpolación", lambda: abrir_submenu_interpolacion(root)),
    ("📏 Integración Numérica", lambda: abrir_submenu_integracion(root)),
    ("🚪 Salir", salir)
]


    def on_enter(e):
        e.widget["background"] = "#3498db"
        e.widget["fg"] = "white"

    def on_leave(e):
        e.widget["background"] = "#ecf0f1"
        e.widget["fg"] = "black"

    for i, (texto, comando) in enumerate(botones):
        fila = i // 2
        columna = i % 2
        btn = tk.Button(
            menu_frame,
            text=texto,
            font=("Arial", 11, "bold"),
            width=25,
            height=2,
            bg="#ecf0f1",
            fg="black",
            relief="raised",
            bd=2,
            command=comando
        )
        btn.grid(row=fila, column=columna, padx=15, pady=10)
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    root.mainloop()

    if cap.isOpened():
        cap.release()


# ==================================================
# PANTALLA DE BIENVENIDA
# ==================================================
def pantalla_bienvenida():
    """Pantalla inicial con logo."""
    bienvenida = tk.Tk()
    bienvenida.title("Bienvenida")
    bienvenida.geometry("1000x600")
    bienvenida.resizable(False, False)
    bienvenida.config(bg="black")

    try:
        logo = Image.open("op1.png")
        logo = logo.resize((1000, 250), Image.Resampling.LANCZOS)
        logo_tk = ImageTk.PhotoImage(logo)
        logo_label = tk.Label(bienvenida, image=logo_tk, bg="black")
        logo_label.image = logo_tk
        logo_label.pack(expand=True)
    except Exception as e:
        print(f"⚠️ Error al cargar el logo: {e}")
        bienvenida.destroy()
        mostrar_interfaz()
        return

    bienvenida.after(2500, lambda: [logo_label.destroy(), mostrar_mensaje_bienvenida(bienvenida)])
    bienvenida.mainloop()


def mostrar_mensaje_bienvenida(bienvenida):
    """Mensaje antes de abrir el menú principal."""
    mensaje = tk.Label(
        bienvenida,
        text="Proyecto Final - Métodos Numéricos\nGrupo 8 - UNJBG/FAIN-ESIS",
        font=("Arial", 20, "bold"),
        fg="white", bg="black",
        padx=20, pady=20
    )
    mensaje.pack(expand=True)
    bienvenida.after(2500, lambda: [bienvenida.destroy(), mostrar_interfaz()])


# ==================================================
# EJECUCIÓN
# ==================================================
if __name__ == "__main__":
    pantalla_bienvenida()
