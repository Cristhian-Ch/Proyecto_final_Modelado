# euler_app.py
# Método de Euler con interfaz gráfica:
# - Entrada manual y ejemplo
# - Tabla con x, y_euler, y_exact, error
# - Gráfica integrada

import math
import matplotlib
from matplotlib.figure import Figure
import pandas as pd

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    matplotlib.use("TkAgg")
except ImportError:
    tk = None
    ttk = None
    messagebox = None
    FigureCanvasTkAgg = None
    # En entorno sin Tk, no llames a la GUI de este archivo.



# --------------------------
# FUNCIÓN SEGURA PARA f(x, y)
# --------------------------
def crear_funcion_doble(funcion_str):
    """
    Crea una función f(x,y) desde un string como: "x + y" o "x*y + math.sin(x)".
    """
    def f(x, y):
        return eval(funcion_str, {"__builtins__": {}, "math": math, "x": x, "y": y})
    return f


# --------------------------
# MÉTODO DE EULER
# --------------------------
def euler_method(f, x0, y0, h, steps):
    xs = [x0]
    ys = [y0]
    x, y = x0, y0

    for _ in range(steps):
        y = y + h * f(x, y)
        x = x + h

        xs.append(round(x, 12))
        ys.append(y)

    return xs, ys


# ===========================
#     INTERFAZ GRÁFICA
# ===========================
class EulerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Método de Euler (EDO)")
        self.geometry("1100x620")

        # Panel izquierdo
        frame_left = tk.Frame(self, padx=10, pady=10)
        frame_left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(frame_left, text="MÉTODO DE EULER", font=("Arial", 15, "bold")).pack()

        # ------------ CAMPOS DE ENTRADA -------------
        tk.Label(frame_left, text="f(x, y):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.entry_f = tk.Entry(frame_left, width=35)
        self.entry_f.insert(0, "y")  # predeterminado
        self.entry_f.pack(pady=3)

        tk.Label(frame_left, text="x0:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.entry_x0 = tk.Entry(frame_left, width=15)
        self.entry_x0.insert(0, "0")
        self.entry_x0.pack(pady=3)

        tk.Label(frame_left, text="y0:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.entry_y0 = tk.Entry(frame_left, width=15)
        self.entry_y0.insert(0, "1")
        self.entry_y0.pack(pady=3)

        tk.Label(frame_left, text="h (paso):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.entry_h = tk.Entry(frame_left, width=15)
        self.entry_h.insert(0, "0.1")
        self.entry_h.pack(pady=3)

        tk.Label(frame_left, text="Número de pasos:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.entry_steps = tk.Entry(frame_left, width=15)
        self.entry_steps.insert(0, "2")
        self.entry_steps.pack(pady=3)

        # Botones
        frame_buttons = tk.Frame(frame_left)
        frame_buttons.pack(pady=8)

        tk.Button(frame_buttons, text="Calcular", width=15, bg="#3498db", fg="white",
                  command=self.calcular).grid(row=0, column=0, padx=5)

        tk.Button(frame_buttons, text="Ejemplo", width=15, bg="#2ecc71", fg="white",
                  command=self.cargar_ejemplo).grid(row=0, column=1, padx=5)

        # Resultado
        self.label_result = tk.Label(
            frame_left,
            text="Resultado:\n—",
            font=("Arial", 11, "bold"),
            fg="darkgreen"
        )
        self.label_result.pack(pady=10)

        # ----------------- TABLA -----------------
        tk.Label(frame_left, text="Tabla de valores", font=("Arial", 10, "bold")).pack()

        self.tree = ttk.Treeview(
            frame_left, columns=("x", "y_e", "y_ex", "err"), show="headings", height=12
        )

        for col, txt, w in [
            ("x", "x", 80),
            ("y_e", "y_euler", 90),
            ("y_ex", "y_exact", 90),
            ("err", "error", 90),
        ]:
            self.tree.heading(col, text=txt)
            self.tree.column(col, width=w, anchor="center")

        self.tree.pack()

        # ----------------- GRÁFICA -----------------
        frame_plot = tk.Frame(self, padx=10, pady=10)
        frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, frame_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------------- EJEMPLO PREDETERMINADO ----------------
    def cargar_ejemplo(self):
        self.entry_f.delete(0, tk.END)
        self.entry_f.insert(0, "y")

        self.entry_x0.delete(0, tk.END)
        self.entry_x0.insert(0, "0")

        self.entry_y0.delete(0, tk.END)
        self.entry_y0.insert(0, "1")

        self.entry_h.delete(0, tk.END)
        self.entry_h.insert(0, "0.1")

        self.entry_steps.delete(0, tk.END)
        self.entry_steps.insert(0, "2")

        self.calcular()

    # ------------------ BOTÓN CALCULAR ------------------
    def calcular(self):
        try:
            f_str = self.entry_f.get().strip()
            f = crear_funcion_doble(f_str)

            x0 = float(self.entry_x0.get())
            y0 = float(self.entry_y0.get())
            h = float(self.entry_h.get())
            steps = int(self.entry_steps.get())

            xs, ys = euler_method(f, x0, y0, h, steps)

            # Exacta (siempre exp(x) para el ejemplo)
            y_exact = [math.exp(x) for x in xs]

            df = pd.DataFrame({
                "x": xs,
                "y_euler": ys,
                "y_exact": y_exact,
                "error": [y_exact[i] - ys[i] for i in range(len(xs))]
            })

            # Mostrar en etiqueta
            self.label_result.config(text=f"Resultado:\nEuler con {steps} pasos, h={h}")

            # Limpiar tabla
            for row in self.tree.get_children():
                self.tree.delete(row)

            # Rellenar tabla
            for _, row in df.iterrows():
                self.tree.insert("", "end", values=(
                    f"{row['x']:.6f}",
                    f"{row['y_euler']:.6f}",
                    f"{row['y_exact']:.6f}",
                    f"{row['error']:.6f}",
                ))

            # Graficar
            self.graficar(xs, ys, y_exact)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ----------------- GRÁFICA -----------------
    def graficar(self, xs, ys, y_exact):
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_title("Método de Euler")

        self.ax.plot(xs, y_exact, label="y exacta", linewidth=2)
        self.ax.plot(xs, ys, "o--", label="Euler numérico")

        self.ax.legend()
        self.canvas.draw()


# ----------------- EJECUCIÓN -----------------
if __name__ == "__main__":
    app = EulerApp()
    app.mainloop()
