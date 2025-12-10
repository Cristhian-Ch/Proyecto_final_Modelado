# taylor2_app.py
# Interfaz completa para el Método de Taylor 2º orden (modo ejemplo T2 y modo manual)
# Opción B: el usuario escribe expresiones sin "math." (sin, cos, exp, etc. disponibles)
# No imprime nada en consola; todo se muestra en la interfaz.

import tkinter as tk
from tkinter import ttk, messagebox
import math
import pandas as pd
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use("TkAgg")

# -----------------------
# Funciones matemáticas seguras expuestas (sin escribir math.)
# -----------------------
SAFE_NAMES = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "log": math.log,      # natural log
    "log10": math.log10,
    "sqrt": math.sqrt,
    "pi": math.pi,
    "e": math.e,
    "pow": pow,
    "abs": abs
}


# -----------------------
# Método de Taylor 2º orden
# -----------------------
def taylor2_method(f, ypp_func, x0, y0, h, steps):
    xs = [x0]
    ys = [y0]
    x, y = x0, y0

    for _ in range(steps):
        yprime = f(x, y)
        ypp = ypp_func(x, y, yprime)
        y = y + h * yprime + (h ** 2 / 2.0) * ypp
        x = x + h
        xs.append(round(x, 12))
        ys.append(y)

    return xs, ys


# -----------------------
# Crear función f(x,y) a partir de string (sin "math.")
# Se usa eval con locals controlados que incluyen x,y y SAFE_NAMES
# -----------------------
def crear_funcion_xy(expr):
    expr = expr.strip()
    def f(x, y):
        local = {"x": x, "y": y}
        local.update(SAFE_NAMES)
        try:
            return eval(expr, {"__builtins__": None}, local)
        except Exception as e:
            raise ValueError(f"Error evaluando f(x,y): {e}")
    return f


def crear_funcion_ypp(expr):
    expr = expr.strip()
    def ypp(x, y, yp):
        local = {"x": x, "y": y, "yp": yp}
        local.update(SAFE_NAMES)
        try:
            return eval(expr, {"__builtins__": None}, local)
        except Exception as e:
            raise ValueError(f"Error evaluando y''(x,y,yp): {e}")
    return ypp


# -----------------------
# Clase principal de la app (idéntica en UX a TrapecioApp)
# -----------------------
class Taylor2App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Método de Taylor 2º Orden - Interfaz")
        self.geometry("1000x640")
        self.resizable(False, False)
        self.config(bg="#f5f6fa")

        # Título
        tk.Label(self, text="MÉTODO DE TAYLOR 2° ORDEN",
                 font=("Arial", 16, "bold"), bg="#f5f6fa").pack(pady=8)

        # Frame principal (izquierda inputs - derecha gráfica)
        main_frame = tk.Frame(self, bg="#f5f6fa")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        left = tk.Frame(main_frame, bg="#f5f6fa")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        right = tk.Frame(main_frame, bg="#f5f6fa")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # -----------------------
        # Panel izquierdo: entradas y botones
        # -----------------------
        frame_inputs = tk.Frame(left, bg="#f5f6fa")
        frame_inputs.pack(fill=tk.X)

        # Función y' = f(x,y)
        tk.Label(frame_inputs, text="y' = f(x,y):", font=("Arial", 11), bg="#f5f6fa").grid(row=0, column=0, sticky="w")
        self.entry_f = tk.Entry(frame_inputs, width=28)
        self.entry_f.grid(row=0, column=1, pady=4)
        self.entry_f.insert(0, "x + y")  # placeholder

        # y''(x,y,yp)
        tk.Label(frame_inputs, text="y'' = y''(x,y,yp):", font=("Arial", 11), bg="#f5f6fa").grid(row=1, column=0, sticky="w")
        self.entry_ypp = tk.Entry(frame_inputs, width=28)
        self.entry_ypp.grid(row=1, column=1, pady=4)
        self.entry_ypp.insert(0, "1 + x + y")  # placeholder

        # x0, y0, h, steps
        tk.Label(frame_inputs, text="x0:", font=("Arial", 11), bg="#f5f6fa").grid(row=2, column=0, sticky="w")
        self.entry_x0 = tk.Entry(frame_inputs, width=12)
        self.entry_x0.grid(row=2, column=1, sticky="w", pady=4, padx=(0, 60))
        self.entry_x0.insert(0, "0.0")

        tk.Label(frame_inputs, text="y0:", font=("Arial", 11), bg="#f5f6fa").grid(row=3, column=0, sticky="w")
        self.entry_y0 = tk.Entry(frame_inputs, width=12)
        self.entry_y0.grid(row=3, column=1, sticky="w", pady=4, padx=(0, 60))
        self.entry_y0.insert(0, "1.0")

        tk.Label(frame_inputs, text="h (paso):", font=("Arial", 11), bg="#f5f6fa").grid(row=4, column=0, sticky="w")
        self.entry_h = tk.Entry(frame_inputs, width=12)
        self.entry_h.grid(row=4, column=1, sticky="w", pady=4, padx=(0, 60))
        self.entry_h.insert(0, "0.1")

        tk.Label(frame_inputs, text="Pasos:", font=("Arial", 11), bg="#f5f6fa").grid(row=5, column=0, sticky="w")
        self.entry_steps = tk.Entry(frame_inputs, width=12)
        self.entry_steps.grid(row=5, column=1, sticky="w", pady=4, padx=(0, 60))
        self.entry_steps.insert(0, "2")

        # Botones
        btn_frame = tk.Frame(left, bg="#f5f6fa")
        btn_frame.pack(pady=8)

        tk.Button(btn_frame, text="Calcular (Manual)",
                  bg="#2ecc71", fg="white", width=18, font=("Arial", 10, "bold"),
                  command=self.on_calcular_manual).grid(row=0, column=0, padx=6, pady=4)

        tk.Button(btn_frame, text="Ejemplo T2 (Cargar y Ejecutar)",
                  bg="#3498db", fg="white", width=28, font=("Arial", 10, "bold"),
                  command=self.on_ejemplo_t2).grid(row=0, column=1, padx=6, pady=4)

        # Resultado label
        self.label_result = tk.Label(left, text="Resultado:\n—", font=("Arial", 11, "bold"),
                                     fg="darkgreen", bg="#f5f6fa", justify="left")
        self.label_result.pack(pady=6)

        # Tabla (Treeview)
        frame_tabla = tk.Frame(left, bg="#f5f6fa")
        frame_tabla.pack(fill=tk.BOTH, expand=True, pady=6)

        tk.Label(frame_tabla, text="Tabla de puntos (x, y_taylor2)",
                 font=("Arial", 10, "bold"), bg="#f5f6fa").pack(anchor="w")

        self.tree = ttk.Treeview(frame_tabla, columns=("i", "x", "y"), show="headings", height=12)
        self.tree.heading("i", text="i")
        self.tree.heading("x", text="x")
        self.tree.heading("y", text="y_taylor2")

        self.tree.column("i", width=30, anchor="center")
        self.tree.column("x", width=100, anchor="center")
        self.tree.column("y", width=120, anchor="center")

        self.tree.pack(fill=tk.BOTH, expand=True)

        # -----------------------
        # Panel derecho: gráfica
        # -----------------------
        frame_plot = tk.Frame(right, bg="#f5f6fa")
        frame_plot.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Inicial: cargar ejemplo pero no ejecutar automáticamente until user presses button.
        # We set fields to the example T2 by default so user sees inputs.
        self._load_example_T2_fields_only()

    # -----------------------
    # Cargar valores del ejemplo T2 en campos (sin ejecutar)
    # -----------------------
    def _load_example_T2_fields_only(self):
        # T2: y' = x + y, y'' = 1 + x + y, y(0)=1, h=0.1, 2 pasos
        self.entry_f.delete(0, tk.END)
        self.entry_f.insert(0, "x + y")

        self.entry_ypp.delete(0, tk.END)
        self.entry_ypp.insert(0, "1 + x + y")

        self.entry_x0.delete(0, tk.END)
        self.entry_x0.insert(0, "0.0")

        self.entry_y0.delete(0, tk.END)
        self.entry_y0.insert(0, "1.0")

        self.entry_h.delete(0, tk.END)
        self.entry_h.insert(0, "0.1")

        self.entry_steps.delete(0, tk.END)
        self.entry_steps.insert(0, "2")

    # -----------------------
    # Ejecuta el ejemplo T2: carga campos y ejecuta cálculo
    # -----------------------
    def on_ejemplo_t2(self):
        self._load_example_T2_fields_only()
        self.on_calcular_manual()  # ejecutar con los campos cargados

    # -----------------------
    # Calcular usando entradas manuales (f, y'', x0, y0, h, steps)
    # -----------------------
    def on_calcular_manual(self):
        # leer y validar entradas
        expr_f = self.entry_f.get().strip()
        expr_ypp = self.entry_ypp.get().strip()

        if expr_f == "" or expr_ypp == "":
            messagebox.showerror("Error", "Debe ingresar f(x,y) y y''(x,y,yp).")
            return

        try:
            x0 = float(self.entry_x0.get())
            y0 = float(self.entry_y0.get())
            h = float(self.entry_h.get())
            steps = int(self.entry_steps.get())
            if steps < 0:
                raise ValueError("Pasos debe ser entero no negativo.")
        except Exception as e:
            messagebox.showerror("Error", f"Valores numéricos inválidos:\n{e}")
            return

        # crear funciones seguras
        try:
            f = crear_funcion_xy(expr_f)
            ypp = crear_funcion_ypp(expr_ypp)
        except Exception as e:
            messagebox.showerror("Error", f"Error al crear funciones:\n{e}")
            return

        # ejecutar método
        try:
            xs, ys = taylor2_method(f, ypp, x0, y0, h, steps)
        except Exception as e:
            messagebox.showerror("Error durante la ejecución del método:\n" + str(e))
            return

        # mostrar resultado final
        self.label_result.config(text=f"Último valor: x={xs[-1]:.8f}, y={ys[-1]:.8f}")

        # actualizar tabla
        for row in self.tree.get_children():
            self.tree.delete(row)

        for i, (xx, yy) in enumerate(zip(xs, ys)):
            self.tree.insert("", "end", values=(i, f"{xx:.8f}", f"{yy:.8f}"))

        # actualizar gráfica (línea con marcadores)
        self._actualizar_grafica(xs, ys)

    # -----------------------
    # Graficar
    # -----------------------
    def _actualizar_grafica(self, xs, ys):
        self.ax.clear()
        self.ax.set_title("Aproximación por Taylor 2° Orden")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True)

        # línea y puntos
        self.ax.plot(xs, ys, marker="o", linestyle="--", label="Taylor 2°")
        # conectar puntos con una línea suave por interpolación simple si hay suficientes puntos
        if len(xs) > 2:
            try:
                # interpolación lineal fina para suavizar visualmente
                import numpy as _np
                t = _np.linspace(xs[0], xs[-1], 300)
                # simple piecewise linear interpolation from discrete points
                y_interp = _np.interp(t, xs, ys)
                self.ax.plot(t, y_interp, linewidth=1.0, alpha=0.6)
            except Exception:
                pass

        self.ax.legend()
        self.canvas.draw()


# -----------------------
# Ejecutar la aplicación
# -----------------------
if __name__ == "__main__":
    app = Taylor2App()
    app.mainloop()
