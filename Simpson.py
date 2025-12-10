# simpson13.py
# Interfaz con Simpson 1/3 Simple y Simpson 1/3 Compuesto ("doble")
# Incluye: entrada de datos, botón de ejemplo, tabla de puntos y gráfica.

import tkinter as tk
from tkinter import ttk, messagebox
import math
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Usar backend de Tkinter
matplotlib.use("TkAgg")


def crear_funcion(funcion_str):
    """
    Convierte un string en una función f(x) usando eval de forma limitada.
    Ejemplo: 'x**2 + math.sin(x)'.
    """
    def f(x):
        return eval(funcion_str, {"__builtins__": {}, "math": math, "x": x})
    return f


class Simpson13App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Método de Simpson 1/3 - Simple y Compuesto")
        self.geometry("1000x600")

        # =========================
        #  PANEL IZQUIERDO: ENTRADAS
        # =========================
        frame_inputs = tk.Frame(self, padx=10, pady=10)
        frame_inputs.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            frame_inputs,
            text="MÉTODOS DE SIMPSON 1/3",
            font=("Arial", 14, "bold")
        ).pack(pady=5)

        # Selector de método
        tk.Label(frame_inputs, text="Seleccione el método:",
                 font=("Arial", 10, "bold")).pack(anchor="w")
        self.metodo = ttk.Combobox(
            frame_inputs,
            values=["Simpson 1/3 Simple", "Simpson 1/3 Compuesto (doble)"],
            state="readonly",
            width=28
        )
        self.metodo.current(0)
        self.metodo.pack(pady=3)

        # f(x)
        tk.Label(frame_inputs, text="f(x):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.entry_func = tk.Entry(frame_inputs, width=30)
        self.entry_func.insert(0, "x**2")
        self.entry_func.pack(pady=3)

        # a
        tk.Label(frame_inputs, text="Límite inferior a:",
                 font=("Arial", 10, "bold")).pack(anchor="w")
        self.entry_a = tk.Entry(frame_inputs, width=15)
        self.entry_a.insert(0, "0")
        self.entry_a.pack(pady=3)

        # b
        tk.Label(frame_inputs, text="Límite superior b:",
                 font=("Arial", 10, "bold")).pack(anchor="w")
        self.entry_b = tk.Entry(frame_inputs, width=15)
        self.entry_b.insert(0, "2")
        self.entry_b.pack(pady=3)

        # n (solo para compuesto)
        tk.Label(frame_inputs,
                 text="Número de subintervalos n (SOLO para compuesto, n PAR):",
                 font=("Arial", 9, "bold"),
                 wraplength=250,
                 justify="left").pack(anchor="w", pady=(5, 0))
        self.entry_n = tk.Entry(frame_inputs, width=15)
        self.entry_n.insert(0, "4")
        self.entry_n.pack(pady=3)

        # Botones
        btn_frame = tk.Frame(frame_inputs)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame,
            text="Calcular",
            width=15,
            bg="#3498db",
            fg="white",
            command=self.calcular
        ).grid(row=0, column=0, padx=5)

        tk.Button(
            btn_frame,
            text="Ejemplo",
            width=15,
            bg="#2ecc71",
            fg="white",
            command=self.cargar_ejemplo
        ).grid(row=0, column=1, padx=5)

        # Resultado
        self.label_result = tk.Label(
            frame_inputs,
            text="Resultado:\n—",
            font=("Arial", 11, "bold"),
            fg="darkgreen",
            justify="left"
        )
        self.label_result.pack(pady=10)

        # =========================
        #  TABLA DE PUNTOS
        # =========================
        frame_tabla = tk.Frame(frame_inputs)
        frame_tabla.pack(fill=tk.BOTH, expand=True, pady=10)

        tk.Label(frame_tabla, text="Tabla de puntos (xᵢ, f(xᵢ))",
                 font=("Arial", 10, "bold")).pack()

        self.tree = ttk.Treeview(
            frame_tabla,
            columns=("i", "x", "fx"),
            show="headings",
            height=10
        )
        self.tree.heading("i", text="i")
        self.tree.heading("x", text="xᵢ")
        self.tree.heading("fx", text="f(xᵢ)")

        self.tree.column("i", width=30, anchor="center")
        self.tree.column("x", width=90, anchor="center")
        self.tree.column("fx", width=120, anchor="center")

        self.tree.pack(fill=tk.BOTH, expand=True)

        # =========================
        #  PANEL DERECHO: GRÁFICA
        # =========================
        frame_plot = tk.Frame(self, padx=10, pady=10)
        frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ==========================================================
    #  BOTÓN EJEMPLO
    # ==========================================================
    def cargar_ejemplo(self):
        metodo = self.metodo.get()

        if metodo == "Simpson 1/3 Simple":
            self.entry_func.delete(0, tk.END)
            self.entry_func.insert(0, "math.sin(x)")
            self.entry_a.delete(0, tk.END)
            self.entry_a.insert(0, "0")
            self.entry_b.delete(0, tk.END)
            self.entry_b.insert(0, "3.1416")
            # n no se usa, pero ponemos algo por defecto
            self.entry_n.delete(0, tk.END)
            self.entry_n.insert(0, "2")

        elif metodo == "Simpson 1/3 Compuesto (doble)":
            self.entry_func.delete(0, tk.END)
            self.entry_func.insert(0, "math.cos(x)")
            self.entry_a.delete(0, tk.END)
            self.entry_a.insert(0, "0")
            self.entry_b.delete(0, tk.END)
            self.entry_b.insert(0, "3.1416")
            self.entry_n.delete(0, tk.END)
            self.entry_n.insert(0, "8")  # n par

        self.calcular()

    # ==========================================================
    #  BOTÓN CALCULAR
    # ==========================================================
    def calcular(self):
        try:
            funcion_str = self.entry_func.get().strip()
            if not funcion_str:
                raise ValueError("Debe ingresar una función f(x).")

            a = float(self.entry_a.get())
            b = float(self.entry_b.get())

            if a == b:
                raise ValueError("a y b no pueden ser iguales.")

            f = crear_funcion(funcion_str)
            metodo = self.metodo.get()

            if metodo == "Simpson 1/3 Simple":
                self.calcular_simpson_simple(f, a, b)
            elif metodo == "Simpson 1/3 Compuesto (doble)":
                self.calcular_simpson_compuesto(f, a, b)
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un problema:\n{e}")

    # ==========================================================
    #  SIMPSON 1/3 SIMPLE
    # ==========================================================
    def calcular_simpson_simple(self, f, a, b):
        m = (a + b) / 2.0
        fa = f(a)
        fm = f(m)
        fb = f(b)

        integral = (b - a) / 6.0 * (fa + 4 * fm + fb)

        self.label_result.config(
            text=(
                "Simpson 1/3 SIMPLE\n"
                f"∫[{a}, {b}] f(x) dx ≈ {integral:.6f}\n"
                f"f(a) = {fa:.6f}, f(m) = {fm:.6f}, f(b) = {fb:.6f}"
            )
        )

        # Tabla
        for fila in self.tree.get_children():
            self.tree.delete(fila)

        datos = [(0, a, fa), (1, m, fm), (2, b, fb)]
        for i, x_i, y_i in datos:
            self.tree.insert("", "end", values=(i, f"{x_i:.6f}", f"{y_i:.6f}"))

        # Gráfica
        self.actualizar_grafica(f, a, b, [a, m, b], [fa, fm, fb])

    # ==========================================================
    #  SIMPSON 1/3 COMPUESTO ("DOBLE")
    # ==========================================================
    def calcular_simpson_compuesto(self, f, a, b):
        n = int(self.entry_n.get())
        if n <= 0 or n % 2 != 0:
            raise ValueError("Para Simpson compuesto, n debe ser PAR y positivo (2, 4, 6, ...).")

        h = (b - a) / n
        xs = [a + i * h for i in range(n + 1)]
        ys = [f(x) for x in xs]

        suma_pares = 0.0
        suma_impares = 0.0
        for i in range(1, n):
            if i % 2 == 0:
                suma_pares += ys[i]
            else:
                suma_impares += ys[i]

        integral = (h / 3.0) * (ys[0] + 2 * suma_pares + 4 * suma_impares + ys[-1])

        self.label_result.config(
            text=(
                "Simpson 1/3 COMPUESTO (DOBLE)\n"
                f"∫[{a}, {b}] f(x) dx ≈ {integral:.6f}\n"
                f"h = {h:.6f}, n = {n}\n"
                f"Suma impares = {suma_impares:.6f}\n"
                f"Suma pares   = {suma_pares:.6f}"
            )
        )

        # Tabla
        for fila in self.tree.get_children():
            self.tree.delete(fila)

        for i, (x_i, y_i) in enumerate(zip(xs, ys)):
            self.tree.insert("", "end", values=(i, f"{x_i:.6f}", f"{y_i:.6f}"))

        # Gráfica
        self.actualizar_grafica_compuesto(f, a, b, xs)

    # ==========================================================
    #  GRÁFICA SIMPLE / GENERAL
    # ==========================================================
    def actualizar_grafica(self, f, a, b, xs_pts, ys_pts):
        self.ax.clear()
        self.ax.set_title("Aproximación por Simpson 1/3")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True)

        # Curva suave
        puntos = 300
        xs = [a + i * (b - a) / (puntos - 1) for i in range(puntos)]
        ys = [f(x) for x in xs]
        self.ax.plot(xs, ys, label="f(x)")

        # Puntos Simpson
        self.ax.plot(xs_pts, ys_pts, "o--", label="Puntos Simpson")

        # Área sombreada entre a y b
        self.ax.fill_between(xs, ys, 0,
                             where=[(x >= a and x <= b) for x in xs],
                             alpha=0.3)

        self.ax.legend()
        self.canvas.draw()

    # ==========================================================
    #  GRÁFICA COMPUESTO
    # ==========================================================
    def actualizar_grafica_compuesto(self, f, a, b, xs_puntos):
        self.ax.clear()
        self.ax.set_title("Simpson 1/3 Compuesto")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True)

        # Curva suave
        puntos = 300
        xs = [a + i * (b - a) / (puntos - 1) for i in range(puntos)]
        ys = [f(x) for x in xs]
        self.ax.plot(xs, ys, label="f(x)")

        # Tramos de Simpson (cada 2 subintervalos)
        for i in range(0, len(xs_puntos) - 1, 2):
            x0 = xs_puntos[i]
            x1 = xs_puntos[i + 1]
            x2 = xs_puntos[i + 2]

            y0 = f(x0)
            y1 = f(x1)
            y2 = f(x2)

            xs_local = [x0, x1, x2]
            ys_local = [y0, y1, y2]

            self.ax.plot(xs_local, ys_local, "o--", alpha=0.8)

            xs_somb = [x for x in xs if x >= x0 and x <= x2]
            ys_somb = [f(x) for x in xs_somb]
            self.ax.fill_between(xs_somb, ys_somb, 0, alpha=0.15)

        self.ax.legend()
        self.canvas.draw()


if __name__ == "__main__":
    app = Simpson13App()
    app.mainloop()
