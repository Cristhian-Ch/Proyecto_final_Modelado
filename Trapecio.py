# trapecio.py
# Método del Trapecio compuesto con interfaz gráfica,
# tabla de puntos y gráfica de la función.

import math
import matplotlib
from matplotlib.figure import Figure

# Intentar importar Tkinter y el backend de Matplotlib para Tk.
# Si no están disponibles (por ejemplo en Streamlit Cloud), no se detiene el programa.
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    tk = None
    ttk = None
    messagebox = None
    FigureCanvasTkAgg = None



# ---------- Función auxiliar para crear f(x) segura ----------
def crear_funcion(funcion_str):
    """
    Recibe un string como 'x**2 + 3*x' y devuelve una función f(x).
    Solo expone 'math' y 'x' dentro de eval para mayor seguridad.
    """
    def f(x):
        return eval(funcion_str, {"__builtins__": {}, "math": math, "x": x})
    return f


class TrapecioApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Método del Trapecio (Integración Numérica)")
        self.geometry("1000x600")

        # =========================
        #  PANEL IZQUIERDO: ENTRADAS
        # =========================
        frame_inputs = tk.Frame(self, padx=10, pady=10)
        frame_inputs.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            frame_inputs,
            text="MÉTODO DEL TRAPECIO COMPUESTO",
            font=("Arial", 14, "bold")
        ).pack(pady=5)

        # f(x)
        tk.Label(frame_inputs, text="f(x):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.entry_func = tk.Entry(frame_inputs, width=30)
        self.entry_func.insert(0, "x**2")  # Ejemplo por defecto
        self.entry_func.pack(pady=3)

        # a
        tk.Label(frame_inputs, text="Límite inferior a:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.entry_a = tk.Entry(frame_inputs, width=15)
        self.entry_a.insert(0, "0")
        self.entry_a.pack(pady=3)

        # b
        tk.Label(frame_inputs, text="Límite superior b:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.entry_b = tk.Entry(frame_inputs, width=15)
        self.entry_b.insert(0, "2")
        self.entry_b.pack(pady=3)

        # n
        tk.Label(frame_inputs, text="Número de subintervalos n:", font=("Arial", 10, "bold")).pack(anchor="w")
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

        # Resultado numérico
        self.label_result = tk.Label(
            frame_inputs,
            text="Resultado:\n—",
            font=("Arial", 11, "bold"),
            fg="darkgreen",
            justify="left"
        )
        self.label_result.pack(pady=10)

        # =========================
        #  PANEL INFERIOR IZQUIERDO: TABLA
        # =========================
        frame_tabla = tk.Frame(frame_inputs)
        frame_tabla.pack(fill=tk.BOTH, expand=True, pady=10)

        tk.Label(frame_tabla, text="Tabla de puntos (xᵢ, f(xᵢ))",
                 font=("Arial", 10, "bold")).pack()

        self.tree = ttk.Treeview(frame_tabla, columns=("i", "x", "fx"), show="headings", height=10)
        self.tree.heading("i", text="i")
        self.tree.heading("x", text="xᵢ")
        self.tree.heading("fx", text="f(xᵢ)")

        self.tree.column("i", width=30, anchor="center")
        self.tree.column("x", width=90, anchor="center")
        self.tree.column("fx", width=90, anchor="center")

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

    # ---------- Botón EJEMPLO ----------
    def cargar_ejemplo(self):
        """
        Coloca un ejemplo en los campos y luego calcula.
        """
        self.entry_func.delete(0, tk.END)
        self.entry_func.insert(0, "math.sin(x) + x**2")

        self.entry_a.delete(0, tk.END)
        self.entry_a.insert(0, "0")

        self.entry_b.delete(0, tk.END)
        self.entry_b.insert(0, "3.1416")

        self.entry_n.delete(0, tk.END)
        self.entry_n.insert(0, "6")

        self.calcular()

    # ---------- Botón CALCULAR ----------
    def calcular(self):
        try:
            funcion_str = self.entry_func.get().strip()
            if not funcion_str:
                raise ValueError("Debe ingresar una función f(x).")

            a = float(self.entry_a.get())
            b = float(self.entry_b.get())
            n = int(self.entry_n.get())

            if n <= 0:
                raise ValueError("n debe ser un entero positivo.")

            if a == b:
                raise ValueError("a y b no pueden ser iguales.")

            f = crear_funcion(funcion_str)

            # Cálculo por Trapecio compuesto
            h = (b - a) / n

            xs = [a + i * h for i in range(n + 1)]
            ys = [f(x) for x in xs]

            suma_extremos = ys[0] + ys[-1]
            suma_interior = sum(ys[1:-1])

            integral = (h / 2.0) * (suma_extremos + 2 * suma_interior)

            # Mostrar resultado
            self.label_result.config(
                text=(
                    f"Resultado:\n"
                    f"∫[{a}, {b}] f(x) dx ≈ {integral:.6f}\n"
                    f"h = {h:.6f}, n = {n}"
                )
            )

            # Actualizar tabla
            for fila in self.tree.get_children():
                self.tree.delete(fila)

            for i, (x_i, y_i) in enumerate(zip(xs, ys)):
                self.tree.insert("", "end", values=(i, f"{x_i:.6f}", f"{y_i:.6f}"))

            # Actualizar gráfica
            self.actualizar_grafica(f, a, b, xs)

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un problema:\n{e}")

    # ---------- Gráfica ----------
    def actualizar_grafica(self, f, a, b, xs_puntos):
        # Limpiar
        self.ax.clear()
        self.ax.set_title("Método del Trapecio")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True)

        # Curva "suave" de la función
        puntos = 300
        xs = [a + i * (b - a) / (puntos - 1) for i in range(puntos)]
        ys = [f(x) for x in xs]
        self.ax.plot(xs, ys, label="f(x)")

        # Trapecios (con relleno)
        for i in range(len(xs_puntos) - 1):
            x0 = xs_puntos[i]
            x1 = xs_puntos[i + 1]
            y0 = f(x0)
            y1 = f(x1)
            self.ax.fill_between([x0, x1], [y0, y1], 0, alpha=0.3)

        self.ax.legend()
        self.canvas.draw()


if __name__ == "__main__":
    app = TrapecioApp()
    app.mainloop()
