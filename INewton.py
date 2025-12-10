import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

# ==================================================
# INTERPOLACIÓN DE NEWTON (Inewton)
# ==================================================
"""
Implementación gráfica del Método de Interpolación de Newton
con diferencias divididas. Interfaz coherente con los demás módulos.
"""

def divided_differences(xs, ys):
    n = len(xs)
    table = np.zeros((n, n))
    table[:, 0] = ys
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i+1, j-1] - table[i, j-1]) / (xs[i+j] - xs[i])
    coef = table[0, :]
    return coef, table

def newton_eval(coef, xs, x):
    n = len(coef)
    result = coef[n-1]
    for k in range(n-2, -1, -1):
        result = result * (x - xs[k]) + coef[k]
    return result

def newton_iterations(xs, ys, x_eval):
    coef, table = divided_differences(xs, ys)
    rows = []
    for i, (xi, yi) in enumerate(zip(xs, ys), start=1):
        rows.append((i, f"{xi:.4f}", f"{yi:.4f}"))
    return rows, coef, table

def graficar_newton(xs, ys, x_eval, resultado, ax=None, canvas=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    x_min, x_max = min(xs) - 1, max(xs) + 1
    x_plot = np.linspace(x_min, x_max, 200)
    coef, _ = divided_differences(xs, ys)
    y_plot = [newton_eval(coef, xs, x) for x in x_plot]
    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='P(x) - Polinomio de Newton')
    ax.plot(xs, ys, 'ro', markersize=8, label='Puntos dados', zorder=5)
    ax.plot(x_eval, resultado, 'gs', markersize=10, label=f'P({x_eval:.4f}) = {resultado:.4f}', zorder=5)
    ax.vlines(x_eval, 0, resultado, colors='green', linestyles='--', alpha=0.5)
    ax.hlines(resultado, x_min, x_eval, colors='green', linestyles='--', alpha=0.5)
    ax.set_xlabel("x", fontsize=12, fontweight='bold')
    ax.set_ylabel("y", fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title("Interpolación de Newton (Inewton)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if canvas is not None:
        canvas.draw()
    else:
        plt.show()
    return ax


class InewtonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📐 Interpolación de Newton - Inewton")
        self.root.geometry("1100x850")
        self.root.configure(bg='#f0f0f0')
        self.xs = np.array([1.0, 2.0, 3.0, 4.0])
        self.ys = np.array([1.0, 4.0, 9.0, 16.0])
        top_frame = tk.Frame(root, bg='#2c3e50', bd=5)
        top_frame.pack(fill='x', padx=10, pady=10)
        tk.Label(top_frame, text="INTERPOLACIÓN DE NEWTON (Inewton)", font=("Arial", 14, "bold"), fg="white", bg='#2c3e50').pack(pady=10)
        input_frame = ttk.LabelFrame(root, text="Datos de Entrada", padding=10)
        input_frame.pack(fill='x', padx=10, pady=5)
        tk.Label(input_frame, text="Valores X (separados por comas):", font=("Arial", 10)).grid(row=0, column=0, sticky='w')
        self.entry_xs = tk.Entry(input_frame, width=50)
        self.entry_xs.insert(0, "1.0,2.0,3.0,4.0")
        self.entry_xs.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(input_frame, text="Valores Y (separados por comas):", font=("Arial", 10)).grid(row=1, column=0, sticky='w')
        self.entry_ys = tk.Entry(input_frame, width=50)
        self.entry_ys.insert(0, "1.0,4.0,9.0,16.0")
        self.entry_ys.grid(row=1, column=1, padx=5, pady=5)
        tk.Label(input_frame, text="Evaluar en x =", font=("Arial", 10)).grid(row=2, column=0, sticky='w')
        self.entry_x_eval = tk.Entry(input_frame, width=50)
        self.entry_x_eval.insert(0, "2.5")
        self.entry_x_eval.grid(row=2, column=1, padx=5, pady=5)
        btn_frame = tk.Frame(root, bg='#f0f0f0')
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="▶ Calcular", font=("Arial", 11, "bold"), bg="#27ae60", fg="white", padx=20, pady=10, command=self.calcular).pack(side='left', padx=5)
        tk.Button(btn_frame, text="🔄 Limpiar", font=("Arial", 11, "bold"), bg="#e74c3c", fg="white", padx=20, pady=10, command=self.limpiar).pack(side='left', padx=5)
        result_frame = ttk.LabelFrame(root, text="Resultado", padding=10)
        result_frame.pack(fill='x', padx=10, pady=5)
        self.label_resultado = tk.Label(result_frame, text="Resultado: Ingrese los datos y presione Calcular", font=("Arial", 12, "bold"), fg="#2c3e50")
        self.label_resultado.pack(pady=10)
        table_frame = ttk.LabelFrame(root, text="Tabla de Diferencias Divididas", padding=10)
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.tree = ttk.Treeview(table_frame, columns=("Punto", "xi", "yi"), height=6, show='headings')
        self.tree.column("#0", width=0, stretch=tk.NO)
        self.tree.column("Punto", anchor=tk.CENTER, width=60)
        self.tree.column("xi", anchor=tk.CENTER, width=100)
        self.tree.column("yi", anchor=tk.CENTER, width=100)
        self.tree.heading("Punto", text="Punto")
        self.tree.heading("xi", text="xi")
        self.tree.heading("yi", text="yi")
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        coef_frame = ttk.LabelFrame(root, text="Coeficientes de Newton (f[x0], f[x0,x1], ...)", padding=10)
        coef_frame.pack(fill='x', padx=10, pady=5)
        self.label_coef = tk.Label(coef_frame, text="Coeficientes: ", font=("Arial", 10), wraplength=1000, justify=tk.LEFT)
        self.label_coef.pack(anchor='w')
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
        self.mostrar_ejemplo()

    def mostrar_ejemplo(self):
        try:
            coef, table = divided_differences(self.xs, self.ys)
            x_eval = 2.5
            resultado = newton_eval(coef, self.xs, x_eval)
            self.label_resultado.config(text=f"P({x_eval}) = {resultado:.6f}")
            for item in self.tree.get_children():
                self.tree.delete(item)
            rows, coef, table = newton_iterations(self.xs, self.ys, x_eval)
            for punto, xi, yi in rows:
                self.tree.insert('', 'end', values=(f"{punto}", xi, yi))
            coef_str = ", ".join([f"{c:.6f}" for c in coef])
            self.label_coef.config(text=f"Coeficientes: [{coef_str}]")
            self.ax.clear()
            graficar_newton(self.xs, self.ys, x_eval, resultado, self.ax, self.canvas)
        except Exception as e:
            messagebox.showerror("Error", f"Error en el cálculo:\n{str(e)}")

    def calcular(self):
        try:
            xs_str = self.entry_xs.get().replace(' ', '')
            ys_str = self.entry_ys.get().replace(' ', '')
            x_eval_str = self.entry_x_eval.get().strip()
            xs = np.array([float(x) for x in xs_str.split(',')])
            ys = np.array([float(y) for y in ys_str.split(',')])
            x_eval = float(x_eval_str)
            if len(xs) != len(ys):
                raise ValueError("Los arrays X e Y deben tener la misma longitud")
            if len(xs) < 2:
                raise ValueError("Se necesitan al menos 2 puntos")
            coef, table = divided_differences(xs, ys)
            resultado = newton_eval(coef, xs, x_eval)
            self.label_resultado.config(text=f"P({x_eval:.6f}) = {resultado:.6f}", fg="#27ae60")
            for item in self.tree.get_children():
                self.tree.delete(item)
            rows, coef, table = newton_iterations(xs, ys, x_eval)
            for punto, xi, yi in rows:
                self.tree.insert('', 'end', values=(f"{punto}", xi, yi))
            coef_str = ", ".join([f"{c:.6f}" for c in coef])
            self.label_coef.config(text=f"Coeficientes: [{coef_str}]")
            self.ax.clear()
            graficar_newton(xs, ys, x_eval, resultado, self.ax, self.canvas)
        except ValueError as e:
            messagebox.showerror("Error de entrada", f"Error: {str(e)}")
            self.label_resultado.config(text="Error en los datos de entrada", fg="#e74c3c")
        except Exception as e:
            messagebox.showerror("Error", f"Error en el cálculo:\n{str(e)}")
            self.label_resultado.config(text="Error en el cálculo", fg="#e74c3c")

    def limpiar(self):
        self.entry_xs.delete(0, tk.END)
        self.entry_xs.insert(0, "1.0,2.0,3.0,4.0")
        self.entry_ys.delete(0, tk.END)
        self.entry_ys.insert(0, "1.0,4.0,9.0,16.0")
        self.entry_x_eval.delete(0, tk.END)
        self.entry_x_eval.insert(0, "2.5")
        self.label_resultado.config(text="Resultado: Ingrese los datos y presione Calcular", fg="#2c3e50")
        self.label_coef.config(text="Coeficientes: ")
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.ax.clear()
        self.canvas.draw()
        self.mostrar_ejemplo()


if __name__ == "__main__":
    root = tk.Tk()
    app = InewtonApp(root)
    root.mainloop()