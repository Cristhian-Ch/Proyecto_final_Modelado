import numpy as np
import matplotlib.pyplot as plt
import math

# Intentar importar Tkinter y los componentes gráficos.
# Si no están disponibles (por ejemplo en Streamlit Cloud), se ignora el error.
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    tk = None
    ttk = None
    messagebox = None
    FigureCanvasTkAgg = None



# ==================================================
# MÉTODO DE GAUSS-JORDAN
# ==================================================
"""
El método de Gauss-Jordan extiende la eliminación de Gauss
llevando la matriz aumentada [A|b] hasta la forma reducida
de identidad. Permite obtener la solución directamente.
"""

def gauss_jordan(A, b):
    """
    Resuelve el sistema A·x = b usando el método de Gauss-Jordan.
    """
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1, 1)])

    for k in range(n):
        # Pivoteo parcial
        if abs(M[k][k]) < 1e-12:
            for i in range(k + 1, n):
                if abs(M[i][k]) > abs(M[k][k]):
                    M[[k, i]] = M[[i, k]]
                    break
        
        # Normalización de la fila pivote
        M[k] = M[k] / M[k][k]

        # Eliminación de otras filas
        for i in range(n):
            if i != k:
                M[i] = M[i] - M[i][k] * M[k]

    return M[:, -1], M


# ==================================================
# INTERFAZ GRÁFICA
# ==================================================
class GaussJordanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧮 Método de Gauss-Jordan - Sistemas Lineales")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="MÉTODO DE GAUSS-JORDAN",
                  font=("Arial", 16, "bold"),
                  foreground="#2c3e50").pack(pady=(0, 10))

        ttk.Label(main_frame, text="Resuelve sistemas lineales por reducción total de la matriz aumentada",
                  font=("Arial", 10), foreground="#7f8c8d",
                  justify=tk.CENTER).pack(pady=(0, 15))

        # Número de incógnitas
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(control_frame, text="Número de incógnitas:",
                  font=("Arial", 10, "bold")).grid(row=0, column=0, padx=(0, 10))
        self.entry_n = ttk.Entry(control_frame, width=10)
        self.entry_n.grid(row=0, column=1, padx=(0, 20))
        self.entry_n.insert(0, "3")

        ttk.Button(control_frame, text="🎯 Generar Sistema",
                   command=self.generar_campos).grid(row=0, column=2, padx=(0, 10))

        ttk.Button(control_frame, text="⚡ Resolver Sistema",
                   command=self.resolver).grid(row=0, column=3, padx=(0, 10))

        # Frame del sistema
        self.sistema_frame = ttk.LabelFrame(main_frame, text="Sistema de Ecuaciones", padding="15")
        self.sistema_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        # Resultados
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.result_frame = ttk.LabelFrame(result_frame, text="Solución del Sistema", padding="10")
        self.result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.result_text = tk.Text(self.result_frame, height=15, width=45,
                                   font=("Consolas", 10), bg="#f8f9fa")
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Matriz aumentada final
        self.graph_frame = ttk.LabelFrame(result_frame, text="Matriz Aumentada Final", padding="10")
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.axis("off")
        self.ax.text(0.5, 0.5, "Genera un sistema para ver la matriz final",
                     ha="center", va="center", transform=self.ax.transAxes)
        self.canvas.draw()

    def generar_campos(self):
        try:
            n = int(self.entry_n.get())
            if n < 2 or n > 6:
                messagebox.showerror("Error", "El número de incógnitas debe estar entre 2 y 6.")
                return
        except ValueError:
            messagebox.showerror("Error", "Ingrese un número válido.")
            return

        for widget in self.sistema_frame.winfo_children():
            widget.destroy()

        self.entradas_A = []
        self.entradas_b = []

        for i in range(n):
            fila = []
            for j in range(n):
                e = ttk.Entry(self.sistema_frame, width=8, font=("Arial", 9))
                e.grid(row=i, column=j, padx=5, pady=3)
                e.insert(0, "1" if i == j else "0")
                fila.append(e)
            self.entradas_A.append(fila)

            ttk.Label(self.sistema_frame, text="=").grid(row=i, column=n, padx=5)
            e_b = ttk.Entry(self.sistema_frame, width=8, font=("Arial", 9))
            e_b.grid(row=i, column=n + 1, padx=5)
            e_b.insert(0, "1")
            self.entradas_b.append(e_b)

    def resolver(self):
        try:
            n = int(self.entry_n.get())
            A = np.zeros((n, n))
            b = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    A[i, j] = float(self.entradas_A[i][j].get())
                b[i] = float(self.entradas_b[i].get())

            x, M_final = gauss_jordan(A, b)

            # Mostrar resultados
            self.mostrar_resultado(A, b, x, M_final)
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {e}")

    def mostrar_resultado(self, A, b, x, M_final):
        self.result_text.delete(1.0, tk.END)
        n = len(b)

        self.result_text.insert(tk.END, "SISTEMA DE ECUACIONES:\n" + "="*40 + "\n")
        for i in range(n):
            ecuacion = " + ".join([f"{A[i,j]:.2f}x{j+1}" for j in range(n)])
            self.result_text.insert(tk.END, f"{ecuacion} = {b[i]:.2f}\n")

        self.result_text.insert(tk.END, "\nSOLUCIÓN:\n" + "="*40 + "\n")
        for i in range(n):
            self.result_text.insert(tk.END, f"x{i+1} = {x[i]:.6f}\n")

        self.result_text.insert(tk.END, "\nMATRIZ AUMENTADA FINAL:\n" + "="*40 + "\n")
        self.result_text.insert(tk.END, str(np.round(M_final, 4)) + "\n")

        # Mostrar matriz en gráfica
        self.ax.clear()
        self.ax.axis("off")
        table_data = np.round(M_final, 4)
        self.ax.table(cellText=table_data, loc="center", cellLoc="center")
        self.canvas.draw()


# ==================================================
# EJECUCIÓN PRINCIPAL
# ==================================================
if __name__ == "__main__":
    plt.style.use("default")
    root = tk.Tk()
    app = GaussJordanApp(root)
    root.mainloop()
