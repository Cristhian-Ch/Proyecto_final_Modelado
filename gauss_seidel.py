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
# MÉTODO ITERATIVO DE GAUSS-SEIDEL
# ==================================================
"""
El método iterativo de Gauss-Seidel resuelve sistemas Ax = b
usando una estrategia de actualización secuencial:
los nuevos valores de x se usan inmediatamente en el mismo ciclo.

x_i^(k+1) = (b_i - Σ_{j<i} a_ij * x_j^(k+1) - Σ_{j>i} a_ij * x_j^(k)) / a_ii
"""

def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    """
    Resuelve Ax = b mediante el método de Gauss-Seidel.
    Retorna las iteraciones, errores y la solución final.
    """
    n = len(b)
    x = np.zeros(n)
    iteraciones = []

    for k in range(1, max_iter + 1):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))        # parte anterior
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))     # parte posterior
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        error = np.linalg.norm(x_new - x, ord=np.inf)
        iteraciones.append((k, *x_new, error))
        if error < tol:
            break
        x = x_new

    return iteraciones, x_new


# ==================================================
# INTERFAZ GRÁFICA
# ==================================================
class GaussSeidelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔁 Método Iterativo de Gauss-Seidel - Sistemas Lineales")
        self.root.geometry("950x700")
        self.root.configure(bg="#f0f0f0")
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="MÉTODO ITERATIVO DE GAUSS-SEIDEL",
                  font=("Arial", 16, "bold"), foreground="#2c3e50").pack(pady=(0, 10))

        ttk.Label(main_frame,
                  text="Calcula soluciones aproximadas de Ax = b usando el método de Gauss-Seidel",
                  font=("Arial", 10), foreground="#7f8c8d",
                  justify=tk.CENTER).pack(pady=(0, 15))

        # Control superior
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

        # Parámetros
        ttk.Label(control_frame, text="Tolerancia:",
                  font=("Arial", 10, "bold")).grid(row=1, column=0, pady=5)
        self.entry_tol = ttk.Entry(control_frame, width=10)
        self.entry_tol.grid(row=1, column=1, padx=5)
        self.entry_tol.insert(0, "1e-6")

        ttk.Label(control_frame, text="Máx. iteraciones:",
                  font=("Arial", 10, "bold")).grid(row=1, column=2, padx=5)
        self.entry_max = ttk.Entry(control_frame, width=10)
        self.entry_max.grid(row=1, column=3, padx=5)
        self.entry_max.insert(0, "50")

        # Sistema
        self.sistema_frame = ttk.LabelFrame(main_frame, text="Sistema de Ecuaciones", padding="15")
        self.sistema_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        # Resultados
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(fill=tk.BOTH, expand=True)

        # Tabla iteraciones
        self.result_frame = ttk.LabelFrame(result_frame, text="Iteraciones", padding="10")
        self.result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.tree = ttk.Treeview(self.result_frame, show='headings', height=15)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.scroll = ttk.Scrollbar(self.result_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=self.scroll.set)
        self.scroll.pack(side="right", fill="y")

        # Gráfica error
        self.graph_frame = ttk.LabelFrame(result_frame, text="Convergencia del Error", padding="10")
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.text(0.5, 0.5, "Genera un sistema y resuélvelo para ver la gráfica",
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
            e_b.grid(row=i, column=n+1, padx=5)
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

            tol = float(self.entry_tol.get())
            max_it = int(self.entry_max.get())

            iteraciones, x_final = gauss_seidel(A, b, tol, max_it)
            self.mostrar_resultados(iteraciones, x_final)
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {e}")

    def mostrar_resultados(self, iteraciones, x_final):
        # Configurar tabla
        self.tree.delete(*self.tree.get_children())
        n = len(x_final)
        cols = ["Iteración"] + [f"x{i+1}" for i in range(n)] + ["Error"]
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=80)
        
        errores = []
        for fila in iteraciones:
            self.tree.insert("", "end", values=[f"{v:.6f}" if isinstance(v, float) else v for v in fila])
            errores.append(fila[-1])

        # Mostrar resultado final
        resumen = "SOLUCIÓN FINAL\n" + "="*40 + "\n"
        for i, xi in enumerate(x_final):
            resumen += f"x{i+1} = {xi:.8f}\n"
        resumen += f"\nIteraciones: {len(iteraciones)}\nError final: {errores[-1]:.2e}"
        messagebox.showinfo("Resultado", resumen)

        # Gráfica del error
        self.ax.clear()
        self.ax.plot(range(1, len(errores)+1), errores, marker='o', linewidth=2, color='green')
        self.ax.set_xlabel("Iteraciones", fontweight='bold')
        self.ax.set_ylabel("Error (∞-norma)", fontweight='bold')
        self.ax.set_title("Convergencia del Método de Gauss-Seidel", fontweight='bold')
        self.ax.grid(True, alpha=0.4)
        self.canvas.draw()


# ==================================================
# EJECUCIÓN PRINCIPAL
# ==================================================
if __name__ == "__main__":
    plt.style.use("default")
    root = tk.Tk()
    app = GaussSeidelApp(root)
    root.mainloop()
