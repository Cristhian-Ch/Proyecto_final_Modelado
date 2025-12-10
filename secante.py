import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

# ==================================================
# MÉTODO DE LA SECANTE
# ==================================================
"""
El método de la Secante aproxima la raíz de una función f(x)
usando una recta que pasa por dos puntos (x0, f(x0)) y (x1, f(x1)).
No requiere derivadas y suele converger más rápido que la bisección.
"""

def secant_iterations(f, x0, x1, tol=1e-8, maxit=100):
    """
    Implementa el método de la secante.
    Retorna una lista con las iteraciones.
    """
    rows = []
    for k in range(1, maxit + 1):
        fx0, fx1 = f(x0), f(x1)
        if fx1 == fx0:
            raise ZeroDivisionError("f(x1) - f(x0) = 0. División por cero.")
        
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        fx2 = f(x2)
        error = abs(x2 - x1)
        
        rows.append((k, x0, x1, x2, fx2, error))
        
        if abs(fx2) < tol or error < tol:
            break
        
        x0, x1 = x1, x2
    return rows


# ==================================================
# GRÁFICA DEL MÉTODO
# ==================================================
def graficar_secante(f, iteraciones, ax=None, canvas=None):
    """Grafica la función y las líneas secantes de las primeras iteraciones."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determinar rango de graficación
    x_vals = [x for _, x0, x1, x2, _, _ in iteraciones for x in (x0, x1, x2)]
    x_min, x_max = min(x_vals) - 1, max(x_vals) + 1
    x_plot = np.linspace(x_min, x_max, 500)
    y_plot = [f(x) for x in x_plot]
    
    # Graficar función
    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    
    # Graficar iteraciones
    colores = ['red', 'green', 'orange', 'purple', 'brown']
    for i, (k, x0, x1, x2, fx2, error) in enumerate(iteraciones[:5]):
        y0, y1 = f(x0), f(x1)
        ax.plot([x0, x1], [y0, y1], color=colores[i % len(colores)], linestyle='--', 
                label=f'Iteración {k}')
        ax.plot(x2, fx2, 'o', color=colores[i % len(colores)], markersize=8)
    
    # Raíz final
    if iteraciones:
        _, _, _, x_final, f_final, _ = iteraciones[-1]
        ax.plot(x_final, 0, 'ro', markersize=10, label=f'Raíz ≈ {x_final:.6f}')
    
    ax.set_xlabel("x", fontsize=12, fontweight='bold')
    ax.set_ylabel("f(x)", fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title("Método de la Secante - Proceso Iterativo", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if canvas:
        canvas.draw()
    else:
        plt.show()


# ==================================================
# INTERFAZ GRÁFICA
# ==================================================
class SecantSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ Método de la Secante - Buscador de Raíces")
        self.root.geometry("900x750")
        self.root.configure(bg='#f0f0f0')
        self.setup_ui()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="MÉTODO DE LA SECANTE", 
                  font=('Arial', 16, 'bold'), foreground='#2c3e50').pack(pady=(0, 10))
        
        ttk.Label(main_frame, text="Encuentra raíces sin derivadas mediante la aproximación secante",
                  font=('Arial', 10), foreground='#7f8c8d', justify=tk.CENTER).pack(pady=(0, 20))
        
        # Función de entrada
        func_frame = ttk.LabelFrame(main_frame, text="Definición de la función f(x)", padding="10")
        func_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(func_frame, text="f(x) =", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=10)
        self.entry_func = ttk.Entry(func_frame, width=40)
        self.entry_func.grid(row=0, column=1, padx=10)
        self.entry_func.insert(0, "x**3 - x - 2")
        
        # Parámetros iniciales
        param_frame = ttk.LabelFrame(main_frame, text="Parámetros del método", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(param_frame, text="x₀:", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5)
        self.entry_x0 = ttk.Entry(param_frame, width=10)
        self.entry_x0.grid(row=0, column=1, padx=10)
        self.entry_x0.insert(0, "1.0")
        
        ttk.Label(param_frame, text="x₁:", font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=5)
        self.entry_x1 = ttk.Entry(param_frame, width=10)
        self.entry_x1.grid(row=0, column=3, padx=10)
        self.entry_x1.insert(0, "2.0")
        
        ttk.Label(param_frame, text="Tolerancia:", font=('Arial', 10, 'bold')).grid(row=0, column=4, padx=5)
        self.entry_tol = ttk.Entry(param_frame, width=10)
        self.entry_tol.grid(row=0, column=5, padx=10)
        self.entry_tol.insert(0, "1e-8")
        
        ttk.Label(param_frame, text="Máx. iteraciones:", font=('Arial', 10, 'bold')).grid(row=0, column=6, padx=5)
        self.entry_maxit = ttk.Entry(param_frame, width=10)
        self.entry_maxit.grid(row=0, column=7, padx=10)
        self.entry_maxit.insert(0, "100")
        
        # Botones
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(btn_frame, text="⚡ Resolver", command=self.resolver).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="📊 Graficar", command=self.graficar_funcion).pack(side=tk.LEFT, padx=10)
        
        # Resultados
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tabla de iteraciones
        self.table_frame = ttk.LabelFrame(result_frame, text="Iteraciones del Método", padding="10")
        self.table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        cols = ("Iter", "x₀", "x₁", "x₂", "f(x₂)", "Error")
        self.tree = ttk.Treeview(self.table_frame, columns=cols, show='headings', height=15)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=90)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Gráfica
        self.graph_frame = ttk.LabelFrame(result_frame, text="Visualización", padding="10")
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax.text(0.5, 0.5, "Ingresa f(x) y valores iniciales", ha='center', va='center', transform=self.ax.transAxes)
        self.canvas.draw()

    def obtener_funcion(self):
        """Crea la función f(x) de la entrada."""
        expr = self.entry_func.get().strip()
        expr = expr.replace("^", "**")
        safe_dict = {"math": math, "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "sqrt": math.sqrt}
        def f(x): return eval(expr, {"x": x, **safe_dict})
        return f

    def resolver(self):
        try:
            f = self.obtener_funcion()
            x0 = float(self.entry_x0.get())
            x1 = float(self.entry_x1.get())
            tol = float(self.entry_tol.get())
            maxit = int(self.entry_maxit.get())
            iters = secant_iterations(f, x0, x1, tol, maxit)
            
            # Mostrar en tabla
            for item in self.tree.get_children():
                self.tree.delete(item)
            for k, x0, x1, x2, fx2, err in iters:
                self.tree.insert('', 'end', values=(k, f"{x0:.6f}", f"{x1:.6f}", f"{x2:.6f}", f"{fx2:.2e}", f"{err:.2e}"))
            
            # Resumen final
            kf, _, _, xf, ff, errf = iters[-1]
            messagebox.showinfo("Resultado", f"Raíz encontrada: {xf:.10f}\nf(x)={ff:.2e}\nError={errf:.2e}\nIteraciones={kf}")
            self.ax.clear()
            graficar_secante(f, iters, self.ax, self.canvas)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def graficar_funcion(self):
        try:
            f = self.obtener_funcion()
            x0 = float(self.entry_x0.get())
            x1 = float(self.entry_x1.get())
            xs = np.linspace(x0 - 3, x1 + 3, 500)
            ys = [f(x) for x in xs]
            self.ax.clear()
            self.ax.plot(xs, ys, 'b-', linewidth=2)
            self.ax.axhline(0, color='black', linewidth=0.5)
            self.ax.axvline(0, color='black', linewidth=0.5)
            self.ax.set_title("Gráfica de la función", fontsize=14)
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))


# ==================================================
# EJECUCIÓN PRINCIPAL
# ==================================================
if __name__ == "__main__":
    plt.style.use('default')
    root = tk.Tk()
    app = SecantSolverApp(root)
    root.mainloop()
