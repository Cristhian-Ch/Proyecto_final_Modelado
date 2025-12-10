import numpy as np
import matplotlib.pyplot as plt
import math

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


# ==================================================
# MÉTODO DE INTERPOLACIÓN DE LAGRANGE
# ==================================================
"""
DESCRIPCIÓN:
El Método de Interpolación de Lagrange es un método que permite obtener un 
polinomio que pasa exactamente por un conjunto de puntos dados. El polinomio 
resultante puede evaluarse en cualquier valor de x.
"""

def lagrange_iterations(xs, ys, x_eval):
    """
    Implementa el método de interpolación de Lagrange.
    
    Parámetros:
    xs: array de valores x
    ys: array de valores y
    x_eval: valor donde evaluar el polinomio
    
    Retorna:
    rows: lista con datos de cada término de Lagrange
    """
    n = len(xs)
    rows = []
    
    for i in range(n):
        li = 1.0
        terminos = []
        for j in range(n):
            if j != i:
                li *= (x_eval - xs[j]) / (xs[i] - xs[j])
                terminos.append(f"({x_eval:.4f}-{xs[j]:.4f})/({xs[i]:.4f}-{xs[j]:.4f})")
        
        contribucion = ys[i] * li
        rows.append((i+1, xs[i], ys[i], li, contribucion))
    
    return rows

def lagrange_eval(x, xs, ys):
    """
    Evalúa el polinomio de Lagrange en un punto x.
    """
    n = len(xs)
    total = 0.0
    for i in range(n):
        li = 1.0
        for j in range(n):
            if j != i:
                li *= (x - xs[j]) / (xs[i] - xs[j])
        total += ys[i] * li
    return total

# ==================================================
# VISUALIZACIÓN GRÁFICA
# ==================================================
def graficar_lagrange(xs, ys, x_eval, resultado, ax=None, canvas=None):
    """
    Genera gráficas del método de Lagrange.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Rango de graficación
    x_min, x_max = min(xs) - 1, max(xs) + 1
    x_plot = np.linspace(x_min, x_max, 200)
    y_plot = [lagrange_eval(x, xs, ys) for x in x_plot]
    
    # Graficar el polinomio de Lagrange
    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='P(x) - Polinomio de Lagrange')
    
    # Graficar los puntos dados
    ax.plot(xs, ys, 'ro', markersize=8, label='Puntos dados', zorder=5)
    
    # Graficar el punto evaluado
    ax.plot(x_eval, resultado, 'gs', markersize=10, label=f'P({x_eval:.4f}) = {resultado:.4f}', zorder=5)
    
    # Líneas verticales de referencia
    ax.vlines(x_eval, 0, resultado, colors='green', linestyles='--', alpha=0.5)
    ax.hlines(resultado, x_min, x_eval, colors='green', linestyles='--', alpha=0.5)
    
    ax.set_xlabel("x", fontsize=12, fontweight='bold')
    ax.set_ylabel("y", fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title("Método de Interpolación de Lagrange", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if canvas is not None:
        canvas.draw()
    else:
        plt.show()
    
    return ax

# ==================================================
# INTERFAZ GRÁFICA
# ==================================================
class LagrangeSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📐 Método de Interpolación de Lagrange")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.xs = np.array([1.0, 2.0, 4.0])
        self.ys = np.array([2.0, 3.0, 1.0])
        
        # ========== PANEL SUPERIOR ==========
        top_frame = tk.Frame(root, bg='#2c3e50', bd=5)
        top_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(top_frame, text="MÉTODO DE INTERPOLACIÓN DE LAGRANGE",
                font=("Arial", 14, "bold"), fg="white", bg='#2c3e50').pack(pady=10)
        
        # ========== PANEL DE ENTRADA ==========
        input_frame = ttk.LabelFrame(root, text="Datos de Entrada", padding=10)
        input_frame.pack(fill='x', padx=10, pady=5)
        
        # Puntos X
        tk.Label(input_frame, text="Valores X (separados por comas):", font=("Arial", 10)).grid(row=0, column=0, sticky='w')
        self.entry_xs = tk.Entry(input_frame, width=50)
        self.entry_xs.insert(0, "1.0,2.0,4.0")
        self.entry_xs.grid(row=0, column=1, padx=5, pady=5)
        
        # Puntos Y
        tk.Label(input_frame, text="Valores Y (separados por comas):", font=("Arial", 10)).grid(row=1, column=0, sticky='w')
        self.entry_ys = tk.Entry(input_frame, width=50)
        self.entry_ys.insert(0, "2.0,3.0,1.0")
        self.entry_ys.grid(row=1, column=1, padx=5, pady=5)
        
        # Punto de evaluación
        tk.Label(input_frame, text="Evaluar en x =", font=("Arial", 10)).grid(row=2, column=0, sticky='w')
        self.entry_x_eval = tk.Entry(input_frame, width=50)
        self.entry_x_eval.insert(0, "3.0")
        self.entry_x_eval.grid(row=2, column=1, padx=5, pady=5)
        
        # Botón Calcular
        btn_frame = tk.Frame(root, bg='#f0f0f0')
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="▶ Calcular", font=("Arial", 11, "bold"),
                 bg="#27ae60", fg="white", padx=20, pady=10,
                 command=self.calcular).pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="🔄 Limpiar", font=("Arial", 11, "bold"),
                 bg="#e74c3c", fg="white", padx=20, pady=10,
                 command=self.limpiar).pack(side='left', padx=5)
        
        # ========== RESULTADO ==========
        result_frame = ttk.LabelFrame(root, text="Resultado", padding=10)
        result_frame.pack(fill='x', padx=10, pady=5)
        
        self.label_resultado = tk.Label(result_frame, text="Resultado: Ingrese los datos y presione Calcular",
                                       font=("Arial", 12, "bold"), fg="#2c3e50")
        self.label_resultado.pack(pady=10)
        
        # ========== TABLA DE ITERACIONES ==========
        table_frame = ttk.LabelFrame(root, text="Términos de Lagrange", padding=10)
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Crear treeview
        self.tree = ttk.Treeview(table_frame, columns=("Punto", "xi", "yi", "Li", "Contribución"),
                                height=8, show='headings')
        
        self.tree.column("#0", width=0, stretch=tk.NO)
        self.tree.column("Punto", anchor=tk.CENTER, width=60)
        self.tree.column("xi", anchor=tk.CENTER, width=80)
        self.tree.column("yi", anchor=tk.CENTER, width=80)
        self.tree.column("Li", anchor=tk.CENTER, width=100)
        self.tree.column("Contribución", anchor=tk.CENTER, width=100)
        
        self.tree.heading("Punto", text="Punto")
        self.tree.heading("xi", text="xi")
        self.tree.heading("yi", text="yi")
        self.tree.heading("Li", text="Li (Lagrange)")
        self.tree.heading("Contribución", text="yi * Li")
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # ========== GRÁFICA ==========
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
        
        # Mostrar ejemplo inicial
        self.mostrar_ejemplo()
    
    def mostrar_ejemplo(self):
        """Muestra un ejemplo con los datos predefinidos."""
        try:
            resultado = lagrange_eval(3.0, self.xs, self.ys)
            self.label_resultado.config(text=f"P(3.0) = {resultado:.6f}")
            
            # Mostrar tabla
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            rows = lagrange_iterations(self.xs, self.ys, 3.0)
            for punto, xi, yi, li, contrib in rows:
                self.tree.insert('', 'end', values=(f"{punto}", f"{xi:.4f}", f"{yi:.4f}", 
                                                   f"{li:.6f}", f"{contrib:.6f}"))
            
            # Graficar
            self.ax.clear()
            graficar_lagrange(self.xs, self.ys, 3.0, resultado, self.ax, self.canvas)
        except Exception as e:
            messagebox.showerror("Error", f"Error en el cálculo:\n{str(e)}")
    
    def calcular(self):
        """Calcula la interpolación de Lagrange con los datos ingresados."""
        try:
            # Obtener datos
            xs_str = self.entry_xs.get().replace(' ', '')
            ys_str = self.entry_ys.get().replace(' ', '')
            x_eval_str = self.entry_x_eval.get().strip()
            
            # Convertir a arrays
            xs = np.array([float(x) for x in xs_str.split(',')])
            ys = np.array([float(y) for y in ys_str.split(',')])
            x_eval = float(x_eval_str)
            
            # Validación
            if len(xs) != len(ys):
                raise ValueError("Los arrays X e Y deben tener la misma longitud")
            
            if len(xs) < 2:
                raise ValueError("Se necesitan al menos 2 puntos")
            
            # Calcular
            resultado = lagrange_eval(x_eval, xs, ys)
            self.label_resultado.config(text=f"P({x_eval:.6f}) = {resultado:.6f}", fg="#27ae60")
            
            # Mostrar tabla
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            rows = lagrange_iterations(xs, ys, x_eval)
            for punto, xi, yi, li, contrib in rows:
                self.tree.insert('', 'end', values=(f"{punto}", f"{xi:.4f}", f"{yi:.4f}", 
                                                   f"{li:.6f}", f"{contrib:.6f}"))
            
            # Graficar
            self.ax.clear()
            graficar_lagrange(xs, ys, x_eval, resultado, self.ax, self.canvas)
        
        except ValueError as e:
            messagebox.showerror("Error de entrada", f"Error: {str(e)}")
            self.label_resultado.config(text="Error en los datos de entrada", fg="#e74c3c")
        except Exception as e:
            messagebox.showerror("Error", f"Error en el cálculo:\n{str(e)}")
            self.label_resultado.config(text="Error en el cálculo", fg="#e74c3c")
    
    def limpiar(self):
        """Limpia los campos de entrada y restablece valores por defecto."""
        self.entry_xs.delete(0, tk.END)
        self.entry_xs.insert(0, "1.0,2.0,4.0")
        
        self.entry_ys.delete(0, tk.END)
        self.entry_ys.insert(0, "2.0,3.0,1.0")
        
        self.entry_x_eval.delete(0, tk.END)
        self.entry_x_eval.insert(0, "3.0")
        
        self.label_resultado.config(text="Resultado: Ingrese los datos y presione Calcular", fg="#2c3e50")
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.ax.clear()
        self.canvas.draw()
        self.mostrar_ejemplo()

# ==================================================
# EJECUCIÓN
# ==================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = LagrangeSolverApp(root)
    root.mainloop()