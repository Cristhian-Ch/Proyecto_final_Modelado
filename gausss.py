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
# MÉTODO DE ELIMINACIÓN GAUSSIANA
# ==================================================
def gauss_elimination(A, b):
    """
    Resuelve el sistema de ecuaciones Ax = b usando Eliminación Gaussiana
    con pivoteo parcial para mejorar la estabilidad numérica.
    
    Parámetros:
    A: matriz de coeficientes (n x n)
    b: vector de términos independientes (n)
    
    Retorna:
    x: vector solución (n)
    """
    n = len(b)
    # Crear matriz aumentada [A|b]
    M = np.hstack([A.astype(float), b.reshape(-1, 1)])

    # Fase de eliminación hacia adelante
    for k in range(n):
        # Pivoteo parcial: encontrar la fila con el mayor elemento en la columna k
        max_row = np.argmax(abs(M[k:, k])) + k
        if max_row != k:
            M[[k, max_row]] = M[[max_row, k]]  # Intercambiar filas
        
        # Eliminación
        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i] = M[i] - factor * M[k]

    # Fase de sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:n])) / M[i, i]
    
    return x

# ==================================================
# VISUALIZACIÓN GRÁFICA
# ==================================================
def graficar_sistema(A, b, ax=None, canvas=None):
    """
    Genera gráficas 2D o 3D del sistema de ecuaciones según el número de variables.
    
    Parámetros:
    A: matriz de coeficientes
    b: vector de términos independientes
    ax: eje para plotting (opcional)
    canvas: canvas de tkinter para embedding (opcional)
    """
    n = A.shape[1]
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        if n == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = plt.gca()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    if n == 2:
        # Sistema 2D
        x_vals = np.linspace(-10, 10, 400)
        for i in range(len(b)):
            if A[i, 1] != 0:
                y_vals = (b[i] - A[i, 0] * x_vals) / A[i, 1]
                ax.plot(x_vals, y_vals, 
                       color=colors[i % len(colors)], 
                       linewidth=2, 
                       label=f"{A[i,0]:.1f}x₁ + {A[i,1]:.1f}x₂ = {b[i]:.1f}")
        
        ax.set_xlabel("x₁", fontsize=12, fontweight='bold')
        ax.set_ylabel("x₂", fontsize=12, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_title("Sistema de 2 Ecuaciones Lineales", fontsize=14, fontweight='bold')
        
    elif n == 3:
        # Sistema 3D
        x_vals = np.linspace(-10, 10, 20)
        y_vals = np.linspace(-10, 10, 20)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        for i in range(len(b)):
            if A[i, 2] != 0:
                Z = (b[i] - A[i, 0] * X - A[i, 1] * Y) / A[i, 2]
                ax.plot_surface(X, Y, Z, 
                              alpha=0.6, 
                              color=colors[i % len(colors)],
                              label=f"Ecuación {i+1}")
        
        ax.set_xlabel("x₁", fontsize=12, fontweight='bold')
        ax.set_ylabel("x₂", fontsize=12, fontweight='bold')
        ax.set_zlabel("x₃", fontsize=12, fontweight='bold')
        ax.set_title("Sistema de 3 Ecuaciones Lineales", fontsize=14, fontweight='bold')
    
    else:
        messagebox.showinfo("Aviso", "Solo se pueden graficar sistemas de 2 o 3 variables.")
        return None
    
    plt.tight_layout()
    
    if canvas is None:
        plt.show()
    else:
        canvas.draw()
    
    return ax

# ==================================================
# INTERFAZ GRÁFICA MEJORADA
# ==================================================
class GaussSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧮 Solucionador de Sistemas Lineales - Método de Gauss")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.entradas_A = []
        self.entradas_b = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(main_frame, 
                               text="MÉTODO DE ELIMINACIÓN GAUSSIANA", 
                               font=('Arial', 16, 'bold'),
                               foreground='#2c3e50')
        title_label.pack(pady=(0, 20))
        
        # Descripción
        desc_text = "Resuelve sistemas de ecuaciones lineales usando el método de Eliminación Gaussiana\ncon pivoteo parcial para máxima precisión."
        desc_label = ttk.Label(main_frame, 
                              text=desc_text, 
                              font=('Arial', 10),
                              foreground='#7f8c8d',
                              justify=tk.CENTER)
        desc_label.pack(pady=(0, 20))
        
        # Frame de controles
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Entrada para número de incógnitas
        ttk.Label(control_frame, text="Número de incógnitas:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=(0, 10))
        
        self.entry_n = ttk.Entry(control_frame, width=10, font=('Arial', 10))
        self.entry_n.grid(row=0, column=1, padx=(0, 20))
        
        # Botón generar
        self.btn_generar = ttk.Button(control_frame, 
                                     text="🎯 Generar Sistema", 
                                     command=self.generar_campos)
        self.btn_generar.grid(row=0, column=2, padx=(0, 10))
        
        # Botón resolver
        self.btn_resolver = ttk.Button(control_frame, 
                                      text="⚡ Resolver Sistema", 
                                      command=self.resolver,
                                      state='disabled')
        self.btn_resolver.grid(row=0, column=3)
        
        # Frame para ecuaciones
        self.frame_ecuaciones = ttk.LabelFrame(main_frame, 
                                              text="Sistema de Ecuaciones", 
                                              padding="15")
        self.frame_ecuaciones.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Frame para resultados y gráfica
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Resultados
        self.result_frame = ttk.LabelFrame(results_frame, 
                                          text="Solución", 
                                          padding="10")
        self.result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.result_text = tk.Text(self.result_frame, 
                                  height=8, 
                                  width=30, 
                                  font=('Consolas', 10),
                                  bg='#f8f9fa',
                                  relief='solid')
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Gráfica
        self.graph_frame = ttk.LabelFrame(results_frame, 
                                         text="Visualización", 
                                         padding="10")
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas para matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mensaje inicial en gráfica
        self.ax.text(0.5, 0.5, 'Genera un sistema\nde ecuaciones\npara ver la gráfica', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, fontsize=12, style='italic')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
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
        
        # Limpiar campos anteriores
        for widget in self.frame_ecuaciones.winfo_children():
            widget.destroy()
        
        self.entradas_A = []
        self.entradas_b = []
        
        # Crear matriz de coeficientes
        for i in range(n):
            fila_entries = []
            for j in range(n):
                e = ttk.Entry(self.frame_ecuaciones, width=8, font=('Arial', 10))
                e.grid(row=i, column=j, padx=3, pady=3)
                # Insertar valores por defecto para testing
                if i == j:
                    e.insert(0, "1")
                else:
                    e.insert(0, "0")
                fila_entries.append(e)
            self.entradas_A.append(fila_entries)
            
            # Separador
            ttk.Label(self.frame_ecuaciones, text="=").grid(row=i, column=n, padx=5)
            
            # Vector b
            e_b = ttk.Entry(self.frame_ecuaciones, width=8, font=('Arial', 10))
            e_b.grid(row=i, column=n+1, padx=3, pady=3)
            e_b.insert(0, "1")
            self.entradas_b.append(e_b)
        
        # Etiquetas de variables
        for j in range(n):
            ttk.Label(self.frame_ecuaciones, text=f"x{j+1}", 
                     font=('Arial', 8, 'bold')).grid(row=n, column=j, pady=(5, 0))
        
        self.btn_resolver['state'] = 'normal'
    
    def resolver(self):
        try:
            n = int(self.entry_n.get())
            A = np.zeros((n, n))
            b = np.zeros(n)
            
            # Leer valores de la interfaz
            for i in range(n):
                for j in range(n):
                    A[i, j] = float(self.entradas_A[i][j].get())
                b[i] = float(self.entradas_b[i].get())
            
            # Resolver sistema
            solucion = gauss_elimination(A, b)
            
            # Mostrar resultados
            self.mostrar_resultado(A, b, solucion)
            
            # Actualizar gráfica
            self.actualizar_grafica(A, b)
            
        except ValueError as e:
            messagebox.showerror("Error", f"Entrada inválida: {e}\nAsegúrese de ingresar números válidos.")
        except np.linalg.LinAlgError as e:
            messagebox.showerror("Error", f"Error matemático: {e}\nEl sistema puede ser singular.")
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado: {e}")
    
    def mostrar_resultado(self, A, b, solucion):
        """Muestra la solución formateada en el área de texto"""
        self.result_text.delete(1.0, tk.END)
        
        # Mostrar sistema original
        self.result_text.insert(tk.END, "SISTEMA ORIGINAL:\n")
        self.result_text.insert(tk.END, "="*30 + "\n")
        
        n = len(solucion)
        for i in range(n):
            ecuacion = ""
            for j in range(n):
                signo = " + " if A[i, j] >= 0 and j > 0 else ""
                ecuacion += f"{signo}{A[i, j]:.2f}x{j+1}"
            ecuacion += f" = {b[i]:.2f}\n"
            self.result_text.insert(tk.END, ecuacion)
        
        # Mostrar solución
        self.result_text.insert(tk.END, "\nSOLUCIÓN:\n")
        self.result_text.insert(tk.END, "="*30 + "\n")
        for i in range(n):
            self.result_text.insert(tk.END, f"x{i+1} = {solucion[i]:.6f}\n")
        
        
    def actualizar_grafica(self, A, b):
        """Actualiza la gráfica con el sistema actual"""
        self.ax.clear()
        n = A.shape[1]
        
        if n in [2, 3]:
            graficar_sistema(A, b, self.ax, self.canvas)
        else:
            self.ax.text(0.5, 0.5, 'Sistema resuelto\n(gráfica disponible\nsolo para 2-3 variables)', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=self.ax.transAxes, fontsize=12)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()

# ==================================================
# EJECUCIÓN PRINCIPAL
# ==================================================
if __name__ == "__main__":
    # Configurar estilo de matplotlib
    plt.style.use('default')
    
    # Crear y ejecutar aplicación
    root = tk.Tk()
    app = GaussSolverApp(root)
    root.mainloop()