import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==================================================
# MÉTODO DE MATRIZ INVERSA
# ==================================================
def resolver_matriz_inversa(A, b):
    """
    Resuelve el sistema de ecuaciones lineales A · x = b usando matriz inversa.
    """
    # Verificar que A es cuadrada
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada")
    
    # Verificar que las dimensiones coinciden
    if A.shape[0] != len(b):
        raise ValueError("Las dimensiones de A y b no coinciden")
    
    # Verificar si la matriz es invertible
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        raise ValueError("La matriz A es singular (determinante ≈ 0). No tiene inversa.")
    
    # Calcular matriz inversa
    A_inv = np.linalg.inv(A)
    
    # Calcular solución
    x = np.dot(A_inv, b)
    
    return x, A_inv, det

# ==================================================
# VISUALIZACIÓN GRÁFICA (Solo para 2 variables)
# ==================================================
def graficar_sistema_2d(A, b, solucion, ax=None, canvas=None):
    """
    Genera gráfica 2D del sistema de ecuaciones para 2 variables.
    """
    if A.shape[1] != 2:
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Configurar límites del gráfico
    x1_sol, x2_sol = solucion
    margin = max(abs(x1_sol), abs(x2_sol)) * 1.5
    margin = max(margin, 5)  # Mínimo de 5 unidades
    
    x_vals = np.linspace(-margin, margin, 400)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Graficar cada ecuación
    for i in range(len(b)):
        if A[i, 1] != 0:  # Evitar división por cero
            if A[i, 0] == 0:  # Línea vertical
                x2_vals = (b[i] - A[i, 1] * x_vals) / A[i, 1]
                ax.axvline(b[i] / A[i, 1] if A[i, 1] != 0 else 0, 
                          color=colors[i % len(colors)], linewidth=2,
                          label=f'{A[i,0]:.1f}x₁ + {A[i,1]:.1f}x₂ = {b[i]:.1f}')
            else:
                x2_vals = (b[i] - A[i, 0] * x_vals) / A[i, 1]
                ax.plot(x_vals, x2_vals, color=colors[i % len(colors)], linewidth=2,
                       label=f'{A[i,0]:.1f}x₁ + {A[i,1]:.1f}x₂ = {b[i]:.1f}')
    
    # Marcar la solución
    ax.plot(x1_sol, x2_sol, 'ro', markersize=10, label=f'Solución: ({x1_sol:.2f}, {x2_sol:.2f})')
    ax.axhline(x2_sol, color='red', linestyle='--', alpha=0.3)
    ax.axvline(x1_sol, color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel("x₁", fontsize=12, fontweight='bold')
    ax.set_ylabel("x₂", fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title("Sistema de Ecuaciones Lineales (2 Variables)", fontsize=14, fontweight='bold')
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    
    plt.tight_layout()
    
    if canvas is not None:
        canvas.draw()
    else:
        plt.show()
    
    return ax

# ==================================================
# INTERFAZ GRÁFICA
# ==================================================
class MatrizInversaSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧮 Método de Matriz Inversa - Solucionador de Sistemas")
        self.root.geometry("900x700")
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
                               text="MÉTODO DE MATRIZ INVERSA", 
                               font=('Arial', 16, 'bold'),
                               foreground='#2c3e50')
        title_label.pack(pady=(0, 10))
        
        # Descripción
        desc_text = "Resuelve sistemas de ecuaciones lineales usando la matriz inversa\nA · x = b  →  x = A⁻¹ · b"
        desc_label = ttk.Label(main_frame, 
                              text=desc_text, 
                              font=('Arial', 10),
                              foreground='#7f8c8d',
                              justify=tk.CENTER)
        desc_label.pack(pady=(0, 20))
        
        # Frame de controles
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Entrada para número de variables
        ttk.Label(control_frame, text="Número de variables:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=(0, 10))
        
        self.entry_n = ttk.Entry(control_frame, width=10)
        self.entry_n.grid(row=0, column=1, padx=(0, 20))
        self.entry_n.insert(0, "2")
        
        # Botones de control
        self.btn_generar = ttk.Button(control_frame, 
                                     text="🎯 Generar Sistema", 
                                     command=self.generar_campos)
        self.btn_generar.grid(row=0, column=2, padx=(0, 10))
        
        self.btn_resolver = ttk.Button(control_frame, 
                                      text="⚡ Resolver con Matriz Inversa", 
                                      command=self.resolver)
        self.btn_resolver.grid(row=0, column=3, padx=(0, 10))
        
        self.btn_ejemplo = ttk.Button(control_frame, 
                                     text="📚 Ejemplo Predefinido", 
                                     command=self.cargar_ejemplo)
        self.btn_ejemplo.grid(row=0, column=4)
        
        # Frame para el sistema de ecuaciones
        self.sistema_frame = ttk.LabelFrame(main_frame, 
                                          text="Sistema de Ecuaciones", 
                                          padding="15")
        self.sistema_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Frame para resultados y gráfica
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame de resultados (izquierda)
        self.result_frame = ttk.LabelFrame(results_frame, 
                                         text="Solución del Sistema", 
                                         padding="10")
        self.result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Área de texto para resultados
        self.sol_text = tk.Text(self.result_frame, wrap=tk.WORD, width=45, height=15,
                               font=('Consolas', 10), bg='#f8f9fa')
        sol_scroll = ttk.Scrollbar(self.result_frame, orient="vertical", command=self.sol_text.yview)
        self.sol_text.configure(yscrollcommand=sol_scroll.set)
        
        self.sol_text.pack(side="left", fill="both", expand=True)
        sol_scroll.pack(side="right", fill="y")
        
        # Frame de gráfica (derecha)
        self.graph_frame = ttk.LabelFrame(results_frame, 
                                        text="Gráfica del Sistema (2 Variables)", 
                                        padding="10")
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas para matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mensaje inicial
        self.ax.text(0.5, 0.5, 'Genera un sistema\n de ecuaciones\npara ver la gráfica', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, fontsize=12, style='italic')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
        
        # Generar campos iniciales
        self.generar_campos()
    
    def generar_campos(self):
        """Genera los campos de entrada para la matriz A y vector b"""
        try:
            n = int(self.entry_n.get())
            if n < 2:
                messagebox.showerror("Error", "Debe haber al menos 2 variables.")
                return
            if n > 6:
                messagebox.showwarning("Advertencia", "Sistemas muy grandes pueden ser lentos.")
        except ValueError:
            messagebox.showerror("Error", "Ingrese un número válido.")
            return
        
        # Limpiar campos anteriores
        for widget in self.sistema_frame.winfo_children():
            widget.destroy()
        
        self.entradas_A = []
        self.entradas_b = []
        
        # Crear encabezados
        for j in range(n):
            ttk.Label(self.sistema_frame, text=f"x{j+1}", 
                     font=('Arial', 9, 'bold')).grid(row=0, column=j, padx=3, pady=2)
        ttk.Label(self.sistema_frame, text="=", 
                 font=('Arial', 9, 'bold')).grid(row=0, column=n, padx=5, pady=2)
        ttk.Label(self.sistema_frame, text="b", 
                 font=('Arial', 9, 'bold')).grid(row=0, column=n+1, padx=3, pady=2)
        
        # Crear entradas para la matriz A y vector b
        for i in range(n):
            fila_entries = []
            for j in range(n):
                e = ttk.Entry(self.sistema_frame, width=8, font=('Arial', 9))
                e.grid(row=i+1, column=j, padx=3, pady=2)
                # Valor por defecto: matriz identidad
                if i == j:
                    e.insert(0, "1")
                else:
                    e.insert(0, "0")
                fila_entries.append(e)
            self.entradas_A.append(fila_entries)
            
            # Signo igual
            ttk.Label(self.sistema_frame, text="=").grid(row=i+1, column=n, padx=5, pady=2)
            
            # Vector b
            e_b = ttk.Entry(self.sistema_frame, width=8, font=('Arial', 9))
            e_b.grid(row=i+1, column=n+1, padx=3, pady=2)
            e_b.insert(0, "1")
            self.entradas_b.append(e_b)
    
    def cargar_ejemplo(self):
        """Carga un ejemplo predefinido de sistema 2x2"""
        self.entry_n.delete(0, tk.END)
        self.entry_n.insert(0, "2")
        self.generar_campos()
        
        # Ejemplo: 2x + 3y = 8, x - y = 1
        valores_A = [
            ["2", "3"],
            ["1", "-1"]
        ]
        valores_b = ["8", "1"]
        
        for i in range(2):
            for j in range(2):
                self.entradas_A[i][j].delete(0, tk.END)
                self.entradas_A[i][j].insert(0, valores_A[i][j])
            self.entradas_b[i].delete(0, tk.END)
            self.entradas_b[i].insert(0, valores_b[i])
    
    def resolver(self):
        """Método principal para resolver el sistema"""
        try:
            n = int(self.entry_n.get())
            A = np.zeros((n, n))
            b = np.zeros(n)
            
            print(f"Leyendo {n}x{n} sistema...")  # DEBUG
            
            # Leer valores actuales de la interfaz
            for i in range(n):
                for j in range(n):
                    valor = self.entradas_A[i][j].get()
                    print(f"A[{i},{j}] = '{valor}'")  # DEBUG
                    A[i, j] = float(valor) if valor.strip() != '' else 0.0
                
                valor_b = self.entradas_b[i].get()
                print(f"b[{i}] = '{valor_b}'")  # DEBUG
                b[i] = float(valor_b) if valor_b.strip() != '' else 0.0
            
            print(f"Matriz A:\n{A}")  # DEBUG
            print(f"Vector b: {b}")  # DEBUG
            
            # Resolver sistema
            solucion, A_inv, determinante = resolver_matriz_inversa(A, b)
            print(f"Solución: {solucion}")  # DEBUG
            
            # Mostrar resultados - FORZAR ACTUALIZACIÓN
            self.mostrar_solucion(A, b, solucion, determinante)
            
            # Graficar si es sistema 2D
            if n == 2:
                self.actualizar_grafica(A, b, solucion)
            else:
                self.ax.clear()
                self.ax.text(0.5, 0.5, 'Gráfica disponible\nsolo para 2 variables', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=self.ax.transAxes, fontsize=12)
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.canvas.draw()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Error en los datos: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado: {e}")
    
    def mostrar_solucion(self, A, b, solucion, determinante):
        """Muestra la solución en el área de texto"""
        n = len(solucion)
        
        print(f"Mostrando solución para sistema {n}x{n}")  # DEBUG
        
        # Limpiar texto anterior COMPLETAMENTE
        self.sol_text.config(state=tk.NORMAL)  # Habilitar edición
        self.sol_text.delete(1.0, tk.END)
        
        self.sol_text.insert(tk.END, "SISTEMA DE ECUACIONES:\n")
        self.sol_text.insert(tk.END, "=" * 40 + "\n")
        
        # Mostrar sistema de ecuaciones
        for i in range(n):
            ecuacion = ""
            for j in range(n):
                signo = " + " if A[i, j] >= 0 and j > 0 else ""
                num = f"{A[i, j]:g}"  # Formato sin decimales innecesarios
                if abs(A[i, j]) == 1 and j > 0:
                    num = "" if A[i, j] == 1 else "-"
                ecuacion += f"{signo}{num}x{j+1}"
            ecuacion += f" = {b[i]:g}\n"
            self.sol_text.insert(tk.END, ecuacion)
        
        self.sol_text.insert(tk.END, "\nINFORMACIÓN DE LA MATRIZ:\n")
        self.sol_text.insert(tk.END, "=" * 40 + "\n")
        self.sol_text.insert(tk.END, f"Determinante: {determinante:.6e}\n")
        self.sol_text.insert(tk.END, f"¿Es invertible? {'Sí' if abs(determinante) > 1e-12 else 'No'}\n")
        
        self.sol_text.insert(tk.END, "\nSOLUCIÓN ENCONTRADA:\n")
        self.sol_text.insert(tk.END, "=" * 40 + "\n")
        for i in range(n):
            self.sol_text.insert(tk.END, f"x{i+1} = {solucion[i]:.8f}\n")
        
        # Forzar actualización de la interfaz
        self.sol_text.update()
        self.sol_text.config(state=tk.DISABLED)
        
        print("Solución mostrada correctamente")  # DEBUG
    
    def actualizar_grafica(self, A, b, solucion):
        """Actualiza la gráfica para sistemas 2D"""
        self.ax.clear()
        if A.shape[1] == 2:
            graficar_sistema_2d(A, b, solucion, self.ax, self.canvas)
        else:
            self.ax.text(0.5, 0.5, 'Gráfica disponible\nsolo para 2 variables', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=self.ax.transAxes, fontsize=12)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()

# ==================================================
# EJECUCIÓN PRINCIPAL
# ==================================================
if __name__ == "__main__":
    plt.style.use('default')
    root = tk.Tk()
    app = MatrizInversaSolverApp(root)
    root.mainloop()