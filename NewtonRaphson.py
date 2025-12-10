import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

# ==================================================
# MÉTODO DE NEWTON-RAPHSON
# ==================================================


def newton_raphson_iterations(f, df, x0, tol=1e-8, maxit=100):
    """
    Implementa el método de Newton-Raphson para encontrar raíces de una función.
    
    Parámetros:
    f: función a evaluar
    df: derivada de la función
    x0: valor inicial
    tol: tolerancia para el criterio de parada
    maxit: número máximo de iteraciones
    
    Retorna:
    rows: lista de tuplas con los datos de cada iteración
    """
    rows = []
    x = x0
    
    for k in range(1, maxit + 1):
        fx = f(x)
        dfx = df(x)
        
        # Evitar división por cero
        if abs(dfx) < 1e-12:
            raise ValueError("Derivada cercana a cero. El método puede divergir.")
        
        # Calcular nuevo punto
        x_new = x - fx / dfx
        error = abs(x_new - x)
        
        rows.append((k, x, fx, dfx, x_new, error))
        
        # Verificar criterio de parada
        if abs(fx) < tol or error < tol:
            break
            
        x = x_new
            
    return rows

# ==================================================
# VISUALIZACIÓN GRÁFICA
# ==================================================
def graficar_newton(f, df, iteraciones, ax=None, canvas=None):
    """
    Genera gráficas del método de Newton-Raphson mostrando la función y las iteraciones.
    
    Parámetros:
    f: función a graficar
    df: derivada de la función
    iteraciones: lista con datos de cada iteración
    ax: eje para plotting
    canvas: canvas de tkinter
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Encontrar rango de graficación
    x_vals = []
    for _, x, _, _, x_new, _ in iteraciones:
        x_vals.extend([x, x_new])
    
    x_min, x_max = min(x_vals), max(x_vals)
    margin = 0.2 * (x_max - x_min) if x_max != x_min else 2
    x_plot = np.linspace(x_min - margin, x_max + margin, 1000)
    y_plot = [f(x) for x in x_plot]
    
    # Graficar la función
    ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.7)
    
    # Graficar iteraciones (máximo las primeras 5 para claridad)
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (k, x, fx, dfx, x_new, error) in enumerate(iteraciones[:5]):
        # Punto actual
        ax.plot(x, fx, color=colors[i % len(colors)], 
                marker=markers[i % len(markers)], markersize=8, 
                label=f'Iteración {k}')
        
        # Línea tangente
        x_tangent = np.linspace(x - 0.5, x + 0.5, 100)
        y_tangent = fx + dfx * (x_tangent - x)
        ax.plot(x_tangent, y_tangent, color=colors[i % len(colors)], 
                linestyle='--', alpha=0.7)
        
        # Proyección al eje x
        ax.plot([x, x_new], [fx, 0], color=colors[i % len(colors)], 
                linestyle=':', alpha=0.5)
    
    # Graficar la raíz final
    if iteraciones:
        ultima_iter = iteraciones[-1]
        x_final = ultima_iter[4]  # x_new de la última iteración
        ax.plot(x_final, 0, 'ro', markersize=10, label='Raíz aproximada')
    
    ax.set_xlabel("x", fontsize=12, fontweight='bold')
    ax.set_ylabel("f(x)", fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title("Método de Newton-Raphson - Proceso Iterativo", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if canvas is not None:
        canvas.draw()
    else:
        plt.show()
    
    return ax

# ==================================================
# INTERFAZ GRÁFICA
# ==================================================
class NewtonRaphsonSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ Método de Newton-Raphson - Buscador de Raíces")
        self.root.geometry("900x750")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.funciones_predefinidas = {
            "x² - 4": {
                "funcion": "x**2 - 4",
                "derivada": "2*x"
            },
            "x³ - 2x - 5": {
                "funcion": "x**3 - 2*x - 5", 
                "derivada": "3*x**2 - 2"
            },
            "cos(x) - x": {
                "funcion": "math.cos(x) - x",
                "derivada": "-math.sin(x) - 1"
            },
            "eˣ - 2": {
                "funcion": "math.exp(x) - 2",
                "derivada": "math.exp(x)"
            },
            "sin(x)": {
                "funcion": "math.sin(x)",
                "derivada": "math.cos(x)"
            },
            "x³ - x - 1": {
                "funcion": "x**3 - x - 1",
                "derivada": "3*x**2 - 1"
            }
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(main_frame, 
                               text="MÉTODO DE NEWTON-RAPHSON", 
                               font=('Arial', 16, 'bold'),
                               foreground='#2c3e50')
        title_label.pack(pady=(0, 10))
        
        # Descripción
        desc_text = "Encuentra raíces de funciones usando el método de Newton-Raphson\nConvergencia rápida mediante el uso de derivadas"
        desc_label = ttk.Label(main_frame, 
                              text=desc_text, 
                              font=('Arial', 10),
                              foreground='#7f8c8d',
                              justify=tk.CENTER)
        desc_label.pack(pady=(0, 20))
        
        # Frame de controles superiores
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Selección de función
        ttk.Label(control_frame, text="Función:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=(0, 10), sticky='w')
        
        self.funcion_var = tk.StringVar()
        self.combo_funciones = ttk.Combobox(control_frame, 
                                           textvariable=self.funcion_var,
                                           values=list(self.funciones_predefinidas.keys()),
                                           state="readonly",
                                           width=20)
        self.combo_funciones.grid(row=0, column=1, padx=(0, 20))
        self.combo_funciones.set("x² - 4")
        self.combo_funciones.bind('<<ComboboxSelected>>', self.actualizar_funcion)
        
        # Frame para entradas de funciones
        func_frame = ttk.LabelFrame(main_frame, text="Definición de Función", padding="10")
        func_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Función f(x)
        ttk.Label(func_frame, text="f(x) =", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=(0, 10))
        
        self.entry_funcion = ttk.Entry(func_frame, width=30, font=('Arial', 10))
        self.entry_funcion.grid(row=0, column=1, padx=(0, 20))
        self.entry_funcion.insert(0, "x**2 - 4")
        
        # Derivada f'(x)
        ttk.Label(func_frame, text="f'(x) =", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=(0, 10))
        
        self.entry_derivada = ttk.Entry(func_frame, width=30, font=('Arial', 10))
        self.entry_derivada.grid(row=0, column=3, padx=(0, 10))
        self.entry_derivada.insert(0, "2*x")
        
        # Frame para parámetros
        param_frame = ttk.LabelFrame(main_frame, text="Parámetros del Método", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Valor inicial
        ttk.Label(param_frame, text="Valor inicial x₀:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=(0, 10))
        self.entry_x0 = ttk.Entry(param_frame, width=15)
        self.entry_x0.grid(row=0, column=1, padx=(0, 20))
        self.entry_x0.insert(0, "2.0")
        
        # Tolerancia
        ttk.Label(param_frame, text="Tolerancia:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=(0, 10))
        self.entry_tol = ttk.Entry(param_frame, width=15)
        self.entry_tol.grid(row=0, column=3, padx=(0, 20))
        self.entry_tol.insert(0, "1e-8")
        
        # Máximo iteraciones
        ttk.Label(param_frame, text="Máx iteraciones:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=4, padx=(0, 10))
        self.entry_maxit = ttk.Entry(param_frame, width=15)
        self.entry_maxit.grid(row=0, column=5, padx=(0, 10))
        self.entry_maxit.insert(0, "100")
        
        # Botones
        botones_frame = ttk.Frame(main_frame)
        botones_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.btn_resolver = ttk.Button(botones_frame, 
                                      text="⚡ Resolver con Newton-Raphson", 
                                      command=self.resolver)
        self.btn_resolver.pack(side=tk.LEFT, padx=(0, 10))
        
        self.btn_graficar = ttk.Button(botones_frame, 
                                      text="📊 Graficar Proceso", 
                                      command=self.graficar_proceso)
        self.btn_graficar.pack(side=tk.LEFT, padx=(0, 10))
        
        self.btn_graficar_funcion = ttk.Button(botones_frame, 
                                             text="📈 Graficar Función", 
                                             command=self.graficar_funcion)
        self.btn_graficar_funcion.pack(side=tk.LEFT)
        
        # Frame para resultados y gráfica
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Resultados en tabla
        self.result_frame = ttk.LabelFrame(results_frame, 
                                          text="Iteraciones del Método", 
                                          padding="10")
        self.result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Crear tabla de resultados
        columns = ('Iter', 'xₙ', 'f(xₙ)', "f'(xₙ)", 'xₙ₊₁', 'Error')
        self.tree = ttk.Treeview(self.result_frame, columns=columns, show='headings', height=15)
        
        # Definir columnas
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        
        # Scrollbar para la tabla
        scrollbar = ttk.Scrollbar(self.result_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Gráfica
        self.graph_frame = ttk.LabelFrame(results_frame, 
                                         text="Visualización del Método", 
                                         padding="10")
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas para matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mensaje inicial en gráfica
        self.ax.text(0.5, 0.5, 'Ingresa una función\n y su derivada\npara comenzar', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, fontsize=12, style='italic')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
        
        # Actualizar con la función por defecto
        self.actualizar_funcion()
    
    def actualizar_funcion(self, event=None):
        """Actualiza los campos de función y derivada cuando se selecciona una predefinida"""
        funcion_seleccionada = self.funcion_var.get()
        if funcion_seleccionada in self.funciones_predefinidas:
            datos = self.funciones_predefinidas[funcion_seleccionada]
            self.entry_funcion.delete(0, tk.END)
            self.entry_funcion.insert(0, datos["funcion"])
            self.entry_derivada.delete(0, tk.END)
            self.entry_derivada.insert(0, datos["derivada"])
    
    def obtener_funciones(self):
        """Obtiene la función y su derivada desde la interfaz"""
        try:
            # Procesar función f(x)
            expr_func = self.entry_funcion.get()
            expr_func = expr_func.replace('^', '**').replace('e', 'math.e')
            expr_func = expr_func.replace('π', 'math.pi')
            
            f = eval(f"lambda x: {expr_func}", {
                'math': math, 'exp': math.exp, 'sin': math.sin, 
                'cos': math.cos, 'tan': math.tan, 'log': math.log,
                'log10': math.log10, 'sqrt': math.sqrt
            })
            
            # Procesar derivada f'(x)
            expr_deriv = self.entry_derivada.get()
            expr_deriv = expr_deriv.replace('^', '**').replace('e', 'math.e')
            expr_deriv = expr_deriv.replace('π', 'math.pi')
            
            df = eval(f"lambda x: {expr_deriv}", {
                'math': math, 'exp': math.exp, 'sin': math.sin, 
                'cos': math.cos, 'tan': math.tan, 'log': math.log,
                'log10': math.log10, 'sqrt': math.sqrt
            })
            
            return f, df
            
        except Exception as e:
            raise ValueError(f"Error en las funciones: {e}")
    
    def resolver(self):
        try:
            # Obtener parámetros
            f, df = self.obtener_funciones()
            x0 = float(self.entry_x0.get())
            tol = float(self.entry_tol.get())
            maxit = int(self.entry_maxit.get())
            
            # Ejecutar método de Newton-Raphson
            iteraciones = newton_raphson_iterations(f, df, x0, tol, maxit)
            
            # Mostrar resultados en tabla
            self.mostrar_resultados(iteraciones)
            
            # Actualizar gráfica
            self.actualizar_grafica(f, df, iteraciones)
            
        except ValueError as e:
            messagebox.showerror("Error", f"Error en los datos: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado: {e}")
    
    def mostrar_resultados(self, iteraciones):
        """Muestra los resultados en la tabla"""
        # Limpiar tabla anterior
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Llenar con nuevos datos
        for k, x, fx, dfx, x_new, error in iteraciones:
            self.tree.insert('', 'end', values=(
                k, 
                f"{x:.8f}", 
                f"{fx:.2e}", 
                f"{dfx:.2e}", 
                f"{x_new:.8f}", 
                f"{error:.2e}"
            ))
        
        # Mostrar resumen final
        if iteraciones:
            ultima_iter = iteraciones[-1]
            k, x_final, fx_final, dfx_final, x_new_final, error_final = ultima_iter
            
            resumen = f"RESUMEN DEL MÉTODO NEWTON-RAPHSON:\n"
            resumen += "="*45 + "\n"
            resumen += f"Raíz encontrada: {x_new_final:.10f}\n"
            resumen += f"f(raíz) = {fx_final:.2e}\n"
            resumen += f"Error final: {error_final:.2e}\n"
            resumen += f"Iteraciones realizadas: {k}\n"
            resumen += f"Valor inicial: {iteraciones[0][1]:.4f}"
            
            messagebox.showinfo("Resultado Final", resumen)
    
    def actualizar_grafica(self, f, df, iteraciones):
        """Actualiza la gráfica con el proceso de Newton-Raphson"""
        self.ax.clear()
        graficar_newton(f, df, iteraciones, self.ax, self.canvas)
    
    def graficar_proceso(self):
        """Grafica el proceso iterativo completo"""
        try:
            f, df = self.obtener_funciones()
            x0 = float(self.entry_x0.get())
            tol = float(self.entry_tol.get())
            maxit = int(self.entry_maxit.get())
            
            iteraciones = newton_raphson_iterations(f, df, x0, tol, maxit)
            self.actualizar_grafica(f, df, iteraciones)
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo graficar: {e}")
    
    def graficar_funcion(self):
        """Grafica solo la función y su derivada"""
        try:
            f, df = self.obtener_funciones()
            x0 = float(self.entry_x0.get())
            
            self.ax.clear()
            
            # Determinar rango de graficación
            x_min, x_max = x0 - 3, x0 + 3
            x_vals = np.linspace(x_min, x_max, 1000)
            y_vals_f = [f(x) for x in x_vals]
            y_vals_df = [df(x) for x in x_vals]
            
            # Graficar función y derivada
            self.ax.plot(x_vals, y_vals_f, 'b-', linewidth=2, label='f(x)')
            self.ax.plot(x_vals, y_vals_df, 'r-', linewidth=2, label="f'(x)")
            self.ax.axhline(0, color='black', linewidth=0.5, alpha=0.7)
            self.ax.axvline(0, color='black', linewidth=0.5, alpha=0.7)
            
            # Marcar valor inicial
            f_x0 = f(x0)
            self.ax.plot(x0, f_x0, 'go', markersize=8, label=f'x₀ = {x0:.2f}')
            
            self.ax.set_xlabel("x", fontsize=12, fontweight='bold')
            self.ax.set_ylabel("y", fontsize=12, fontweight='bold')
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_title("Función y su Derivada", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo graficar: {e}")

# ==================================================
# EJECUCIÓN PRINCIPAL
# ==================================================
if __name__ == "__main__":
    # Configurar estilo de matplotlib
    plt.style.use('default')
    
    # Crear y ejecutar aplicación
    root = tk.Tk()
    app = NewtonRaphsonSolverApp(root)
    root.mainloop()