import numpy as np
import matplotlib.pyplot as plt
import math

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
# MÉTODO DE BISECCIÓN
# ==================================================
"""
DESCRIPCIÓN:
El método de Bisección es un algoritmo de búsqueda de raíces que funciona dividiendo
repetidamente un intervalo a la mitad y seleccionando el subintervalo que contiene la raíz.
"""

def bisection_iterations(f, a, b, tol=1e-8, maxit=100):
    """
    Implementa el método de bisección para encontrar raíces de una función.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0: 
        raise ValueError("No hay cambio de signo en el intervalo [a,b]")
    
    rows = []
    for k in range(1, maxit + 1):
        r = (a + b) / 2.0
        fr = f(r)
        error = (b - a) / 2.0
        rows.append((k, a, b, r, fr, error))
        
        if abs(fr) < tol or error < tol: 
            break
            
        if fa * fr < 0:
            b, fb = r, fr
        else:
            a, fa = r, fr
            
    return rows

# ==================================================
# VISUALIZACIÓN GRÁFICA
# ==================================================
def graficar_biseccion(f, a, b, iteraciones, ax=None, canvas=None):
    """
    Genera gráficas del método de bisección mostrando la función y las iteraciones.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Crear puntos para la gráfica
    x_vals = np.linspace(a - 0.1 * (b - a), b + 0.1 * (b - a), 1000)
    y_vals = [f(x) for x in x_vals]
    
    # Graficar la función
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.7)
    
    # Graficar intervalo inicial
    ax.axvline(a, color='red', linestyle='--', alpha=0.7, label='Intervalo inicial')
    ax.axvline(b, color='red', linestyle='--', alpha=0.7)
    
    # Graficar puntos de las iteraciones
    iteraciones_para_grafica = min(5, len(iteraciones))
    colors = ['green', 'orange', 'purple', 'brown', 'pink']
    
    for i in range(iteraciones_para_grafica):
        k, a_i, b_i, r, fr, error = iteraciones[i]
        ax.plot(r, fr, 'o', color=colors[i % len(colors)], 
                markersize=8, label=f'Iteración {k}')
        ax.axvline(r, color=colors[i % len(colors)], linestyle=':', alpha=0.5)
    
    # Graficar la raíz final
    if iteraciones:
        k_final, a_final, b_final, r_final, fr_final, error_final = iteraciones[-1]
        ax.plot(r_final, fr_final, 'ro', markersize=10, label='Raíz aproximada')
    
    ax.set_xlabel("x", fontsize=12, fontweight='bold')
    ax.set_ylabel("f(x)", fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title("Método de Bisección - Proceso Iterativo", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if canvas is not None:
        canvas.draw()
    else:
        plt.show()
    
    return ax

# ==================================================
# INTERFAZ GRÁFICA
# ==================================================
class BisectionSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 Método de Bisección - Buscador de Raíces")
        self.root.geometry("900x750")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.funciones_predefinidas = {
            "x² - 4": "x**2 - 4",
            "x³ - 2x - 5": "x**3 - 2*x - 5", 
            "cos(x) - x": "math.cos(x) - x",
            "eˣ - 2": "math.exp(x) - 2",
            "sin(x)": "math.sin(x)",
            "x³ - x - 1": "x**3 - x - 1"
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(main_frame, 
                               text="MÉTODO DE BISECCIÓN", 
                               font=('Arial', 16, 'bold'),
                               foreground='#2c3e50')
        title_label.pack(pady=(0, 10))
        
        # Descripción
        desc_text = "Encuentra raíces de funciones usando el método de bisección\nBasado en el Teorema del Valor Intermedio"
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
        self.combo_funciones.bind('<<ComboboxSelected>>', self.actualizar_funcion_desde_combo)
        
        # Entrada personalizada
        ttk.Label(control_frame, text="Función personalizada f(x):", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=(0, 10))
        
        self.entry_funcion_personalizada = ttk.Entry(control_frame, width=25)
        self.entry_funcion_personalizada.grid(row=0, column=3, padx=(0, 20))
        self.entry_funcion_personalizada.insert(0, "x**2 - 4")
        
        # Frame para intervalo
        intervalo_frame = ttk.Frame(main_frame)
        intervalo_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(intervalo_frame, text="Intervalo [a, b]:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Label(intervalo_frame, text="a =").grid(row=0, column=1, padx=(0, 5))
        self.entry_a = ttk.Entry(intervalo_frame, width=10)
        self.entry_a.grid(row=0, column=2, padx=(0, 15))
        self.entry_a.insert(0, "1")
        
        ttk.Label(intervalo_frame, text="b =").grid(row=0, column=3, padx=(0, 5))
        self.entry_b = ttk.Entry(intervalo_frame, width=10)
        self.entry_b.grid(row=0, column=4, padx=(0, 15))
        self.entry_b.insert(0, "3")
        
        # Frame para parámetros
        parametros_frame = ttk.Frame(main_frame)
        parametros_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(parametros_frame, text="Tolerancia:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=(0, 10))
        self.entry_tol = ttk.Entry(parametros_frame, width=10)
        self.entry_tol.grid(row=0, column=1, padx=(0, 20))
        self.entry_tol.insert(0, "1e-8")
        
        ttk.Label(parametros_frame, text="Máx iteraciones:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=(0, 10))
        self.entry_maxit = ttk.Entry(parametros_frame, width=10)
        self.entry_maxit.grid(row=0, column=3, padx=(0, 20))
        self.entry_maxit.insert(0, "100")
        
        # Botones
        botones_frame = ttk.Frame(main_frame)
        botones_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.btn_resolver = ttk.Button(botones_frame, 
                                      text="⚡ Resolver con Bisección", 
                                      command=self.resolver)
        self.btn_resolver.pack(side=tk.LEFT, padx=(0, 10))
        
        self.btn_graficar = ttk.Button(botones_frame, 
                                      text="📊 Graficar Función", 
                                      command=self.graficar_funcion)
        self.btn_graficar.pack(side=tk.LEFT)
        
        # Frame para resultados y gráfica
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Resultados en tabla
        self.result_frame = ttk.LabelFrame(results_frame, 
                                          text="Iteraciones del Método", 
                                          padding="10")
        self.result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Crear tabla de resultados
        columns = ('Iter', 'a', 'b', 'Raíz (r)', 'f(r)', 'Error')
        self.tree = ttk.Treeview(self.result_frame, columns=columns, show='headings', height=15)
        
        # Definir columnas
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80)
        
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
        self.ax.text(0.5, 0.5, 'Ingresa una función\ny un intervalo\npara comenzar', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, fontsize=12, style='italic')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
    
    def actualizar_funcion_desde_combo(self, event=None):
        """Actualiza la entrada personalizada cuando se selecciona una función predefinida"""
        funcion_seleccionada = self.funcion_var.get()
        if funcion_seleccionada in self.funciones_predefinidas:
            self.entry_funcion_personalizada.delete(0, tk.END)
            self.entry_funcion_personalizada.insert(0, self.funciones_predefinidas[funcion_seleccionada])
    
    def obtener_funcion(self):
        """Obtiene la función desde la interfaz - CORREGIDO"""
        try:
            # Obtener la expresión de la entrada personalizada
            expr = self.entry_funcion_personalizada.get().strip()
            
            if not expr:
                raise ValueError("La función no puede estar vacía")
            
            print(f"Expresión original: {expr}")  # DEBUG
            
            # Reemplazar notación matemática común
            expr = expr.replace('^', '**')
            expr = expr.replace('²', '**2')
            expr = expr.replace('³', '**3')
            expr = expr.replace('e', 'math.e')
            expr = expr.replace('π', 'math.pi')
            expr = expr.replace('pi', 'math.pi')
            
            print(f"Expresión procesada: {expr}")  # DEBUG
            
            # Crear un entorno seguro para eval
            safe_dict = {
                'math': math,
                'exp': math.exp,
                'sin': math.sin, 
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'log10': math.log10,
                'sqrt': math.sqrt,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'sinh': math.sinh,
                'cosh': math.cosh,
                'tanh': math.tanh
            }
            
            # Verificar que la expresión es válida
            try:
                # Probar la expresión con un valor de prueba
                test_x = 1.0
                test_result = eval(expr, {'x': test_x, **safe_dict})
                print(f"Prueba con x={test_x}: {test_result}")  # DEBUG
            except Exception as e:
                raise ValueError(f"Expresión inválida: {e}")
            
            # Crear la función lambda
            def f(x):
                return eval(expr, {'x': x, **safe_dict})
            
            return f
            
        except Exception as e:
            raise ValueError(f"Error en la función: {str(e)}")
    
    def resolver(self):
        try:
            # Obtener parámetros
            f = self.obtener_funcion()
            a = float(self.entry_a.get())
            b = float(self.entry_b.get())
            tol = float(self.entry_tol.get())
            maxit = int(self.entry_maxit.get())
            
            # Validar intervalo
            if a >= b:
                messagebox.showerror("Error", "El valor de 'a' debe ser menor que 'b'")
                return
            
            # Ejecutar método de bisección
            iteraciones = bisection_iterations(f, a, b, tol, maxit)
            
            # Mostrar resultados en tabla
            self.mostrar_resultados(iteraciones)
            
            # Actualizar gráfica
            self.actualizar_grafica(f, a, b, iteraciones)
            
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
        for k, a, b, r, fr, error in iteraciones:
            self.tree.insert('', 'end', values=(
                k, 
                f"{a:.6f}", 
                f"{b:.6f}", 
                f"{r:.8f}", 
                f"{fr:.2e}", 
                f"{error:.2e}"
            ))
        
        # Mostrar resumen final
        if iteraciones:
            ultima_iter = iteraciones[-1]
            k, a_final, b_final, r_final, fr_final, error_final = ultima_iter
            
            # Crear ventana de resumen
            resumen = f"RESUMEN DE LA BÚSQUEDA:\n"
            resumen += "="*40 + "\n"
            resumen += f"Raíz encontrada: {r_final:.10f}\n"
            resumen += f"f(raíz) = {fr_final:.2e}\n"
            resumen += f"Error estimado: {error_final:.2e}\n"
            resumen += f"Iteraciones realizadas: {k}\n"
            resumen += f"Intervalo final: [{a_final:.6f}, {b_final:.6f}]"
            
            messagebox.showinfo("Resultado Final", resumen)
    
    def actualizar_grafica(self, f, a, b, iteraciones):
        """Actualiza la gráfica con el proceso de bisección"""
        self.ax.clear()
        graficar_biseccion(f, a, b, iteraciones, self.ax, self.canvas)
    
    def graficar_funcion(self):
        """Grafica solo la función sin el proceso iterativo"""
        try:
            f = self.obtener_funcion()
            a = float(self.entry_a.get())
            b = float(self.entry_b.get())
            
            self.ax.clear()
            x_vals = np.linspace(a - 0.1 * (b - a), b + 0.1 * (b - a), 1000)
            y_vals = [f(x) for x in x_vals]
            
            self.ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
            self.ax.axhline(0, color='black', linewidth=0.5, alpha=0.7)
            self.ax.axvline(0, color='black', linewidth=0.5, alpha=0.7)
            self.ax.axvline(a, color='red', linestyle='--', alpha=0.7, label='Límite a')
            self.ax.axvline(b, color='green', linestyle='--', alpha=0.7, label='Límite b')
            
            # Evaluar en los extremos
            fa, fb = f(a), f(b)
            self.ax.plot(a, fa, 'ro', markersize=8, label=f'f(a) = {fa:.2f}')
            self.ax.plot(b, fb, 'go', markersize=8, label=f'f(b) = {fb:.2f}')
            
            self.ax.set_xlabel("x", fontsize=12, fontweight='bold')
            self.ax.set_ylabel("f(x)", fontsize=12, fontweight='bold')
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_title("Gráfica de la Función", fontsize=14, fontweight='bold')
            
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
    app = BisectionSolverApp(root)
    root.mainloop()