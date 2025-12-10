import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import math

# ========= Lógica del método de punto fijo multivariable =========

def punto_fijo_multivariable(g_func, x0, tol=1e-8, maxiter=100):
    x = x0.astype(float).copy()
    history = []
    for k in range(1, maxiter + 1):
        x_new = np.asarray(g_func(x))
        err = np.linalg.norm(x_new - x, ord=np.inf)
        history.append((k, x.copy(), x_new.copy(), err))
        x = x_new
        if err < tol:
            break
    return x, err, k, history

def mostrar_historial_puntofijo(history):
    rows = []
    for k, x_old, x_new, err in history:
        row = {"Iteración": k}
        for i, val in enumerate(x_new):
            row[f"x{i}"] = round(val, 6)
        row["Error"] = round(err, 6)
        rows.append(row)
    return pd.DataFrame(rows)

# ========= Interfaz gráfica =========

class PuntoFijoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Punto Fijo Multivariable")
        self.root.geometry("900x600")

        # Inputs
        frame_inputs = tk.Frame(root)
        frame_inputs.pack(pady=10)

        tk.Label(frame_inputs, text="Funciones g(x) (una por componente):").grid(row=0, column=0, sticky="w")
        self.funciones_entry = tk.Text(frame_inputs, height=4, width=60)
        self.funciones_entry.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.funciones_entry.insert("1.0", '["cos(x[1]*x[2])", "sqrt(x[0]**2 + x[2]**2)", "sin(x[0] + x[1])"]')

        tk.Label(frame_inputs, text="Vector inicial x0 (ej: [0.5, 0.5, 0.5])").grid(row=2, column=0, sticky="w")
        self.x0_entry = tk.Entry(frame_inputs, width=50)
        self.x0_entry.grid(row=3, column=0, columnspan=2, pady=5)
        self.x0_entry.insert(0, "[0.5, 0.5, 0.5]")

        tk.Label(frame_inputs, text="Tolerancia:").grid(row=4, column=0, sticky="w")
        self.tol_entry = tk.Entry(frame_inputs, width=20)
        self.tol_entry.grid(row=5, column=0, sticky="w", padx=5)
        self.tol_entry.insert(0, "1e-8")

        tk.Label(frame_inputs, text="Máx iteraciones:").grid(row=4, column=1, sticky="w")
        self.maxiter_entry = tk.Entry(frame_inputs, width=20)
        self.maxiter_entry.grid(row=5, column=1, sticky="w", padx=5)
        self.maxiter_entry.insert(0, "100")

        tk.Button(root, text="Ejecutar", command=self.ejecutar).pack(pady=10)

        # Resultados
        self.resultado_label = tk.Label(root, text="", font=("Arial", 10))
        self.resultado_label.pack()
        self.btn_procedimiento = tk.Button(root, text="Ver procedimiento", command=self.mostrar_procedimiento, state="disabled")
        self.btn_procedimiento.pack(pady=5)


        # Tabla
        self.tree = ttk.Treeview(root)
        self.tree.pack(expand=True, fill="both", padx=10, pady=10)

    def ejecutar(self):
        try:
            funciones_texto = self.funciones_entry.get("1.0", tk.END).strip()
            x0_texto = self.x0_entry.get().strip()
            tol = float(self.tol_entry.get())
            maxiter = int(self.maxiter_entry.get())

            funciones = eval(funciones_texto, {"__builtins__": None})
            x0 = np.array(eval(x0_texto, {"__builtins__": None}))

            if len(funciones) != len(x0):
                raise ValueError("El número de funciones debe coincidir con el tamaño del vector inicial.")

            # Construir función g(x)
            def g(x):
                local_dict = {"x": x, "np": np, **math.__dict__}
                return [eval(expr, {"__builtins__": None}, local_dict) for expr in funciones]

            sol, err, iters, hist = punto_fijo_multivariable(g, x0, tol, maxiter)
            df = mostrar_historial_puntofijo(hist)

            self.resultado_label.config(
                text=f"Solución: {np.round(sol, 6)} | Iteraciones: {iters} | Error final: {err:.2e}"
            )

            self.mostrar_tabla(df)
            
            self.historial = hist
            self.btn_procedimiento.config(state="normal")  # Activar botón

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def mostrar_tabla(self, df):
        for i in self.tree.get_children():
            self.tree.delete(i)

        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"

        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        for _, row in df.iterrows():
            self.tree.insert("", "end", values=list(row))

    def mostrar_procedimiento(self):
        if not hasattr(self, 'historial'):
            return

        ventana_proc = tk.Toplevel(self.root)
        ventana_proc.title("Procedimiento Iterativo")
        ventana_proc.geometry("500x400")

        text_widget = tk.Text(ventana_proc, wrap="word", font=("Courier", 10))
        text_widget.pack(expand=True, fill="both", padx=10, pady=10)

        for k, x_old, x_new, err in self.historial:
            texto = f"Iteración {k}:\n"
            texto += f"x(k)     = {np.round(x_old, 6)}\n"
            texto += f"x(k+1)   = {np.round(x_new, 6)}\n"
            texto += f"Error    = {err:.2e}\n"
            texto += "-"*40 + "\n"
            text_widget.insert(tk.END, texto)

        text_widget.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = PuntoFijoApp(root)
    root.mainloop()
