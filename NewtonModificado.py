import numpy as np
import pandas as pd

# Intentar importar Tkinter solo si está disponible.
# En Streamlit Cloud NO existe, así que se evita el error.
try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    tk = None
    messagebox = None


def newton_modificado(f, J, x0, tol=1e-8, maxiter=50):
    x = x0.astype(float).copy()
    J0 = np.asarray(J(x0))
    history = []
    try:
        for k in range(1, maxiter + 1):
            fx = np.asarray(f(x))
            delta = np.linalg.solve(J0, -fx)
            x_new = x + delta
            err = np.linalg.norm(delta, ord=np.inf)
            resnorm = np.linalg.norm(fx, ord=2)
            history.append((k, x.copy(), delta.copy(), err, resnorm))
            x = x_new
            if err < tol and resnorm < tol:
                break
        return x, err, k, history
    except Exception as e:
        return None, str(e), 0, history

def mostrar_historial_newton(history):
    rows = []
    for k, x_old, delta, err, resnorm in history:
        row = {"Iteración": k}
        for i, val in enumerate(x_old):
            row[f"x{i}"] = val
        for i, d in enumerate(delta):
            row[f"delta{i}"] = d
        row["||delta||_inf"] = err
        row["||f(x)||_2"] = resnorm
        rows.append(row)
    return pd.DataFrame(rows)

# Interfaz gráfica
class NewtonModificadoApp:
    def __init__(self, root):
        self.root = root
        root.title("Método de Newton-Raphson Modificado")
        root.geometry("700x700")

        # Funciones
        tk.Label(root, text="Funciones f(x): (una por línea, usar x[0], x[1], ...)", font=("Arial", 10)).pack()
        self.fx_text = tk.Text(root, height=5, width=80)
        self.fx_text.pack()

        # Jacobiano
        tk.Label(root, text="Jacobiano J(x): (una fila por línea, cada derivada separada por coma)", font=("Arial", 10)).pack()
        self.jac_text = tk.Text(root, height=5, width=80)
        self.jac_text.pack()

        # Vector inicial
        tk.Label(root, text="Vector inicial (ej: 0.5, 0.5):", font=("Arial", 10)).pack()
        self.x0_entry = tk.Entry(root, width=40)
        self.x0_entry.pack()

        # Parámetros
        tk.Label(root, text="Tolerancia:", font=("Arial", 10)).pack()
        self.tol_entry = tk.Entry(root, width=20)
        self.tol_entry.insert(0, "1e-8")
        self.tol_entry.pack()

        tk.Label(root, text="Máximo de iteraciones:", font=("Arial", 10)).pack()
        self.iter_entry = tk.Entry(root, width=20)
        self.iter_entry.insert(0, "50")
        self.iter_entry.pack()

        # Botón Ejecutar
        tk.Button(root, text="Ejecutar", command=self.ejecutar).pack(pady=10)

        # Resultado
        self.resultado_label = tk.Label(root, text="", font=("Arial", 11, "bold"))
        self.resultado_label.pack()

        # Botón para mostrar procedimiento
        self.btn_proc = tk.Button(root, text="Ver procedimiento", command=self.mostrar_procedimiento, state="disabled")
        self.btn_proc.pack(pady=5)

        self.historial = None

    def ejecutar(self):
        try:
            # Funciones
            f_expr = self.fx_text.get("1.0", tk.END).strip().splitlines()
            f = lambda x: [eval(expr, {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp}) for expr in f_expr]

            # Jacobiano
            j_lines = self.jac_text.get("1.0", tk.END).strip().splitlines()
            J = lambda x: [ [eval(expr, {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp}) for expr in row.split(",")] for row in j_lines ]

            # Vector inicial
            x0 = np.fromstring(self.x0_entry.get(), sep=",")
            tol = float(self.tol_entry.get())
            maxiter = int(self.iter_entry.get())

            # Ejecutar
            sol, err, iters, hist = newton_modificado(f, J, x0, tol, maxiter)

            if sol is None:
                self.resultado_label.config(text=f"Error: {err}")
                self.btn_proc.config(state="disabled")
            else:
                self.resultado_label.config(
                    text=f"Solución: {np.round(sol, 6)}\nIteraciones: {iters}\nError final: {err:.2e}"
                )
                self.historial = hist
                self.btn_proc.config(state="normal")

        except Exception as e:
            messagebox.showerror("Error", f"Ha ocurrido un error: {e}")
            self.btn_proc.config(state="disabled")

    def mostrar_procedimiento(self):
        if not self.historial:
            return

        ventana = tk.Toplevel(self.root)
        ventana.title("Procedimiento Iterativo")
        ventana.geometry("600x400")

        text = tk.Text(ventana, font=("Courier", 10))
        text.pack(expand=True, fill="both", padx=10, pady=10)

        for k, x_old, delta, err, resnorm in self.historial:
            text.insert(tk.END, f"Iteración {k}:\n")
            text.insert(tk.END, f"x(k)     = {np.round(x_old, 6)}\n")
            text.insert(tk.END, f"delta    = {np.round(delta, 6)}\n")
            text.insert(tk.END, f"Error    = {err:.2e}, ||f(x)||₂ = {resnorm:.2e}\n")
            text.insert(tk.END, "-"*50 + "\n")
        text.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = NewtonModificadoApp(root)
    root.mainloop()
