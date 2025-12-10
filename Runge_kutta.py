# Importar Tkinter solo si está disponible
try:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext
except ImportError:
    tk = None
    messagebox = None
    scrolledtext = None


class RungeKuttaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Método de Runge-Kutta (RK2 y RK4)")
        self.root.geometry("900x720")
        self.root.configure(bg="#f0f0f0")

        # --- Entradas ---
        frame_input = tk.Frame(root, bg="#e8e8e8", padx=10, pady=10, relief="ridge", bd=3)
        frame_input.pack(side="top", fill="x")

        tk.Label(frame_input, text="Función f(t, y):", bg="#e8e8e8", font=("Arial", 12)).grid(row=0, column=0, sticky="w")
        self.entry_f = tk.Entry(frame_input, width=45, font=("Arial", 12))
        self.entry_f.insert(0, "y - t**2 + 1")
        self.entry_f.grid(row=0, column=1)

        tk.Label(frame_input, text="t0:", bg="#e8e8e8", font=("Arial", 12)).grid(row=1, column=0, sticky="w")
        self.entry_t0 = tk.Entry(frame_input, width=10, font=("Arial", 12))
        self.entry_t0.insert(0, "0")
        self.entry_t0.grid(row=1, column=1, sticky="w")

        tk.Label(frame_input, text="y0:", bg="#e8e8e8", font=("Arial", 12)).grid(row=2, column=0, sticky="w")
        self.entry_y0 = tk.Entry(frame_input, width=10, font=("Arial", 12))
        self.entry_y0.insert(0, "0.5")
        self.entry_y0.grid(row=2, column=1, sticky="w")

        tk.Label(frame_input, text="h:", bg="#e8e8e8", font=("Arial", 12)).grid(row=3, column=0, sticky="w")
        self.entry_h = tk.Entry(frame_input, width=10, font=("Arial", 12))
        self.entry_h.insert(0, "0.1")
        self.entry_h.grid(row=3, column=1, sticky="w")

        tk.Label(frame_input, text="Pasos:", bg="#e8e8e8", font=("Arial", 12)).grid(row=4, column=0, sticky="w")
        self.entry_steps = tk.Entry(frame_input, width=10, font=("Arial", 12))
        self.entry_steps.insert(0, "1")
        self.entry_steps.grid(row=4, column=1, sticky="w")

        btn = tk.Button(frame_input, text="Ejecutar Runge-Kutta", font=("Arial", 12), command=self.ejecutar, bg="#a0c8ff")
        btn.grid(row=5, columnspan=2, pady=10)

        # --- SALIDA con scroll ---
        frame_output = tk.Frame(root, bg="#dcdcdc", relief="ridge", bd=3)
        frame_output.pack(fill="both", expand=True, padx=10, pady=10)

        self.text_output = scrolledtext.ScrolledText(frame_output, wrap="word", font=("Consolas", 12))
        self.text_output.pack(fill="both", expand=True)

    def ejecutar(self):
        try:
            # Leer entradas
            func_str = self.entry_f.get()
            f = lambda t, y: eval(func_str, {"t": t, "y": y})

            t0 = float(self.entry_t0.get())
            y0 = float(self.entry_y0.get())
            h = float(self.entry_h.get())
            steps = int(self.entry_steps.get())

            salida = ""

            # =================== RK2 ===================
            salida += "=========== RUNGE–KUTTA 2 (Heun) ===========\n"
            y_rk2 = y0
            t_rk2 = t0

            for n in range(steps):
                salida += f"\nPaso {n+1}:\n"

                salida += "\nPASO 1 → Calcular k1\n"
                k1 = f(t_rk2, y_rk2)
                salida += f"k1 = f({t_rk2:.4f}, {y_rk2:.4f}) = {k1:.6f}\n"

                salida += "\nPASO 2 → Calcular k2\n"
                k2 = f(t_rk2 + h, y_rk2 + h * k1)
                salida += f"k2 = f({t_rk2+h:.4f}, {y_rk2 + h*k1:.6f}) = {k2:.6f}\n"

                y_rk2 = y_rk2 + (h / 2) * (k1 + k2)
                salida += f"\nResultado del paso → y_{n+1} (RK2) = {y_rk2:.6f}\n"

                t_rk2 += h

            # =================== RK4 ===================
            salida += "\n\n=========== RUNGE–KUTTA 4 ===========\n"
            y_rk4 = y0
            t_rk4 = t0

            for n in range(steps):
                salida += f"\nPaso {n+1}:\n"

                salida += "\nPASO 1 → k1\n"
                k1 = f(t_rk4, y_rk4)
                salida += f"k1 = f({t_rk4:.4f}, {y_rk4:.4f}) = {k1:.6f}\n"

                salida += "\nPASO 2 → k2\n"
                k2 = f(t_rk4 + h/2, y_rk4 + (h/2)*k1)
                salida += f"k2 = f({t_rk4+h/2:.4f}, {y_rk4 + (h/2)*k1:.6f}) = {k2:.6f}\n"

                salida += "\nPASO 3 → k3\n"
                k3 = f(t_rk4 + h/2, y_rk4 + (h/2)*k2)
                salida += f"k3 = f({t_rk4+h/2:.4f}, {y_rk4 + (h/2)*k2:.6f}) = {k3:.6f}\n"

                salida += "\nPASO 4 → k4\n"
                k4 = f(t_rk4 + h, y_rk4 + h*k3)
                salida += f"k4 = f({t_rk4+h:.4f}, {y_rk4 + h*k3:.6f}) = {k4:.6f}\n"

                y_rk4 = y_rk4 + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
                salida += f"\nResultado del paso → y_{n+1} (RK4) = {y_rk4:.6f}\n"

                t_rk4 += h

            # ================= RESULTADOS ==================
            salida += "\n\n=========== RESULTADOS FINALES ===========\n"
            salida += f"• Resultado final RK2 = {y_rk2:.10f}\n"
            salida += f"• Resultado final RK4 = {y_rk4:.10f}\n"
            salida += "============================================\n"

            # Mostrar salida
            self.text_output.delete("1.0", tk.END)
            self.text_output.insert(tk.END, salida)

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un problema:\n{e}")


# Ejecutar ventana
if __name__ == "__main__":
    root = tk.Tk()
    app = RungeKuttaGUI(root)
    root.mainloop()
