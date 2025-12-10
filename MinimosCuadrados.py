import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

"""
Interfaz para Ajuste por Mínimos Cuadrados (regresión polinomial).

Permite:
- Ingresar manualmente puntos X e Y (comas)
- Seleccionar el grado del polinomio
- Cargar datos desde archivo (dos columnas x y)
- Mostrar coeficientes, tabla de valores y errores, y gráfica
"""

def fit_least_squares(xs, ys, degree):
	coefs = np.polyfit(xs, ys, degree)
	p = np.poly1d(coefs)
	ys_pred = p(xs)
	errors = ys - ys_pred
	rmse = np.sqrt(np.mean(errors ** 2))
	sse = np.sum(errors ** 2)
	return coefs, p, ys_pred, errors, rmse, sse

def plot_fit(xs, ys, poly, ax, canvas, x_eval=None, y_eval=None):
	ax.clear()
	x_min, x_max = min(xs) - 1, max(xs) + 1
	x_plot = np.linspace(x_min, x_max, 300)
	y_plot = poly(x_plot)

	ax.plot(x_plot, y_plot, '-b', label='Ajuste polinomial')
	ax.scatter(xs, ys, color='red', label='Datos')
	if x_eval is not None and y_eval is not None:
		ax.plot(x_eval, y_eval, 'gs', label=f'P({x_eval})={y_eval:.4f}')

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title('Ajuste por Mínimos Cuadrados')
	ax.grid(True, alpha=0.3)
	ax.legend()
	canvas.draw()


class MinimosCuadradosApp:
	def __init__(self, root):
		self.root = root
		self.root.title('📈 Ajuste por Mínimos Cuadrados')
		self.root.geometry('1000x760')

		frame_top = tk.Frame(root, bg='#2c3e50')
		frame_top.pack(fill='x')
		tk.Label(frame_top, text='MÍNIMOS CUADRADOS - AJUSTE POLINOMIAL', fg='white', bg='#2c3e50', font=('Arial', 14, 'bold')).pack(padx=8, pady=8)

		input_frame = ttk.LabelFrame(root, text='Entradas', padding=8)
		input_frame.pack(fill='x', padx=10, pady=6)

		tk.Label(input_frame, text='X (coma separada):').grid(row=0, column=0, sticky='w')
		self.entry_x = tk.Entry(input_frame, width=60)
		self.entry_x.insert(0, '1,2,3,4,5')
		self.entry_x.grid(row=0, column=1, padx=6, pady=4)

		tk.Label(input_frame, text='Y (coma separada):').grid(row=1, column=0, sticky='w')
		self.entry_y = tk.Entry(input_frame, width=60)
		self.entry_y.insert(0, '2.1,3.9,6.2,7.8,10.1')
		self.entry_y.grid(row=1, column=1, padx=6, pady=4)

		tk.Label(input_frame, text='Grado del polinomio:').grid(row=2, column=0, sticky='w')
		self.entry_deg = tk.Spinbox(input_frame, from_=1, to=10, width=5)
		self.entry_deg.delete(0, tk.END)
		self.entry_deg.insert(0, '1')
		self.entry_deg.grid(row=2, column=1, sticky='w', padx=6, pady=4)

		btns = tk.Frame(root)
		btns.pack(fill='x', padx=10, pady=4)
		tk.Button(btns, text='▶ Calcular', bg='#27ae60', fg='white', command=self.calcular).pack(side='left', padx=6)
		tk.Button(btns, text='📂 Cargar archivo', command=self.cargar_archivo).pack(side='left', padx=6)
		tk.Button(btns, text='🔄 Limpiar', bg='#e74c3c', fg='white', command=self.limpiar).pack(side='left', padx=6)

		res_frame = ttk.LabelFrame(root, text='Resumen', padding=8)
		res_frame.pack(fill='x', padx=10, pady=6)
		self.label_res = tk.Label(res_frame, text='Estado: listo', anchor='w')
		self.label_res.pack(fill='x')

		table_frame = ttk.LabelFrame(root, text='Tabla de resultados', padding=8)
		table_frame.pack(fill='both', expand=True, padx=10, pady=6)

		cols = ('i','x','y','y_pred','error')
		self.tree = ttk.Treeview(table_frame, columns=cols, show='headings')
		for c in cols:
			self.tree.heading(c, text=c)
			self.tree.column(c, anchor='center')
		self.tree.pack(side='left', fill='both', expand=True)
		ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview).pack(side='right', fill='y')

		coef_frame = ttk.LabelFrame(root, text='Coeficientes', padding=8)
		coef_frame.pack(fill='x', padx=10, pady=6)
		self.label_coef = tk.Label(coef_frame, text='Coeficientes: []', anchor='w', justify='left', wraplength=900)
		self.label_coef.pack(fill='x')

		# Gráfica
		self.fig, self.ax = plt.subplots(figsize=(9,3))
		self.canvas = FigureCanvasTkAgg(self.fig, master=root)
		self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=6)

		# mostrar ejemplo inicial
		self.calcular_ejemplo()

	def parse_inputs(self):
		xs_text = self.entry_x.get().strip()
		ys_text = self.entry_y.get().strip()
		deg = int(self.entry_deg.get())
		try:
			xs = np.array([float(s) for s in xs_text.split(',')])
			ys = np.array([float(s) for s in ys_text.split(',')])
		except Exception:
			raise ValueError('Formato inválido en X o Y. Usa números separados por comas.')
		if xs.size != ys.size:
			raise ValueError('X e Y deben tener la misma cantidad de elementos')
		return xs, ys, deg

	def calcular(self):
		try:
			xs, ys, deg = self.parse_inputs()
			coefs, poly, ys_pred, errors, rmse, sse = fit_least_squares(xs, ys, deg)

			# actualizar resumen
			coef_str = ', '.join([f'{c:.6f}' for c in coefs])
			self.label_coef.config(text=f'Coeficientes: [{coef_str}]')
			self.label_res.config(text=f'RMSE: {rmse:.6f} | SSE: {sse:.6f}', fg='#27ae60')

			# tabla
			for it in self.tree.get_children():
				self.tree.delete(it)
			for i, (xi, yi, ypi, err) in enumerate(zip(xs, ys, ys_pred, errors), start=1):
				self.tree.insert('', 'end', values=(i, f'{xi:.4f}', f'{yi:.4f}', f'{ypi:.4f}', f'{err:.4f}'))

			# graficar
			plot_fit(xs, ys, poly, self.ax, self.canvas)

		except Exception as e:
			messagebox.showerror('Error', str(e))
			self.label_res.config(text='Error', fg='#e74c3c')

	def calcular_ejemplo(self):
		# calcular con los datos de entrada por defecto
		try:
			self.calcular()
		except Exception:
			pass

	def limpiar(self):
		self.entry_x.delete(0, tk.END)
		self.entry_x.insert(0, '1,2,3,4,5')
		self.entry_y.delete(0, tk.END)
		self.entry_y.insert(0, '2.1,3.9,6.2,7.8,10.1')
		self.entry_deg.delete(0, tk.END)
		self.entry_deg.insert(0, '1')
		self.label_res.config(text='Estado: listo', fg='black')
		self.label_coef.config(text='Coeficientes: []')
		for it in self.tree.get_children():
			self.tree.delete(it)
		self.ax.clear()
		self.canvas.draw()

	def cargar_archivo(self):
		path = filedialog.askopenfilename(title='Abrir archivo', filetypes=[('Text','*.txt'), ('CSV','*.csv'), ('All','*.*')])
		if not path:
			return
		xs = []
		ys = []
		with open(path, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				parts = line.replace(',', ' ').split()
				if len(parts) >= 2:
					xs.append(float(parts[0])); ys.append(float(parts[1]))
		self.entry_x.delete(0, tk.END); self.entry_x.insert(0, ','.join(map(str, xs)))
		self.entry_y.delete(0, tk.END); self.entry_y.insert(0, ','.join(map(str, ys)))


if __name__ == '__main__':
	root = tk.Tk()
	app = MinimosCuadradosApp(root)
	root.mainloop()
