import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
from collections import Counter
import csv

# ==========================================
# 1. CLASES MATEMÁTICAS (SIN LIBRERÍAS ML)
# ==========================================

class RegresionLinealManual:
    def __init__(self):
        self.m = 0.0
        self.b = 0.0

    def entrenar(self, X, Y):
        n = len(X)
        sum_x = sum(X)
        sum_y = sum(Y)
        sum_xy = sum(x * y for x, y in zip(X, Y))
        sum_x2 = sum(x ** 2 for x in X)

        # Manejo de error si todos los puntos X son iguales (evita división por cero)
        if (n * sum_x2 - sum_x ** 2) == 0:
            self.m = 0
        else:
            self.m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
        self.b = (sum_y - self.m * sum_x) / n

    def predecir(self, x):
        return self.m * x + self.b

    def calcular_mse(self, X, Y):
        n = len(X)
        error_cuadratico_total = sum((y - self.predecir(x)) ** 2 for x, y in zip(X, Y))
        return error_cuadratico_total / n

class KNNManual:
    def __init__(self, k=3):
        self.k = k
        self.X_entrenamiento = []
        self.Y_entrenamiento = []

    def entrenar(self, X, y):
        self.X_entrenamiento = X
        self.Y_entrenamiento = y

    def distancia_euclidiana(self, punto1, punto2):
        return math.sqrt((punto2[0] - punto1[0])**2 + (punto2[1] - punto1[1])**2)

    def predecir(self, nuevo_punto):
        distancias = []
        for i in range(len(self.X_entrenamiento)):
            dist = self.distancia_euclidiana(self.X_entrenamiento[i], nuevo_punto)
            distancias.append((dist, self.Y_entrenamiento[i]))

        distancias.sort(key=lambda x: x[0])
        k_vecinos = distancias[:self.k]
        clases_vecinos = [vecino[1] for vecino in k_vecinos]
        
        votos = Counter(clases_vecinos)
        clase_ganadora = votos.most_common(1)[0][0]
        return clase_ganadora

# ==========================================
# 2. INTERFAZ GRÁFICA Y LÓGICA DE CONEXIÓN
# ==========================================

class AplicacionML:
    def __init__(self, root):
        self.root = root
        self.root.title("Modelos de IA - Proyecto Final")
        self.root.geometry("900x750")

        self.tipo_modelo = None  # Guardará "Regresion" o "KNN"
        self.X_data = []
        self.Y_data = []
        self.clases_data = []

        # --- MÓDULO DE CARGA ---
        frame_carga = tk.LabelFrame(self.root, text="1. Módulo de Carga", padx=10, pady=10)
        frame_carga.pack(fill="x", padx=10, pady=5)

        self.btn_csv = tk.Button(frame_carga, text="Subir Archivo CSV", command=self.cargar_csv)
        self.btn_csv.grid(row=0, column=0, pady=5, sticky="w")
        
        self.lbl_archivo = tk.Label(frame_carga, text="Ningún archivo cargado", fg="gray")
        self.lbl_archivo.grid(row=0, column=1, padx=5, sticky="w")

        tk.Label(frame_carga, text="Ingreso Manual\n(Ej: '5' para Regresión o '2.5, 3.1' para KNN):").grid(row=1, column=0, sticky="e")
        self.entrada_manual = tk.Entry(frame_carga, width=15)
        self.entrada_manual.grid(row=1, column=1, padx=5, sticky="w")

        tk.Label(frame_carga, text="Valor K (Solo KNN):").grid(row=1, column=2, sticky="e")
        self.entrada_k = tk.Entry(frame_carga, width=5)
        self.entrada_k.insert(0, "3") # Valor por defecto
        self.entrada_k.grid(row=1, column=3, padx=5)

        self.btn_calcular = tk.Button(frame_carga, text="Ejecutar Predicción", bg="lightblue", font=("Arial", 10, "bold"), command=self.ejecutar_modelo)
        self.btn_calcular.grid(row=1, column=4, padx=10)

        # --- PANEL DE RESULTADOS ---
        frame_resultados = tk.LabelFrame(self.root, text="2. Panel de Resultados", padx=10, pady=10)
        frame_resultados.pack(fill="x", padx=10, pady=5)

        self.lbl_resultado = tk.Label(frame_resultados, text="Esperando datos...", font=("Arial", 12), fg="blue")
        self.lbl_resultado.pack()

        # --- ÁREA DE GRÁFICOS ---
        frame_graficos = tk.LabelFrame(self.root, text="3. Área de Gráficos", padx=10, pady=10)
        frame_graficos.pack(fill="both", expand=True, padx=10, pady=5)

        self.figura, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figura, master=frame_graficos)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def cargar_csv(self):
        ruta_archivo = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=(("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*"))
        )
        if not ruta_archivo:
            return

        self.X_data.clear()
        self.Y_data.clear()
        self.clases_data.clear()

        try:
            with open(ruta_archivo, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                encabezados = next(reader) # Leer la primera fila (nombres de columnas)

                if len(encabezados) == 2:
                    self.tipo_modelo = "Regresion"
                    for row in reader:
                        self.X_data.append(float(row[0]))
                        self.Y_data.append(float(row[1]))
                    self.lbl_archivo.config(text="Modo: Regresión Lineal", fg="green")

                elif len(encabezados) >= 3:
                    self.tipo_modelo = "KNN"
                    for row in reader:
                        self.X_data.append((float(row[0]), float(row[1]))) # Guardamos X e Y como una tupla (coordenada)
                        self.clases_data.append(row[2]) # La tercera columna es la clase
                    self.lbl_archivo.config(text="Modo: K-Nearest Neighbors (KNN)", fg="green")
                else:
                    messagebox.showerror("Error", "El CSV debe tener 2 columnas (Regresión) o 3 columnas (KNN).")
                    
            messagebox.showinfo("Éxito", "Datos cargados correctamente. Ahora ingresa un valor manual y presiona Ejecutar.")
        except Exception as e:
            messagebox.showerror("Error al leer CSV", f"Asegúrate de que los datos sean numéricos.\nDetalle: {e}")

    def ejecutar_modelo(self):
        valor_ingresado = self.entrada_manual.get()
        
        if not self.tipo_modelo:
            messagebox.showwarning("Faltan Datos", "Primero debes subir un archivo CSV para entrenar el modelo.")
            return
        if not valor_ingresado:
            messagebox.showwarning("Faltan Datos", "Ingresa un valor manual para hacer la predicción.")
            return

        self.ax.clear()

        # ==========================================
        # EJECUCIÓN: REGRESIÓN LINEAL
        # ==========================================
        if self.tipo_modelo == "Regresion":
            try:
                x_nuevo = float(valor_ingresado)
            except ValueError:
                messagebox.showerror("Error", "Para Regresión, el valor manual debe ser un solo número (Ej: 5.2)")
                return

            modelo = RegresionLinealManual()
            modelo.entrenar(self.X_data, self.Y_data)
            y_pred = modelo.predecir(x_nuevo)
            mse = modelo.calcular_mse(self.X_data, self.Y_data)

            # Mostrar resultados en texto
            texto_res = (f"Ecuación: Y = {modelo.m:.4f}X + {modelo.b:.4f}   |   MSE: {mse:.4f}\n"
                         f"Predicción: Si X = {x_nuevo}, entonces Y = {y_pred:.4f}")
            self.lbl_resultado.config(text=texto_res)

            # Dibujar gráfico
            self.ax.set_title("Regresión Lineal Simple")
            self.ax.scatter(self.X_data, self.Y_data, color='blue', label='Datos Históricos')
            
            # Línea de tendencia
            x_min, x_max = min(self.X_data), max(self.X_data)
            self.ax.plot([x_min, x_max], [modelo.predecir(x_min), modelo.predecir(x_max)], color='red', label='Línea de Tendencia')
            
            # Punto predicho
            self.ax.scatter([x_nuevo], [y_pred], color='green', marker='*', s=150, label='Tu Predicción')
            
            self.ax.legend()
            self.canvas.draw()

        # ==========================================
        # EJECUCIÓN: K-NN
        # ==========================================
        elif self.tipo_modelo == "KNN":
            try:
                # Separar las coordenadas separadas por coma
                coords = valor_ingresado.split(',')
                punto_nuevo = (float(coords[0].strip()), float(coords[1].strip()))
                k_val = int(self.entrada_k.get())
            except ValueError:
                messagebox.showerror("Error", "Para KNN, ingresa dos coordenadas separadas por coma (Ej: 2.5, 3.1) y un K entero.")
                return

            modelo = KNNManual(k=k_val)
            modelo.entrenar(self.X_data, self.clases_data)
            clase_predicha = modelo.predecir(punto_nuevo)

            # Mostrar resultados en texto
            texto_res = f"Analizando {k_val} vecinos...\nEl nuevo punto {punto_nuevo} pertenece a la Clase: {clase_predicha}"
            self.lbl_resultado.config(text=texto_res)

            # Dibujar gráfico
            self.ax.set_title(f"Clasificación K-NN (K={k_val})")
            
            # Colorear puntos según su clase
            clases_unicas = list(set(self.clases_data))
            colores = ['blue', 'orange', 'purple', 'brown'] # Soporta hasta 4 clases visualmente
            
            for idx, clase_actual in enumerate(clases_unicas):
                x_clase = [p[0] for i, p in enumerate(self.X_data) if self.clases_data[i] == clase_actual]
                y_clase = [p[1] for i, p in enumerate(self.X_data) if self.clases_data[i] == clase_actual]
                color_actual = colores[idx % len(colores)]
                self.ax.scatter(x_clase, y_clase, color=color_actual, label=f'Clase {clase_actual}')
            
            # Punto predicho
            self.ax.scatter([punto_nuevo[0]], [punto_nuevo[1]], color='green', marker='*', s=200, label=f'Predicción: {clase_predicha}', edgecolor='black')
            
            self.ax.legend()
            self.canvas.draw()

if __name__ == "__main__":
    ventana_principal = tk.Tk()
    app = AplicacionML(ventana_principal)
    ventana_principal.mainloop()