# EVALUACI-N-PR-CTICA-PARCIAL-II

# Implementación de Modelos de IA con Interfaz Gráfica (GUI)

## 📌 Descripción del Proyecto
Esta es una aplicación de escritorio desarrollada en Python que implementa algoritmos de aprendizaje supervisado desde cero. Cumpliendo con estrictas restricciones técnicas, **no se utilizan librerías de Machine Learning automáticas** (como Scikit-learn o PyCaret). Toda la matemática subyacente de los modelos ha sido programada manualmente.

El proyecto integra una interfaz de usuario interactiva que facilita la carga de datos, la interacción con los modelos y la visualización dinámica de los resultados.

## 🚀 Características Principales

### 1. Modelos Matemáticos Manuales
* **Regresión Lineal Simple:** Implementada mediante el método de Mínimos Cuadrados para hallar la pendiente y el intercepto. Incluye el cálculo del Error Cuadrático Medio (MSE).
* **K-Nearest Neighbors (K-NN):** Implementado utilizando la Distancia Euclidiana. Permite al usuario configurar el valor de `K` (número de vecinos) y clasifica nuevos puntos mediante votación por mayoría.

### 2. Interfaz Gráfica (GUI)
* **Módulo de Carga:** Permite subir datasets en formato `.csv` y cuenta con campos para el ingreso manual de datos a predecir.
* **Panel de Resultados:** Muestra claramente las ecuaciones matemáticas generadas, las métricas de error y la clase asignada.
* **Área de Gráficos Dinámicos:** Integración directa para dibujar *scatter plots* con líneas de tendencia (Regresión) y mapas de dispersión coloreados por etiquetas de clase (K-NN).

## 🛠️ Tecnologías y Librerías Utilizadas
* **Python 3.x** (Lenguaje principal)
* **Tkinter** (Desarrollo de la interfaz gráfica)
* **Matplotlib / FigureCanvasTkAgg** (Generación de gráficos dinámicos incrustados)
* **Math / Collections / CSV** (Módulos nativos de Python para cálculos y lectura de datos)

## ⚙️ Instrucciones de Ejecución

1. Clonar o descargar este repositorio.
2. Instalar la única dependencia gráfica necesaria ejecutando en la terminal:
   ```bash
   pip install matplotlib
