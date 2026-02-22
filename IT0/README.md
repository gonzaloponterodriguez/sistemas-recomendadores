# Sistema de Recomendación de Playlists - Baseline de Popularidad (Iteración 0)

**Integrantes del grupo:**
1. Pablo Fernandez Rubal
2. Noura El Morchid
3. Gonzalo Ponte Rodriguez

---

## 1. Descripción de la Implementación
Este proyecto implementa un sistema de recomendación base (baseline) para playlists de Spotify utilizando la popularidad global de las canciones.

La lógica se divide en tres fases principales:
* **Creación de la Matriz (`creacion_matrix.py`):** Procesa el dataset de entrenamiento en bruto (`spotify_train_dataset.zip`) y construye una matriz dispersa (CSR matrix) donde las filas son las playlists y las columnas las canciones. También genera diccionarios JSON para mapear las URIs de las canciones a los índices de la matriz.
* **Cálculo y Generación (`baseline_popularidad.py`):** Calcula la popularidad sumando las apariciones de cada canción en la matriz de entrenamiento. Extrae un Top 1000 general y, para cada playlist del conjunto de test, recomienda hasta 500 canciones de ese Top que el usuario no tenga ya en su lista (semillas). **Optimización:** Lee el archivo de test directamente desde `spotify_test_playlists.zip` sin necesidad de extracción previa.
* **Evaluación (`evaluacion.py`):** Compara el archivo de predicciones generado con el *ground truth* oculto (leyéndolo también directamente desde el `.zip`) y calcula las tres métricas solicitadas.

---

## 2. Estructura del Proyecto
Para mantener el código limpio y organizado, el proyecto tiene una estructura modular:
* `main.py`: Script orquestador principal.
* `scripts/`: Carpeta que contiene la lógica del programa (`creacion_matrix.py`, `baseline_popularidad.py`, `evaluacion.py`).
* `datos/`: **[NO INCLUIDA EN LA ENTREGA POR TAMAÑO]** Carpeta que deberá contener los datasets originales proporcionados (`spotify_train_dataset.zip` y `spotify_test_playlists.zip`). No es necesario descomprimirlos.
* `matrix/`: Carpeta generada automáticamente durante la ejecución para guardar los diccionarios y la matriz dispersa.

---

## 3. Instrucciones de Ejecución
Para reproducir los resultados y generar el archivo de recomendaciones, el proceso está completamente automatizado.

**REQUISITO PREVIO (IMPORTANTE):** Debido al tamaño masivo de los datasets, estos no se incluyen en el archivo de entrega. Antes de ejecutar el código, **es necesario crear una carpeta llamada `datos` en la raíz del proyecto** y colocar dentro los dos archivos originales de la práctica:
- `spotify_train_dataset.zip`
- `spotify_test_playlists.zip`

**Preparación del entorno**
1. Abrir una terminal en la raíz del proyecto.
2. Crear un entorno virtual.
3. Activar el entorno virtual.
4. Instalar las librerías necesarias:
   > `pip install -r requirements.txt`

**Ejecución:**
Desde la terminal (con el entorno activado), situado en la carpeta raíz del proyecto, ejecuta uel archivo `main.py`

Este script se encargará de ejecutar secuencialmente todo el *pipeline*:
1. Leerá los datos de entrenamiento y creará la carpeta `matrix/`.
2. Generará el fichero de recomendaciones llamado `resultado_baseline.csv` en la raíz.
3. Evaluará las métricas y las mostrará por pantalla.

*(Nota: Si se prefiere la ejecución manual, se pueden correr los scripts de la carpeta `scripts/` de forma individual respetando el orden mencionado).*

---

## 4. Resultados Obtenidos
Al evaluar las recomendaciones generadas por nuestro modelo de popularidad (Iteración 0) contra el conjunto de evaluación oculto, se han obtenido las siguientes métricas medias:

* **R-Precision media:** 0.0257
* **NDCG media:** 0.0904
* **Clicks medios:** 18.1028