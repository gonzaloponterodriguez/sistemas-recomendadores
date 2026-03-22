# Sistema de Recomendacion de Playlists - Iteraciones 0 y 1

**Integrantes del grupo:**
1. Pablo Fernandez Rubal
2. Noura El Morchid
3. Gonzalo Ponte Rodriguez

---

## 1. Descripción de la Implementación
Este proyecto implementa un sistema de recomendacion para playlists de Spotify en dos iteraciones:

* **Iteracion 0 (Baseline de popularidad):** recomienda canciones por popularidad global.
* **Iteracion 1 (Filtrado colaborativo KNN):** incluye un modelo `user-based` y un modelo `item-based`.

La logica se organiza en los siguientes scripts:

* **`creacion_matrix.py`:** procesa `spotify_train_dataset.zip` y genera una matriz dispersa (CSR) de interacciones playlist-cancion, junto a los diccionarios de indices.
* **`baseline_popularidad.py`:** calcula popularidad global de canciones (Top-N) y genera recomendaciones para test filtrando semillas.
* **`calcular_vecinos.py`:** precomputa vecinos similares por coseno para cada playlist de test y guarda el resultado intermedio en `vecinos.json`.
* **`user_based.py`:** genera recomendaciones con KNN user-based usando los vecinos precomputados; rellena con popularidad cuando no alcanza 500 tracks.
* **`item_based.py`:** genera recomendaciones con KNN item-based usando operaciones sparse por lotes; tambien rellena con popularidad cuando es necesario.
* **`evaluacion.py`:** evalua predicciones frente al ground truth oculto y calcula R-Precision, NDCG y Clicks.

---

## 2. Estructura del Proyecto
Para mantener el código limpio y organizado, el proyecto tiene una estructura modular:
* `main.py`: Script orquestador principal.
* `scripts/`: Carpeta con la logica del pipeline (`creacion_matrix.py`, `baseline_popularidad.py`, `calcular_vecinos.py`, `user_based.py`, `item_based.py`, `evaluacion.py`).
* `datos/`: **[NO INCLUIDA EN LA ENTREGA POR TAMAÑO]** Carpeta que deberá contener los datasets originales proporcionados (`spotify_train_dataset.zip` y `spotify_test_playlists.zip`). No es necesario descomprimirlos.
* `matrix/`: Carpeta generada automáticamente durante la ejecución para guardar los diccionarios y la matriz dispersa.
* `resultados/`: Carpeta de salida con los CSV generados por cada modelo (`baseline.csv`, `user_based.csv`, `item_based.csv`).
* `vecinos.json`: Artefacto intermedio para el modelo user-based (vecinos precomputados).

---

## 3. Instrucciones de Ejecución
Para reproducir los resultados y generar recomendaciones, el proceso esta automatizado desde `main.py`.

**REQUISITOS PREVIOS (IMPORTANTE):** Debido al tamaño masivo de los datasets, estos no se incluyen en el archivo de entrega. Antes de ejecutar el código, **es necesario crear una carpeta llamada `datos` en la raíz del proyecto** y colocar dentro los dos archivos originales de la práctica:
- `spotify_train_dataset.zip`
- `spotify_test_playlists.zip`

Ademas, para ejecutar correctamente la Iteracion 1, debes tener generada la matriz de entrenamiento (`matrix/`) , resultado de haber ejecutado el script `creacion_matrix.py`. Esta carpeta ya se encuentra en el zip para no tener que volver a ejecutar `creacion_matrix.py`
Nota: no es obligatorio tener `resultados/baseline.csv` para Iteracion 1; los modelos `user-based` e `item-based` calculan el relleno por popularidad directamente desde la matriz.

**Preparación del entorno**
1. Abrir una terminal en la raíz del proyecto.
2. Crear un entorno virtual.
3. Activar el entorno virtual.
4. Instalar las librerías necesarias:
   > `pip install -r requirements.txt`

**Ejecución:**
Desde la raiz del proyecto (con el entorno activado), ejecutar:

> `python main.py`

El script pedira la iteracion a ejecutar.

**Flujo Iteracion 0 (Baseline):**
1. `creacion_matrix.py`
2. `baseline_popularidad.py`
3. `evaluacion.py resultados/baseline.csv`

Salida principal: `resultados/baseline.csv`

**Flujo Iteracion 1 (KNN):**
Al elegir iteracion `1`, se debe escoger modelo `user` o `item`.

* Modelo `user`:
1. `calcular_vecinos.py`
2. `user_based.py`
3. `evaluacion.py resultados/user_based.csv`

Salida principal: `resultados/user_based.csv`

* Modelo `item`:
1. `item_based.py`
2. `evaluacion.py resultados/item_based.csv`

Salida principal: `resultados/item_based.csv`

Nota: si se prefiere ejecucion manual, se pueden lanzar los scripts de `scripts/` respetando el orden de cada flujo.

---

## 4. Resultados Obtenidos
La siguiente tabla resume las metricas medias por modelo. 
| Modelo | R-Precision | NDCG | Clicks |
|---|---:|---:|---:|
| Baseline Popularidad (It 0) | 0.0257 | 0.0904 | 18.1028 |
| User-based KNN (Iteracion 1)| 0.1542 | 0.3335 | 5.7268 |
| Item-based KNN (Iteracion 1)| 0.1540 | 0.3373 | 6.4012 |

## 5. Justificacion de K=500 en User-based
La seleccion de `K=500` para `user_based.py` se hizo tras comparar varios valores de K con las tres metricas principales (R-Precision, NDCG y Clicks).

Resumen de tendencia observada :

| K (User-based) | R-Precision | NDCG | Clicks |
|---:|---:|---:|---:|
| 100 | 0.1424 | 0.3033 | 6.2949 |
| 200 | 0.1492 | 0.3202 | 5.9770 |
| 250 | 0.1507 | 0.3246 | 5.8882 |
| 300 | 0.1516 | 0.3276 | 5.8417 |
| 400 | 0.1533 | 0.3314 | 5.7700 |
| 500 | 0.1542 | 0.3335 | 5.7268 |

Conclusion:
* `K=500` da el mejor equilibrio y los mejores valores finales en las tres metricas.
* El tiempo de ejecucion real fue muy parecido entre configuraciones, por lo que aumentar K no supuso una penalizacion relevante en nuestro entorno.