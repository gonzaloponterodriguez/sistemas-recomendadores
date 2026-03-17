import json
import sys
import numpy as np
import os
import zipfile
from scipy.sparse import load_npz

# --- CONFIGURACIÓN ---
input_dir = "matrix"
test_zip_path = "datos/spotify_test_playlists.zip"
output_file_path = "resultado_item_knn.csv"

if os.path.exists(output_file_path):
    print("Las recomendaciones ya estaban creadas en el directorio.")
    print("Se cancela la ejecución para evitar duplicar el trabajo.")
    sys.exit()

print("INICIANDO MODELO ITEM-BASED KNN")

# --- CARGA DE DATOS ---
try:
    print("Cargando matriz de entrenamiento...")
    train_matrix = load_npz(os.path.join(input_dir, "sparse_matrix_train.npz"))
    
    # IMPORTANTE: Convertimos a formato CSC (Compressed Sparse Column). 
    # Como el Item-KNN trabaja analizando columnas (canciones), esto acelera el proceso un 1000%
    print("Convirtiendo matriz a formato columnar (CSC)...")
    train_matrix = train_matrix.tocsc()
    
    with open(os.path.join(input_dir, "track_to_index_train.json"), "r") as f:
        track_to_index = json.load(f)
    index_to_track = {v: k for k, v in track_to_index.items()}
except FileNotFoundError:
    print("Error: No se encuentran los archivos en la carpeta 'matrix'")
    exit()

num_train_playlists, num_tracks = train_matrix.shape

# --- PRECOMPUTACIÓN DE NORMAS ---
print("Precomputando normas de los ÍTEMS (Canciones)...")
# Sumamos por columnas (axis=0) para obtener la magnitud de cada canción
item_norms = np.sqrt(train_matrix.power(2).sum(axis=0).A1)
item_norms[item_norms == 0] = 1e-9 # Evitar divisiones por cero

# Precalcular Popularidad Global (Plan B para el Cold Start)
print("Calculando Top Popularidad para playlists vacías...")
popularity_counts = np.array(train_matrix.sum(axis=0)).flatten()
top_popular_indices = np.argsort(popularity_counts)[-500:][::-1]
top_popular_uris = [index_to_track[idx] for idx in top_popular_indices]

# --- PROCESAR TEST Y GENERAR RECOMENDACIONES ---
print(f"Generando recomendaciones para {test_zip_path}...")
results = {}

with zipfile.ZipFile(test_zip_path, "r") as zipf:
    for filename in zipf.namelist():
        if filename.endswith("test_input_playlists.json"):
            with zipf.open(filename) as f:
                test_data = json.loads(f.read())
            break

playlists = test_data.get("playlists", [])
total = len(playlists)

for i, playlist in enumerate(playlists):
    pid = playlist["pid"]
    
    # Extraer las semillas de esta playlist (canciones que ya tiene)
    semillas = [track_to_index[t["track_uri"]] for t in playlist["tracks"] if t["track_uri"] in track_to_index]
    
    if semillas:
        # MAGIA ALGEBRAICA: Calculamos el Item-KNN exacto de golpe
        # 1. Extraemos solo las columnas de las canciones que el usuario ya tiene
        seeds_matrix = train_matrix[:, semillas]
        
        # 2. Dividimos cada columna por su norma (Primera parte del Coseno)
        seeds_matrix_normalized = seeds_matrix.multiply(1 / item_norms[semillas])
        
        # 3. Sumamos horizontalmente para crear un "Perfil de Gustos" ponderado de los usuarios
        user_taste_profile = seeds_matrix_normalized.sum(axis=1) # Matriz densa de Nx1
        
        # 4. Multiplicamos la matriz transpuesta entera por el perfil para obtener las similitudes
        raw_item_scores = train_matrix.T.dot(user_taste_profile)
        
        # 5. Dividimos entre la norma de cada canción (Segunda parte del Coseno)
        item_scores = raw_item_scores.A1 / item_norms
        
        # Penalizamos fuertemente las semillas para no recomendarlas
        item_scores[semillas] = -np.inf
        
        # Obtenemos el Top 500
        if len(item_scores) >= 500:
            top_500_idx = np.argpartition(item_scores, -500)[-500:]
            top_500_idx = top_500_idx[np.argsort(item_scores[top_500_idx])[::-1]]
        else:
            top_500_idx = np.argsort(item_scores)[::-1]
            
        # Filtramos
        recommendations = [index_to_track[idx] for idx in top_500_idx if item_scores[idx] > 0]
        
        # Fallback de popularidad si falta alguna
        if len(recommendations) < 500:
            actuales = set(recommendations)
            semillas_uris = {index_to_track[s] for s in semillas}
            for uri in top_popular_uris:
                if uri not in actuales and uri not in semillas_uris:
                    recommendations.append(uri)
                    if len(recommendations) == 500:
                        break
    else:
        # Cold Start: Playlist 100% vacía
        recommendations = top_popular_uris.copy()
        
    results[pid] = recommendations
    
    if (i + 1) % 500 == 0:
        print(f"Procesado: {i + 1}/{total}")

# --- GUARDAR RESULTADOS ---
print(f"Guardando CSV en {output_file_path}...")
with open(output_file_path, "w") as f:
    f.write("team_info, Pablo Fernandez Rubal - Noura el Morchid - Gonzalo Ponte Rodriguez, pablo.fernandez.rubal@udc.es - n.elmorchid@udc.es - g.ponte@udc.es\n")
    for pid, tracks in results.items():
        f.write(f"{pid}," + ",".join(tracks) + "\n")
        
print("¡Archivo Item-KNN generado con éxito!")