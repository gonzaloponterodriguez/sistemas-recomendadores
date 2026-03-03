import json
import numpy as np
import os
import zipfile
from scipy.sparse import load_npz, csr_matrix

# --- CONFIGURACIÓN ---
input_dir = "matrix"
test_zip_path = "datos/spotify_test_playlists.zip"
output_file_path = "vecinos_test.json"
MAX_VECINOS_A_GUARDAR = 1000 # Guardamos los 1000 mejores por si luego quieres probar K=10, 50, 100 o 500

print("--- INICIANDO FASE 1: PRECOMPUTACIÓN DE VECINOS KNN ---")

# --- CARGA DE DATOS ---
try:
    print("Cargando matriz de entrenamiento...")
    train_matrix = load_npz(os.path.join(input_dir, "sparse_matrix_train.npz"))
    with open(os.path.join(input_dir, "track_to_index_train.json"), "r") as f:
        track_to_index = json.load(f)
except FileNotFoundError:
    print("Error: No se encuentran los archivos en la carpeta 'matrix'")
    exit()

num_train_playlists, num_tracks = train_matrix.shape

# Precomputar normas de entrenamiento para el Coseno
print("Precomputando normas de las playlists de entrenamiento...")
train_norms = np.sqrt(train_matrix.power(2).sum(axis=1).A1)
train_norms[train_norms == 0] = 1e-9 # Evitar divisiones por cero

# --- PROCESAR TEST ---
print(f"Buscando vecinos para el archivo {test_zip_path}...")
vecinos_guardados = {}

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
    user_track_indices = [track_to_index[t["track_uri"]] for t in playlist["tracks"] if t["track_uri"] in track_to_index]
    
    if user_track_indices:
        # Vector disperso del usuario
        data = np.ones(len(user_track_indices))
        rows = np.zeros(len(user_track_indices))
        cols = np.array(user_track_indices)
        user_vector = csr_matrix((data, (rows, cols)), shape=(1, num_tracks))
        
        # Similitud del Coseno: Producto punto / (Norma A * Norma B)
        dot_products = train_matrix.dot(user_vector.T).toarray().flatten()
        user_norm = np.sqrt(len(user_track_indices)) 
        similarities = dot_products / (user_norm * train_norms)
        
        # Extraer Top N vecinos
        best_neighbors_idx = np.argsort(similarities)[-MAX_VECINOS_A_GUARDAR:][::-1]
        best_similarities = similarities[best_neighbors_idx]
        
        # Guardamos convirtiendo a tipos nativos de Python para el JSON
        vecinos_guardados[pid] = {
            "indices": best_neighbors_idx.tolist(),
            "similitudes": best_similarities.tolist()
        }
    else:
        # Cold start: si no conocemos ninguna canción
        vecinos_guardados[pid] = {"indices": [], "similitudes": []}
        
    if (i + 1) % 100 == 0:
        print(f"Calculados vecinos para: {i + 1}/{total} playlists")

# --- GUARDAR JSON ---
print(f"Guardando similitudes en {output_file_path}...")
with open(output_file_path, "w") as f:
    json.dump(vecinos_guardados, f)
print("¡Fase 1 completada!")