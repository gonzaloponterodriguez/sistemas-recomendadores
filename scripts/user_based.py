import json
import numpy as np
import os
import zipfile
from scipy.sparse import load_npz

# --- CONFIGURACIÓN ---
input_dir = "matrix"
test_zip_path = "datos/spotify_test_playlists.zip"
vecinos_path = "vecinos_test.json"
output_file_path = "resultado_user_knn.csv"

# HIPERPARÁMETRO: Puedes cambiar este número (entre 1 y 1000) y volver a ejecutar 
# este script sin tener que recalcular todas las similitudes.
K_NEIGHBORS = 150 

print(f"--- INICIANDO FASE 2: GENERACIÓN DE RECOMENDACIONES (K={K_NEIGHBORS}) ---")

# --- CARGA DE DATOS ---
try:
    train_matrix = load_npz(os.path.join(input_dir, "sparse_matrix_train.npz"))
    with open(os.path.join(input_dir, "track_to_index_train.json"), "r") as f:
        track_to_index = json.load(f)
    index_to_track = {v: k for k, v in track_to_index.items()}
    
    with open(vecinos_path, "r") as f:
        vecinos_data = json.load(f)
except FileNotFoundError:
    print("Error: Asegúrate de haber ejecutado 'calcular_vecinos.py' primero.")
    exit()

# Extraer las semillas del test para no recomendarlas de nuevo
print("Cargando semillas de test...")
test_seeds = {}
with zipfile.ZipFile(test_zip_path, "r") as zipf:
    for filename in zipf.namelist():
        if filename.endswith("test_input_playlists.json"):
            with zipf.open(filename) as f:
                test_raw = json.loads(f.read())
            break
for pl in test_raw.get("playlists", []):
    test_seeds[pl["pid"]] = [track_to_index[t["track_uri"]] for t in pl["tracks"] if t["track_uri"] in track_to_index]

# --- GENERAR RECOMENDACIONES ---
print("Aplicando KNN y optimización de vecinos...")
results = {}
total = len(vecinos_data)

for i, (pid_str, data) in enumerate(vecinos_data.items()):
    pid = int(pid_str)
    semillas = test_seeds.get(pid, [])
    
    # Recortamos a los K vecinos que queremos usar en esta ejecución
    k_indices = data["indices"][:K_NEIGHBORS]
    k_similitudes = np.array(data["similitudes"][:K_NEIGHBORS]).reshape(1, -1)
    
    if len(k_indices) > 0:
        # OPTIMIZACIÓN APLICADA: 
        # Extraemos la submatriz solo de estos K vecinos
        submatriz_vecinos = train_matrix[k_indices]
        
        # Al multiplicar una matriz dispersa, matemáticamente ignora todas 
        # las columnas (canciones) que están a 0 en el perfil de estos vecinos.
        # Devuelve un array con puntuaciones solo para los items candidatos.
        item_scores = (k_similitudes @ submatriz_vecinos).flatten()
        
        # Penalizamos fuertemente las canciones que el usuario ya tiene en su playlist
        item_scores[semillas] = -np.inf
        
        # Obtenemos los índices de las 500 mejores canciones
        # np.argpartition es mucho más rápido que np.argsort para sacar el top 500
        if len(item_scores) >= 500:
            top_500_idx = np.argpartition(item_scores, -500)[-500:]
            # Argpartition no ordena internamente los 500, así que los ordenamos ahora
            top_500_idx = top_500_idx[np.argsort(item_scores[top_500_idx])[::-1]]
        else:
            top_500_idx = np.argsort(item_scores)[::-1]
            
        # Filtramos canciones con puntuación 0 (nadie de los vecinos las tenía)
        recommendations = [index_to_track[idx] for idx in top_500_idx if item_scores[idx] > 0]
    else:
        recommendations = []
        
    results[pid] = recommendations
    
    if (i + 1) % 1000 == 0:
        print(f"Generado: {i + 1}/{total}")

# --- GUARDAR RESULTADOS ---
print(f"Guardando CSV en {output_file_path}...")
with open(output_file_path, "w") as f:
    f.write("team_info,Pablo Noura Gonzalo,contacto@email.com\n")
    for pid, tracks in results.items():
        f.write(f"{pid}," + ",".join(tracks) + "\n")
        
print("¡Archivo de recomendaciones KNN generado!")