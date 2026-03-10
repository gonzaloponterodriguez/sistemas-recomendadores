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

# Bajamos el K a 40 para hacer las recomendaciones más específicas y evitar sesgo de popularidad
K_NEIGHBORS = 40 

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

# --- NUEVO: PRECALCULAR BASELINE PARA COLD START ---
print("Calculando Top Popularidad global para playlists vacías (Cold Start)...")
popularity_counts = np.array(train_matrix.sum(axis=0)).flatten()
top_popular_indices = np.argsort(popularity_counts)[-500:][::-1]
top_popular_uris = [index_to_track[idx] for idx in top_popular_indices]

# Extraer las semillas del test
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
    
    k_indices = data["indices"][:K_NEIGHBORS]
    
    if len(k_indices) > 0:
        # POTENCIAMOS LAS SIMILITUDES: Elevamos al cubo para dar más peso a los vecinos muy cercanos
        similitudes_base = np.array(data["similitudes"][:K_NEIGHBORS])
        pesos_potenciados = np.power(similitudes_base, 3).reshape(len(k_indices), 1)
        
        submatriz_vecinos = train_matrix[k_indices]
        
        # Multiplicamos la transpuesta de vecinos por los pesos potenciados
        item_scores = np.array(submatriz_vecinos.T.dot(pesos_potenciados)).flatten()
        
        # Penalizamos fuertemente las canciones que el usuario ya tiene
        item_scores[semillas] = -np.inf
        
        # Obtenemos los índices de las 500 mejores canciones
        if len(item_scores) >= 500:
            top_500_idx = np.argpartition(item_scores, -500)[-500:]
            top_500_idx = top_500_idx[np.argsort(item_scores[top_500_idx])[::-1]]
        else:
            top_500_idx = np.argsort(item_scores)[::-1]
            
        # Filtramos canciones con puntuación 0
        recommendations = [index_to_track[idx] for idx in top_500_idx if item_scores[idx] > 0]
        
        # Fallback. Si KNN encontró menos de 500, rellenamos con las populares
        if len(recommendations) < 500:
            actuales = set(recommendations)
            semillas_uris = {index_to_track[s] for s in semillas}
            
            for uri in top_popular_uris:
                if uri not in actuales and uri not in semillas_uris:
                    recommendations.append(uri)
                    if len(recommendations) == 500:
                        break
    else:
        # Playlists 100% vacías (Cold Start) se llevan la popularidad directamente
        recommendations = top_popular_uris.copy()
        
    results[pid] = recommendations
    
    if (i + 1) % 1000 == 0:
        print(f"Generado: {i + 1}/{total}")

# --- GUARDAR RESULTADOS ---
print(f"Guardando CSV en {output_file_path}...")
with open(output_file_path, "w") as f:
    f.write("team_info, Pablo Fernandez Rubal - Noura el Morchid - Gonzalo Ponte Rodriguez, pablo.fernandez.rubal@udc.es - n.elmorchid@udc.es - g.ponte@udc.es\n")
    for pid, tracks in results.items():
        f.write(f"{pid}," + ",".join(tracks) + "\n")
        
print("¡Archivo de recomendaciones KNN generado!")