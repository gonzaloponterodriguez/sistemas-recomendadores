import os
# ¡SÚPER IMPORTANTE! Obligamos a NumPy a usar 1 solo hilo por proceso.
# Así los múltiples procesos de Python no se pelean entre ellos por la CPU.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import numpy as np
import zipfile
from scipy.sparse import load_npz
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- CONFIGURACIÓN ---
input_dir = "matrix"
test_zip_path = "datos/spotify_test_playlists.zip"
output_file_path = "resultado_item_knn_paralelo.csv"

# --- VARIABLES GLOBALES PARA COMPARTIR EN RAM (SOLO LECTURA) ---
train_matrix = None
item_norms = None
top_popular_uris = None
index_to_track = None

# --- FUNCIÓN QUE EJECUTARÁ CADA TRABAJADOR (NÚCLEO) ---
def procesar_playlist(args):
    pid, semillas, semillas_uris_set = args
    
    if semillas:
        # 1. Extraemos solo las columnas de las canciones que el usuario ya tiene
        seeds_matrix = train_matrix[:, semillas]
        
        # 2. Dividimos cada columna por su norma
        seeds_matrix_normalized = seeds_matrix.multiply(1 / item_norms[semillas])
        
        # 3. Sumamos horizontalmente para crear el "Perfil de Gustos"
        user_taste_profile = seeds_matrix_normalized.sum(axis=1) 
        
        # 4. Multiplicamos la matriz transpuesta entera por el perfil para obtener las similitudes
        raw_item_scores = train_matrix.T.dot(user_taste_profile)
        
        # 5. Dividimos entre la norma de cada canción 
        item_scores = raw_item_scores.A1 / item_norms
        
        # Penalizamos fuertemente las semillas para no recomendarlas
        item_scores[semillas] = -np.inf
        
        # Obtenemos el Top 500 rápido
        if len(item_scores) >= 500:
            top_500_idx = np.argpartition(item_scores, -500)[-500:]
            top_500_idx = top_500_idx[np.argsort(item_scores[top_500_idx])[::-1]]
        else:
            top_500_idx = np.argsort(item_scores)[::-1]
            
        # Filtrado rápido en NumPy
        valid_idx = top_500_idx[item_scores[top_500_idx] > 0]
        recommendations = [index_to_track[idx] for idx in valid_idx]
        
        # Fallback de popularidad
        if len(recommendations) < 500:
            actuales = set(recommendations)
            for uri in top_popular_uris:
                if uri not in actuales and uri not in semillas_uris_set:
                    recommendations.append(uri)
                    if len(recommendations) == 500:
                        break
    else:
        # Cold Start: Playlist 100% vacía
        recommendations = top_popular_uris.copy()
        
    return pid, recommendations


# --- BLOQUE PRINCIPAL (Obligatorio para multiprocessing) ---
if __name__ == '__main__':
    print("--- INICIANDO MODELO ITEM-BASED KNN PARALELIZADO ---")

    try:
        print("Cargando matriz de entrenamiento...")
        train_matrix = load_npz(os.path.join(input_dir, "sparse_matrix_train.npz"))
        print("Convirtiendo matriz a formato columnar (CSC)...")
        train_matrix = train_matrix.tocsc()
        
        with open(os.path.join(input_dir, "track_to_index_train.json"), "r") as f:
            track_to_index = json.load(f)
        index_to_track = {v: k for k, v in track_to_index.items()}
    except FileNotFoundError:
        print("Error: No se encuentran los archivos en la carpeta 'matrix'")
        exit()

    print("Precomputando normas de los ÍTEMS (Canciones)...")
    item_norms = np.sqrt(train_matrix.getnnz(axis=0))
    item_norms[item_norms == 0] = 1e-9 # Evitar divisiones por cero

    print("Calculando Top Popularidad para playlists vacías...")
    popularity_counts = np.array(train_matrix.sum(axis=0)).flatten()
    top_popular_indices = np.argsort(popularity_counts)[-500:][::-1]
    top_popular_uris = [index_to_track[idx] for idx in top_popular_indices]

    print(f"Cargando semillas de prueba de {test_zip_path}...")
    with zipfile.ZipFile(test_zip_path, "r") as zipf:
        for filename in zipf.namelist():
            if filename.endswith("test_input_playlists.json"):
                with zipf.open(filename) as f:
                    test_data = json.loads(f.read())
                break

    # PREPARAR TAREAS PARA EL POOL
    tareas = []
    playlists = test_data.get("playlists", [])
    
    for playlist in playlists:
        pid = playlist["pid"]
        valid_tracks = [t["track_uri"] for t in playlist["tracks"] if t["track_uri"] in track_to_index]
        semillas = [track_to_index[uri] for uri in valid_tracks]
        semillas_uris_set = set(valid_tracks)
        tareas.append((pid, semillas, semillas_uris_set))

    # EJECUCIÓN EN PARALELO
    n_cores = max(1, cpu_count() - 1) # Usamos todos menos 1 para no congelar el PC
    print(f"Iniciando Pool con {n_cores} procesos usando tqdm...")
    
    results = {}

    with Pool(processes=n_cores) as pool:
        # Usamos imap_unordered para ganar un extra de velocidad
        for result in tqdm(pool.imap_unordered(procesar_playlist, tareas), total=len(tareas), desc="Recomendando"):
            pid, recommendations = result
            results[pid] = recommendations

    # GUARDAR RESULTADOS
    print(f"\nGuardando CSV en {output_file_path}...")
    with open(output_file_path, "w") as f:
        f.write("team_info, Pablo Fernandez Rubal - Noura el Morchid - Gonzalo Ponte Rodriguez, pablo.fernandez.rubal@udc.es - n.elmorchid@udc.es - g.ponte@udc.es\n")
        # Ordenamos las claves antes de guardar para que el CSV quede organizado por PID
        for pid in sorted(results.keys()):
            f.write(f"{pid}," + ",".join(results[pid]) + "\n")
            
    print("¡Archivo Item-KNN generado con éxito a toda velocidad!")