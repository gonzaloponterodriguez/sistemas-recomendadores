import os
import json
import numpy as np
import zipfile
import sys
from scipy.sparse import load_npz, csr_matrix
from tqdm import tqdm

# CONFIGURACIÓN 
input_dir = "matrix"
test_zip_path = "datos/spotify_test_playlists.zip"
output_file_path = "resultados/item_based.csv"
BATCH_SIZE = 500  

os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

if os.path.exists(output_file_path):
    print("Las recomendaciones ya estaban creadas en el directorio.")
    sys.exit()

print("INICIANDO MODELO ITEM-BASED")

# CARGA DE DATOS
try:
    print("Cargando matriz de entrenamiento...")
    train_matrix = load_npz(os.path.join(input_dir, "sparse_matrix_train.npz")).astype(np.float32).tocsr()
    
    print("Pre-calculando transpuesta para optimizar...")
    train_matrix_T = train_matrix.T.tocsr()
    
    num_playlists, num_tracks = train_matrix.shape
    
    with open(os.path.join(input_dir, "track_to_index_train.json"), "r") as f:
        track_to_index = json.load(f)
    index_to_track = {v: k for k, v in track_to_index.items()}
except FileNotFoundError:
    print("Error: No se encuentran los archivos en la carpeta 'matrix'")
    sys.exit()

# PRECOMPUTACIÓN MATEMÁTICA
print("Precomputando normas de los ÍTEMS (Canciones)...")
item_norms = np.sqrt(train_matrix.getnnz(axis=0)).astype(np.float32)
item_norms[item_norms == 0] = 1e-9 
item_norms_inv = 1.0 / item_norms 

print("Calculando Top Popularidad...")
popularity_counts = np.array(train_matrix.sum(axis=0)).flatten()
top_popular_indices = np.argsort(popularity_counts)[-500:][::-1]
top_popular_uris = [index_to_track[idx] for idx in top_popular_indices]

# LECTURA DEL TEST
print(f"Cargando archivo de prueba: {test_zip_path}...")
with zipfile.ZipFile(test_zip_path, "r") as zipf:
    for filename in zipf.namelist():
        if filename.endswith("test_input_playlists.json"):
            with zipf.open(filename) as f:
                test_data = json.loads(f.read())
            break

playlists = test_data.get("playlists", [])
total_playlists = len(playlists)
results = {}

print(f"\nIniciando procesamiento (Batch = {BATCH_SIZE})...")

# PROCESAMIENTO POR LOTES 
for i in tqdm(range(0, total_playlists, BATCH_SIZE), desc="Procesando Lotes"):
    batch = playlists[i:i + BATCH_SIZE]
    
    data, rows, cols = [], [], []
    batch_pids = []
    batch_seeds = []
    batch_seeds_uris = []
    
    # Preparamos los datos
    for row_idx, playlist in enumerate(batch):
        pid = playlist["pid"]
        batch_pids.append(pid)
        
        valid_tracks = [t["track_uri"] for t in playlist["tracks"] if t["track_uri"] in track_to_index]
        semillas = [track_to_index[uri] for uri in valid_tracks]
        
        batch_seeds.append(semillas)
        batch_seeds_uris.append(set(valid_tracks))
        
        if semillas:
            data.extend(item_norms_inv[semillas])
            rows.extend([row_idx] * len(semillas))
            cols.extend(semillas)

    if not data:
        for pid in batch_pids:
            results[pid] = top_popular_uris.copy()
        continue

    # Matriz del lote (muy ligera)
    W_batch = csr_matrix((data, (rows, cols)), shape=(len(batch), num_tracks), dtype=np.float32)

    # Multiplicaciones dispersas (devuelven matrices dispersas CSR)
    profile_batch = W_batch.dot(train_matrix_T)
    raw_scores_batch = profile_batch.dot(train_matrix)

    # EXTRACCIÓN DIRECTA SIN .toarray() 
    # raw_scores_batch ya es una CSR matrix. 
    for row_idx in range(len(batch)):
        pid = batch_pids[row_idx]
        semillas = batch_seeds[row_idx]
        
        if not semillas:
            results[pid] = top_popular_uris.copy()
            continue
            
        # Punteros para saber dónde empiezan y acaban los datos de este usuario
        start_ptr = raw_scores_batch.indptr[row_idx]
        end_ptr = raw_scores_batch.indptr[row_idx + 1]
        
        # Extraemos solo las canciones que tienen una puntuación > 0
        canciones_posibles = raw_scores_batch.indices[start_ptr:end_ptr]
        puntuaciones_brutas = raw_scores_batch.data[start_ptr:end_ptr]
        
        if len(canciones_posibles) > 0:
            # Aplicamos la segunda parte del Coseno solo a esas canciones
            puntuaciones = puntuaciones_brutas * item_norms_inv[canciones_posibles]
            
            # Filtramos las semillas usando una máscara booleana (más rápido)
            mask = ~np.isin(canciones_posibles, semillas)
            canciones_posibles = canciones_posibles[mask]
            puntuaciones = puntuaciones[mask]
            
            # Sacamos el Top 500
            if len(canciones_posibles) >= 500:
                top_500_local = np.argpartition(puntuaciones, -500)[-500:]
                top_500_local = top_500_local[np.argsort(puntuaciones[top_500_local])[::-1]]
                valid_idx = canciones_posibles[top_500_local]
            else:
                top_500_local = np.argsort(puntuaciones)[::-1]
                valid_idx = canciones_posibles[top_500_local]
                
            recommendations = [index_to_track[idx] for idx in valid_idx]
        else:
            recommendations = []
            
        # Rellenamos si no llegamos a 500
        if len(recommendations) < 500:
            actuales = set(recommendations)
            semillas_uris = batch_seeds_uris[row_idx]
            for uri in top_popular_uris:
                if uri not in actuales and uri not in semillas_uris:
                    recommendations.append(uri)
                    if len(recommendations) == 500:
                        break
                        
        results[pid] = recommendations

# GUARDADO
print(f"\nGuardando CSV en {output_file_path}...")
with open(output_file_path, "w") as f:
    f.write("team_info, Pablo Fernandez Rubal - Noura el Morchid - Gonzalo Ponte Rodriguez, pablo.fernandez.rubal@udc.es - n.elmorchid@udc.es - g.ponte@udc.es\n")
    for pid in (results.keys()):
        f.write(f"{pid}," + ",".join(results[pid]) + "\n")
        
