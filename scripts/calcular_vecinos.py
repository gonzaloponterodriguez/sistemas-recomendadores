import json
import sys
import numpy as np
import os
import zipfile
from scipy.sparse import load_npz, csr_matrix
from tqdm import tqdm

# CONFIGURACIÓN 
input_dir = "matrix"
test_zip_path = "datos/spotify_test_playlists.zip"
output_file_path = "vecinos.json"
MAX_VECINOS_A_GUARDAR = 500
BATCH_SIZE = 500  

# COMPROBACIÓN INICIAL
if os.path.exists(output_file_path):
    print("Los vecinos ya estaban creados en el directorio.")
    print("Se cancela la ejecución para evitar duplicar el trabajo.")
    sys.exit()

print("INICIANDO FASE 1: PRECOMPUTACIÓN DE VECINOS")

# CARGA DE DATOS
try:
    print("Cargando matriz de entrenamiento...")
    train_matrix = load_npz(os.path.join(input_dir, "sparse_matrix_train.npz"))
    with open(os.path.join(input_dir, "track_to_index_train.json"), "r") as f:
        track_to_index = json.load(f)
except FileNotFoundError:
    print("Error: No se encuentran los archivos en la carpeta 'matrix'")
    sys.exit()

num_train_playlists, num_tracks = train_matrix.shape

# Precomputar normas de entrenamiento para el Coseno (y su inversa para multiplicar más rápido)
print("Precomputando normas de las playlists de entrenamiento...")
train_norms = np.sqrt(train_matrix.getnnz(axis=1))
train_norms[train_norms == 0] = 1e-9 # Evitar divisiones por cero
train_norms_inv = 1.0 / train_norms  # Multiplicar es mucho más rápido que dividir

# PROCESAR TEST 
print(f"Leyendo archivo de test: {test_zip_path}...")
vecinos_guardados = {}

with zipfile.ZipFile(test_zip_path, "r") as zipf:
    for filename in zipf.namelist():
        if filename.endswith("test_input_playlists.json"):
            with zipf.open(filename) as f:
                test_data = json.loads(f.read())
            break
            
playlists = test_data.get("playlists", [])
total_playlists = len(playlists)

print(f"Calculando similitudes en lotes de {BATCH_SIZE}...")

# Iteramos en bloques (batches)
for i in tqdm(range(0, total_playlists, BATCH_SIZE), desc="Procesando Lotes"):
    batch = playlists[i:i + BATCH_SIZE]
    
    # Construir la matriz dispersa para este lote
    data, rows, cols = [], [], []
    batch_pids = []
    user_norms = []
    
    for row_idx, playlist in enumerate(batch):
        pid = playlist["pid"]
        batch_pids.append(pid)
        
        user_track_indices = [track_to_index[t["track_uri"]] for t in playlist["tracks"] if t["track_uri"] in track_to_index]
        
        if user_track_indices:
            data.extend([1.0] * len(user_track_indices))
            rows.extend([row_idx] * len(user_track_indices))
            cols.extend(user_track_indices)
            user_norms.append(np.sqrt(len(user_track_indices)))
        else:
            # Cold start
            user_norms.append(1.0) # Evita división por cero
            
    # Si el lote entero está vacío , lo saltamos
    if not data:
        for pid in batch_pids:
            vecinos_guardados[pid] = {"indices": [], "similitudes": []}
        continue

    # Creamos la matriz del lote entero
    batch_matrix = csr_matrix((data, (rows, cols)), shape=(len(batch), num_tracks))
    
    # MULTIPLICACIÓN MATRICIAL
    # Multiplicamos el lote de test (ej: 1000xN) por toda la de train transpuesta (NxM)
    # Resultado: matriz densa de 1000 x M (M = num playlists entrenamiento)
    dot_products = batch_matrix.dot(train_matrix.T).toarray()
    
    # CÁLCULO DE SIMILITUD VECTORIZADO 
    user_norms_inv = 1.0 / np.array(user_norms)
    
    # Fórmula del coseno optimizada con numpy broadcasting
    similarities = dot_products * train_norms_inv
    similarities *= user_norms_inv[:, np.newaxis] 
    
    # EXTRACCIÓN DEL TOP K PARA TODO EL LOTE A LA VEZ 
    # Extraemos los 200 mejores índices para cada fila del lote de golpe
    top_k_idx = np.argpartition(similarities, -MAX_VECINOS_A_GUARDAR, axis=1)[:, -MAX_VECINOS_A_GUARDAR:]
    
    # Guardamos los resultados
    for row_idx in range(len(batch)):
        pid = batch_pids[row_idx]
        
        # Verificamos si era cold start
        if user_norms[row_idx] == 1.0 and np.sum(batch_matrix[row_idx]) == 0:
            vecinos_guardados[pid] = {"indices": [], "similitudes": []}
            continue
            
        row_sims = similarities[row_idx]
        row_top_k_idx = top_k_idx[row_idx]
        
        # Ordenamos solo los ganadores de mayor a menor
        sorted_idx = row_top_k_idx[np.argsort(row_sims[row_top_k_idx])[::-1]]
        sorted_sims = row_sims[sorted_idx]
        
        vecinos_guardados[pid] = {
            "indices": sorted_idx.tolist(),
            "similitudes": sorted_sims.tolist()
        }

# GUARDAR JSON 
print(f"\nGuardando similitudes en {output_file_path}...")
with open(output_file_path, "w") as f:
    json.dump(vecinos_guardados, f)
print("¡Fase 1 completada con éxito!")