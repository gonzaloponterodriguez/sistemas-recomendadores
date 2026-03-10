import sys
import zipfile
import json
import numpy as np
import os
from scipy.sparse import csr_matrix, save_npz

zip_path = "datos/spotify_train_dataset.zip" 
output_dir = "matrix"
matrix_path = os.path.join(output_dir, "sparse_matrix_train.npz")
track_json_path = os.path.join(output_dir, "track_to_index_train.json")
pid_json_path = os.path.join(output_dir, "pid_to_index_train.json")

# COMPROBACIÓN INICIAL
# Comprobamos si los 3 archivos ya existen en el disco
if os.path.exists(matrix_path) and os.path.exists(track_json_path) and os.path.exists(pid_json_path):
    print("La matriz y los diccionarios ya estaban creados en el directorio.")
    print("Se cancela la ejecución para evitar duplicar el trabajo.")
    sys.exit()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


rows = []
cols = []
pid_to_index = {}
track_to_index = {}
current_pid_idx = 0
current_track_idx = 0

print(f"Procesando archivo {zip_path}...")

with zipfile.ZipFile(zip_path, "r") as zipf:
    for file in zipf.namelist():
        if file.endswith(".json"):
            with zipf.open(file) as f:
                file_data = json.loads(f.read())
                
                for playlist in file_data["playlists"]:
                    pid = playlist["pid"]
                    
                    # 1. REGISTRO DE PLAYLIST
                    if pid not in pid_to_index:
                        pid_to_index[pid] = current_pid_idx
                        current_pid_idx += 1
                    
                    p_idx = pid_to_index[pid]
                    
                    # 2. PROCESAMIENTO DE TRACKS
                    # Usamos .get() para evitar errores si no hay clave 'tracks'
                    tracks = playlist.get("tracks", [])
                    
                    if tracks:
                        for track in tracks:
                            track_uri = track["track_uri"]
                            
                            if track_uri not in track_to_index:
                                track_to_index[track_uri] = current_track_idx
                                current_track_idx += 1
                            
                            t_idx = track_to_index[track_uri]
                            
                            rows.append(p_idx)
                            cols.append(t_idx)

print("Convirtiendo a arrays...")

rows_np = np.array(rows, dtype=np.int32)
cols_np = np.array(cols, dtype=np.int32)
data_np = np.ones(len(rows_np), dtype=np.int8)

num_playlists = len(pid_to_index)
num_tracks = len(track_to_index)

print(f"Playlists encontradas: {num_playlists}")
print(f"Tracks únicos encontrados: {num_tracks}")

# Creación y guardado de la matriz CSR
matrix = csr_matrix((data_np, (rows_np, cols_np)), 
                    shape=(num_playlists, num_tracks),
                    dtype=np.int8)

matrix_path = os.path.join(output_dir, "sparse_matrix_train.npz")
save_npz(matrix_path, matrix)
print(f"Matriz CSR guardada en: {matrix_path}")


# Guardado de diccionarios de mapeo para futuro uso
print("Guardando diccionarios de mapeo...")

# Guardado track_to_index (URI -> ID) para saber qué canción es el "track 0", "track 1", etc.
track_json_path = os.path.join(output_dir, "track_to_index_train.json")
with open(track_json_path, "w") as f:
    json.dump(track_to_index, f)
print(f"Diccionario de Tracks guardado en: {track_json_path}")

# Guardamos pid_to_index (PID real -> ID fila)
pid_json_path = os.path.join(output_dir, "pid_to_index_train.json")
with open(pid_json_path, "w") as f:
    json.dump(pid_to_index, f)
print(f"Diccionario de Playlists guardado en: {pid_json_path}")
