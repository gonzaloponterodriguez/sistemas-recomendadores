import zipfile
import json
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz

zip_path = "datos/spotify_test_playlists.zip"

# Usamos listas estándar de Python (son más rápidas para 'append' que numpy arrays en bucle)
rows = []
cols = []

# Mapeos
pid_to_index = {}
track_to_index = {}

current_pid_idx = 0
current_track_idx = 0

print("Procesando archivo zip...")

with zipfile.ZipFile(zip_path, "r") as zipf:
    for file in zipf.namelist():
        if file.endswith(".json"):
            with zipf.open(file) as f:
                # Carga el JSON completo en memoria (cuidado si el JSON es gigante)
                file_data = json.loads(f.read())
                
                for playlist in file_data["playlists"]:
                    pid = playlist["pid"]
                    
                    # 1. REGISTRO DE PLAYLIST
                    # Mapeamos el PID a un índice incremental (0, 1, 2...)
                    if pid not in pid_to_index:
                        pid_to_index[pid] = current_pid_idx
                        current_pid_idx += 1
                    
                    p_idx = pid_to_index[pid]
                    
                    # 2. PROCESAMIENTO DE TRACKS
                    # Solo procesamos si hay tracks. Si no, p_idx se queda reservado pero sin entradas en rows/cols
                    if playlist["tracks"]:
                        for track in playlist["tracks"]:
                            track_uri = track["track_uri"]
                            
                            if track_uri not in track_to_index:
                                track_to_index[track_uri] = current_track_idx
                                current_track_idx += 1
                            
                            t_idx = track_to_index[track_uri]
                            
                            rows.append(p_idx)
                            cols.append(t_idx)
                            # No hacemos append a 'data' aquí para ahorrar RAM

print("Convirtiendo a arrays...")

# Conversión a arrays de numpy
rows_np = np.array(rows, dtype=np.int32) # int32 es suficiente para índices hasta 2 mil millones
cols_np = np.array(cols, dtype=np.int32)

# Generamos la data directamente como un array de 1s
# Usamos int8 porque solo necesitamos 1 o 0
data_np = np.ones(len(rows_np), dtype=np.int8)

num_playlists = len(pid_to_index)
num_tracks = len(track_to_index)

print(f"Playlists encontradas (Total filas): {num_playlists}")
print(f"Tracks únicos encontrados (Total columnas): {num_tracks}")

# Creación de la matriz CSR
# Especificamos dtype=np.int8 para que la matriz final ocupe poco espacio
matrix = csr_matrix((data_np, (rows_np, cols_np)), 
                    shape=(num_playlists, num_tracks),
                    dtype=np.int8)

save_npz("matrix/sparse_matrix_test.npz", matrix)
print("CSR matrix created and saved.")


# --- VERIFICACIÓN ---
matrix2 = load_npz("matrix/sparse_matrix_test.npz")

# Verificamos filas vacías
# axis=1 suma a través de las columnas (para cada fila)
# Aplanar el resultado (A1) para tener un array 1D
items_por_playlist = matrix2.getnnz(axis=1) 
playlists_vacias = np.sum(items_por_playlist == 0)

print(f"\nVerificación:")
print(f"Dimensiones: {matrix2.shape}")
print(f"Playlists vacías detectadas: {playlists_vacias}")


# --- EXPLICACIÓN ---
# Logramos representar las playlists vacías asignando un índice de fila a CADA playlist encontrada (en pid_to_index),
# pero solo añadiendo entradas a las listas de coordenadas (rows/cols) si existen canciones.
# Al instanciar la csr_matrix con shape=(num_playlists, num_tracks), Scipy genera la matriz del tamaño total
# y rellena automáticamente con 0s aquellas filas que reservamos pero para las que no aportamos datos.