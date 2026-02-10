import zipfile
import json
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz

# Path to the ZIP file
zip_path = "spotify_test_playlists.zip"

# Listas para coordenadas de la matriz dispersa (Filas, Columnas, Valores)
rows = []
cols = []
data = []

# Diccionarios para mapear URIs/PIDs a índices enteros (0, 1, 2...)
pid_to_index = {}
track_to_index = {}

current_pid_idx = 0
current_track_idx = 0

print("Procesando archivo zip...")

# Open the ZIP file
with zipfile.ZipFile(zip_path, "r") as zipf:
    for file in zipf.namelist():
        if file.endswith(".json"):
            with zipf.open(file) as f:
                file_data = json.loads(f.read())
                
                for playlist in file_data["playlists"]:
                    pid = playlist["pid"]
                    
                    # 1. REGISTRO DE PLAYLIST (Aunque esté vacía)
                    # Si no hemos visto esta playlist, le asignamos un índice nuevo
                    if pid not in pid_to_index:
                        pid_to_index[pid] = current_pid_idx
                        current_pid_idx += 1
                    
                    # Obtenemos el índice numérico de la fila actual
                    p_idx = pid_to_index[pid]
                    
                    # 2. PROCESAMIENTO DE TRACKS (Solo si existen)
                    if playlist["tracks"]:
                        for track in playlist["tracks"]:
                            track_uri = track["track_uri"]
                            
                            # Si no hemos visto este track, le asignamos un índice nuevo
                            if track_uri not in track_to_index:
                                track_to_index[track_uri] = current_track_idx
                                current_track_idx += 1
                            
                            t_idx = track_to_index[track_uri]
                            
                            # Guardamos la coordenada (Fila, Columna, Valor)
                            rows.append(p_idx)
                            cols.append(t_idx)
                            data.append(1)

# Conversión a arrays de numpy para eficiencia
rows_np = np.array(rows)
cols_np = np.array(cols)
data_np = np.array(data)

# Dimensiones reales de la matriz
# Cantidad de playlists únicas y tracks únicos encontrados
num_playlists = len(pid_to_index)
num_tracks = len(track_to_index)

print(f"Playlists encontradas: {num_playlists}")
print(f"Tracks únicos encontrados: {num_tracks}")

# Creación de la matriz CSR
# IMPORTANTE: El shape se define con los contadores totales, no con len(rows)
matrix = csr_matrix((data_np, (rows_np, cols_np)), 
                    shape=(num_playlists, num_tracks))

# Save the matrix
save_npz("sparse_matrix_test.npz", matrix)
print("CSR matrix created and saved to 'sparse_matrix_test.npz'.")

# --- VERIFICACIÓN ---
matrix2 = load_npz("sparse_matrix_test.npz")
print("\nMatrix loaded verification:")
print(f"Shape: {matrix2.shape} (Filas [Playlists] x Columnas [Tracks])")
print(f"Non-zero elements: {matrix2.nnz}")

# Verificamos si hay filas vacías (playlists sin canciones)
# Calculamos la suma por fila, si es 0, la playlist está vacía
filas_con_datos = np.diff(matrix2.indptr) # Forma rápida de contar elementos por fila en CSR
playlists_vacias = np.sum(filas_con_datos == 0)

print(f"Playlists vacías detectadas e incluidas en la matriz: {playlists_vacias}")