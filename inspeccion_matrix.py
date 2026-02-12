import numpy as np
from scipy.sparse import load_npz
import sys

file_path = "matrix/sparse_matrix_test.npz"

print(f"--- Cargando matriz desde {file_path} ---")
try:
    matrix = load_npz(file_path)
except FileNotFoundError:
    print("Error: No se encuentra el archivo .npz")
    sys.exit()

n_playlists, n_tracks = matrix.shape
print(f"Matriz cargada. Dimensiones: {n_playlists} x {n_tracks}")

# --- 1. DETECCIÓN DE VACÍAS ---
print("\n--- Buscando playlists vacías ---")
# getnnz(axis=1) cuenta cuántos valores no-cero hay en cada fila. Es instantáneo.
canciones_por_playlist = matrix.getnnz(axis=1)

# Buscamos los índices donde el recuento es 0
indices_vacias = np.where(canciones_por_playlist == 0)[0]
num_vacias = len(indices_vacias)

if num_vacias > 0:
    print(f"⚠️ SE ENCONTRARON {num_vacias} PLAYLISTS VACÍAS.")
    print(f"Ejemplos de índices vacíos: {indices_vacias[:5]} ...")
else:
    print("✅ No hay playlists vacías. Todas tienen al menos 1 canción.")


# --- 2. FUNCIÓN DE VISUALIZACIÓN ---
def inspeccionar_fila(idx, descripcion):
    if idx >= n_playlists:
        return # Evitar errores de índice fuera de rango
        
    print(f"\n[{descripcion}] Inspeccionando Playlist (Fila {idx}):")
    
    # Extraemos la fila (sin convertir a densa para no explotar la RAM)
    fila = matrix[idx]
    
    # .indices nos da los IDs de columnas (tracks) que tienen un 1
    track_ids = fila.indices
    
    if len(track_ids) == 0:
        print(f"   estado: ❌ VACÍA (0 tracks)")
        print(f"   contenido: []")
    else:
        print(f"   estado: ✅ CON DATOS ({len(track_ids)} tracks)")
        # Mostramos los primeros 5 y los últimos 5 para no saturar
        if len(track_ids) > 10:
            vista = f"[{', '.join(map(str, track_ids[:5]))}, ..., {', '.join(map(str, track_ids[-5:]))}]"
        else:
            vista = track_ids
        print(f"   contenido (IDs de columnas): {vista}")

# --- 3. EJECUCIÓN DE PRUEBAS ---
print("\n--- Muestreo de Datos ---")

# A) El principio
inspeccionar_fila(0, "INICIO")
inspeccionar_fila(1, "INICIO")

# B) El medio
idx_medio = n_playlists // 2
inspeccionar_fila(idx_medio, "MEDIO")

# C) El final
inspeccionar_fila(n_playlists - 1, "FINAL")

# D) Prueba específica de vacía (Si existe alguna)
if num_vacias > 0:
    idx_vacia = indices_vacias[0] # Cogemos la primera que encontramos
    inspeccionar_fila(idx_vacia, "PRUEBA DE VACÍA")