import json
import sys
import numpy as np
import os
import zipfile
from scipy.sparse import load_npz

input_dir = "matrix"
output_file_path = "resultados/baseline.csv"
test_zip_path = "datos/spotify_test_playlists.zip"  

os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# COMPROBACIÓN INICIAL
if os.path.exists(output_file_path) :
    print("El baseline ya estaba creado en el directorio.")
    print("Se cancela la ejecución para evitar duplicar el trabajo.")
    sys.exit()



# CARGA DE DATOS
try:
    # Carga de la matriz 
    train_matrix = load_npz(os.path.join(input_dir, "sparse_matrix_train.npz"))
    
    # Carga e inversión del diccionario para traducir de ID a URI de la cancion. 
    with open(os.path.join(input_dir, "track_to_index_train.json"), "r") as f:
        track_to_index = json.load(f)
    index_to_track = {v: k for k, v in track_to_index.items()}
    
except FileNotFoundError:
    print("No se encuentran los archivos en la carpeta 'matrix'")
    exit()

# CALCULO POPULARIDAD
# Sumamos todas las columnas (axis=0) para ver cuántas veces aparece cada canción
popularity_counts = np.array(train_matrix.sum(axis=0)).flatten()

# Obtenemos los índices ordenados de MAYOR a MENOR popularidad con argsort invertido
sorted_indices = np.argsort(popularity_counts)[::-1]

# PREPARAR LA LISTA 
# En lugar de revisar las 2 millones de canciones para cada usuario,
# cogemos las Top 1000. Como solo necesitamos recomendar 500, 
# 1000 es suficiente margen por si el usuario ya tiene muchas de las populares.
print("Generando lista maestra de Top Tracks...")
top_n_indices = sorted_indices[:1000]
top_n_uris = [index_to_track[idx] for idx in top_n_indices]

# GENERAR RECOMENDACIONES PARA TEST 
print(f"Procesando archivo {test_zip_path}...")

results = {}

try:
    with zipfile.ZipFile(test_zip_path, "r") as zipf:
        for filename in zipf.namelist():
            if filename.endswith("test_input_playlists.json"):
                with zipf.open(filename) as f:
                    test_data = json.loads(f.read())
                break
        
    # El archivo de test suele tener una estructura {"playlists": [...]}
    playlists = test_data.get("playlists", [])
    
    total = len(playlists)
    print(f"Generando recomendaciones para {total} playlists...")
    
    for i, playlist in enumerate(playlists):
        pid = playlist["pid"]
        
        # Obtenemos las canciones que YA están en la playlist (semillas)
        # Nota: Dependiendo del formato de test, 'tracks' puede ser una lista de objetos o URIs.
        # Asumimos formato estándar MPD donde tracks es una lista de objetos con "track_uri"
        current_tracks = set()
        for t in playlist["tracks"]:
            current_tracks.add(t["track_uri"])
            
        recommendations = []
        
        # Recorremos nuestra lista de populares
        for track_uri in top_n_uris:
            # Si no tiene la cancion ya, la recomendamos
            if track_uri not in current_tracks:
                recommendations.append(track_uri)
            
            if len(recommendations) == 500:
                break
        
        # Guardamos el resultado para este PID
        results[pid] = recommendations
        
        if (i + 1) % 1000 == 0:
            print(f"Procesado: {i + 1}/{total}")

    # GUARDAR RESULTADOS
    print(f"Guardando resultados en {output_file_path}...")
    with open(output_file_path, "w") as f:
        f.write("team_info, Pablo Fernandez Rubal - Noura el Morchid - Gonzalo Ponte Rodriguez, pablo.fernandez.rubal@udc.es - n.elmorchid@udc.es - g.ponte@udc.es\n")
        for pid, tracks in results.items():
            linea = f"{pid}," + ",".join(tracks)
            f.write(linea + "\n")
            
    print("Archivo generado correctamente")

except FileNotFoundError:
    print(f"No se encuentra el archivo {test_zip_path}")