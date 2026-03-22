import os
import sys
import json
import numpy as np
import zipfile
from scipy.sparse import load_npz
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Obligar a NumPy a usar 1 solo hilo por proceso.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# CONFIGURACIÓN 
K_NEIGHBORS = 500
input_dir = "matrix"
test_zip_path = "datos/spotify_test_playlists.zip"
vecinos_path = "vecinos.json"
output_file_path = "resultados/user_based.csv"

os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# VARIABLES GLOBALES (Para que los trabajadores compartan RAM y no dupliquen la matriz)
train_matrix = None
index_to_track = None
top_popular_uris = None

# LA FUNCIÓN QUE EJECUTARÁ CADA HILO
def procesar_playlist(args):
    pid, data, semillas, semillas_uris_set = args
    
    k_indices = data["indices"][:K_NEIGHBORS]
    
    if len(k_indices) > 0:
        similitudes_base = np.array(data["similitudes"][:K_NEIGHBORS])
        pesos_potenciados = np.power(similitudes_base, 3) 
        
        # Multiplicación 
        submatriz_vecinos = train_matrix[k_indices]
        item_scores = submatriz_vecinos.T.dot(pesos_potenciados).flatten()
        
        # Sacamos SOLO los IDs de las canciones que los vecinos realmente tienen
        canciones_posibles = np.unique(submatriz_vecinos.indices)
        
        # Quitamos las semillas de esa lista para no recomendarlas
        if semillas:
            canciones_posibles = np.setdiff1d(canciones_posibles, semillas, assume_unique=True)
        
        # Extraemos las puntuaciones solo de esas canciones válidas
        puntuaciones_posibles = item_scores[canciones_posibles]
        
        # Buscamos el Top 500 en esta lista reducida de canciones posibles
        if len(puntuaciones_posibles) >= 500:
            top_500_local = np.argpartition(puntuaciones_posibles, -500)[-500:]
            top_500_local = top_500_local[np.argsort(puntuaciones_posibles[top_500_local])[::-1]]
            # Recuperamos el ID real de la canción
            top_500_idx = canciones_posibles[top_500_local] 
        else:
            top_500_local = np.argsort(puntuaciones_posibles)[::-1]
            top_500_idx = canciones_posibles[top_500_local]
            
        # Filtramos (por si hubiera alguna con score <= 0)
        valid_idx = top_500_idx[item_scores[top_500_idx] > 0]
        recommendations = [index_to_track[idx] for idx in valid_idx]
        
        knn_count = len(recommendations)
        
        if knn_count < 500:
            actuales = set(recommendations)
            for uri in top_popular_uris:
                if uri not in actuales and uri not in semillas_uris_set:
                    recommendations.append(uri)
                    if len(recommendations) == 500:
                        break
    else:
        recommendations = top_popular_uris.copy()
        knn_count = 0 
        
    return pid, recommendations, knn_count, len(semillas)

# BLOQUE PRINCIPAL
if __name__ == '__main__':

    if os.path.exists(output_file_path):
        print("Las recomendaciones ya estaban creadas en el directorio.")
        print("Se cancela la ejecución para evitar duplicar el trabajo.")
        sys.exit()

    print(f"INICIANDO FASE 2: GENERACIÓN DE RECOMENDACIONES USER-BASED (K={K_NEIGHBORS}) ---")
    # CARGA DE DATOS A VARIABLES GLOBALES
    train_matrix = load_npz(os.path.join(input_dir, "sparse_matrix_train.npz")).tocsr()
    with open(os.path.join(input_dir, "track_to_index_train.json"), "r") as f:
        track_to_index = json.load(f)
    index_to_track = {v: k for k, v in track_to_index.items()}
    with open(vecinos_path, "r") as f:
        vecinos_data = json.load(f)

    popularity_counts = np.array(train_matrix.sum(axis=0)).flatten()
    top_popular_indices = np.argsort(popularity_counts)[-500:][::-1]
    top_popular_uris = [index_to_track[idx] for idx in top_popular_indices]

    test_seeds = {}
    test_seeds_uris = {}
    with zipfile.ZipFile(test_zip_path, "r") as zipf:
        for filename in zipf.namelist():
            if filename.endswith("test_input_playlists.json"):
                with zipf.open(filename) as f:
                    test_raw = json.loads(f.read())
                break
                
    for pl in test_raw.get("playlists", []):
        pid = pl["pid"]
        valid_tracks = [t["track_uri"] for t in pl["tracks"] if t["track_uri"] in track_to_index]
        test_seeds_uris[pid] = set(valid_tracks) 
        test_seeds[pid] = [track_to_index[uri] for uri in valid_tracks]

    # PREPARAR TAREAS PARA EL POOL
    tareas = []
    for pid_str, data in vecinos_data.items():
        pid = int(pid_str)
        tareas.append((
            pid, 
            data, 
            test_seeds.get(pid, []), 
            test_seeds_uris.get(pid, set())
        ))

    # EJECUCIÓN EN PARALELO
    n_cores = max(1, cpu_count() - 1) # Dejamos 1 núcleo libre para que tu PC no se congele
    print(f"Iniciando Pool con {n_cores} procesos usando tqdm...")
    
    results = {}
    total_knn_aportado = 0  
    total_playlists_con_semillas = 0

    with Pool(processes=n_cores) as pool:
        # El mapeo paralelo con la barra de progreso
        for result in tqdm(pool.imap_unordered(procesar_playlist, tareas), total=len(tareas), desc="Recomendando"):
            pid, recommendations, knn_count, num_semillas = result
            results[pid] = recommendations
            
            # Recopilamos estadísticas globalmente
            if num_semillas > 0:
                total_playlists_con_semillas += 1
                total_knn_aportado += knn_count

    # ESTADÍSTICAS GLOBALES
    if total_playlists_con_semillas > 0:
        media_knn = total_knn_aportado / total_playlists_con_semillas
        print(f"\n--- ESTADÍSTICAS DEL KNN (K={K_NEIGHBORS}) ---")
        print(f"De las {total_playlists_con_semillas} playlists que SÍ tenían canciones previas:")
        print(f"El modelo KNN aportó de media {media_knn:.2f} canciones por playlist.")
        print(f"La popularidad rellenó de media {500 - media_knn:.2f} canciones por playlist.\n")

    # GUARDAR
    print(f"Guardando CSV en {output_file_path}...")
    with open(output_file_path, "w") as f:
        f.write("team_info, Pablo Fernandez Rubal - Noura el Morchid - Gonzalo Ponte Rodriguez, pablo.fernandez.rubal@udc.es - n.elmorchid@udc.es - g.ponte@udc.es\n")
        for pid in (results.keys()): 
            f.write(f"{pid}," + ",".join(results[pid]) + "\n")
            
    print("¡Proceso multiproceso completado con éxito!")