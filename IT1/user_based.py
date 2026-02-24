import json
import numpy as np
import os
import zipfile
from scipy.sparse import load_npz, csr_matrix

# --- CONFIGURACIÓN ---
input_dir = "matrix"
test_zip_path = "datos/spotify_test_playlists.zip"
output_file_path = "resultado_user_knn.csv"
K_NEIGHBORS = 150 # Hiperparámetro K: Cuántos vecinos similares vamos a buscar

print(f"--- INICIANDO MODELO USER-BASED KNN (K={K_NEIGHBORS}) ---")

# --- 1. CARGA DE DATOS ---
try:
    print("Cargando matriz de entrenamiento...")
    train_matrix = load_npz(os.path.join(input_dir, "sparse_matrix_train.npz"))
    
    with open(os.path.join(input_dir, "track_to_index_train.json"), "r") as f:
        track_to_index = json.load(f)
    index_to_track = {v: k for k, v in track_to_index.items()}
    
except FileNotFoundError:
    print("Error: No se encuentran los archivos en la carpeta 'matrix'")
    exit()

num_train_playlists, num_tracks = train_matrix.shape

# --- 2. PRECOMPUTAR NORMAS (Para Similitud del Coseno) ---
print("Precomputando normas de las playlists de entrenamiento...")
# La similitud del coseno es: (A · B) / (||A|| * ||B||)
# Calculamos ||B|| para todas las playlists de entrenamiento de golpe.
# Al elevar al cuadrado, sumar y hacer raíz cuadrada obtenemos la norma de cada fila.
train_norms = np.sqrt(train_matrix.power(2).sum(axis=1).A1)

# Evitamos divisiones por cero en playlists vacías (si las hubiera)
train_norms[train_norms == 0] = 1e-9 

# --- 3. PROCESAR TEST Y GENERAR RECOMENDACIONES ---
print(f"Procesando archivo comprimido {test_zip_path}...")
results = {}

try:
    with zipfile.ZipFile(test_zip_path, "r") as zipf:
        for filename in zipf.namelist():
            if filename.endswith("test_input_playlists.json"):
                with zipf.open(filename) as f:
                    test_data = json.loads(f.read())
                break
        
    playlists = test_data.get("playlists", [])
    total = len(playlists)
    print(f"Generando recomendaciones KNN para {total} playlists...")
    
    for i, playlist in enumerate(playlists):
        pid = playlist["pid"]
        
        # 1. Identificar qué canciones tiene el usuario (semillas)
        user_track_indices = []
        current_tracks_uris = set()
        
        for t in playlist["tracks"]:
            uri = t["track_uri"]
            current_tracks_uris.add(uri)
            # Solo podemos usar canciones que existían en el entrenamiento
            if uri in track_to_index:
                user_track_indices.append(track_to_index[uri])
                
        # Si la playlist tiene canciones conocidas, aplicamos KNN
        if user_track_indices:
            # 2. Crear el vector del usuario actual (sparse matrix de 1 fila)
            data = np.ones(len(user_track_indices))
            rows = np.zeros(len(user_track_indices))
            cols = np.array(user_track_indices)
            user_vector = csr_matrix((data, (rows, cols)), shape=(1, num_tracks))
            
            # 3. Calcular Similitud del Coseno con TODAS las playlists de train
            # Producto punto (A · B)
            dot_products = train_matrix.dot(user_vector.T).toarray().flatten()
            
            # Norma del usuario ||A||
            user_norm = np.sqrt(len(user_track_indices)) 
            
            # Similitud del coseno: (A · B) / (||A|| * ||B||)
            similarities = dot_products / (user_norm * train_norms)
            
            # 4. Obtener los K Vecinos más cercanos (Top K)
            # argsort ordena de menor a mayor, cogemos los últimos K y le damos la vuelta
            best_neighbors_idx = np.argsort(similarities)[-K_NEIGHBORS:][::-1]
            best_similarities = similarities[best_neighbors_idx]
            
            # 5. Calcular la predicción (Puntuación de las canciones)
            # Aplicamos tu optimización: Aislamos la submatriz de los K vecinos
            # Al multiplicar (1 x K) * (K x Num_Tracks), SciPy solo calcula
            # las puntuaciones para las columnas que tienen datos.
            neighbors_matrix = train_matrix[best_neighbors_idx] 
            
            # Multiplicamos la similitud de cada vecino por las canciones que tiene
            item_scores = best_similarities.dot(neighbors_matrix.toarray())
            
            # 6. Eliminar canciones que el usuario ya tiene (ponemos score a -1)
            item_scores[user_track_indices] = -1
            
            # 7. Extraer las Top 500 canciones recomendadas
            recommended_indices = np.argsort(item_scores)[-500:][::-1]
            
            recommendations = []
            for idx in recommended_indices:
                # Si llegamos a canciones con score <= 0 (nadie de los vecinos la tenía), paramos.
                # (Opcional: Si no llegamos a 500, se podrían rellenar con las más populares)
                if item_scores[idx] <= 0:
                    break
                recommendations.append(index_to_track[idx])
                
        else:
            # Si la playlist no tiene canciones que conozcamos (Cold Start), 
            # no podemos buscar vecinos. (Aquí idealmente iría el Baseline de Popularidad).
            recommendations = []
            
        # Guardamos en el diccionario
        results[pid] = recommendations
        
        if (i + 1) % 100 == 0:
            print(f"Procesado: {i + 1}/{total}")

    # --- GUARDAR RESULTADOS (FORMATO CSV COMPETICIÓN) ---
    print(f"Guardando resultados en {output_file_path}...")
    with open(output_file_path, "w") as f:
        f.write("team_info,Pablo Noura Gonzalo,contacto@email.com\n")
        for pid, tracks in results.items():
            linea = f"{pid}," + ",".join(tracks)
            f.write(linea + "\n")
            
    print("Archivo CSV de KNN generado correctamente.")

except FileNotFoundError:
    print(f"Error: No se encuentra el archivo {test_zip_path}")