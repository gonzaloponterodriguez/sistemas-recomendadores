import json
import numpy as np
import math
import os
import sys
import zipfile

predictions_path = sys.argv[1] if len(sys.argv) > 1 else "resultados/baseline.csv"
ground_truth_zip = "datos/spotify_test_playlists.zip" 

def r_precision(predicted, actual):
    """Calcula R-Precision: % de aciertos ajustado al tamaño de la verdad."""
    if not actual: return 0.0
    # Solo miramos tantas predicciones como canciones reales faltan (R)
    r = len(actual)
    predicted_r = set(predicted[:r])
    actual_set = set(actual)
    
    intersection = predicted_r.intersection(actual_set)
    return len(intersection) / r

def dcg(relevance_scores, k):
    """Discounted Cumulative Gain"""
    relevance_scores = np.asarray(relevance_scores, dtype=float)[:k]
    
    if relevance_scores.size == 0:
        return 0.0
    
    # Formula: rel_i / log2(i + 2)
    return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))

def ndcg(predicted, actual, k=500):
    """Normalized DCG: Qué tan bien ordenadas están las recomendaciones."""
    if not actual: return 0.0
    
    actual_set = set(actual)
    # Creamos un vector de 1s y 0s: 1 si la canción predicha es correcta, 0 si no
    relevance = [1 if x in actual_set else 0 for x in predicted]
    
    # DCG real (nuestra puntuación)
    dcg_val = dcg(relevance, k)
    
    # IDCG ideal (puntuación perfecta)
    ideal_relevance = [1] * len(actual) + [0] * (k - len(actual))
    idcg_val = dcg(ideal_relevance, k)
    
    if idcg_val == 0: return 0.0
    return dcg_val / idcg_val

def recommended_songs_clicks(predicted, actual):
    """
    Métrica específica de Spotify: 
    ¿Cuántos 'refrescos' de 10 canciones necesita el usuario 
    para encontrar la primera relevante?
    """
    if not actual: return 51 # Valor de penalización máximo + 1
    
    actual_set = set(actual)
    for i, track in enumerate(predicted):
        if track in actual_set:
            # Si está en la pos 0-9 -> 1 click
            # Si está en la pos 10-19 -> 2 clicks, etc.
            return math.floor(i / 10) + 1
            
    return 51 # Si no encontramos nada en las 500

# MAIN
print("Cargando archivos...")

if not os.path.exists(predictions_path):
    print(f"No se encuentra el archivo de predicciones: {predictions_path}")
    sys.exit(1)

# Cargar Predicciones
preds_data = {}
with open(predictions_path, 'r') as f:
    for line in f:
        line = line.strip()
        # Ignorar comentarios, líneas vacías y la cabecera del equipo
        if not line or line.startswith("#") or line.startswith("team_info"):
            continue
            
        # Separar por comas
        parts = line.split(",")
        pid = parts[0].strip() # Guardamos el PID como string temporalmente
        tracks = [t.strip() for t in parts[1:]] # Limpiamos posibles espacios en las URIs
        
        preds_data[pid] = tracks

# Cargar Ground Truth
with zipfile.ZipFile(ground_truth_zip, 'r') as zipf:
    for filename in zipf.namelist():
        if filename.endswith("test_eval_playlists.json"):
            with zipf.open(filename) as f:
                gt_data = json.loads(f.read())
            break

# Convertir GT a formato diccionario fácil si no lo es
gt_dict = {}
if "playlists" in gt_data:
    for pl in gt_data["playlists"]:
        # Extraemos SOLO las URIs de los tracks ocultos
        gt_dict[pl["pid"]] = [t["track_uri"] for t in pl["tracks"]]
else:
    print("Formato de test_eval.json desconocido.")
    exit()

print(f"Evaluando {len(preds_data)} playlists...")

r_precisions = []
ndcgs = []
clicks = []

for pid_str, predicted_tracks in preds_data.items():
    # El PID en el json a veces es string, a veces int. Aseguramos consistencia.
    pid = int(pid_str) 
    # Buscamos este PID en la verdad
    actual_tracks = gt_dict.get(pid) or gt_dict.get(str(pid))
    
    if actual_tracks is None:
        continue # Si no hay datos de verdad para esta playlist, saltamos
    
    # Calcular métricas para esta playlist
    rp = r_precision(predicted_tracks, actual_tracks)
    nd = ndcg(predicted_tracks, actual_tracks)
    cl = recommended_songs_clicks(predicted_tracks, actual_tracks)
    
    r_precisions.append(rp)
    ndcgs.append(nd)
    clicks.append(cl)

# RESULTADOS FINALES
print(f" EVALUACIÓN ({predictions_path}) \n")
print(f"R-Precision media: {np.mean(r_precisions):.4f}")
print(f"NDCG media:        {np.mean(ndcgs):.4f}")
print(f"Clicks medios:     {np.mean(clicks):.4f}")
