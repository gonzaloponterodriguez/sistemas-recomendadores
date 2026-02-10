import zipfile
import json
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz

# Path to the ZIP file
zip_path = "spotify_test_playlists.zip"

playlist_id, track_id, value = [], [], []

# Open the ZIP file
with zipfile.ZipFile(zip_path, "r") as zipf:
    # Iterate over each file in the ZIP
    for file in zipf.namelist():
        # Check if the file is a JSON
        if file.endswith(".json"):
            # Read the content of the JSON file
            with zipf.open(file) as f:
                file_data = json.loads(f.read())
                # Extract row, column, and value data
                for playlist in file_data["playlists"]:
                    if playlist["tracks"]:
                        for tracks in playlist["tracks"]:
                            playlist_id.append(playlist["pid"])
                            track_id.append(tracks["track_uri"])
                            value.append(1)

def string_to_int(lista: list):
    unique_values = sorted(set(lista))
    mapeo = {valor: indice for indice, valor in enumerate(unique_values)}
    lista_int = [mapeo[s] for s in lista]

    return lista_int, mapeo

playlist_id_int, playlist_id_mapeo = string_to_int(playlist_id)
track_id_int, track_id_mapeo = string_to_int(track_id)

matrix = csr_matrix((value, (playlist_id_int, track_id_int)),
                     shape=(len(playlist_id_mapeo, len(track_id_mapeo))))


# Save the matrix to a .npz file
save_npz("sparse_matrix_test.npz", matrix, compressed=False)
print("CSR matrix created and saved to 'sparse_matrix_test.npz'.")

# Load the matrix (keep it sparse!)
matrix2 = load_npz("sparse_matrix_test.npz")
print("\nMatrix loaded from 'sparse_matrix_test.npz':")
print(f"Shape: {matrix2.shape}")
print(f"Non-zero elements: {matrix2.nnz}")
print(f"Sparsity: {(1 - matrix2.nnz / (matrix2.shape[0] * matrix2.shape[1])) * 100:.4f}%")
print(f"Data type: {matrix2.dtype}")

# View a small sample (e.g., first 10x10 block)
print("\nFirst 10x10 block:")
print(matrix2[:10, :1000].toarray())
