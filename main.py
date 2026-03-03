import subprocess
import sys
import os

SCRIPTS_DIR = "scripts"

def ejecutar_script(nombre_script):
    ruta_script = os.path.join(SCRIPTS_DIR, nombre_script)
    
    print(f"Running -> {nombre_script}")
    
    if not os.path.exists(ruta_script):
        print(f"No se encuentra el archivo {ruta_script}")
        sys.exit(1)

    try:
        # Ejecutamos el script manteniendo el directorio de trabajo en la RAÍZ
        # Esto es vital para que las rutas "datos/" y "matrix/" dentro de tus
        # scripts sigan funcionando sin cambios.
        subprocess.run([sys.executable, ruta_script], check=True)
        
    except subprocess.CalledProcessError:
        print(f"Error al ejecutar -> {nombre_script}.")
        sys.exit(1)

if __name__ == "__main__":
    input_iteracion = input("Ingresa el número de la iteración a ejecutar: ")
    if input_iteracion not in ["0", "1"]:
        print("Iteración no válida. Por favor, ingresa 0 o 1.")
        sys.exit(1)
    if input_iteracion == "0":
        ejecutar_script("creacion_matrix.py")
        ejecutar_script("baseline_popularidad.py")
        ejecutar_script("evaluacion.py")
        