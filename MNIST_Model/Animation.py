import cv2
import os
import re

def ordenar_por_numero(archivo):
    # Extrae el número después de "x0_" para ordenar correctamente
    numero = re.search(r'x0_(\d+)', archivo)
    return int(numero.group(1)) if numero else float('inf')

def crear_video(ruta_carpeta = 'default/samples', nombre_video="Animation.mp4", fps=100):
    # Obtiene todas las imágenes en la carpeta y las ordena numéricamente
    archivos = sorted([os.path.join(ruta_carpeta, archivo) for archivo in os.listdir(ruta_carpeta) if archivo.endswith(('png', 'jpg', 'jpeg'))], key=ordenar_por_numero, reverse=True)

    # Verifica si hay imágenes en la carpeta
    if not archivos:
        print("No se encontraron imágenes en la carpeta.")
        return
    
    # Lee la primera imagen para obtener el tamaño del video
    frame = cv2.imread(archivos[0])
    alto, ancho, _ = frame.shape
    tamano = (ancho, alto)

    # Define el codec y crea el objeto VideoWriter
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(nombre_video, codec, fps, tamano)

    # Agrega cada imagen al video
    for archivo in archivos:
        frame = cv2.imread(archivo)
        video.write(frame)

    # Libera el objeto VideoWriter
    video.release()
    print(f"Video guardado como {nombre_video}")

# Ejemplo de uso
if __name__ == '__main__':
    crear_video()
