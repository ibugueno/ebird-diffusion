import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Lista de rutas a CSV junto con el nombre del modelo
rutas = [
    (Path("metrics_model25") / "resultados_metricas.csv", "Conditional 25%"),
    (Path("metrics_model50") / "resultados_metricas.csv", "Conditional 50%"),
    (Path("metrics_model75") / "resultados_metricas.csv", "Conditional 75%"),
    (Path("metrics_model100") / "resultados_metricas.csv", "Conditional 100%")
]

# Métricas a graficar
metricas = ['SSIM', 'PSNR', 'MSE']

# Diccionario donde guardar los DataFrames por modelo
dataframes = {}

# Leer y almacenar cada CSV
for ruta, nombre in rutas:
    if not ruta.exists():
        print(f"❌ Archivo no encontrado: {ruta}")
        continue
    df = pd.read_csv(ruta)
    dataframes[nombre] = df

# Verificar que haya al menos un archivo cargado
if not dataframes:
    raise ValueError("No se pudo cargar ningún CSV. Verifica las rutas.")

# Obtener clases
primer_nombre = list(dataframes.keys())[0]
clases = sorted(dataframes[primer_nombre]['Clase'].unique())

# Colores para los modelos
colores = plt.cm.tab10.colors

# Ancho de cada barra
n_modelos = len(dataframes)
bar_width = 0.8 / n_modelos  # Total ancho por grupo es ~0.8

# === 1. Gráficos de barras por clase ===
for metrica in metricas:
    plt.figure(figsize=(12, 6))
    x = np.arange(len(clases))  # posiciones base

    for i, (nombre_modelo, df) in enumerate(dataframes.items()):
        df_sorted = df.sort_values("Clase")
        valores = df_sorted[metrica].values

        # Ajuste horizontal para agrupar barras
        posiciones = x + (i - n_modelos / 2) * bar_width + bar_width / 2
        plt.bar(posiciones, valores,
                width=bar_width,
                color=colores[i % len(colores)],
                label=nombre_modelo)

    plt.title(f'Comparación de {metrica} por clase')
    plt.xlabel('Clase')
    plt.ylabel(metrica)
    plt.xticks(x, clases)
    plt.grid(True, linestyle='--', axis='y', alpha=0.6)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.12), ncol=n_modelos)
    plt.tight_layout()
    plt.savefig(f'barras_{metrica}.png')
    plt.show()

# === 2. Calcular promedios por modelo y métrica ===
promedios = []

for nombre_modelo, df in dataframes.items():
    fila = {'Modelo': nombre_modelo}
    for metrica in metricas:
        fila[metrica] = df[metrica].mean()
    promedios.append(fila)

df_promedios = pd.DataFrame(promedios)
df_promedios.to_csv("promedios_metricas.csv", index=False)
print("\n✅ Tabla de promedios guardada en 'promedios_metricas.csv'")
print(df_promedios.round(4))

# === 3. Gráficos de barras por métrica con promedios ===
# === 3. Gráficos de barras por métrica con promedios ===
for metrica in metricas:
    plt.figure(figsize=(8, 5))
    modelos = df_promedios['Modelo']
    valores = df_promedios[metrica]

    # Calcular rango Y con margen para mejor precisión visual
    ymin = valores.min()
    ymax = valores.max()
    margen = (ymax - ymin) * 0.1 if ymax != ymin else 0.1
    plt.ylim(ymin - margen, ymax + margen)

    plt.bar(modelos, valores, color=colores[:len(modelos)], edgecolor='black')
    plt.title(f'Average {metrica} per model')
    plt.ylabel(f'Average {metrica}')
    plt.xticks(rotation=45)

    # Mejorar precisión y formato de ticks Y
    # Ejemplo: generar 6 ticks en el eje Y
    from matplotlib.ticker import MaxNLocator
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))  # 6 ticks máximo, sin extremos sobrantes

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'promedio_{metrica}.png')
    plt.show()

