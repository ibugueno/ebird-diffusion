import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Leer el archivo CSV
archivo_entrada = "Metrics.csv"  # Reemplaza con la ruta de tu archivo CSV
df = pd.read_csv(archivo_entrada)

# Agrupar por 'Number' y calcular los promedios de 'MSE' y 'SSIM'
resumen = df.groupby("Number")[["MSE", "SSIM"]].mean().reset_index()
resumen[["MSE", "SSIM"]] = resumen[["MSE", "SSIM"]].round(4)
# Guardar la tabla de resumen en un nuevo archivo CSV
archivo_salida = "resumen.csv"
resumen.to_csv(archivo_salida, index=False)


print(f"Tabla de resumen guardada en {archivo_salida}")

# Calcular la media de MSE y SSIM por clase
mean_mse = df.groupby("Number")["MSE"].mean().reset_index()
mean_ssim = df.groupby("Number")["SSIM"].mean().reset_index()
# Filtrar los datos de la clase 5
df_clase_5 = df[df["Number"] == 5]

# Calcular los valores de Q1, Q3 y IQR solo para la clase 5
Q1 = df_clase_5["MSE"].quantile(0.25)
Q3 = df_clase_5["MSE"].quantile(0.75)
IQR = Q3 - Q1

# Filtrar los outliers solo en la clase 5
df_clase_5_filtrada = df_clase_5[(df_clase_5["MSE"] >= Q1 -30 * IQR) & (df_clase_5["MSE"] <= Q3 +30 * IQR)]

# Obtener los datos de las otras clases (sin filtrado)
df_otros = df[df["Number"] != 5]

# Combinar los datos filtrados de la clase 5 con el resto de las clases
df_filtrado_completo = pd.concat([df_otros, df_clase_5_filtrada])

# Verificar cuántos registros fueron eliminados
outliers_eliminados = len(df_clase_5) - len(df_clase_5_filtrada)
print(f"Outliers eliminados en la clase 5: {outliers_eliminados}")


# Crear gráfico de violín para MSE (sin outliers)
plt.figure(figsize=(8, 6))
sns.violinplot(
    x="Number", 
    y="MSE", 
    data=df_filtrado_completo, 
    inner="quart",  # Muestra la mediana y los cuartiles internos
    color="#4169E1",  # Color de las violinas
    linewidth=1.2,
    bw=0.2  # Reducción de la suavidad para quitar los outliers
)



# Configuración del gráfico
plt.title("MSE Distribution per Class", fontsize=16)
plt.xlabel("Class", fontsize=14)
plt.ylabel("MSE", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Crear gráfico de violín para SSIM con separación de 0.05 en el eje Y
plt.figure(figsize=(8, 6))
sns.violinplot(
    x="Number", 
    y="SSIM", 
    data=df, 
    inner="quart",  # Muestra la mediana y los cuartiles internos
    color="#4169E1",  # Color de las violinas
    linewidth=1.2,
    bw=0.2  # Reducción de la suavidad para mejorar la visualización
)



# Configuración del gráfico para SSIM
plt.title("SSIM Distribution per Class", fontsize=16)
plt.xlabel("Class", fontsize=14)
plt.ylabel("SSIM", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Establecer la separación del eje Y de 0.05 en 0.05
plt.yticks([i / 20 for i in range(0, 21)], fontsize=12)

plt.grid(axis="y", linestyle="--", alpha=0.7)

# Mostrar el gráfico
plt.tight_layout()
plt.show()