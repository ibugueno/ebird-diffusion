import re

import numpy as np
import matplotlib.pyplot as plt

# Función para leer y procesar un archivo .log
def read_log_file(filepath):
    epochs = []
    losses = []
    with open(filepath, 'r') as file:
        for line in file:
            match = re.match(r"Epoch (\d+): Loss = ([0-9.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return epochs, losses



logs = ["/home/fbn/fv_event-based_diffusion_modeling_image_reconstruction/Automatizated_Trained/default_0",
        "/home/fbn/fv_event-based_diffusion_modeling_image_reconstruction/Automatizated_Trained/default_1",
        "/home/fbn/fv_event-based_diffusion_modeling_image_reconstruction/Automatizated_Trained/default_2",
        "/home/fbn/fv_event-based_diffusion_modeling_image_reconstruction/Automatizated_Trained/default_3",
        "/home/fbn/fv_event-based_diffusion_modeling_image_reconstruction/Automatizated_Trained/default_4",
        "/home/fbn/fv_event-based_diffusion_modeling_image_reconstruction/Automatizated_Trained/default_5",
        "/home/fbn/fv_event-based_diffusion_modeling_image_reconstruction/Automatizated_Trained/default_6",
        "/home/fbn/fv_event-based_diffusion_modeling_image_reconstruction/Automatizated_Trained/default_7",
        "/home/fbn/fv_event-based_diffusion_modeling_image_reconstruction/Automatizated_Trained/default_8",
        "/home/fbn/fv_event-based_diffusion_modeling_image_reconstruction/Automatizated_Trained/default_9"]


num = int(len(logs)/2) 
fig, axes = plt.subplots(num, 2, figsize=(15, 20), constrained_layout=True) 
pos = 0
col = 0
for i in range(0,num*2):
    file1, file2 = logs[i]+"/training.log", logs[i]+"/C_training.log"
    epochs1, losses1 = read_log_file(file1)

    epochs2, losses2 = read_log_file(file2)
    epochs2 = [epoch+40 for epoch in epochs2]



    print(pos, col," par")    
    ax = axes[pos][col]
    if col == 1:
        col = 0
    else:
        col+=1
    if col==0:
        pos+=1  
    
    ax.plot(epochs1, losses1, label=f"Main Branch", marker='.')
    ax.plot(epochs2, losses2, label=f"Event Branch", marker='.')
    ax.set_title(f"Epoch vs Log Loss - Number {i}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log Loss")
    ax.legend()
    ax.grid()
    # Configuración adicional del eje Y: incrementos de 0.05
    y_min, y_max = min(min(losses1), min(losses2)), max(max(losses1), max(losses2))
    y_ticks = np.arange(np.floor(y_min * 20) / 20, np.ceil(y_max * 20) / 20 + 0.05, 0.1)
    ax.set_yticks(y_ticks)  # Establecer ticks en intervalos de 0.05
    ax.set_yscale('log')  # Cambiar el eje Y a escala logarítmica

# Ajustar diseño
plt.tight_layout()
plt.subplots_adjust(hspace=0.9)
plt.show()
