import torch
import torchvision
import argparse
import yaml
import os
import numpy as np
from torchvision.utils import make_grid
from tqdm import tqdm
from UnetClass import Unet
from UnetClass import CombinedUnet
from Scheduler import LinearNoiseScheduler
from PIL import Image
import torchvision.transforms as transforms
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
import pandas as pd
from openpyxl import load_workbook

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def agregar_datos_csv(archivo, datos_filas, columnas):
    """
    Agrega nuevas filas a un archivo CSV existente, respetando las columnas existentes.

    Parameters:
    archivo (str): El nombre del archivo CSV.
    datos_filas (list of lists): Lista de listas, donde cada sublista representa una fila completa.
    columnas (list): Lista de nombres de las columnas en el CSV.
    """
    if os.path.exists(archivo):
        # Leer el archivo CSV existente
        df = pd.read_csv(archivo)

        # Verificar que las columnas en el archivo coincidan con las esperadas
        if not all(col in df.columns for col in columnas):
            print("Las columnas especificadas no coinciden con las del archivo.")
            return

        # Crear un DataFrame para las nuevas filas
        nuevas_filas = pd.DataFrame(datos_filas, columns=columnas)

        # Concatenar el DataFrame original con las nuevas filas
        df = pd.concat([df, nuevas_filas], ignore_index=True)

        # Guardar el DataFrame actualizado en el archivo CSV
        df.to_csv(archivo, index=False)
        #print("Nuevas filas agregadas exitosamente.")
    else:
        print(f"El archivo {archivo} no existe.")

        
def load_single_image(image_path, im_size=(28, 28)):
        """Cargar y transformar una sola imagen desde el archivo."""
        im = Image.open(image_path).convert('L') #Escala de grises
        #im = im.resize(im_size, Image.Resampling.LANCZOS)
        im_tensor = transforms.ToTensor()(im)
        #im_tensor = (2 * im_tensor) - 1  #[0,1] to [-1, 1]
        return im_tensor#.unsqueeze(0)  # Añadir una dimensión para el batch    

def Metrics(number):

    Number_list = []
    MSE_list = []
    SSIM_list = []
    Truth_route_list = []
    Generation_route_list = []
    file_list = []
    
    
    
    
    
    rute = number+'/Generations'
           
    ########################### METRICS #############################################
    groth_truth_folder = os.path.join('/home/fbn/Dataset Mnist - NMnist/MNIST/Test/',number[-1])
    generation_folder = os.path.join('/home/fbn/Dataset Mnist - NMnist/TEST GENERATIONS 10K/',rute)
    
    #print(groth_truth_folder)
    #print(generation_folder)


    ## Obtenemos los nombres de los archivos en cada carpeta
    files_truth = set(os.listdir(groth_truth_folder))
    file_generated = set(os.listdir(generation_folder))
    
    # Comparamos archivos
    Common_files = files_truth.intersection(file_generated)
    total_len = len(Common_files) 
    iteration = 0
    
    #print(Common_files)
    #print(total_len)
    
    for file in Common_files:
        
        
        ## XLXS ##
        file_list.append(file)
        Number_list.append(int(number[-1]))
        Truth_route_list.append(groth_truth_folder)
        Generation_route_list.append(generation_folder)
        ##
        
        iteration+= 1
        #print(f"Sampling progress: {iteration}/{total_len}")
        #print(f"Trabajando el archivo {file}")
        
        file_truth = os.path.join(groth_truth_folder, file)
        file_generated = os.path.join(generation_folder, file)
        
        generation = load_single_image(file_generated).to(device) #Range [0, 1]            
        truth = load_single_image(file_truth).to(device) #Range [0, 1]   
        
        mse = torch.mean((truth - generation) ** 2)
        #print(f"Mean Squared Error (MSE): {mse.item()}")
        MSE_list.append(mse.item())
        
        truth_np = truth.permute(1, 2, 0).cpu().numpy()
        generation_np = generation.permute(1, 2, 0).cpu().numpy()
        
        #print("truth: ",np.shape(truth_np))
        #print("generation: ", np.shape(generation_np))
    
    
        ssim_value, _ = compare_ssim(truth_np.squeeze(-1), generation_np.squeeze(-1), full=True, multichannel=False, win_size=7, data_range=1.0)
        #print(f"Structural Similarity Index (SSIM): {ssim_value}")
        SSIM_list.append(ssim_value)
        
        # fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        # axs[0].imshow(truth_np, cmap='gray')  # Cambiar orden de dimensiones para Matplotlib (C, H, W -> H, W, C)
        # axs[0].set_title("Ground Truth")
        # axs[0].axis("off")
        # axs[1].imshow(generation_np , cmap='gray')
        # axs[1].set_title("Generation")
        # axs[1].axis("off")

        # axs[1].text(0.5, -0.05, f"MSE = {mse.cpu():.6f}", fontsize=12, color='blue', ha='center', transform=axs[1].transAxes)
        # axs[1].text(0.5, -0.1, f"SSIM = {ssim_value:.6f}", fontsize=12, color='blue', ha='center', transform=axs[1].transAxes)
        # plt.show()
        #print("#"*20)
    
    datos_columnas = {
        "Number": Number_list,
        "MSE": MSE_list,
        "SSIM": SSIM_list,
        "file": file_list,
        "Truth_route": Truth_route_list,
        "Generation_route": Generation_route_list
    }
    
    #print(datos_columnas)
    columnas = ["Number", "MSE", "SSIM", "file", "Truth_route", "Generation_route"]
    agregar_datos_csv(archivo, datos_columnas, columnas)    

if __name__ == '__main__':
    
    archivo = "Metrics.csv"
    
    if not os.path.exists(archivo):
        # Crear un DataFrame vacío con columnas iniciales
        columnas = ["Number", "MSE", "SSIM", "file", "Truth_route", "Generation_route"]  # Cambia según tus necesidades
        df = pd.DataFrame(columns=columnas)
        df.to_csv(archivo, index=False)  # Guardar el DataFrame en un archivo CSV
        print(f"Archivo {archivo} creado con columnas: {columnas}.")
    else:
        print(f"El archivo {archivo} ya existe.")
        
    
    
     
    number = ["default_0",
              "default_1",
              "default_2",
              "default_3",
              "default_4",
              "default_5",
              "default_6",
              "default_7",
              "default_8",
              "default_9",]
    for n in number:
        print(n)
        Metrics(n)
    