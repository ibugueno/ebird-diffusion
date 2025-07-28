import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from UnetClass2 import Unet
from UnetClass2 import CombinedUnet
from Scheduler import LinearNoiseScheduler
from PIL import Image
import torchvision.transforms as transforms
from timeit import default_timer as timer
from math import ceil
import gc
from pathlib import Path
import logging
import platform
import psutil
import cpuinfo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(
level=logging.INFO,
format='[%(asctime)s] %(levelname)s - %(message)s',
handlers=[
    logging.FileHandler(os.path.join("Rislab_Event_influence_volume","inferencia.log")),   # Log en archivo
    logging.StreamHandler()                  # Y también en consola
    ]
)

def log_system_info():
    logging.info("=== Información del sistema ===")
    
    # Información general del sistema
    logging.info(f"Sistema operativo: {platform.system()} {platform.release()}")
    logging.info(f"Versión de Python: {platform.python_version()}")

    # CPU
    cpu = cpuinfo.get_cpu_info()
    logging.info(f"Procesador: {cpu.get('brand_raw', 'Desconocido')}")
    logging.info(f"Núcleos físicos: {psutil.cpu_count(logical=False)}")
    logging.info(f"Núcleos lógicos: {psutil.cpu_count(logical=True)}")

    # RAM
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    logging.info(f"RAM total: {ram_gb:.2f} GB")

    # GPU (si está disponible)
    if torch.cuda.is_available():
        logging.info("GPU disponible: SÍ")
        logging.info(f"Nombre GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        logging.info("GPU disponible: NO")

log_system_info()


def load_images_in_batch(image_paths, im_size=(28, 28)):
    batch = []
    for path in image_paths:
        im = Image.open(path).convert('L')
        im = im.resize(im_size, Image.Resampling.LANCZOS)
        im_tensor = transforms.ToTensor()(im)
        im_tensor = (2 * im_tensor) - 1
        batch.append(im_tensor)
    return torch.stack(batch)  

def sample_batch(output_route, model, c_batch, scheduler, model_config, diffusion_config, nombres_archivos, class_name):
    batch_size = c_batch.size(0)
    xt = torch.randn((batch_size,
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)

    sum_time = 0
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        start_time = timer()
        noise_pred = model(xt, c_batch, torch.full((batch_size,), i, device=device))
        torch.cuda.synchronize()
        inference_time = (timer() - start_time) * 1000
        sum_time += inference_time

        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.full((batch_size,), i, device=device))

    ims = torch.clamp(xt, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2

    subset_route = output_route
    class_route = os.path.join(subset_route, class_name)

    os.makedirs(class_route, exist_ok=True)

    for idx, im in enumerate(ims):
        im = im.float()
        img = torchvision.transforms.ToPILImage()(im)
        base_name = os.path.splitext(nombres_archivos[idx])[0]  # sin extensión
        filename = base_name + ".png"  # fuerza extensión .png
        img = img.convert('L')  # asegúrate de que sea 8-bit grayscale
        img.save(os.path.join(class_route, filename))
        img.close()


    mean_time_inference = round(sum_time / diffusion_config['num_timesteps'],4)
    logging.info(f"Tiempo de inferencia promedio por paso: {mean_time_inference} ms para {batch_size} imagenes.")
    logging.info(f"Tiempo total de generacion: {round(sum_time, 2)} ms. Para {batch_size} imagenes.")

def infer(args, DDPM_route,c_route,subset_ratio_=0.05, batch_size=768):
    # Leer config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    conditional_config = config['conditional_params']

    main_route = Path(c_route) / f"subset_{int(subset_ratio_ * 100)}"
    combinet_route = main_route / "checkpoints" / f"Combinetsubset_{int(subset_ratio_ * 100)}.pth"
    output_route = main_route / "samples"

    logging.info(f"Iniciando la generacion en: {output_route}")

    #Cargar los Modelos Unet + Controlnet
    model = CombinedUnet(model_config, model_config).to(device)

    logging.info(f"...Cargando los pesos de Unet ubicados en: {DDPM_route}")
    try:
        ckpt = torch.load(combinet_route, map_location='cpu')
        model.unet.load_state_dict(ckpt['unet'])
        logging.info("The main weights have been loaded")
    except Exception as e:
        logging.error(f"Error al cargar los pesos de Unet: {e}")

    logging.info(f"...Cargando los pesos de ControlNet ubicados en: {combinet_route}")
    try:
        ckpt_partial = torch.load(combinet_route, map_location='cpu')
        model.partial_unet.load_state_dict(ckpt_partial['partial_unet'])
        logging.info("Conditional weights have been loaded")
    except Exception as e:
        logging.error(f"Error al cargar los pesos de ControlNet: {e}")

    model.eval() #Modo Evaluacion

    scheduler = LinearNoiseScheduler(**diffusion_config)

    with torch.no_grad(): 
        for route in conditional_config['path_conditional']:
            archivos = os.listdir(route)
            full_paths = [os.path.join(route, archivo) for archivo in archivos]

            for i in range(0, len(full_paths), batch_size):
                batch_paths = full_paths[i:i + batch_size]
                c_batch = load_images_in_batch(batch_paths).to(device)

                class_name = route[-1]

                try:
                    sample_batch(output_route, model, c_batch, scheduler, model_config, diffusion_config,
                                [os.path.basename(p) for p in batch_paths], class_name)
                
                except RuntimeError as e:
                    print(f" Error en batch {i}-{i+batch_size}: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

                #Liberar memoria explícitamente tras cada batch
                del c_batch
                torch.cuda.empty_cache()
                gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='src/default.yaml', type=str)
    args = parser.parse_args()
    
    DDPM_route = "Rislab_Event_influence_volume/DDPM/checkpoints/ddpm_ckpt.pth"

    Conditonal_routes = [
        "Rislab_Event_influence_volume/all",
        "Rislab_Event_influence_volume/0",
        "Rislab_Event_influence_volume/1",
        "Rislab_Event_influence_volume/2",
        "Rislab_Event_influence_volume/3",
        "Rislab_Event_influence_volume/4",
        "Rislab_Event_influence_volume/5",
        "Rislab_Event_influence_volume/6",
        "Rislab_Event_influence_volume/7",
        "Rislab_Event_influence_volume/8", 
        "Rislab_Event_influence_volume/9"
    ]

    b_size = 768

    for c_route in Conditonal_routes:
        infer(args,DDPM_route, c_route, subset_ratio_ = 0.25, batch_size = b_size)
        infer(args,DDPM_route, c_route, subset_ratio_ = 0.50, batch_size = b_size)
        infer(args,DDPM_route, c_route, subset_ratio_ = 0.75, batch_size = b_size)
        infer(args,DDPM_route, c_route, subset_ratio_ =  1.0, batch_size = b_size)