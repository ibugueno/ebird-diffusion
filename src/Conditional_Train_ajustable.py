import torch
import torch
import yaml
import argparse
import os
import logging
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from UnetClass import Unet
from UnetClass import CombinedUnet
from Scheduler import LinearNoiseScheduler
from PairedDataSet_ajustable import PairedImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def configurar_logger(subset_ratio_, nueva_ruta):
    """Configura un nuevo logger con una nueva ruta y cierra el anterior."""
    
    # Crear la nueva carpeta si no existe
    os.makedirs(nueva_ruta, exist_ok=True)

    # Nombre del archivo de log con la nueva ruta
    text_title_login = os.path.join(nueva_ruta, f"logfile_{int(subset_ratio_ * 100)}.log")

    # Obtener el logger raíz
    logger = logging.getLogger()

    # Cerrar handlers existentes
    while logger.hasHandlers():
        logger.handlers[0].close()
        logger.removeHandler(logger.handlers[0])

    # Configurar el nuevo logger
    logging.basicConfig(
        filename=text_title_login,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    return logger

def load_checkpoint(model, checkpoint_path_Unet, checkpoint_path_Partial=None):
    model.unet.load_state_dict(torch.load(checkpoint_path_Unet,map_location=device))
    print("Si")
    # try:
    #     model.partial_unet.load_state_dict(torch.load(checkpoint_path_Partial, map_location=device))
    # except RuntimeError as e:
    #     print(f"Error al cargar los pesos de PartialUnet: {e}")
        
        
def main(args, subset_ratio_):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    #print(config)
    ########################

    logger = configurar_logger(subset_ratio_, "Rislab_Event_influence_volume")

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    dataset_paths = dataset_config['paths']
    dataset_event_path = dataset_config['paths_event']        
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    # Create the dataset
    dataset = PairedImageDataset(dataset_paths, dataset_event_path, im_size=(28, 28), subset_ratio = subset_ratio_)
    mnist_loader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)

    #model = Unet(model_config).to(device)   
    model = CombinedUnet(model_config, model_config).to(device)
    load_checkpoint(model, "Rislab_Event_influence_volume/ddpm_ckpt.pth")

   # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # Specify training parameters
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()
    
    # Run training
    for epoch_idx in range(num_epochs):
        losses = []
        for batch_idx, (im, c) in enumerate(tqdm(mnist_loader, desc="Entrenando")):
            im = im.float().to(device)  # Asegúrate de que im esté en la GPU
            c = c.float().to(device)  
            optimizer.zero_grad()
            im = im.float().to(device)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, c, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))
        text_log = "Epoch-"+str(epoch_idx + 1)
        logger.info(f"{text_log}: {np.mean(losses)}")
        torch.save(model.partial_unet.state_dict(), os.path.join(train_config['task_name'],
                                                    "Combinet"+str(int(subset_ratio_*100))+".pth"))
    print('Done Training ...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='src/default.yaml', type=str)
    args = parser.parse_args()

    main(args, subset_ratio_ = 0.25)
    main(args, subset_ratio_ = 0.50)
    main(args, subset_ratio_ = 0.75)
    main(args, subset_ratio_ = 1.0)