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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_checkpoint(model, checkpoint_path_Unet, checkpoint_path_Partial):
    model.unet.load_state_dict(torch.load(checkpoint_path_Unet,map_location=device))
    print("Loading Model...")
    try:
        model.partial_unet.load_state_dict(torch.load(checkpoint_path_Partial, map_location=device))
    except RuntimeError as e:
        print(f"Error al cargar los pesos de PartialUnet: {e}")

def load_single_image(image_path, im_size=(28, 28)):
        """Cargar y transformar una sola imagen desde el archivo."""
        im = Image.open(image_path).convert('L') #Escala de grises
        im = im.resize(im_size, Image.Resampling.LANCZOS)
        im_tensor = transforms.ToTensor()(im)
        im_tensor = (2 * im_tensor) - 1  #[0,1] to [-1, 1]
        return im_tensor#.unsqueeze(0)  # Añadir una dimensión para el batch    

def sample(Model_original, model, c ,scheduler, train_config, model_config, diffusion_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    xt = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)

    sum_time = 0
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        #noise_pred = Model_original(xt, torch.as_tensor(i).unsqueeze(0).to(device)) + 0.5*(model(xt, c ,torch.as_tensor(i).unsqueeze(0).to(device)) - Model_original(xt, torch.as_tensor(i).unsqueeze(0).to(device)))

        noise_pred = model(xt, c ,torch.as_tensor(i).unsqueeze(0).to(device)) #- Model_original(xt, torch.as_tensor(i).unsqueeze(0).to(device))
      
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
    ims = torch.clamp(xt, -1., 1.).detach().cpu() #Cortar en -1 y 1
    return ims


def Metrics(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    #print(config)
    ######################## MODEL ##################################################
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    conditional_config = config['conditional_params'] 
    
    # Load model with checkpoint
    Model_original = Unet(model_config).to(device)
    Model_original.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                str(train_config["num_model"])+"_"+train_config['ckpt_name']), map_location=device))         
    Model_original.eval()
    
    route_original = train_config['task_name']+"/"+str(train_config["num_model"])+"_"+train_config['ckpt_name']
    route_conditional = train_config['task_name']+"/"+str(train_config["num_model"])+"_"+train_config['ckpt_event_branch']
    
    model = CombinedUnet(model_config, model_config).to(device)
    load_checkpoint(model, route_original,route_conditional)
    model.eval()       
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
           
    ########################### METRICS #############################################
    groth_truth_folder = os.path.join(conditional_config['path_ground_truth'], str(train_config['task_name'][-1]))
    condition_folder = os.path.join(conditional_config['path_condition'], str(train_config['task_name'][-1]))
    
    print(groth_truth_folder, condition_folder)
    if not os.path.exists(groth_truth_folder) or not os.path.exists(condition_folder):
        print(f"La carpeta {train_config['task_name'][-1]} no existe en una de las rutas.")

    # Obtenemos los nombres de los archivos en cada carpeta
    files_truth = set(os.listdir(groth_truth_folder))
    file_condition = set(os.listdir(condition_folder))
    # Comparamos archivos
    Common_files = files_truth.intersection(file_condition)
    total_len = len(Common_files) 
    iteration = 0
    
    for file in iter(Common_files):  #Sacar el iter para generar en todo el conjunto y borrar el break de abajo
        iteration+= 1
        print(f"Sampling progress: {iteration}/{total_len}")
        print(f"Trabajando el archivo {file}")
        file_truth = os.path.join(groth_truth_folder, file)
        file_condition = os.path.join(condition_folder, file)
        
        c = load_single_image(file_condition).to(device) #Range [-1, 1]            
        truth = load_single_image(file_truth).to(device) #Range [-1, 1]
        
        with torch.no_grad():
            ims = sample(Model_original ,model,c ,scheduler, train_config, model_config, diffusion_config) 
            # Mostrar las imágenes
            c = (c + 1) / 2 #[0,1]
            #print("Rango max-min C",torch.max(c), torch.min(c))
            
            truth = (truth + 1) / 2
            #print("Rango max-min truth",torch.max(c), torch.min(c))
            ims = ((ims[0] + 1) / 2).to(device) 

            #print("Rango max-min truth",torch.max(ims), torch.min(ims))
            
            #print(c.shape, truth.shape, ims.shape)
            #mse = torch.mean((truth - ims) ** 2)

            #print(f"Mean Squared Error (MSE): {mse.item()}")
            
            truth_np = truth.permute(1, 2, 0).cpu().numpy()
            ims_np = ims.permute(1, 2, 0).cpu().numpy()
            c_np = c.permute(1,2,0).cpu().numpy()
           
            print("truth: ",np.shape(truth_np))
            print("ims: ", np.shape(ims_np))
            print("C: ",np.shape(c_np))            
            
            # ##GENERATION
            ims_np = (ims_np * 255).astype(np.uint8)
            
            image = Image.fromarray(ims_np.squeeze(), 'L')
            if not os.path.exists(os.path.join(train_config['task_name'], "Generations")):
                os.mkdir(os.path.join(train_config['task_name'], "Generations"))
            save_rute = os.path.join(train_config['task_name'], "Generations", file)            
            image.save(save_rute)

            
    

            #ssim_value, _ = compare_ssim(truth.squeeze(0).cpu().numpy(), ims.squeeze(0).cpu().numpy(), full=True, multichannel=False, win_size=7, data_range=1.0)
            #print(f"Structural Similarity Index (SSIM): {ssim_value}")
            
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(truth_np, cmap='gray')  # Cambiar orden de dimensiones para Matplotlib (C, H, W -> H, W, C)
            axs[0].set_title("Ground Truth")
            axs[0].axis("off")
            axs[1].imshow(c_np , cmap='gray')
            axs[1].set_title("Condition")
            axs[1].axis("off")
            axs[2].imshow(ims_np, cmap='gray')
            axs[2].set_title("Generation")
            axs[2].axis("off")
            
            
            # axs[2].text(0.5, -0.1, f"MSE = {mse.cpu():.6f}", fontsize=12, color='blue', ha='center', transform=axs[2].transAxes)
            # axs[2].text(0.5, -0.2, f"SSIM = {ssim_value:.6f}", fontsize=12, color='blue', ha='center', transform=axs[2].transAxes)

            plt.show()
            print("#"*20)
        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    
    
    yamls = ['Training_yaml/0.yaml',
             'Training_yaml/1.yaml',
             'Training_yaml/2.yaml',
             'Training_yaml/3.yaml',
             'Training_yaml/4.yaml',
             'Training_yaml/5.yaml',
             'Training_yaml/6.yaml',
             'Training_yaml/7.yaml',
             'Training_yaml/8.yaml',
             'Training_yaml/9.yaml']
    
    
    for route_yamls in yamls:
        args = parser.parse_args(['--config', route_yamls])
        Metrics(args)
    