import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from UnetClass import Unet
from UnetClass import CombinedUnet
from Scheduler import LinearNoiseScheduler
from PIL import Image
import torchvision.transforms as transforms
from timeit import default_timer as timer

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
        im = Image.open(image_path).convert('L')
        im = im.resize(im_size, Image.Resampling.LANCZOS)
        im_tensor = transforms.ToTensor()(im)
        im_tensor = (2 * im_tensor) - 1
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

        start_time = timer()
        noise_pred = model(xt, c ,torch.as_tensor(i).unsqueeze(0).to(device)) #- Model_original(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        torch.cuda.synchronize()  # Sincroniza la GPU
        inference_time = (timer() - start_time) * 1000
        sum_time += inference_time
        #print(f"Tiempo de inferencia: {inference_time:.2f} ms")
        #noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join(train_config['task_name'], 'samples'+"Conditional")):
            os.mkdir(os.path.join(train_config['task_name'], 'samples'+"Conditional"))
        img.save(os.path.join(train_config['task_name'], 'samples'+"Conditional", 'x0_{}.png'.format(i)))
        img.close()
    average_inference_time = sum_time / diffusion_config['num_timesteps']
    print(f"Tiempo de inferencia: {average_inference_time:.2f} ms")

def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    conditional_config = config['conditional_params']
    
    c = load_single_image(conditional_config['path_conditional']).to(device)
    
    # Load model with checkpoint
    Model_original = Unet(model_config).to(device)
    Model_original.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                str(train_config["num_model"])+"_"+train_config['ckpt_name']), map_location=device))
    Model_original.eval()
    
    ##
    route_original = train_config['task_name']+"/"+str(train_config["num_model"])+"_"+train_config['ckpt_name']
    route_conditional = train_config['task_name']+"/"+str(train_config["num_model"])+"_"+train_config['ckpt_event_branch']
    
    model = CombinedUnet(model_config, model_config).to(device)
    load_checkpoint(model, route_original,route_conditional)
    model.eval()
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample(Model_original ,model,c ,scheduler, train_config, model_config, diffusion_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    
    
    yamls = ['Training_yaml/0.yaml']
    # yamls = ['Training_yaml/0.yaml',
    #          'Training_yaml/1.yaml',
    #          'Training_yaml/2.yaml',
    #          'Training_yaml/3.yaml',
    #          'Training_yaml/4.yaml',
    #          'Training_yaml/5.yaml',
    #          'Training_yaml/6.yaml',
    #          'Training_yaml/7.yaml',
    #          'Training_yaml/8.yaml',
    #          'Training_yaml/9.yaml']
    
    
    for route_yamls in yamls:
        args = parser.parse_args(['--config', route_yamls])
        infer(args)
    