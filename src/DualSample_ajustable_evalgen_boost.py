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
from math import ceil
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_checkpoint(model, checkpoint_path_Unet, checkpoint_path_Partial):
    model.unet.load_state_dict(torch.load(checkpoint_path_Unet,map_location=device))
    print("The main weights have been loaded")
    try:
        model.partial_unet.load_state_dict(torch.load(checkpoint_path_Partial, map_location=device))
        print("Conditional weights have been loaded")
    except RuntimeError as e:
        print(f"Error al cargar los pesos de PartialUnet: {e}")

def load_images_in_batch(image_paths, im_size=(28, 28)):
    batch = []
    for path in image_paths:
        im = Image.open(path).convert('L')
        im = im.resize(im_size, Image.Resampling.LANCZOS)
        im_tensor = transforms.ToTensor()(im)
        im_tensor = (2 * im_tensor) - 1
        batch.append(im_tensor)
    return torch.stack(batch)  

def sample(Model_original, model, c ,scheduler, train_config, model_config, diffusion_config, subset_ratio_, archivo, class_name):
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
        
        samples_route = os.path.join(train_config['task_name'], "samples")
        aux = "subset_"+str(int(subset_ratio_*100))+"_porcent"
        subset_route = os.path.join(samples_route, aux)
        class_route = os.path.join(subset_route,class_name)

        # Save x0
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)

        if not os.path.exists(samples_route):
            os.mkdir(samples_route)

        if not os.path.exists(subset_route):
            os.mkdir(subset_route)

        if not os.path.exists(class_route):
            os.mkdir(class_route)

        if i == 0:
            img.save(os.path.join(class_route, '{}'.format(archivo)))
            img.close()
            return 
    average_inference_time = sum_time / diffusion_config['num_timesteps']
    print(f"Tiempo de inferencia: {average_inference_time:.2f} ms")

def sample_batch(Model_original, model, c_batch, scheduler, train_config, model_config, diffusion_config, subset_ratio_, nombres_archivos, class_name):
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

    samples_route = os.path.join(train_config['task_name'], "samples")
    aux = "subset_" + str(int(subset_ratio_ * 100)) + "_porcent"
    subset_route = os.path.join(samples_route, aux)
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

    print(f"Tiempo de inferencia promedio por paso: {sum_time / diffusion_config['num_timesteps']:.2f} ms")


def infer(args, subset_ratio_=0.05, batch_size=768):
    # Leer config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    conditional_config = config['conditional_params']

    # Cargar modelos
    Model_original = Unet(model_config).to(device)
    Model_original.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                           train_config['ckpt_name']), map_location=device))
    Model_original.eval()

    model = CombinedUnet(model_config, model_config).to(device)
    combinet = f"Rislab_Event_influence_volume/Combinet{int(subset_ratio_ * 100)}.pth"
    load_checkpoint(model, "Rislab_Event_influence_volume/ddpm_ckpt.pth", combinet)
    model.eval()

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
                    sample_batch(Model_original, model, c_batch, scheduler,
                                 train_config, model_config, diffusion_config,
                                 subset_ratio_, [os.path.basename(p) for p in batch_paths], class_name)
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
                        default='default.yaml', type=str)
    args = parser.parse_args()
    
    infer(args, subset_ratio_ = 0.25, batch_size=768)
    infer(args, subset_ratio_ = 0.50, batch_size=768)
    infer(args, subset_ratio_ = 0.75, batch_size=768)
    infer(args, subset_ratio_ = 1.0, batch_size=768)