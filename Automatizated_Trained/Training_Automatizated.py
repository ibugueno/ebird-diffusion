import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from UnetClass import Unet
from Scheduler import LinearNoiseScheduler
from Datasets import SingleImageDataset

from UnetClass import CombinedUnet
from Scheduler import LinearNoiseScheduler
from PairedDataSet import PairedImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Train_MainBranch(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    #print(config)
    ########################
    
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
    mnist = SingleImageDataset(im_paths=dataset_paths)
    #mnist = SingleImageDataset(im_path=dataset_config['im_path'])
    mnist_loader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)
    
    # Instantiate the model
    model = Unet(model_config).to(device)
    #print(model)
    model.train()
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # Load checkpoint if found
    if os.path.exists(os.path.join(train_config['task_name'],str(train_config['epoch_resume'])+"_"+train_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        print(train_config['task_name'], str(train_config['epoch_resume'])+"_"+train_config['ckpt_name'])
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      str(train_config['epoch_resume'])+"_"+train_config['ckpt_name']), map_location=device))
    else:
        print("Not found Checkpoint")
    
    # Specify training parameters
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()
    
    # Run training
    graph_losses_epoch = []
    log_file = train_config["log_loss"]
    
    if os.path.exists(log_file):
        with open(log_file, 'w') as file:
            file.write("")  
            
            
    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(mnist_loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        name = "Main_"+str(epoch_idx+1)+"_"+train_config['task_name']+".pth"
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))
        
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                        str(epoch_idx + 1)+"_"+train_config['ckpt_name']))
        
        graph_losses_epoch.append(np.mean(losses))


    for epoch, loss in enumerate(graph_losses_epoch, 1):
        log_message = f"Epoch {epoch}: Loss = {loss:.4f}\n"
        print(log_message, end="")  
        
        # Escribir el log en el archivo
        with open(log_file, 'a') as file:
            file.write(log_message)

    print(f"Log guardado en {log_file}")
    
    for var in list(locals().keys()):
        del locals()[var]

    print("Deleting locals variables...")
    
    print('Done Training ...')
    

def load_checkpoint(model, checkpoint_path_Unet, checkpoint_path_Partial=None):
    model.unet.load_state_dict(torch.load(checkpoint_path_Unet,map_location=device))
    print("Main Model Found")
    # try:
    #     model.partial_unet.load_state_dict(torch.load(checkpoint_path_Partial, map_location=device))
    # except RuntimeError as e:
    #     print(f"Error al cargar los pesos de PartialUnet: {e}")
        
        
def Train_EventBranch(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    #print(config)
    ########################
     
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
    dataset = PairedImageDataset(dataset_paths, dataset_event_path, im_size=(28, 28))
    mnist_loader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)

    #model = Unet(model_config).to(device)   
    model = CombinedUnet(model_config, model_config).to(device)
    load_checkpoint(model, train_config['task_name']+"/"+str(train_config['num_epochs'])+"_"+train_config['ckpt_name'])

   # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # Specify training parameters
    num_epochs = train_config['num_epochs_2']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()
    
    
    graph_losses_epoch = []
    log_file = train_config["C_log_loss"]
    
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

        torch.save(model.partial_unet.state_dict(), os.path.join(train_config['task_name'],
                                                        str(epoch_idx + 1)+"_"+train_config['ckpt_event_branch']))
    
        graph_losses_epoch.append(np.mean(losses))
    
    for epoch, loss in enumerate(graph_losses_epoch, 1):
        log_message = f"Epoch {epoch}: Loss = {loss:.4f}\n"
        print(log_message, end="")  
        
        # Escribir el log en el archivo
        with open(log_file, 'a') as file:
            file.write(log_message)

    print(f"Log guardado en {log_file}")                
    print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='default.yaml', type=str)    

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
        
        print('*'*40)
        print('*'*40)
        print('Model Config by: ',route_yamls)
        print('*'*40)
        print('*'*40)
        
        Train_MainBranch(args)
        Train_EventBranch(args)