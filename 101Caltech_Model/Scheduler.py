import torch

class LinearNoiseScheduler: 
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps  #Numero de pasos
        self.beta_start = beta_start  #beta inicial
        self.beta_end = beta_end  #Beta Final
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps) #Tensor linspace desde beta inicio a beta final
        self.alphas = 1. - self.betas #Tensor de tamaño (numero de pasos) precalculado de todos los 1 - beta, es decir los alphas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0) #Tensor de Alpha gorro, cada elemento es la pitatoria de los alphas desde 0 hasta su indice correspondiente.
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod) #Tensor de la raiz de aplha gorro
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod) #tensor de la raiz de 1 - alpha gorro.


    def add_noise(self, original, noise, t):
        original_shape = original.shape  #Revisar las dimensiones
        batch_size = original_shape[0]  #Tamaño del Batch , recordar que suele venir BxCxHxW
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)   
    
        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
    
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) /
              torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)   #Valores dentro del rango [−1,1]: Permanecen sin cambios.
                                        #Valores menores que −1: Se limitan a −1.
                                        #Valores mayores que 1: Se limitan a 1.
        
        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])
        #Inferencia Pagina 47

        
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            
            # OR
            # variance = self.betas[t]
            # sigma = variance ** 0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0