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
        """
        xt: Tensor of shape (B, C, H, W)
        noise_pred: Tensor of shape (B, C, H, W)
        t: Tensor of shape (B,) with timestep indices for each image in the batch
        """

        B, C, H, W = xt.shape

        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t].view(B, 1, 1, 1)
        alpha_cum_prod = self.alpha_cum_prod.to(xt.device)[t].view(B, 1, 1, 1)
        betas = self.betas.to(xt.device)[t].view(B, 1, 1, 1)
        alphas = self.alphas.to(xt.device)[t].view(B, 1, 1, 1)

        x0 = (xt - sqrt_one_minus_alpha_cum_prod * noise_pred) / torch.sqrt(alpha_cum_prod)
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - (betas * noise_pred) / sqrt_one_minus_alpha_cum_prod
        mean = mean / torch.sqrt(alphas)

        # Mascara para los elementos donde t == 0
        is_t0 = (t == 0).view(B, 1, 1, 1)

        # Calcular la varianza y ruido
        prev_t = torch.clamp(t - 1, min=0)
        alpha_cum_prev = self.alpha_cum_prod.to(xt.device)[prev_t].view(B, 1, 1, 1)
        variance = (1 - alpha_cum_prev) / (1.0 - alpha_cum_prod)
        variance = variance * betas
        sigma = torch.sqrt(variance)

        z = torch.randn_like(xt)

        xt_prev = mean + sigma * z
        xt_prev = torch.where(is_t0, x0, xt_prev)

        return xt_prev, x0
