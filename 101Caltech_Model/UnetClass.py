import torch
import torch.nn as nn
import torch.nn.functional as F

def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DownBlock(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample using 2x2 average pooling
    """
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, out_channels)
             for _ in range(num_layers)]
        )
        
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers)]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
                                          4, 2, 1) if self.down_sample else nn.Identity()
    
    def forward(self, x, t_emb):
        out = x
        for i in range(self.num_layers):
            
            # Resnet block of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            # Attention block of Unet
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn
            
        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    r"""
    Mid conv block with attention.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers+1)
            ]
        )
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers + 1)
        ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers+1)
            ]
        )
        
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)]
        )
        
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers+1)
            ]
        )
    
    def forward(self, x, t_emb):
        out = x
        
        # First resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        for i in range(self.num_layers):
            
            # Attention Block
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn
            
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i+1](out)
            out = out + self.t_emb_layers[i+1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i+1](out)
            out = out + self.residual_input_conv[i+1](resnet_input)
        
        return out


class UpBlock(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        
        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)
            ]
        )
        
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)
        
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

        return out


class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """
    def __init__(self, model_config):
        super().__init__()
        im_channels = model_config['im_channels']
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim,
                                        down_sample=self.down_sample[i], num_layers=self.num_down_layers))
        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim,
                                      num_layers=self.num_mid_layers))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpBlock(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16,
                                    self.t_emb_dim, up_sample=self.down_sample[i], num_layers=self.num_up_layers))
        
        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        #print("Unet x Input: ",x.size())
        out = self.conv_in(x)
        #print("Unet Conv x Input: ",x.size())
        # B x C1 x H x W
        
        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        #print("Unet time embedding: ", t_emb.size())
        
        down_outs = []
        
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)
            #print("Unet Down: ", out.size())
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4
            
        for mid in self.mids:
            out = mid(out, t_emb)
            #print("Unet Mid: ", out.size())
        # out B x C3 x H/4 x W/4
        
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            #print("Unet Up: ", out.size())
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        #print("Unet norm: ", out.size())
        out = nn.SiLU()(out)
        #print("Unet SILU: ", out.size())
        out = self.conv_out(out)
        #print("Unet conv Out: ", out.size())
        # out B x C x H x W
        return out
    
    
#######################
#######################
###### UNET-C #########
#######################
####################### 
    
    
class ZeroConv1x1(nn.Module):
    """Convolución con pesos inicializados en cero."""
    def __init__(self, in_channels, out_channels, spass= False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.out_channels = out_channels
        self.spass = spass
        # Inicializa los pesos y sesgos en cero
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)
        
        # Definir la convolución transpuesta si up_sample es True
        self.up_sample_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

        # Inicializar los pesos de la convolución transpuesta (opcional)
        nn.init.constant_(self.up_sample_conv.weight, 0)
        nn.init.constant_(self.up_sample_conv.bias, 0)          
        
        
    def forward(self, x):
        # Aplicar convolución 1x1
        x = self.conv(x)        
        if self.spass:
            x = self.up_sample_conv(x)
        return x
    
class PartialUnet(nn.Module):
    """Partial Unet que contiene solo DownBlocks, MidBlocks, y ZeroConv1x1 para Skip Connections."""
    def __init__(self, model_config):
        super().__init__()

        # Configuración
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']

        # Proyección inicial
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        # Capa de entrada
        self.conv_in = nn.Conv2d(model_config['im_channels'], self.down_channels[0], kernel_size=3, padding=1)

        # Down Blocks
        self.downs = nn.ModuleList([
            DownBlock(self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
                      down_sample=self.down_sample[i], num_layers=self.num_down_layers)
            for i in range(len(self.down_channels) - 1)
        ])

        # Mid Blocks
        self.mids = nn.ModuleList([
            MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                     num_layers=self.num_mid_layers)
            for i in range(len(self.mid_channels) - 1)
        ])

        # Capa de ZeroConv1x1 para la condición inicial
        self.zero_cond_conv = ZeroConv1x1(model_config['im_channels'], self.down_channels[0])

        # ZeroConv1x1 para las Skip Connections
        self.skip_convs = nn.ModuleList([
        ZeroConv1x1(
        self.down_channels[i] * 2, 
        self.down_channels[i - 1] if i != 0 else 16, 
        spass=(i != 2)  # Solo True en las siguientes veces
        )
        for i in reversed(range(len(self.down_channels) - 1))
        ])
        
        

    def forward(self, x, condition, t):
        # Condición pasa primero por ZeroConv1x1
        #print("Unet Copy-Condition Input: ",condition.size())
        
        cond_out = self.zero_cond_conv(condition)
        #print("Unet Copy-Condition Conv: ",cond_out.size())
        
        # Entrada original pasa por la capa de entrada
        out = self.conv_in(x)
        #print("Unet Copy X input: ",cond_out.size())
        
        # Sumar salida de ZeroConv1x1 a la entrada de la Unet
        out += cond_out  # Asegúrate de que los canales sean compatibles aquí
        #print("Unet Copy X+C: ",out.size())
        #print("Suma de X + C se ha realizado correctamente")
        
        # Procesamiento a través de la red
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        #print("Unet Copy Time embbeding ",t_emb.size())
        
        
        down_outs = []
        for down in self.downs:
            out = down(out, t_emb)
            down_outs.append(out)
            #print("Unet Copy Down out: ", out.size())
        
        for mid in self.mids:
            out = mid(out, t_emb)
            #print("Unet Copy Mid out: ", out.size())
            
        zero_conv_outs = []
        # Aplicación de las conexiones directas mediante ZeroConv1x1
        for i, zero_conv in enumerate(self.skip_convs):
            down_out = down_outs[-(i + 1)]  # Conexión de la misma dimensión espacial desde el downsampling
            zero_out = zero_conv(down_out)  # Aplicar ZeroConv1x1 a la skip connection
            #print("Unet Copy zero out: ", zero_out.size())
            zero_conv_outs.append(zero_out)  # Guardar salida de ZeroConv1x1 para su uso posterior   
        #print("Unet Copy Completamente ejecutada\n")
        
        
        return down_outs, zero_conv_outs
    
class CombinedUnet(nn.Module):
    def __init__(self, unet_config, partial_unet_config):
        super().__init__()
        
        # U-Net Original Congelada
        self.unet = Unet(unet_config)
        for param in self.unet.parameters():
            param.requires_grad = False  # Congelar la red

        # U-Net Clonada para generar salidas de Zero Conv
        self.partial_unet = PartialUnet(partial_unet_config)
    
    def forward(self, x, condition, t):
        # Paso 1: Obtener las salidas de la U-Net congelada
        down_outs_unet, up_outs_unet = [], []
        
        # Extracción de capas de DownBlock y UpBlock en la U-Net congelada
        # Nota: Aquí puedes necesitar ajustar la U-Net original para extraer estas salidas, según la implementación

        # Paso 2: Obtener las salidas de las capas de Zero Conv en PartialUnet
        down_outs_partial, zero_conv_outs = self.partial_unet(x, condition, t)
        
        # Paso 3: Ejecutar a través de los bloques de subida en la U-Net original y sumar las salidas de Zero Conv
        out = self.unet.conv_in(x)
        t_emb = self.unet.t_proj(get_time_embedding(torch.as_tensor(t).long(), self.unet.t_emb_dim))
        
        # Aplicar DownBlocks y recolectar salidas para la fase de Upsampling
        for i, down in enumerate(self.unet.downs):
            down_outs_unet.append(out)
            out = down(out, t_emb)
            #print("Original Unet Down: ",out.size())
            
        # Aplicar MidBlocks
        for mid in self.unet.mids:
            out = mid(out, t_emb)
            #print("Original Mid: ",out.size())
            
        
        # Fase de Upsampling y combinación con las salidas de ZeroConv1x1
        for i, up in enumerate(self.unet.ups):
            #print("-",i,"-")
            down_out = down_outs_unet.pop()
            zero_conv_out = zero_conv_outs[i]  # Salida de ZeroConv correspondiente
            #print("Unet Copy Ups Zero Conv: ",zero_conv_out.size())
            out = up(out, down_out, t_emb)
            #print("Original Unet Ups: ",out.size())

            out += zero_conv_out  # Sumar salida de ZeroConv a la salida del UpBlock correspondiente

        # Aplicar las capas finales de la U-Net original
        out = self.unet.norm_out(out)
        out = nn.SiLU()(out)
        out = self.unet.conv_out(out)
        
        return out