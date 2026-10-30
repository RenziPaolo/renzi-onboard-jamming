import torch
import torch.nn as nn
from spikingjelly.activation_based import  neuron, functional, surrogate, layer
import argparse
    
class CSNNanomaly_image(nn.Module):
    def __init__(self, T: int, in_channels: int, inner_channels: int, use_cupy=False, smaller_kernel=False):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        self.T = T
        
        if smaller_kernel:
            kernel_size = 3
        else:
            kernel_size = 5
        
        # Encoder blocks (without pooling)
        self.downconv_1 = nn.Sequential(
            layer.Conv2d(in_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_1 = layer.MaxPool2d(2, 2, step_mode='m')

        self.downconv_2 = nn.Sequential(
            layer.Conv2d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_2 = layer.MaxPool2d(2, 2, step_mode='m')

        # Decoder blocks (without unpooling)
        
        self.upconv_1 = nn.Sequential(
            layer.ConvTranspose2d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        if smaller_kernel:
            kernel_size = 3
        else:
            kernel_size = 7        
        self.upconv_2 = nn.Sequential(
            layer.ConvTranspose2d(inner_channels, in_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(in_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')
        )

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, N, C, W, H]

        # Encoder
        x_enc_1 = self.downconv_1(x_seq)
        x_enc_1 = self.pool_1(x_enc_1)

        x_enc_2 = self.downconv_2(x_enc_1)
        x_enc_2 = self.pool_2(x_enc_2)

        # Decoder
        x_dec_1 = self.upconv_1(x_enc_2)

        x_dec_2 = self.upconv_2(x_dec_1)

        x_collapsed = x_dec_2.mean(dim=0) 
        return x_dec_2, x_enc_2, x_collapsed
    
class CSNNanomaly_image_VAE(nn.Module):
    def __init__(self, T: int, in_channels: int, inner_channels: int, latent_dim: int, w:int, h:int, use_cupy=False, smaller_kernel=False):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        self.T = T
        
        if smaller_kernel:
            kernel_size = 3
            if inner_channels == 2:
                latent_dim = 6272
            else:
                latent_dim = 5617
        else:
            kernel_size = 5
            if inner_channels == 2:
                latent_dim = 6273
            else:
                latent_dim = 11664
        
        # Encoder blocks (without pooling)
        self.downconv_1 = nn.Sequential(
            layer.Conv2d(in_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_1 = layer.MaxPool2d(2, 2, step_mode='m')

        self.downconv_2 = nn.Sequential(
            layer.Conv2d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_2 = layer.MaxPool2d(2, 2, step_mode='m')

        # Decoder blocks (without unpooling)
        
        self.upconv_1 = nn.Sequential(
            layer.ConvTranspose2d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        if smaller_kernel:
            kernel_size = 3
        else:
            kernel_size = 7        
        self.upconv_2 = nn.Sequential(
            layer.ConvTranspose2d(inner_channels, in_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(in_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')
        )
        
        self.inner_channels = inner_channels
        # Flatten the output for the variational layer
        self.flatten = nn.Flatten()
        #print('mu')
        #self.fc_mu = layer.Linear(inner_channels * (h // 4) * (w // 4), (latent_dim * (h // 16) * (w // 16) )*4, step_mode='m')
        #print('logvar')
        #self.fc_logvar = layer.Linear(inner_channels * (h // 4) * (w // 4), (latent_dim * (h // 16) * (w // 16) )*4, step_mode='m')
        print(inner_channels * ((h-12)  //4) * ((w-12) //4))
        print('mu')
        self.fc_mu = layer.Linear(latent_dim, latent_dim, step_mode='m')
        print('logvar')
        self.fc_logvar = layer.Linear(latent_dim, latent_dim, step_mode='m')        

        print('latent to spikes')
        self.latent_to_spikes = neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, N, C, W, H]
        # Encoder
        x_enc_1 = self.downconv_1(x_seq)
        x_enc_1 = self.pool_1(x_enc_1)
        x_enc_2 = self.downconv_2(x_enc_1)
        x_enc_2 = self.pool_2(x_enc_2)
        
        # Flatten and variational layer
        #print(x_enc_2.shape)
        x_enc_2_flat = x_enc_2.reshape(x_enc_2.shape[0], x_enc_2.shape[1], -1)
        mu = self.fc_mu(x_enc_2_flat)
        log_var = self.fc_logvar(x_enc_2_flat)
        z = self.reparameterize(mu, log_var)
        z = z.reshape(self.T, x_seq.size(1), self.inner_channels, x_enc_2.shape[-2], x_enc_2.shape[-1])  # [T, N, C, latent_dim]

        # Decoder
        x_dec_1 = self.upconv_1(z)

        x_dec_2 = self.upconv_2(x_dec_1)

        x_collapsed = x_dec_2.mean(0)
        return x_dec_2, x_enc_2, x_collapsed    

'''
class CSNNanomaly_image_VAE_old(nn.Module):
    def __init__(self, h: int, w: int, latent_dim: int, T: int, in_channels: int, inner_channels: int, use_cupy=True):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        self.T = T

        # Encoder blocks (without pooling)
        self.downconv_1 = nn.Sequential(
            layer.Conv2d(in_channels, inner_channels, kernel_size=5, padding=1, stride=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_1 = layer.MaxPool2d(2, 2, step_mode='m')

        self.downconv_2 = nn.Sequential(
            layer.Conv2d(inner_channels, inner_channels, kernel_size=5, padding=1, stride=2, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_2 = layer.MaxPool2d(2, 2, step_mode='m')

        # Decoder blocks (without unpooling)
        
        self.upconv_1 = nn.Sequential(
            layer.ConvTranspose2d(inner_channels, inner_channels, kernel_size=5, padding=1, stride=4, output_padding=3, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        
        self.upconv_2 = nn.Sequential(
            layer.ConvTranspose2d(inner_channels, in_channels, kernel_size=7, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(in_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')
        )

        self.inner_channels = inner_channels
        # Flatten the output for the variational layer
        self.flatten = nn.Flatten()
        #print('mu')
        #self.fc_mu = layer.Linear(inner_channels * (h // 4) * (w // 4), (latent_dim * (h // 16) * (w // 16) )*4, step_mode='m')
        #print('logvar')
        #self.fc_logvar = layer.Linear(inner_channels * (h // 4) * (w // 4), (latent_dim * (h // 16) * (w // 16) )*4, step_mode='m')
        print(inner_channels * ((((h-1) //2) //4)) * ((((h-1) //2) //4)))
        print('mu')
        self.fc_mu = layer.Linear(inner_channels * ((((h-1) //2) //4)) * ((((h-1) //2) //4)), inner_channels * ((((h-1) //2) //4)) * ((((h-1) //2) //4)), step_mode='m')
        print('logvar')
        self.fc_logvar = layer.Linear(inner_channels * ((((h-1) //2) //4)) * ((((h-1) //2) //4)), inner_channels * ((((h-1) //2) //4)) * ((((h-1) //2) //4)), step_mode='m')        

        print('latent to spikes')
        self.latent_to_spikes = neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, N, C, W, H]
        # Encoder
        x_enc_1 = self.downconv_1(x_seq)
        x_enc_1 = self.pool_1(x_enc_1)
        x_enc_2 = self.downconv_2(x_enc_1)
        x_enc_2 = self.pool_2(x_enc_2)
        
        # Flatten and variational layer
        #print(x_enc_2.shape)
        x_enc_2_flat = x_enc_2.reshape(x_enc_2.shape[0], x_enc_2.shape[1], -1)
        mu = self.fc_mu(x_enc_2_flat)
        log_var = self.fc_logvar(x_enc_2_flat)
        z = self.reparameterize(mu, log_var)
        z = z.reshape(self.T, x_seq.size(1), self.inner_channels, x_enc_2.shape[-2], x_enc_2.shape[-1])  # [T, N, C, latent_dim]

        # Decoder
        x_dec_1 = self.upconv_1(z)

        x_dec_2 = self.upconv_2(x_dec_1)

        x_collapsed = x_dec_2.mean(0)
        return x_dec_2, x_enc_2, x_collapsed
'''
    
class CSNNanomaly_line(nn.Module):
    def __init__(self, T: int, in_channels: int, inner_channels: int, use_cupy=False, smaller_kernel=False):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        self.T = T

        if smaller_kernel:
            kernel_size = 3
        else:
            kernel_size = 5
        #print(f"kernel_size: {kernel_size}")
        # Encoder blocks (without pooling)
        self.downconv_1 = nn.Sequential(
            layer.Conv1d(in_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_1 = layer.MaxPool1d(2, 2, step_mode='m')

        self.downconv_2 = nn.Sequential(
            layer.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_2 = layer.MaxPool1d(2, 2, step_mode='m')

        # Decoder blocks (without unpooling)
        
        self.upconv_1 = nn.Sequential(
            layer.ConvTranspose1d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )

        if smaller_kernel:
            kernel_size = 3
        else:
            kernel_size = 7
                
        self.upconv_2 = nn.Sequential(
            layer.ConvTranspose1d(inner_channels, in_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(in_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')
        )


    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1)  # [T, N, C, W]

        # Encoder
        x_enc_1 = self.downconv_1(x_seq)
        x_enc_1 = self.pool_1(x_enc_1)

        x_enc_2 = self.downconv_2(x_enc_1)
        x_enc_2 = self.pool_2(x_enc_2)

        # Decoder
        x_dec_1 = self.upconv_1(x_enc_2)

        x_dec_2 = self.upconv_2(x_dec_1)

        x_collapsed = x_dec_2.mean(dim=0) 
        return x_dec_2, x_enc_2, x_collapsed

class CSNNanomaly_line_VAE(nn.Module):
    def __init__(self, T: int, in_channels: int, inner_channels: int, latent_dim: int, w:int, use_cupy=False, smaller_kernel=False):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        self.T = T

        if smaller_kernel:
            kernel_size = 3
            if inner_channels == 2:
                print("test")
                latent_dim = 255
            print("test1")
            latent_dim = 112
        else:
            kernel_size = 5
            if inner_channels == 2:
                latent_dim = 118
            latent_dim = 216
        
        # Encoder blocks (without pooling)
        self.downconv_1 = nn.Sequential(
            layer.Conv1d(in_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_1 = layer.MaxPool1d(2, 2, step_mode='m')

        self.downconv_2 = nn.Sequential(
            layer.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_2 = layer.MaxPool1d(2, 2, step_mode='m')

        # Decoder blocks (without unpooling)
        
        self.upconv_1 = nn.Sequential(
            layer.ConvTranspose1d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )

        if smaller_kernel:
            kernel_size = 3
        else:
            kernel_size = 7
                
        self.upconv_2 = nn.Sequential(
            layer.ConvTranspose1d(inner_channels, in_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(in_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')
        )


        self.inner_channels = inner_channels
        # Flatten the output for the variational layer
        self.fc_mu = layer.Linear(latent_dim, latent_dim , step_mode='m')
        self.fc_logvar = layer.Linear(latent_dim, latent_dim, step_mode='m')
        self.latent_to_spikes = neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1)  # [T, N, C, W]

        # Encoder
        x_enc_1 = self.downconv_1(x_seq)
        x_enc_1 = self.pool_1(x_enc_1)
        x_enc_2 = self.downconv_2(x_enc_1)
        x_enc_2 = self.pool_2(x_enc_2)
        # Flatten and variational layer
        x_enc_2_flat = x_enc_2.reshape(x_enc_2.shape[0], x_enc_2.shape[1], -1)
        mu = self.fc_mu(x_enc_2_flat)
        log_var = self.fc_logvar(x_enc_2_flat)

        z = self.reparameterize(mu, log_var)
        z = z.reshape(self.T, x_seq.size(1), self.inner_channels, -1)  # [T, N, C, latent_dim]

        # Decoder
        x_dec_1 = self.upconv_1(z)

        x_dec_2 = self.upconv_2(x_dec_1)

        x_collapsed = x_dec_2.mean(0)
        return x_dec_2, x_enc_2, x_collapsed


class CSNNanomaly_line_VAE_old(nn.Module):
    def __init__(self, w: int, latent_dim: int, T: int, in_channels: int, inner_channels: int, use_cupy=False):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        self.T = T

        # Encoder blocks (without pooling)
        self.downconv_1 = nn.Sequential(
            layer.Conv1d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_1 = layer.MaxPool1d(2, 2, step_mode='m')

        self.downconv_2 = nn.Sequential(
            layer.Conv1d(inner_channels, inner_channels, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_2 = layer.MaxPool1d(2, 2, step_mode='m')

        # Decoder blocks (without unpooling)
        
        self.upconv_1 = nn.Sequential(
            layer.ConvTranspose1d(inner_channels, inner_channels, kernel_size=3, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        
        self.upconv_2 = nn.Sequential(
            layer.ConvTranspose1d(inner_channels, in_channels, kernel_size=3, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(in_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')
        )
        self.inner_channels = inner_channels
        # Flatten the output for the variational layer
        self.fc_mu = layer.Linear(inner_channels * (w // 4), latent_dim * (w // 2), step_mode='m')
        self.fc_logvar = layer.Linear(inner_channels * (w // 4), latent_dim * (w // 2), step_mode='m')
        self.latent_to_spikes = neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1)  # [T, N, C, W]

        # Encoder
        x_enc_1 = self.downconv_1(x_seq)
        x_enc_1 = self.pool_1(x_enc_1)
        x_enc_2 = self.downconv_2(x_enc_1)
        x_enc_2 = self.pool_2(x_enc_2)
        # Flatten and variational layer
        x_enc_2_flat = x_enc_2.reshape(x_enc_2.shape[0], x_enc_2.shape[1], -1)
        mu = self.fc_mu(x_enc_2_flat)
        log_var = self.fc_logvar(x_enc_2_flat)

        z = self.reparameterize(mu, log_var)
        z = z.reshape(self.T, x_seq.size(1), self.inner_channels, -1)  # [T, N, C, latent_dim]

        # Decoder
        x_dec_1 = self.upconv_1(z)

        x_dec_2 = self.upconv_2(x_dec_1)

        x_collapsed = x_dec_2.mean(0)
        return x_dec_2, x_enc_2, x_collapsed

    
class CSNNanomaly_image_LIF(nn.Module):
    def __init__(self, T: int, in_channels: int, inner_channels: int, use_cupy=False, smaller_kernel=False):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        self.T = T
        
        if smaller_kernel:
            kernel_size = 3
        else:
            kernel_size = 5
        
        # Encoder blocks (without pooling)
        self.downconv_1 = nn.Sequential(
            layer.Conv2d(in_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_1 = layer.MaxPool2d(2, 2, step_mode='m')

        self.downconv_2 = nn.Sequential(
            layer.Conv2d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_2 = layer.MaxPool2d(2, 2, step_mode='m')

        # Decoder blocks (without unpooling)
        
        self.upconv_1 = nn.Sequential(
            layer.ConvTranspose2d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        if smaller_kernel:
            kernel_size = 3
        else:
            kernel_size = 7        
        self.upconv_2 = nn.Sequential(
            layer.ConvTranspose2d(inner_channels, in_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(in_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')
        )

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, N, C, W, H]

        # Encoder
        x_enc_1 = self.downconv_1(x_seq)
        x_enc_1 = self.pool_1(x_enc_1)

        x_enc_2 = self.downconv_2(x_enc_1)
        x_enc_2 = self.pool_2(x_enc_2)

        # Decoder
        x_dec_1 = self.upconv_1(x_enc_2)

        x_dec_2 = self.upconv_2(x_dec_1)

        x_collapsed = x_dec_2.mean(dim=0) 
        return x_dec_2, x_enc_2, x_collapsed
    
class CSNNanomaly_image_VAE_LIF(nn.Module):
    def __init__(self, T: int, in_channels: int, inner_channels: int, latent_dim: int, w:int, h:int, use_cupy=False, smaller_kernel=False):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        self.T = T
        
        if smaller_kernel:
            kernel_size = 3
            latent_dim = 6272
        else:
            kernel_size = 5
            latent_dim = 11664
        
        # Encoder blocks (without pooling)
        self.downconv_1 = nn.Sequential(
            layer.Conv2d(in_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_1 = layer.MaxPool2d(2, 2, step_mode='m')

        self.downconv_2 = nn.Sequential(
            layer.Conv2d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_2 = layer.MaxPool2d(2, 2, step_mode='m')

        # Decoder blocks (without unpooling)
        
        self.upconv_1 = nn.Sequential(
            layer.ConvTranspose2d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        if smaller_kernel:
            kernel_size = 3
        else:
            kernel_size = 7        
        self.upconv_2 = nn.Sequential(
            layer.ConvTranspose2d(inner_channels, in_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(in_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')
        )
        
        self.inner_channels = inner_channels
        # Flatten the output for the variational layer
        self.flatten = nn.Flatten()
        #print('mu')
        #self.fc_mu = layer.Linear(inner_channels * (h // 4) * (w // 4), (latent_dim * (h // 16) * (w // 16) )*4, step_mode='m')
        #print('logvar')
        #self.fc_logvar = layer.Linear(inner_channels * (h // 4) * (w // 4), (latent_dim * (h // 16) * (w // 16) )*4, step_mode='m')
        print(inner_channels * ((h-12)  //4) * ((w-12) //4))
        print('mu')
        self.fc_mu = layer.Linear(latent_dim, latent_dim, step_mode='m')
        print('logvar')
        self.fc_logvar = layer.Linear(latent_dim, latent_dim, step_mode='m')        

        print('latent to spikes')
        self.latent_to_spikes = neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, N, C, W, H]
        # Encoder
        x_enc_1 = self.downconv_1(x_seq)
        x_enc_1 = self.pool_1(x_enc_1)
        x_enc_2 = self.downconv_2(x_enc_1)
        x_enc_2 = self.pool_2(x_enc_2)
        
        # Flatten and variational layer
        #print(x_enc_2.shape)
        x_enc_2_flat = x_enc_2.reshape(x_enc_2.shape[0], x_enc_2.shape[1], -1)
        mu = self.fc_mu(x_enc_2_flat)
        log_var = self.fc_logvar(x_enc_2_flat)
        z = self.reparameterize(mu, log_var)
        z = z.reshape(self.T, x_seq.size(1), self.inner_channels, x_enc_2.shape[-2], x_enc_2.shape[-1])  # [T, N, C, latent_dim]

        # Decoder
        x_dec_1 = self.upconv_1(z)

        x_dec_2 = self.upconv_2(x_dec_1)

        x_collapsed = x_dec_2.mean(0)
        return x_dec_2, x_enc_2, x_collapsed    

'''
class CSNNanomaly_image_VAE_old(nn.Module):
    def __init__(self, h: int, w: int, latent_dim: int, T: int, in_channels: int, inner_channels: int, use_cupy=True):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        self.T = T

        # Encoder blocks (without pooling)
        self.downconv_1 = nn.Sequential(
            layer.Conv2d(in_channels, inner_channels, kernel_size=5, padding=1, stride=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_1 = layer.MaxPool2d(2, 2, step_mode='m')

        self.downconv_2 = nn.Sequential(
            layer.Conv2d(inner_channels, inner_channels, kernel_size=5, padding=1, stride=2, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_2 = layer.MaxPool2d(2, 2, step_mode='m')

        # Decoder blocks (without unpooling)
        
        self.upconv_1 = nn.Sequential(
            layer.ConvTranspose2d(inner_channels, inner_channels, kernel_size=5, padding=1, stride=4, output_padding=3, bias=False, step_mode='m'),
            layer.BatchNorm2d(inner_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        
        self.upconv_2 = nn.Sequential(
            layer.ConvTranspose2d(inner_channels, in_channels, kernel_size=7, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(in_channels, step_mode='m'),
            neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')
        )

        self.inner_channels = inner_channels
        # Flatten the output for the variational layer
        self.flatten = nn.Flatten()
        #print('mu')
        #self.fc_mu = layer.Linear(inner_channels * (h // 4) * (w // 4), (latent_dim * (h // 16) * (w // 16) )*4, step_mode='m')
        #print('logvar')
        #self.fc_logvar = layer.Linear(inner_channels * (h // 4) * (w // 4), (latent_dim * (h // 16) * (w // 16) )*4, step_mode='m')
        print(inner_channels * ((((h-1) //2) //4)) * ((((h-1) //2) //4)))
        print('mu')
        self.fc_mu = layer.Linear(inner_channels * ((((h-1) //2) //4)) * ((((h-1) //2) //4)), inner_channels * ((((h-1) //2) //4)) * ((((h-1) //2) //4)), step_mode='m')
        print('logvar')
        self.fc_logvar = layer.Linear(inner_channels * ((((h-1) //2) //4)) * ((((h-1) //2) //4)), inner_channels * ((((h-1) //2) //4)) * ((((h-1) //2) //4)), step_mode='m')        

        print('latent to spikes')
        self.latent_to_spikes = neuron.IFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, N, C, W, H]
        # Encoder
        x_enc_1 = self.downconv_1(x_seq)
        x_enc_1 = self.pool_1(x_enc_1)
        x_enc_2 = self.downconv_2(x_enc_1)
        x_enc_2 = self.pool_2(x_enc_2)
        
        # Flatten and variational layer
        #print(x_enc_2.shape)
        x_enc_2_flat = x_enc_2.reshape(x_enc_2.shape[0], x_enc_2.shape[1], -1)
        mu = self.fc_mu(x_enc_2_flat)
        log_var = self.fc_logvar(x_enc_2_flat)
        z = self.reparameterize(mu, log_var)
        z = z.reshape(self.T, x_seq.size(1), self.inner_channels, x_enc_2.shape[-2], x_enc_2.shape[-1])  # [T, N, C, latent_dim]

        # Decoder
        x_dec_1 = self.upconv_1(z)

        x_dec_2 = self.upconv_2(x_dec_1)

        x_collapsed = x_dec_2.mean(0)
        return x_dec_2, x_enc_2, x_collapsed
'''
    
class CSNNanomaly_line_LIF(nn.Module):
    def __init__(self, T: int, in_channels: int, inner_channels: int, use_cupy=False, smaller_kernel=False):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        self.T = T

        if smaller_kernel:
            kernel_size = 3
        else:
            kernel_size = 5
        
        # Encoder blocks (without pooling)
        self.downconv_1 = nn.Sequential(
            layer.Conv1d(in_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_1 = layer.MaxPool1d(2, 2, step_mode='m')

        self.downconv_2 = nn.Sequential(
            layer.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_2 = layer.MaxPool1d(2, 2, step_mode='m')

        # Decoder blocks (without unpooling)
        
        self.upconv_1 = nn.Sequential(
            layer.ConvTranspose1d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )

        if smaller_kernel:
            kernel_size = 3
        else:
            kernel_size = 7
                
        self.upconv_2 = nn.Sequential(
            layer.ConvTranspose1d(inner_channels, in_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(in_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')
        )


    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1)  # [T, N, C, W]

        # Encoder
        x_enc_1 = self.downconv_1(x_seq)
        x_enc_1 = self.pool_1(x_enc_1)

        x_enc_2 = self.downconv_2(x_enc_1)
        x_enc_2 = self.pool_2(x_enc_2)

        # Decoder
        x_dec_1 = self.upconv_1(x_enc_2)

        x_dec_2 = self.upconv_2(x_dec_1)

        x_collapsed = x_dec_2.mean(dim=0) 
        return x_dec_2, x_enc_2, x_collapsed

class CSNNanomaly_line_VAE_LIF(nn.Module):
    def __init__(self, T: int, in_channels: int, inner_channels: int, latent_dim: int, w:int, use_cupy=False, smaller_kernel=False):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        self.T = T

        if smaller_kernel:
            kernel_size = 3
            if inner_channels == 2:
                latent_dim = 112
            else:
                latent_dim = 112
        else:
            kernel_size = 5
            if inner_channels == 2:
                latent_dim = 112
            else:
                latent_dim = 216
        
        
        # Encoder blocks (without pooling)
        self.downconv_1 = nn.Sequential(
            layer.Conv1d(in_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_1 = layer.MaxPool1d(2, 2, step_mode='m')

        self.downconv_2 = nn.Sequential(
            layer.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_2 = layer.MaxPool1d(2, 2, step_mode='m')

        # Decoder blocks (without unpooling)
        
        self.upconv_1 = nn.Sequential(
            layer.ConvTranspose1d(inner_channels, inner_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )

        if smaller_kernel:
            kernel_size = 3
        else:
            kernel_size = 7
                
        self.upconv_2 = nn.Sequential(
            layer.ConvTranspose1d(inner_channels, in_channels, kernel_size=kernel_size, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(in_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')
        )


        self.inner_channels = inner_channels
        # Flatten the output for the variational layer
        self.fc_mu = layer.Linear(latent_dim, latent_dim , step_mode='m')
        self.fc_logvar = layer.Linear(latent_dim, latent_dim, step_mode='m')
        self.latent_to_spikes = neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1)  # [T, N, C, W]

        # Encoder
        x_enc_1 = self.downconv_1(x_seq)
        x_enc_1 = self.pool_1(x_enc_1)
        x_enc_2 = self.downconv_2(x_enc_1)
        x_enc_2 = self.pool_2(x_enc_2)
        # Flatten and variational layer
        x_enc_2_flat = x_enc_2.reshape(x_enc_2.shape[0], x_enc_2.shape[1], -1)
        mu = self.fc_mu(x_enc_2_flat)
        log_var = self.fc_logvar(x_enc_2_flat)

        z = self.reparameterize(mu, log_var)
        z = z.reshape(self.T, x_seq.size(1), self.inner_channels, -1)  # [T, N, C, latent_dim]

        # Decoder
        x_dec_1 = self.upconv_1(z)

        x_dec_2 = self.upconv_2(x_dec_1)

        x_collapsed = x_dec_2.mean(0)
        return x_dec_2, x_enc_2, x_collapsed



class CSNNanomaly_line_VAE_old_LIF(nn.Module):
    def __init__(self, w: int, latent_dim: int, T: int, in_channels: int, inner_channels: int, use_cupy=False):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        self.T = T

        # Encoder blocks (without pooling)
        self.downconv_1 = nn.Sequential(
            layer.Conv1d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_1 = layer.MaxPool1d(2, 2, step_mode='m')

        self.downconv_2 = nn.Sequential(
            layer.Conv1d(inner_channels, inner_channels, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        self.pool_2 = layer.MaxPool1d(2, 2, step_mode='m')

        # Decoder blocks (without unpooling)
        
        self.upconv_1 = nn.Sequential(
            layer.ConvTranspose1d(inner_channels, inner_channels, kernel_size=3, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(inner_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m'),
        )
        
        self.upconv_2 = nn.Sequential(
            layer.ConvTranspose1d(inner_channels, in_channels, kernel_size=3, padding=1, stride=2, output_padding=1, bias=False, step_mode='m'),
            layer.BatchNorm1d(in_channels, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')
        )
        self.inner_channels = inner_channels
        # Flatten the output for the variational layer
        self.fc_mu = layer.Linear(inner_channels * (w // 4), latent_dim * (w // 2), step_mode='m')
        self.fc_logvar = layer.Linear(inner_channels * (w // 4), latent_dim * (w // 2), step_mode='m')
        self.latent_to_spikes = neuron.LIFNode(surrogate_function=surrogate.PiecewiseExp(), step_mode='m')

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1)  # [T, N, C, W]

        # Encoder
        x_enc_1 = self.downconv_1(x_seq)
        x_enc_1 = self.pool_1(x_enc_1)
        x_enc_2 = self.downconv_2(x_enc_1)
        x_enc_2 = self.pool_2(x_enc_2)
        # Flatten and variational layer
        x_enc_2_flat = x_enc_2.reshape(x_enc_2.shape[0], x_enc_2.shape[1], -1)
        mu = self.fc_mu(x_enc_2_flat)
        log_var = self.fc_logvar(x_enc_2_flat)

        z = self.reparameterize(mu, log_var)
        z = z.reshape(self.T, x_seq.size(1), self.inner_channels, -1)  # [T, N, C, latent_dim]

        # Decoder
        x_dec_1 = self.upconv_1(z)

        x_dec_2 = self.upconv_2(x_dec_1)

        x_collapsed = x_dec_2.mean(0)
        return x_dec_2, x_enc_2, x_collapsed

    
if __name__ == "__main__":
    from torchviz import make_dot
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='image', type=str, help='type of model image or line')
    parser.add_argument('-vae', default=False, help='use VAE architecture')
    args = parser.parse_args()
    
    channels = 2
    
    # Create an instance of the model
    if args.model == 'line':
        w = 64
        if args.vae:
            model = CSNNanomaly_line_VAE(w=w, latent_dim=2, T=10, in_channels=channels, inner_channels=2)
        else:
            model = CSNNanomaly_line(T=10, in_channels=channels, inner_channels=4)
        # Create a dummy input tensor
        x = torch.randn(1, channels, w)
    elif args.model == 'image':
        h , w = 512, 512
        if args.vae:
            model = CSNNanomaly_image_VAE(h=h, w=w, latent_dim=2, T=10, in_channels=channels, inner_channels=2)
        else:
            model = CSNNanomaly_image(T=10, in_channels=channels, inner_channels=4)
        # Create a dummy input tensor

        x = torch.randn(4, 2, 512, 512)
    

    # Perform a forward pass to generate the computational graph
    x_dec, x_enc, output = model(x)
    print(x.shape)
    print(output.shape)

    # Save the computational graph
    make_dot(output, params=dict(model.named_parameters())).render("csnnanomaly_torchviz", format="png")