import torch
import torch.nn as nn
from norse.torch import LIFRecurrentCell
from norse.torch import LIFCell

class CSNNanomaly_image_Norse(nn.Module):
    def __init__(self, T, in_channels, inner_channels):
        super().__init__()
        self.T = T
        self.downconv_1 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            LIFCell(),  # Use LIFCell for convolutional layers
        )
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.downconv_2 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            LIFCell(),
        )
        self.pool_2 = nn.MaxPool2d(2, 2)
        self.upconv_1 = nn.Sequential(
            nn.ConvTranspose2d(inner_channels, inner_channels, kernel_size=3, padding=1, stride=2, output_padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            LIFCell(),
        )
        self.upconv_2 = nn.Sequential(
            nn.ConvTranspose2d(inner_channels, in_channels, kernel_size=3, padding=1, stride=2, output_padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            LIFCell()
        )

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        outputs = []
        for t in range(self.T):
            x_enc_1 = self.downconv_1(x_seq[t])
            x_enc_1 = self.pool_1(x_enc_1)
            x_enc_2 = self.downconv_2(x_enc_1)
            x_enc_2 = self.pool_2(x_enc_2)
            x_dec_1 = self.upconv_1(x_enc_2)
            x_dec_2 = self.upconv_2(x_dec_1)
            outputs.append(x_dec_2)
        outputs = torch.stack(outputs)
        x_collapsed = outputs.mean(dim=0)
        return outputs, x_enc_2, x_collapsed

class CSNNanomaly_image_VAE_Norse(nn.Module):
    def __init__(self, h, w, latent_dim, T, in_channels, inner_channels):
        super().__init__()
        self.T = T
        self.downconv_1 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=5, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            LIFCell(),
        )
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.downconv_2 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=5, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(inner_channels),
            LIFCell(),
        )
        self.pool_2 = nn.MaxPool2d(2, 2)

        # Calculate flattened size
        flattened_size = inner_channels * ((((h-1) // 2) // 4)) * ((((w-1) // 2) // 4))

        # Use LIFRecurrentCell for fully connected layers
        self.fc_mu = LIFRecurrentCell(flattened_size, latent_dim)
        self.fc_logvar = LIFRecurrentCell(flattened_size, latent_dim)
        self.latent_to_spikes = LIFCell()  # Use LIFCell for stateless processing

        self.upconv_1 = nn.Sequential(
            nn.ConvTranspose2d(inner_channels, inner_channels, kernel_size=5, padding=1, stride=4, output_padding=3, bias=False),
            nn.BatchNorm2d(inner_channels),
            LIFCell(),
        )
        self.upconv_2 = nn.Sequential(
            nn.ConvTranspose2d(inner_channels, in_channels, kernel_size=7, padding=1, stride=2, output_padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            LIFCell()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = []
        for t in range(self.T):
            x_enc_1 = self.downconv_1(x)
            x_enc_1 = self.pool_1(x_enc_1)
            x_enc_2 = self.downconv_2(x_enc_1)
            x_enc_2 = self.pool_2(x_enc_2)
            x_enc_2_flat = x_enc_2.reshape(batch_size, -1)

            # Process through LIFRecurrentCell
            mu, _ = self.fc_mu(x_enc_2_flat)
            log_var, _ = self.fc_logvar(x_enc_2_flat)
            z = self.reparameterize(mu, log_var)
            z = z.reshape(batch_size, -1, 1, 1).repeat(1, 1, x_enc_2.shape[2], x_enc_2.shape[3])
            z = self.latent_to_spikes(z)

            x_dec_1 = self.upconv_1(z)
            x_dec_2 = self.upconv_2(x_dec_1)
            outputs.append(x_dec_2)
            x = x_dec_2  # Optionally feed output back as input

        outputs = torch.stack(outputs)
        x_collapsed = outputs.mean(dim=0)
        return outputs, x_enc_2, x_collapsed

class CSNNanomaly_line_Norse(nn.Module):
    def __init__(self, T: int, in_channels: int, inner_channels: int):
        super().__init__()
        self.T = T

        # Encoder blocks
        self.downconv_1 = nn.Sequential(
            nn.Conv1d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(inner_channels),
            LIFCell(),
        )
        self.pool_1 = nn.MaxPool1d(2, 2)

        self.downconv_2 = nn.Sequential(
            nn.Conv1d(inner_channels, inner_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(inner_channels),
            LIFCell(),
        )
        self.pool_2 = nn.MaxPool1d(2, 2)

        # Decoder blocks
        self.upconv_1 = nn.Sequential(
            nn.ConvTranspose1d(inner_channels, inner_channels, kernel_size=3, padding=1, stride=2, output_padding=1, bias=False),
            nn.BatchNorm1d(inner_channels),
            LIFCell(),
        )

        self.upconv_2 = nn.Sequential(
            nn.ConvTranspose1d(inner_channels, in_channels, kernel_size=3, padding=1, stride=2, output_padding=1, bias=False),
            nn.BatchNorm1d(in_channels),
            LIFCell()
        )

    def forward(self, x):
        # x shape: (batch_size, in_channels, H, W)
        batch_size = x.shape[0]
        outputs = []
        for t in range(self.T):
            x_enc_1, _ = self.downconv_1(x)
            x_enc_1 = self.pool_1(x_enc_1)
            x_enc_2, _ = self.downconv_2(x_enc_1)
            x_enc_2 = self.pool_2(x_enc_2)
            x_dec_1, _ = self.upconv_1(x_enc_2)
            x_dec_2, _ = self.upconv_2(x_dec_1)
            outputs.append(x_dec_2)
            x = x_dec_2  # Optionally feed output back as input

        # Stack outputs: (T, batch_size, in_channels, H, W)
        outputs = torch.stack(outputs)
        x_collapsed = outputs.mean(dim=0)
        return outputs, x_enc_2, x_collapsed


class CSNNanomaly_line_VAE_Norse(nn.Module):
    def __init__(self, w, latent_dim, T, in_channels, inner_channels):
        super().__init__()
        self.T = T
        self.inner_channels = inner_channels
        self.downconv_1 = nn.Sequential(
            nn.Conv1d(in_channels, inner_channels, kernel_size=5, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(inner_channels),
            LIFCell(),
        )
        self.pool_1 = nn.MaxPool1d(2, 2)
        self.downconv_2 = nn.Sequential(
            nn.Conv1d(inner_channels, inner_channels, kernel_size=5, padding=1, stride=2, bias=False),
            nn.BatchNorm1d(inner_channels),
            LIFCell(),
        )
        self.pool_2 = nn.MaxPool1d(2, 2)

        # Calculate flattened size
        flattened_size = inner_channels * ((((w-1) // 2) // 4))

        # Use LIFRecurrentCell for fully connected layers
        self.fc_mu = LIFRecurrentCell(flattened_size, latent_dim* (w // 2))
        self.fc_logvar = LIFRecurrentCell(flattened_size, latent_dim* (w // 2))
        self.latent_to_spikes = LIFCell()  # Use LIFCell for stateless processing

        self.upconv_1 = nn.Sequential(
            nn.ConvTranspose1d(inner_channels, inner_channels, kernel_size=3, padding=1, stride=2, output_padding=1, bias=False),
            nn.BatchNorm1d(inner_channels),
            LIFCell(),
        )
        self.upconv_2 = nn.Sequential(
            nn.ConvTranspose1d(inner_channels, in_channels, kernel_size=3, padding=1, stride=2, output_padding=1, bias=False),
            nn.BatchNorm1d(in_channels),
            LIFCell()
        )


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = []
        print("Input x",x.shape)
        for t in range(self.T):
            x_enc_1, _ = self.downconv_1(x)
            x_enc_1 = self.pool_1(x_enc_1)
            x_enc_2, _ = self.downconv_2(x_enc_1)
            x_enc_2 = self.pool_2(x_enc_2)
            print("x_enc_2",x_enc_2.shape)
            x_enc_2_flat = x_enc_2.reshape(batch_size, -1)
            print("x_enc_2_flat",x_enc_2_flat.shape)
            mu, _ = self.fc_mu(x_enc_2_flat)  # mu is the spikes, _ is the state
            log_var, _ = self.fc_logvar(x_enc_2_flat)  # log_var is the spikes, _ is the state
            z = self.reparameterize(mu, log_var)
            z = z.reshape(batch_size, self.inner_channels, -1)  # [N, C, latent_dim]
            z, _ = self.latent_to_spikes(z)
            x_dec_1, _ = self.upconv_1(z)
            x_dec_2, _ = self.upconv_2(x_dec_1)
            outputs.append(x_dec_2)
            x = x_dec_2  # Optionally feed output back as input
            print("x_dec_2",x_dec_2.shape)

        outputs = torch.stack(outputs)
        x_collapsed = outputs.mean(dim=0)
        return outputs, x_enc_2, x_collapsed



if __name__ == "__main__":
    from torchviz import make_dot
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='image', type=str, help='type of model image or line')
    parser.add_argument('-vae', default=False, help='use VAE architecture')
    args = parser.parse_args()
    
    channels = 2
    
    # Create an instance of the model
    if args.model == 'line':
        w = 64
        if args.vae:
            model = CSNNanomaly_line_VAE_Norse(w=w, latent_dim=2, T=10, in_channels=channels, inner_channels=2)
        else:
            model = CSNNanomaly_line_Norse(T=10, in_channels=channels, inner_channels=4)
        # Create a dummy input tensor
        x = torch.randn(1, channels, w)
    elif args.model == 'image':
        h , w = 512, 512
        if args.vae:
            model = CSNNanomaly_image_VAE_Norse(h=h, w=w, latent_dim=2, T=10, in_channels=channels, inner_channels=2)
        else:
            model = CSNNanomaly_image_Norse(T=10, in_channels=channels, inner_channels=4)
        # Create a dummy input tensor

        x = torch.randn(4, 2, 512, 512)
    

    # Perform a forward pass to generate the computational graph
    x_dec, x_enc, output = model(x)
    print(x.shape)
    print(output.shape)

    # Save the computational graph
    make_dot(output, params=dict(model.named_parameters())).render("csnnanomaly_torchviz", format="png")