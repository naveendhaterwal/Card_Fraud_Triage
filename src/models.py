"""
Autoencoder Models for Credit Card Fraud Detection
Written in clean, beginner-friendly PyTorch.
"""

import torch
import torch.nn as nn
import numpy as np


class Autoencoder(nn.Module):
    """
    Simple, clean 4-layer Autoencoder.
    
    Encoder Funnel: 29 -> 20 -> 12 -> 8 (Bottleneck)
    Decoder Horn:   8 -> 12 -> 20 -> 29 (Reconstructed Output)
    """
    def __init__(self, input_dim=29, latent_dim=8, dropout_rate=0.0):
        super(Autoencoder, self).__init__()
        
        # 1. The Encoder: Squeezes 29 features down to 8 numbers
        self.encoder = nn.Sequential(
            # Step 1: 29 -> 20
            nn.Linear(input_dim, 20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            
            # Step 2: 20 -> 12
            nn.Linear(20, 12),
            nn.BatchNorm1d(12),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            
            # Step 3: 12 -> 8 (Our Latent Bottleneck!)
            nn.Linear(12, latent_dim)
        )
        
        # 2. The Decoder: Expands the 8 numbers back to 29 features
        self.decoder = nn.Sequential(
            # Step 4: 8 -> 12
            nn.Linear(latent_dim, 12),
            nn.BatchNorm1d(12),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            
            # Step 5: 12 -> 20
            nn.Linear(12, 20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            
            # Step 6: 20 -> 29 (Linear output, no Sigmoid!)
            nn.Linear(20, input_dim)
        )
        
    def encode(self, x):
        """Pass input through encoder to get 8-number summary z"""
        return self.encoder(x)
        
    def decode(self, z):
        """Pass 8-number summary z through decoder to rebuild x_hat"""
        return self.decoder(z)
        
    def forward(self, x):
        """Full pass: x -> z -> x_hat"""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class DenoisingAutoencoder(Autoencoder):
    """
    Denoising Autoencoder (DAE).
    Adds a tiny bit of random noise to the input during training,
    forcing the model to learn the real clean shape of normal transactions.
    """
    def __init__(self, input_dim=29, latent_dim=8, dropout_rate=0.0, noise_factor=0.1):
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, dropout_rate=dropout_rate)
        self.noise_factor = noise_factor
        
    def forward(self, x):
        # Only add noise while training
        if self.training and self.noise_factor > 0:
            noise = torch.randn_like(x) * self.noise_factor
            x_noisy = x + noise
            z = self.encode(x_noisy)
        else:
            z = self.encode(x)
        return self.decode(z)


def compute_reconstruction_error(model, x_tensor, device='cpu', loss_type='mse'):
    """
    Calculates reconstruction error for each transaction.
    Higher error = More anomalous / suspicious!
    """
    model.eval()
    with torch.no_grad():
        x_tensor = x_tensor.to(device)
        x_hat = model(x_tensor)
        
        # Calculate squared difference: (x - x_hat)^2
        if loss_type == 'mse':
            error = torch.mean((x_tensor - x_hat) ** 2, dim=1)
        elif loss_type == 'mae':
            error = torch.mean(torch.abs(x_tensor - x_hat), dim=1)
        elif loss_type == 'huber':
            smooth_l1 = nn.SmoothL1Loss(reduction='none')
            error = torch.mean(smooth_l1(x_hat, x_tensor), dim=1)
            
    return error.cpu().numpy()
