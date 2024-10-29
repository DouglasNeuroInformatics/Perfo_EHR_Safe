# gan_models.py
import torch
import torch.nn as nn
from typing import List

class Generator(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        # Build generator layers
        layers = []
        input_dim = noise_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Tanh())  # Scale outputs to [-1, 1]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        # Build discriminator layers
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # Output layer - keep output as [batch_size, 1]
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Do not squeeze the output to maintain [batch_size, 1] shape
        return self.model(x)