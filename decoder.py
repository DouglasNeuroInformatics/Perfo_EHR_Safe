# decoder.py
import torch
import torch.nn as nn
from typing import Dict, Tuple, List

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        static_dim: int,
        temporal_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        # Dimensions
        self.latent_dim = latent_dim
        self.static_dim = static_dim
        self.temporal_dim = temporal_dim
        
        # Build decoder layers
        layers = []
        input_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        self.decoder_layers = nn.Sequential(*layers)
        
        # Output layers
        self.static_output = nn.Linear(hidden_dims[-1], static_dim)
        self.temporal_output = nn.Linear(hidden_dims[-1], temporal_dim)
        self.mask_output = nn.Sequential(
            nn.Linear(hidden_dims[-1], temporal_dim),
            nn.Sigmoid()
        )
        self.time_output = nn.Linear(hidden_dims[-1], 1)
        
        # Temporal attention for sequence generation
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dims[-1], temporal_dim),
            nn.Tanh(),
            nn.Linear(temporal_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def _generate_sequence(
        self,
        hidden: torch.Tensor,
        seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate temporal sequence using attention mechanism"""
        batch_size = hidden.size(0)
        
        # Initialize sequence tensors
        temporal_seq = torch.zeros(
            batch_size, seq_length, self.temporal_dim,
            device=hidden.device
        )
        mask_seq = torch.zeros(
            batch_size, seq_length, self.temporal_dim,
            device=hidden.device
        )
        
        # Generate sequence step by step
        for t in range(seq_length):
            # Generate attention weights
            attention = self.temporal_attention(hidden)
            
            # Generate temporal features and mask for current timestep
            temporal_t = self.temporal_output(hidden * attention)
            mask_t = self.mask_output(hidden * attention)
            
            # Store in sequence tensors
            temporal_seq[:, t, :] = temporal_t
            mask_seq[:, t, :] = mask_t
        
        return temporal_seq, mask_seq
    
    def forward(
        self,
        latent: torch.Tensor,
        seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the decoder
        
        Args:
            latent: [batch_size, latent_dim]
            seq_length: Length of sequence to generate
        
        Returns:
            static_out: [batch_size, static_dim]
            temporal_out: [batch_size, seq_length, temporal_dim]
            mask_out: [batch_size, seq_length, temporal_dim]
            time_out: [batch_size, 1]
        """
        # Pass through decoder layers
        hidden = self.decoder_layers(latent)
        
        # Generate static features
        static_out = self.static_output(hidden)
        
        # Generate temporal sequence
        temporal_out, mask_out = self._generate_sequence(hidden, seq_length)
        
        # Generate time delta
        time_out = self.time_output(hidden)
        
        return static_out, temporal_out, mask_out, time_out