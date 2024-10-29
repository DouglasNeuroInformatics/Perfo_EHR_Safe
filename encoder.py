# encoder.py
import torch
import torch.nn as nn
from typing import Dict, Tuple, List

class Encoder(nn.Module):
    def __init__(
        self,
        static_dim: int,
        temporal_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        # Input dimensions
        self.static_dim = static_dim
        self.temporal_dim = temporal_dim
        self.total_input_dim = static_dim + temporal_dim
        
        # Build encoder layers
        layers = []
        input_dim = self.total_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        # Final layer to latent space
        self.encoder_layers = nn.Sequential(*layers)
        self.latent_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Temporal attention mechanism
        self.temporal_attention = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim),
            nn.Tanh(),
            nn.Linear(temporal_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def _apply_temporal_attention(self, temporal_data: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism to temporal data"""
        # temporal_data shape: [batch_size, sequence_length, feature_dim]
        attention_weights = self.temporal_attention(temporal_data)
        attended_temporal = torch.bmm(
            attention_weights.transpose(1, 2),
            temporal_data
        ).squeeze(1)
        return attended_temporal
    
    def forward(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        mask_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the encoder
        
        Args:
            static_data: [batch_size, static_dim]
            temporal_data: [batch_size, seq_length, temporal_dim]
            mask_data: [batch_size, seq_length, temporal_dim]
        
        Returns:
            latent: [batch_size, latent_dim]
        """
        # Apply attention to temporal data
        attended_temporal = self._apply_temporal_attention(temporal_data)
        
        # Apply mask
        masked_temporal = attended_temporal * mask_data.mean(dim=1)
        
        # Concatenate static and temporal data
        combined_input = torch.cat([static_data, masked_temporal], dim=1)
        
        # Pass through encoder layers
        hidden = self.encoder_layers(combined_input)
        
        # Project to latent space
        latent = self.latent_layer(hidden)
        
        return latent