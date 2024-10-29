# encdec_model.py
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from encoder import Encoder
from decoder import Decoder

class EncoderDecoderModel(nn.Module):
    def __init__(
        self,
        static_dim: int,
        temporal_dim: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int],
        latent_dim: int,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.encoder = Encoder(
            static_dim=static_dim,
            temporal_dim=temporal_dim,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            static_dim=static_dim,
            temporal_dim=temporal_dim,
            hidden_dims=decoder_hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
    
    def forward(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        mask_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the full encoder-decoder model
        
        Args:
            static_data: [batch_size, static_dim]
            temporal_data: [batch_size, seq_length, temporal_dim]
            mask_data: [batch_size, seq_length, temporal_dim]
            
        Returns:
            latent: [batch_size, latent_dim]
            static_out: [batch_size, static_dim]
            temporal_out: [batch_size, seq_length, temporal_dim]
            mask_out: [batch_size, seq_length, temporal_dim]
            time_out: [batch_size, 1]
        """
        # Encode
        latent = self.encoder(static_data, temporal_data, mask_data)
        
        # Decode
        seq_length = temporal_data.size(1)
        static_out, temporal_out, mask_out, time_out = self.decoder(latent, seq_length)
        
        return latent, static_out, temporal_out, mask_out, time_out