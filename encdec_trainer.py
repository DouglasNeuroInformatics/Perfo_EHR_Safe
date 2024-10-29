# encdec_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm
from encdec_config import EncoderDecoderConfig
from encdec_model import EncoderDecoderModel

class EncoderDecoderTrainer:
    def __init__(
        self,
        model: EncoderDecoderModel,
        config: EncoderDecoderConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE
        )
        
        # Loss functions
        self.static_criterion = nn.MSELoss()
        self.temporal_criterion = nn.MSELoss()
        self.mask_criterion = nn.BCELoss()
        self.time_criterion = nn.MSELoss()
    
    def compute_loss(
        self,
        static_out: torch.Tensor,
        temporal_out: torch.Tensor,
        mask_out: torch.Tensor,
        time_out: torch.Tensor,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        mask_data: torch.Tensor,
        time_data: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted loss for all components"""
        
        # Individual losses
        static_loss = self.static_criterion(static_out, static_data)
        temporal_loss = self.temporal_criterion(temporal_out * mask_data, temporal_data * mask_data)
        mask_loss = self.mask_criterion(mask_out, mask_data)
        time_loss = self.time_criterion(time_out, time_data)
        
        # Weighted total loss
        total_loss = (
            self.config.RECONSTRUCTION_WEIGHTS['static'] * static_loss +
            self.config.RECONSTRUCTION_WEIGHTS['temporal'] * temporal_loss +
            self.config.RECONSTRUCTION_WEIGHTS['mask'] * mask_loss +
            self.config.RECONSTRUCTION_WEIGHTS['time'] * time_loss
        )
        
        # Store individual losses
        loss_dict = {
            'static_loss': static_loss.item(),
            'temporal_loss': temporal_loss.item(),
            'mask_loss': mask_loss.item(),
            'time_loss': time_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        for batch in tqdm(dataloader, desc="Training"):
            static_data = batch['static'].to(self.device)
            temporal_data = batch['temporal'].to(self.device)
            mask_data = batch['mask'].to(self.device)
            time_data = batch['time'].to(self.device)
            
            # Forward pass
            latent, static_out, temporal_out, mask_out, time_out = self.model(
                static_data, temporal_data, mask_data
            )
            
            # Compute loss
            loss, loss_dict = self.compute_loss(
                static_out, temporal_out, mask_out, time_out,
                static_data, temporal_data, mask_data, time_data
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss_dict)
        
        # Average losses over epoch
        avg_losses = {
            k: np.mean([d[k] for d in epoch_losses])
            for k in epoch_losses[0].keys()
        }
        
        return avg_losses
    
    def validate(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        validation_losses = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                static_data = batch['static'].to(self.device)
                temporal_data = batch['temporal'].to(self.device)
                mask_data = batch['mask'].to(self.device)
                time_data = batch['time'].to(self.device)
                
                # Forward pass
                latent, static_out, temporal_out, mask_out, time_out = self.model(
                    static_data, temporal_data, mask_data
                )
                
                # Compute loss
                _, loss_dict = self.compute_loss(
                    static_out, temporal_out, mask_out, time_out,
                    static_data, temporal_data, mask_data, time_data
                )
                
                validation_losses.append(loss_dict)
        
        # Average losses
        avg_losses = {
            k: np.mean([d[k] for d in validation_losses])
            for k in validation_losses[0].keys()
        }
        
        return avg_losses
    
    def save_model(self, path: str):
        """Save model and optimizer state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load model and optimizer state"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])