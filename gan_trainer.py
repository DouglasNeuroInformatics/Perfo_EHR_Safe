# gan_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List
from tqdm import tqdm
from gan_config import GANConfig
from gan_models import Generator, Discriminator

class GANTrainer:
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        config: GANConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.config = config
        self.device = device
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            generator.parameters(),
            lr=config.LEARNING_RATE_G,
            betas=(0.5, 0.999)
        )
        self.d_optimizer = optim.Adam(
            discriminator.parameters(),
            lr=config.LEARNING_RATE_D,
            betas=(0.5, 0.999)
        )
    
    def _gradient_penalty(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor
    ) -> torch.Tensor:
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1).to(self.device)
        
        # Get random interpolation between real and fake data
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        
        # Calculate gradients of discriminator w.r.t. interpolates
        d_interpolates = self.discriminator(interpolates)
        
        # Create fake gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_discriminator(
        self,
        real_data: torch.Tensor
    ) -> Dict[str, float]:
        """Train discriminator for one iteration"""
        batch_size = real_data.size(0)
        
        # Generate fake data
        noise = torch.randn(batch_size, self.config.NOISE_DIM).to(self.device)
        fake_data = self.generator(noise)
        
        # Reset gradients
        self.d_optimizer.zero_grad()
        
        # Calculate losses
        d_real = self.discriminator(real_data)
        d_fake = self.discriminator(fake_data.detach())
        
        # Calculate Wasserstein loss
        d_loss = torch.mean(d_fake) - torch.mean(d_real)
        
        # Calculate gradient penalty
        gradient_penalty = self._gradient_penalty(real_data, fake_data)
        
        # Add gradient penalty to discriminator loss
        d_total_loss = d_loss + self.config.GRAD_PENALTY_WEIGHT * gradient_penalty
        
        # Update discriminator
        d_total_loss.backward()
        self.d_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'grad_penalty': gradient_penalty.item(),
            'wasserstein_dist': -d_loss.item()
        }
    
    def train_generator(self) -> float:
        """Train generator for one iteration"""
        batch_size = self.config.BATCH_SIZE
        
        # Generate fake data
        noise = torch.randn(batch_size, self.config.NOISE_DIM).to(self.device)
        fake_data = self.generator(noise)
        
        # Reset gradients
        self.g_optimizer.zero_grad()
        
        # Calculate generator loss
        g_loss = -torch.mean(self.discriminator(fake_data))
        
        # Update generator
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train both networks for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        d_losses = []
        g_losses = []
        wasserstein_distances = []
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
        for i, batch in pbar:
            # Train discriminator
            real_data = batch.to(self.device)
            d_loss_dict = self.train_discriminator(real_data)
            d_losses.append(d_loss_dict['d_loss'])
            wasserstein_distances.append(d_loss_dict['wasserstein_dist'])
            
            # Train generator every n_critic steps
            if (i + 1) % self.config.N_CRITIC == 0:
                g_loss = self.train_generator()
                g_losses.append(g_loss)
                
                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f"{d_losses[-1]:.4f}",
                    'G_loss': f"{g_loss:.4f}",
                    'W_dist': f"{wasserstein_distances[-1]:.4f}"
                })
        
        return {
            'd_loss': np.mean(d_losses),
            'g_loss': np.mean(g_losses) if g_losses else float('nan'),
            'wasserstein_dist': np.mean(wasserstein_distances)
        }
    
    def generate_samples(self, num_samples: int) -> np.ndarray:
        """Generate synthetic samples using trained generator"""
        self.generator.eval()
        synthetic_samples = []
        
        with torch.no_grad():
            for i in range(0, num_samples, self.config.BATCH_SIZE):
                batch_size = min(self.config.BATCH_SIZE, num_samples - i)
                noise = torch.randn(batch_size, self.config.NOISE_DIM).to(self.device)
                fake_data = self.generator(noise)
                synthetic_samples.append(fake_data.cpu().numpy())
        
        return np.concatenate(synthetic_samples, axis=0)
    
    def save_model(self, path: str):
        """Save GAN models and optimizers"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """Load GAN models and optimizers"""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])