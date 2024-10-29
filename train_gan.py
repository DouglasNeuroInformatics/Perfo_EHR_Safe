# train_gan.py
import torch
import numpy as np
from typing import Tuple
import pickle
from tqdm import tqdm

from gan_config import GANConfig
from gan_models import Generator, Discriminator
from gan_trainer import GANTrainer
from gan_data import (
    load_encoded_data,
    create_dataloader,
    scale_data,
    inverse_scale_data
)

def setup_gan(
    input_dim: int,
    config: GANConfig
) -> Tuple[Generator, Discriminator]:
    """Initialize GAN models"""
    generator = Generator(
        noise_dim=config.NOISE_DIM,
        output_dim=input_dim,
        hidden_dims=config.GENERATOR_HIDDEN_DIMS,
        dropout_rate=config.DROPOUT_RATE,
        use_batch_norm=config.USE_BATCH_NORM
    )
    
    discriminator = Discriminator(
        input_dim=input_dim,
        hidden_dims=config.DISCRIMINATOR_HIDDEN_DIMS,
        dropout_rate=config.DROPOUT_RATE,
        use_batch_norm=config.USE_BATCH_NORM
    )
    
    return generator, discriminator

def train_gan(config: GANConfig):
    """Main training function for GAN"""
    print("Starting GAN training process...")
    
    # Load and prepare data
    print("\nLoading encoded data...")
    encoded_data = load_encoded_data('encoded_data.pkl')
    
    print("Scaling data...")
    scaled_data, scaling_params = scale_data(encoded_data)
    
    # Save scaling parameters
    with open('scaling_params.pkl', 'wb') as f:
        pickle.dump(scaling_params, f)
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = create_dataloader(scaled_data, config.BATCH_SIZE)
    
    # Initialize models
    print("\nInitializing models...")
    generator, discriminator = setup_gan(encoded_data.shape[1], config)
    
    # Initialize trainer
    trainer = GANTrainer(generator, discriminator, config)
    
    # Training loop
    print("\nStarting training...")
    best_wasserstein_dist = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        # Train for one epoch
        losses = trainer.train_epoch(dataloader, epoch)
        
        # Print losses
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"Discriminator Loss: {losses['d_loss']:.4f}")
        print(f"Generator Loss: {losses['g_loss']:.4f}")
        print(f"Wasserstein Distance: {losses['wasserstein_dist']:.4f}")
        
        # Save best model based on Wasserstein distance
        if losses['wasserstein_dist'] < best_wasserstein_dist:
            best_wasserstein_dist = losses['wasserstein_dist']
            trainer.save_model(config.MODEL_SAVE_PATH)
            print(f"\nSaved new best model with Wasserstein distance: {best_wasserstein_dist:.4f}")
    
    print("\nTraining complete!")
    return trainer, scaling_params

def generate_synthetic_data(
    trainer: GANTrainer,
    config: GANConfig,
    scaling_params: dict
) -> np.ndarray:
    """Generate synthetic data using trained GAN"""
    print(f"\nGenerating {config.NUM_SYNTHETIC_SAMPLES} synthetic samples...")
    
    # Generate samples in latent space
    synthetic_encoded = trainer.generate_samples(config.NUM_SYNTHETIC_SAMPLES)
    
    # Inverse scale the data
    synthetic_encoded = inverse_scale_data(synthetic_encoded, scaling_params)
    
    # Save synthetic encoded data
    with open(config.SYNTHETIC_DATA_PATH, 'wb') as f:
        pickle.dump(synthetic_encoded, f)
    
    return synthetic_encoded

def main():
    # Initialize configuration
    config = GANConfig()
    
    # Train GAN
    trainer, scaling_params = train_gan(config)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(trainer, config, scaling_params)
    
    print("\nGAN process complete!")
    print(f"- Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"- Synthetic data saved to: {config.SYNTHETIC_DATA_PATH}")
    print(f"- Scaling parameters saved to: scaling_params.pkl")
    
    return trainer, synthetic_data

if __name__ == "__main__":
    trainer, synthetic_data = main()