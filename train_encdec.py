# train_encdec.py
import torch
import numpy as np
from tqdm import tqdm
import pickle
import os
from encdec_config import EncoderDecoderConfig
from encdec_model import EncoderDecoderModel
from encdec_trainer import EncoderDecoderTrainer
from encdec_data import create_dataloaders
from encdec_processor import DataProcessor

def train(config: EncoderDecoderConfig):
    print("Starting encoder-decoder training process...")
    
    # Process data
    print("\nProcessing data...")
    processor = DataProcessor(config)
    static_features, temporal_features, feature_mask, time_data = processor.process_data()
    processor.save_feature_dims()
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        static_features,
        temporal_features,
        feature_mask,
        time_data,
        batch_size=config.BATCH_SIZE
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = EncoderDecoderModel(
        static_dim=processor.feature_dims['static_dim'],
        temporal_dim=processor.feature_dims['temporal_dim'],
        encoder_hidden_dims=config.ENCODER_HIDDEN_DIMS,
        decoder_hidden_dims=config.DECODER_HIDDEN_DIMS,
        latent_dim=config.LATENT_DIM,
        dropout_rate=config.DROPOUT_RATE,
        use_batch_norm=config.USE_BATCH_NORM
    )
    
    # Initialize trainer
    trainer = EncoderDecoderTrainer(model, config)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Train
        train_losses = trainer.train_epoch(train_loader)
        print("\nTraining losses:")
        for loss_name, loss_value in train_losses.items():
            print(f"{loss_name}: {loss_value:.4f}")
        
        # Validate
        val_losses = trainer.validate(val_loader)
        print("\nValidation losses:")
        for loss_name, loss_value in val_losses.items():
            print(f"{loss_name}: {loss_value:.4f}")
        
        # Save best model
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            trainer.save_model(config.MODEL_SAVE_PATH)
            print(f"\nSaved new best model with validation loss: {best_val_loss:.4f}")
    
    print("\nTraining complete!")
    return trainer

def encode_data(trainer: EncoderDecoderTrainer, config: EncoderDecoderConfig):
    """Encode all data using trained model"""
    print("\nEncoding all data...")
    
    # Load data
    processor = DataProcessor(config)
    static_features, temporal_features, feature_mask, time_data = processor.process_data()
    
    # Create dataloader with all data
    full_loader = create_dataloaders(
        static_features,
        temporal_features,
        feature_mask,
        time_data,
        batch_size=config.BATCH_SIZE,
        val_split=0
    )[0]
    
    # Encode data
    encoded_data = []
    trainer.model.eval()
    
    with torch.no_grad():
        for batch in tqdm(full_loader, desc="Encoding"):
            static_data = batch['static'].to(trainer.device)
            temporal_data = batch['temporal'].to(trainer.device)
            mask_data = batch['mask'].to(trainer.device)
            
            # Get latent representations
            latent = trainer.model.encoder(static_data, temporal_data, mask_data)
            encoded_data.append(latent.cpu().numpy())
    
    encoded_data = np.concatenate(encoded_data, axis=0)
    
    # Save encoded data
    print(f"\nSaving encoded data to {config.ENCODED_DATA_PATH}")
    with open(config.ENCODED_DATA_PATH, 'wb') as f:
        pickle.dump(encoded_data, f)
    
    return encoded_data

def main():
    # Initialize configuration
    config = EncoderDecoderConfig()
    
    # Train model
    trainer = train(config)
    
    # Encode data
    encoded_data = encode_data(trainer, config)
    
    print("\nEncoder-decoder process complete!")
    print(f"- Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"- Encoded data saved to: {config.ENCODED_DATA_PATH}")
    
    return trainer, encoded_data

if __name__ == "__main__":
    trainer, encoded_data = main()