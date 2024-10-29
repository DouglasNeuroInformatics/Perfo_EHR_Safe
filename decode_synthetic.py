# decode_synthetic.py
import torch
import numpy as np
import pickle
from typing import Dict
from tqdm import tqdm
from encdec_config import EncoderDecoderConfig
from encdec_model import EncoderDecoderModel

def load_models_and_data():
    """Load all necessary models and data"""
    # Load feature dimensions
    with open('feature_dims.pkl', 'rb') as f:
        feature_dims = pickle.load(f)
    
    # Load synthetic encoded data
    with open('synthetic_data.pkl', 'rb') as f:
        synthetic_encoded = pickle.load(f)
    
    # Initialize encoder-decoder model
    config = EncoderDecoderConfig()
    model = EncoderDecoderModel(
        static_dim=feature_dims['static_dim'],
        temporal_dim=feature_dims['temporal_dim'],
        encoder_hidden_dims=config.ENCODER_HIDDEN_DIMS,
        decoder_hidden_dims=config.DECODER_HIDDEN_DIMS,
        latent_dim=config.LATENT_DIM
    )
    
    # Load trained model
    checkpoint = torch.load(config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, synthetic_encoded, feature_dims

def decode_synthetic_data(
    model: EncoderDecoderModel,
    synthetic_encoded: np.ndarray,
    batch_size: int = 32
) -> Dict[str, np.ndarray]:
    """Decode synthetic data using encoder-decoder model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Initialize lists for decoded data
    static_decoded = []
    temporal_decoded = []
    mask_decoded = []
    time_decoded = []
    
    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(synthetic_encoded), batch_size), desc="Decoding"):
            # Get batch
            batch = synthetic_encoded[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            
            # Decode batch
            static_out, temporal_out, mask_out, time_out = model.decoder(
                batch_tensor,
                model.feature_dims['max_seq_length']
            )
            
            # Store decoded data
            static_decoded.append(static_out.cpu().numpy())
            temporal_decoded.append(temporal_out.cpu().numpy())
            mask_decoded.append(mask_out.cpu().numpy())
            time_decoded.append(time_out.cpu().numpy())
    
    # Concatenate batches
    decoded_data = {
        'static': np.concatenate(static_decoded),
        'temporal': np.concatenate(temporal_decoded),
        'mask': np.concatenate(mask_decoded),
        'time': np.concatenate(time_decoded)
    }
    
    return decoded_data

def main():
    print("Starting synthetic data decoding process...")
    
    # Load models and data
    print("\nLoading models and data...")
    model, synthetic_encoded, feature_dims = load_models_and_data()
    
    # Decode synthetic data
    print("\nDecoding synthetic data...")
    decoded_data = decode_synthetic_data(model, synthetic_encoded)
    
    # Save decoded synthetic data
    print("\nSaving decoded synthetic data...")
    with open('synthetic_decoded_data.pkl', 'wb') as f:
        pickle.dump(decoded_data, f)
    
    print("\nDecoding process complete!")
    print("- Decoded synthetic data saved to: synthetic_decoded_data.pkl")
    
    return decoded_data

if __name__ == "__main__":
    decoded_data = main()