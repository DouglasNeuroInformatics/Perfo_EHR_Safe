import torch
import pickle
from encoder import Encoder
from decoder import Decoder
from encdec_model import EncoderDecoderModel
from encdec_config import EncoderDecoderConfig

class SyntheticDataGenerator:
    def __init__(self):
        # Load synthetic data
        print("Loading synthetic data...")
        with open('synthetic_data.pkl', 'rb') as f:
            self.synthetic_encoded = pickle.load(f)
        
        # Load feature dimensions
        print("Loading feature dimensions...")
        with open('feature_dims.pkl', 'rb') as f:
            self.feature_dims = pickle.load(f)
        
        # Initialize and load encoder-decoder model
        print("Loading encoder-decoder model...")
        self.load_encoder_decoder()

    def load_encoder_decoder(self):
        """Initialize and load the encoder-decoder model"""
        config = EncoderDecoderConfig()
        
        self.model = EncoderDecoderModel(
            static_dim=self.feature_dims['static_dim'],
            temporal_dim=self.feature_dims['temporal_dim'],
            encoder_hidden_dims=config.ENCODER_HIDDEN_DIMS,
            decoder_hidden_dims=config.DECODER_HIDDEN_DIMS,
            latent_dim=config.LATENT_DIM
        )
        
        # Load trained model
        checkpoint = torch.load('encoder_decoder_model.pkl')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def decode_synthetic_data(self):
        """Decode synthetic data using encoder-decoder model"""
        print("Decoding synthetic data...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        with torch.no_grad():
            synthetic_tensor = torch.FloatTensor(self.synthetic_encoded).to(device)
            static_out, temporal_out, mask_out, time_out = self.model.decoder(
                synthetic_tensor,
                self.feature_dims['max_seq_length']
            )
            
        return {
            'static': static_out.cpu().numpy(),
            'temporal': temporal_out.cpu().numpy(),
            'mask': mask_out.cpu().numpy(),
            'time': time_out.cpu().numpy()
        }

    def generate_synthetic_data(self):
        """Generate synthetic data"""
        decoded_data = self.decode_synthetic_data()
        
        # Save decoded data
        print("Saving decoded synthetic data...")
        with open('decoded_synthetic_data.pkl', 'wb') as f:
            pickle.dump(decoded_data, f)
        
        print("Synthetic data generation complete!")

def main():
    try:
        generator = SyntheticDataGenerator()
        generator.generate_synthetic_data()
    except Exception as e:
        print(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()