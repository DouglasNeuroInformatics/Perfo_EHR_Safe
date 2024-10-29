# encdec_config.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional

def get_default_encoder_dims() -> List[int]:
    return [512, 256, 128]

def get_default_decoder_dims() -> List[int]:
    return [128, 256, 512]

@dataclass
class EncoderDecoderConfig:
    # Architecture dimensions
    ENCODER_HIDDEN_DIMS: List[int] = field(default_factory=get_default_encoder_dims)
    DECODER_HIDDEN_DIMS: List[int] = field(default_factory=get_default_decoder_dims)
    LATENT_DIM: int = 64
    
    # Training parameters
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 100
    DROPOUT_RATE: float = 0.2
    
    # Model parameters
    USE_BATCH_NORM: bool = True
    ACTIVATION: str = 'relu'
    
    # File paths
    MODEL_SAVE_PATH: str = 'encoder_decoder_model.pkl'
    ENCODED_DATA_PATH: str = 'encoded_data.pkl'
    
    # Weights for different components in loss function
    RECONSTRUCTION_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            'static': 1.0,
            'temporal': 1.0,
            'mask': 1.0,
            'time': 0.1
        }
    )