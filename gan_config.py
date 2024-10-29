# gan_config.py
from dataclasses import dataclass, field
from typing import List

def get_generator_dims() -> List[int]:
    return [256, 512, 1024]

def get_discriminator_dims() -> List[int]:
    return [1024, 512, 256]

@dataclass
class GANConfig:
    # Architecture dimensions
    NOISE_DIM: int = 128
    GENERATOR_HIDDEN_DIMS: List[int] = field(default_factory=get_generator_dims)
    DISCRIMINATOR_HIDDEN_DIMS: List[int] = field(default_factory=get_discriminator_dims)
    
    # Training parameters
    LEARNING_RATE_G: float = 0.0002
    LEARNING_RATE_D: float = 0.0002
    BATCH_SIZE: int = 64
    NUM_EPOCHS: int = 500
    N_CRITIC: int = 5  # Number of discriminator updates per generator update
    GRAD_PENALTY_WEIGHT: float = 10.0
    
    # Model parameters
    DROPOUT_RATE: float = 0.2
    USE_BATCH_NORM: bool = True
    
    # File paths
    MODEL_SAVE_PATH: str = 'gan_model.pkl'
    SYNTHETIC_DATA_PATH: str = 'synthetic_data.pkl'
    
    # Generation parameters
    NUM_SYNTHETIC_SAMPLES: int = 1000  # Number of synthetic samples to generate