# encdec_processor.py
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List
import pickle
from encdec_config import EncoderDecoderConfig

class DataProcessor:
    def __init__(self, config: EncoderDecoderConfig):
        self.config = config
        self.feature_dims = None
        self.max_seq_length = None
        
    def _load_processed_data(self) -> Dict:
        """Load normalized and embedded data"""
        # Load embedded data
        with open('categorical_embeddings.pkl', 'rb') as f:
            embedded_data = pickle.load(f)
            
        # Load normalized data
        with open('normalized_data.pkl', 'rb') as f:
            normalized_data = pickle.load(f)
            
        return {
            'embedded': embedded_data,
            'normalized': normalized_data
        }
    
    def _combine_static_features(
        self, 
        normalized_static: pd.DataFrame,
        static_embeddings: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Combine normalized numerical and embedded categorical static features"""
        feature_arrays = []
        
        # Add normalized numerical features
        numerical_features = normalized_static.select_dtypes(include=[np.number]).values
        feature_arrays.append(numerical_features)
        
        # Add embedded categorical features
        for feature, embeddings in static_embeddings.items():
            feature_arrays.append(embeddings)
        
        # Concatenate all features
        combined = np.concatenate(feature_arrays, axis=1)
        return combined
    
    def _combine_temporal_features(
        self,
        normalized_temporal: Dict[str, pd.DataFrame],
        temporal_embeddings: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine normalized numerical and embedded categorical temporal features"""
        # Determine max sequence length
        max_seq_length = max(
            df.shape[0] for df in normalized_temporal.values()
        )
        self.max_seq_length = max_seq_length
        
        # Initialize arrays for combined features
        num_samples = next(iter(normalized_temporal.values())).shape[0]
        total_dim = 0
        
        # Calculate total feature dimension
        for visit_type, df in normalized_temporal.items():
            # Add numerical features
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            total_dim += len(numerical_cols)
            
            # Add embedded features dimensions
            if visit_type in temporal_embeddings:
                for embeddings in temporal_embeddings[visit_type].values():
                    total_dim += embeddings.shape[1]
        
        # Initialize arrays
        combined_features = np.zeros((num_samples, max_seq_length, total_dim))
        mask = np.zeros((num_samples, max_seq_length, total_dim))
        
        current_pos = 0
        for visit_type, df in normalized_temporal.items():
            # Process numerical features
            numerical_data = df.select_dtypes(include=[np.number]).values
            if len(numerical_data.shape) == 2:  # Ensure 2D array
                num_features = numerical_data.shape[1]
                seq_length = numerical_data.shape[0]
                
                combined_features[:, :seq_length, current_pos:current_pos+num_features] = \
                    np.nan_to_num(numerical_data)
                mask[:, :seq_length, current_pos:current_pos+num_features] = \
                    ~np.isnan(numerical_data)
                current_pos += num_features
            
            # Process embedded features
            if visit_type in temporal_embeddings:
                for embeddings in temporal_embeddings[visit_type].values():
                    embed_dim = embeddings.shape[1]
                    seq_length = embeddings.shape[0]
                    
                    combined_features[:, :seq_length, current_pos:current_pos+embed_dim] = \
                        embeddings
                    mask[:, :seq_length, current_pos:current_pos+embed_dim] = 1
                    current_pos += embed_dim
        
        return combined_features, mask
    
    def _process_time_data(
        self,
        normalized_temporal: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """Process time data"""
        num_samples = next(iter(normalized_temporal.values())).shape[0]
        max_length = self.max_seq_length
        
        # Create default time intervals (1.0 represents standard time step)
        time_data = np.ones((num_samples, max_length))
        
        # Create time mask
        time_mask = np.zeros((num_samples, max_length))
        
        # Fill time mask based on actual sequence lengths
        for df in normalized_temporal.values():
            seq_length = df.shape[0]
            time_mask[:, :seq_length] = 1
        
        # Apply mask to time data
        time_data = time_data * time_mask
        
        return time_data
    
    def process_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process all data for encoder-decoder model"""
        print("Loading processed data...")
        data = self._load_processed_data()
        
        print("Processing static features...")
        static_features = self._combine_static_features(
            data['normalized']['static_data'],
            data['embedded']['static_embeddings']
        )
        
        print("Processing temporal features...")
        temporal_features, feature_mask = self._combine_temporal_features(
            data['normalized']['temporal_data'],
            data['embedded']['temporal_embeddings']
        )
        
        print("Processing time data...")
        time_data = self._process_time_data(
            data['normalized']['temporal_data']
        )
        
        # Store feature dimensions
        self.feature_dims = {
            'static_dim': static_features.shape[1],
            'temporal_dim': temporal_features.shape[2],
            'max_seq_length': self.max_seq_length
        }
        
        return static_features, temporal_features, feature_mask, time_data
    
    def save_feature_dims(self, path: str = 'feature_dims.pkl'):
        """Save feature dimensions"""
        with open(path, 'wb') as f:
            pickle.dump(self.feature_dims, f)
    
    def load_feature_dims(self, path: str = 'feature_dims.pkl'):
        """Load feature dimensions"""
        with open(path, 'rb') as f:
            self.feature_dims = pickle.load(f)