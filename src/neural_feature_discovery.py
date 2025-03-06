#!/usr/bin/env python3
"""
Neural Network-based Feature Discovery Module

This module implements deep learning techniques to discover meaningful features 
from financial time series data that can be used to enhance trading strategies.
It combines autoencoder architectures, attention mechanisms, and feature importance
analysis to extract hidden patterns and relationships from market data.

Key capabilities:
- Nonlinear feature extraction using deep autoencoders
- Temporal feature learning with LSTM/GRU layers
- Cross-asset feature discovery using attention mechanisms
- Automated feature importance ranking and selection
- Integration with existing trading strategy frameworks
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

# Configure logging
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """
    Dataset class for financial time series data.
    
    Handles both univariate and multivariate time series with optional target values,
    supporting both regression and classification tasks.
    """
    
    def __init__(self, 
                data: np.ndarray, 
                targets: Optional[np.ndarray] = None,
                window_size: int = 60,
                stride: int = 1,
                horizon: int = 1,
                transform: Optional[Callable] = None):
        """
        Initialize the time series dataset.
        
        Args:
            data: Input time series data with shape (n_samples, n_features) or (n_samples,)
            targets: Target values with shape (n_samples,) or (n_samples, n_targets)
            window_size: Size of the input window
            stride: Stride between consecutive windows
            horizon: Prediction horizon (steps ahead to predict)
            transform: Optional transform to apply to each sample
        """
        # Ensure data is 2D
        if len(data.shape) == 1:
            self.data = data.reshape(-1, 1)
        else:
            self.data = data
        
        # Handle targets
        self.has_targets = targets is not None
        if self.has_targets:
            if len(targets.shape) == 1:
                self.targets = targets.reshape(-1, 1)
            else:
                self.targets = targets
        else:
            self.targets = None
            
        self.window_size = window_size
        self.stride = stride
        self.horizon = horizon
        self.transform = transform
        
        # Create valid indices
        self.indices = self._create_indices()
        
    def _create_indices(self) -> List[int]:
        """Create valid starting indices for windows."""
        if self.has_targets:
            max_idx = len(self.data) - self.window_size - self.horizon + 1
        else:
            max_idx = len(self.data) - self.window_size + 1
            
        return list(range(0, max_idx, self.stride))
    
    def __len__(self) -> int:
        """Get dataset length (number of samples)."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_size
        
        # Get input window
        X = self.data[start_idx:end_idx]
        
        # Apply transform if specified
        if self.transform:
            X = self.transform(X)
            
        sample = {"X": torch.FloatTensor(X)}
        
        # Add target if available
        if self.has_targets:
            target_idx = end_idx + self.horizon - 1
            y = self.targets[target_idx]
            sample["y"] = torch.FloatTensor(y)
            
        return sample


class Autoencoder(nn.Module):
    """
    Deep autoencoder for nonlinear feature extraction.
    
    This model compresses the input data into a lower-dimensional representation
    and then reconstructs it, learning meaningful features in the process.
    """
    
    def __init__(self, 
                input_dim: int, 
                hidden_dims: List[int],
                latent_dim: int,
                dropout: float = 0.1,
                activation: str = 'relu'):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim: Dimensionality of input features
            hidden_dims: List of hidden layer dimensions for the encoder (decoder is symmetric)
            latent_dim: Dimensionality of the latent space
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', or 'elu')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(self.activation)
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
            
        # Latent layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (symmetric to encoder)
        decoder_layers = []
        prev_dim = latent_dim
        
        for dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(self.activation)
            decoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
            
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (reconstructed_x, latent_code)
        """
        # Encode input to latent space
        z = self.encoder(x)
        
        # Decode latent representation
        x_reconstructed = self.decoder(z)
        
        return x_reconstructed, z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)


class TemporalAutoencoder(nn.Module):
    """
    Temporal autoencoder for learning time series features.
    
    Uses recurrent layers (LSTM/GRU) to capture temporal patterns
    in financial time series data.
    """
    
    def __init__(self, 
                input_dim: int,
                hidden_dim: int,
                latent_dim: int,
                num_layers: int = 2,
                cell_type: str = 'lstm',
                dropout: float = 0.1,
                bidirectional: bool = False):
        """
        Initialize the temporal autoencoder.
        
        Args:
            input_dim: Dimensionality of input features (per time step)
            hidden_dim: Dimensionality of hidden states
            latent_dim: Dimensionality of the latent space
            num_layers: Number of recurrent layers
            cell_type: Type of recurrent cell ('lstm' or 'gru')
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional recurrent layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Choose recurrent cell type
        if cell_type.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif cell_type.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")
            
        # Direction factor (1 if unidirectional, 2 if bidirectional)
        self.dir_factor = 2 if bidirectional else 1
        
        # Encoder
        self.encoder_rnn = self.rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Latent layer
        self.latent_layer = nn.Linear(hidden_dim * self.dir_factor, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim * self.dir_factor)
        
        self.decoder_rnn = self.rnn_cell(
            input_size=hidden_dim * self.dir_factor,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim * self.dir_factor, input_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the temporal autoencoder.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (reconstructed_x, latent_code)
        """
        batch_size, seq_len, _ = x.shape
        
        # Encode input sequence
        enc_output, enc_hidden = self.encoder_rnn(x)
        
        # Extract final encoder state
        if isinstance(enc_hidden, tuple):  # LSTM returns (h_n, c_n)
            enc_hidden = enc_hidden[0]
            
        # Combine last layer hidden states (if bidirectional)
        if self.bidirectional:
            enc_hidden = enc_hidden.view(self.num_layers, self.dir_factor, batch_size, self.hidden_dim)
            enc_hidden = enc_hidden[-1].transpose(0, 1).contiguous().view(batch_size, -1)
        else:
            enc_hidden = enc_hidden[-1]
            
        # Project to latent space
        z = self.latent_layer(enc_hidden)
        
        # Prepare decoder input
        dec_input = self.decoder_input(z).unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode sequence
        dec_output, _ = self.decoder_rnn(dec_input)
        
        # Project to output space
        x_reconstructed = self.output_layer(dec_output)
        
        return x_reconstructed, z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to latent space."""
        batch_size = x.shape[0]
        
        # Encode input sequence
        _, enc_hidden = self.encoder_rnn(x)
        
        # Extract final encoder state
        if isinstance(enc_hidden, tuple):  # LSTM returns (h_n, c_n)
            enc_hidden = enc_hidden[0]
            
        # Combine last layer hidden states (if bidirectional)
        if self.bidirectional:
            enc_hidden = enc_hidden.view(self.num_layers, self.dir_factor, batch_size, self.hidden_dim)
            enc_hidden = enc_hidden[-1].transpose(0, 1).contiguous().view(batch_size, -1)
        else:
            enc_hidden = enc_hidden[-1]
            
        # Project to latent space
        z = self.latent_layer(enc_hidden)
        
        return z


class AttentionFeatureExtractor(nn.Module):
    """
    Attention-based feature extractor for multivariate time series.
    
    Uses self-attention mechanisms to identify relationships between
    different financial variables and time steps.
    """
    
    def __init__(self, 
                input_dim: int,
                hidden_dim: int,
                output_dim: int,
                num_heads: int = 4,
                num_layers: int = 2,
                dropout: float = 0.1):
        """
        Initialize the attention feature extractor.
        
        Args:
            input_dim: Dimensionality of input features (per time step)
            hidden_dim: Dimensionality of hidden states
            output_dim: Dimensionality of output features
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the attention feature extractor.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional mask tensor [seq_len, seq_len]
            
        Returns:
            Tuple of (output_features, attention_weights)
        """
        # Project input to hidden dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        # Note: In PyTorch 1.8+, the attention weights are not directly accessible
        # We would need a custom implementation to extract them
        transformer_output = self.transformer_encoder(x, mask=mask)
        
        # Project to output dimension
        output_features = self.output_proj(transformer_output)
        
        # Return output features and attention weights (placeholder for now)
        attention_weights = torch.zeros(x.shape[0], x.shape[1], x.shape[1])
        
        return output_features, attention_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Adds information about the position of tokens in the sequence,
    which is important for transformer models that have no inherent
    notion of sequence order.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model: Dimensionality of the model
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to add positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DeepFeatureExtractor:
    """
    High-level class for extracting features from financial data using
    deep learning models. This class coordinates the training and feature
    extraction process.
    """
    
    def __init__(self, 
                model_type: str = 'autoencoder',
                input_dim: int = None,
                window_size: int = 60,
                batch_size: int = 32,
                latent_dim: int = 10,
                learning_rate: float = 0.001,
                num_epochs: int = 100,
                device: str = None):
        """
        Initialize the deep feature extractor.
        
        Args:
            model_type: Type of model to use ('autoencoder', 'temporal', or 'attention')
            input_dim: Input dimension of the data
            window_size: Size of the input window for temporal models
            batch_size: Batch size for training
            latent_dim: Dimensionality of the latent space
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.input_dim = input_dim
        self.window_size = window_size
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize model
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info(f"Initialized DeepFeatureExtractor with {model_type} model on {self.device}")
    
    def _create_model(self) -> nn.Module:
        """Create the appropriate model based on model_type."""
        if self.model_type == 'autoencoder':
            return Autoencoder(
                input_dim=self.input_dim,
                hidden_dims=[128, 64, 32],  # Example architecture
                latent_dim=self.latent_dim,
                dropout=0.1
            )
        elif self.model_type == 'temporal':
            return TemporalAutoencoder(
                input_dim=self.input_dim,
                hidden_dim=64,
                latent_dim=self.latent_dim,
                num_layers=2,
                cell_type='lstm',
                dropout=0.1
            )
        elif self.model_type == 'attention':
            return AttentionFeatureExtractor(
                input_dim=self.input_dim,
                hidden_dim=64,
                output_dim=self.latent_dim,
                num_heads=4,
                num_layers=2,
                dropout=0.1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, data: np.ndarray, validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        Train the feature extractor model.
        
        Args:
            data: Input data (time series data)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training history
        """
        if self.input_dim is None:
            if len(data.shape) == 1:
                self.input_dim = 1
            else:
                self.input_dim = data.shape[1]
                
        # Scale the data
        if len(data.shape) == 1:
            data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        else:
            data_scaled = self.scaler.fit_transform(data)
            
        # Create model
        self.model = self._create_model().to(self.device)
        
        # Prepare data loader
        if self.model_type in ['temporal', 'attention']:
            # For temporal models, we need to prepare sequences
            dataset = TimeSeriesDataset(
                data=data_scaled,
                window_size=self.window_size,
                stride=1
            )
        else:
            # For standard autoencoder, we use the data directly
            tensor_x = torch.FloatTensor(data_scaled)
            dataset = TensorDataset(tensor_x)
            
        # Split into train and validation sets
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        logger.info(f"Starting training for {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                # Get input data
                if self.model_type in ['temporal', 'attention']:
                    x = batch['X'].to(self.device)
                else:
                    x = batch[0].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                if self.model_type == 'attention':
                    output, _ = self.model(x)
                    loss = F.mse_loss(output, x)
                else:
                    output, _ = self.model(x)
                    loss = F.mse_loss(output, x)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    # Get input data
                    if self.model_type in ['temporal', 'attention']:
                        x = batch['X'].to(self.device)
                    else:
                        x = batch[0].to(self.device)
                    
                    # Forward pass
                    if self.model_type == 'attention':
                        output, _ = self.model(x)
                        loss = F.mse_loss(output, x)
                    else:
                        output, _ = self.model(x)
                        loss = F.mse_loss(output, x)
                    
                    val_loss += loss.item()
                    
            # Average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                    
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        self.is_fitted = True
        logger.info("Training completed")
        
        return history
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data into the latent space (extract features).
        
        Args:
            data: Input data to transform
            
        Returns:
            Extracted features in the latent space
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
            
        # Scale the data
        if len(data.shape) == 1:
            data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()
        else:
            data_scaled = self.scaler.transform(data)
            
        # Prepare data loader
        if self.model_type in ['temporal', 'attention']:
            # For temporal models, we need to prepare sequences
            dataset = TimeSeriesDataset(
                data=data_scaled,
                window_size=self.window_size,
                stride=1
            )
        else:
            # For standard autoencoder, we use the data directly
            tensor_x = torch.FloatTensor(data_scaled)
            dataset = TensorDataset(tensor_x)
            
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Extract features
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Get input data
                if self.model_type in ['temporal', 'attention']:
                    x = batch['X'].to(self.device)
                else:
                    x = batch[0].to(self.device)
                
                # Get latent representation
                if self.model_type == 'attention':
                    output, _ = self.model(x)
                    # For attention model, use last time step output as feature
                    batch_features = output[:, -1, :].cpu().numpy()
                else:
                    _, z = self.model(x)
                    batch_features = z.cpu().numpy()
                
                features.append(batch_features)
                
        # Concatenate features
        features = np.concatenate(features, axis=0)
        
        return features
    
    def fit_transform(self, data: np.ndarray, validation_split: float = 0.2) -> np.ndarray:
        """
        Fit the model and transform the data.
        
        Args:
            data: Input data to fit and transform
            validation_split: Fraction of data to use for validation
            
        Returns:
            Extracted features in the latent space
        """
        self.fit(data, validation_split)
        return self.transform(data)
    
    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from the latent space.
        
        Args:
            data: Input data to reconstruct
            
        Returns:
            Reconstructed data
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
            
        # Scale the data
        if len(data.shape) == 1:
            data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()
        else:
            data_scaled = self.scaler.transform(data)
            
        # Prepare data loader
        if self.model_type in ['temporal', 'attention']:
            # For temporal models, we need to prepare sequences
            dataset = TimeSeriesDataset(
                data=data_scaled,
                window_size=self.window_size,
                stride=1
            )
        else:
            # For standard autoencoder, we use the data directly
            tensor_x = torch.FloatTensor(data_scaled)
            dataset = TensorDataset(tensor_x)
            
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Reconstruct data
        self.model.eval()
        reconstructions = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Get input data
                if self.model_type in ['temporal', 'attention']:
                    x = batch['X'].to(self.device)
                else:
                    x = batch[0].to(self.device)
                
                # Get reconstruction
                output, _ = self.model(x)
                reconstructions.append(output.cpu().numpy())
                
        # Concatenate reconstructions
        reconstructions = np.concatenate(reconstructions, axis=0)
        
        # Inverse transform
        if len(data.shape) == 1:
            reconstructions = self.scaler.inverse_transform(reconstructions.reshape(-1, 1)).flatten()
        else:
            reconstructions = self.scaler.inverse_transform(reconstructions)
            
        return reconstructions
    
    def save_model(self, path: str):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state and metadata
        model_state = {
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'window_size': self.window_size,
            'latent_dim': self.latent_dim,
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler
        }
        
        torch.save(model_state, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'DeepFeatureExtractor':
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded feature extractor
        """
        # Load model state
        model_state = torch.load(path)
        
        # Create new instance
        instance = cls(
            model_type=model_state['model_type'],
            input_dim=model_state['input_dim'],
            window_size=model_state['window_size'],
            latent_dim=model_state['latent_dim']
        )
        
        # Create model
        instance.model = instance._create_model().to(instance.device)
        
        # Load model state
        instance.model.load_state_dict(model_state['model_state_dict'])
        
        # Load scaler
        instance.scaler = model_state['scaler']
        
        instance.is_fitted = True
        logger.info(f"Model loaded from {path}")
        
        return instance


class FeatureImportanceAnalyzer:
    """
    Analyzes the importance of extracted features for prediction tasks.
    
    Uses techniques like SHAP values and permutation importance to determine
    which features are most important for predicting target variables.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the feature importance analyzer.
        
        Args:
            model_type: Type of model to use for importance analysis
                        ('random_forest', 'xgboost', or 'linear')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        
    def _create_model(self):
        """Create the model for feature importance analysis."""
        if self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'xgboost':
            import xgboost as xgb
            return xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'linear':
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def fit(self, 
           features: np.ndarray, 
           targets: np.ndarray,
           feature_names: Optional[List[str]] = None):
        """
        Fit the model for feature importance analysis.
        
        Args:
            features: Input features
            targets: Target values
            feature_names: Names of the features (optional)
        """
        self.model = self._create_model()
        
        # Set feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(features.shape[1])]
            
        # Fit the model
        self.model.fit(features, targets)
        
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get basic feature importance from the model.
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")
            
        # Extract importance scores
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            raise ValueError("Model does not have feature importance attribute")
            
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def get_shap_values(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate SHAP values for the features.
        
        Args:
            features: Input features
            
        Returns:
            SHAP values for each feature and sample
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")
            
        # Create explainer
        explainer = shap.Explainer(self.model, features)
        
        # Calculate SHAP values
        shap_values = explainer(features)
        
        return shap_values
    
    def plot_shap_summary(self, features: np.ndarray, max_display: int = 20):
        """
        Plot SHAP summary plot.
        
        Args:
            features: Input features
            max_display: Maximum number of features to display
        """
        shap_values = self.get_shap_values(features)
        
        # Create summary plot
        shap.summary_plot(
            shap_values, 
            features,
            feature_names=self.feature_names, 
            max_display=max_display
        )
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            figsize: Figure size
        """
        importance_df = self.get_feature_importance()
        
        # Select top N features
        plot_df = importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(x='importance', y='feature', data=plot_df)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()


class NeuralFeatureDiscovery:
    """
    High-level class for discovering important features from financial data.
    
    This class brings together the deep feature extraction and importance
    analysis components to discover meaningful features.
    """
    
    def __init__(self,
                extractor_type: str = 'temporal',
                window_size: int = 60,
                latent_dim: int = 10,
                batch_size: int = 32,
                learning_rate: float = 0.001,
                num_epochs: int = 100,
                device: str = None):
        """
        Initialize the neural feature discovery system.
        
        Args:
            extractor_type: Type of feature extractor ('autoencoder', 'temporal', or 'attention')
            window_size: Size of the input window for temporal models
            latent_dim: Dimensionality of the latent space
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            device: Device to use ('cuda' or 'cpu')
        """
        self.extractor_type = extractor_type
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        
        # Initialize components
        self.feature_extractor = None
        self.importance_analyzer = None
        
        # Store discovered features
        self.discovered_features = None
        self.feature_importance = None
        
        logger.info(f"Initialized NeuralFeatureDiscovery with {extractor_type} extractor")
    
    def fit(self, 
           data: Union[pd.DataFrame, np.ndarray],
           target: Optional[Union[pd.Series, np.ndarray]] = None,
           validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Fit the feature discovery system to the data.
        
        Args:
            data: Input data (time series data)
            target: Target values (optional)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with results
        """
        # Convert to numpy arrays
        if isinstance(data, pd.DataFrame):
            feature_names = data.columns.tolist()
            data_np = data.values
        else:
            data_np = data
            feature_names = [f'feature_{i}' for i in range(data_np.shape[1] if len(data_np.shape) > 1 else 1)]
            
        # Handle target
        if target is not None:
            if isinstance(target, pd.Series):
                target_np = target.values
            else:
                target_np = target
        else:
            target_np = None
            
        # Create feature extractor
        input_dim = data_np.shape[1] if len(data_np.shape) > 1 else 1
        self.feature_extractor = DeepFeatureExtractor(
            model_type=self.extractor_type,
            input_dim=input_dim,
            window_size=self.window_size,
            batch_size=self.batch_size,
            latent_dim=self.latent_dim,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            device=self.device
        )
        
        # Train feature extractor
        history = self.feature_extractor.fit(data_np, validation_split)
        
        # Extract features
        discovered_features = self.feature_extractor.transform(data_np)
        
        # Store discovered features
        if len(discovered_features.shape) == 1:
            self.discovered_features = pd.DataFrame(
                discovered_features.reshape(-1, 1),
                columns=['discovered_feature_0']
            )
        else:
            self.discovered_features = pd.DataFrame(
                discovered_features,
                columns=[f'discovered_feature_{i}' for i in range(discovered_features.shape[1])]
            )
            
        # If target is provided, analyze feature importance
        if target_np is not None:
            # Align target with features (for temporal models)
            if self.extractor_type in ['temporal', 'attention']:
                # For temporal models, the features are extracted from windows
                # We need to align the target with the extracted features
                target_aligned = target_np[self.window_size-1:]
                if len(target_aligned) > len(discovered_features):
                    target_aligned = target_aligned[:len(discovered_features)]
            else:
                target_aligned = target_np
                
            # Create importance analyzer
            self.importance_analyzer = FeatureImportanceAnalyzer(model_type='random_forest')
            
            # Fit importance analyzer
            self.importance_analyzer.fit(
                discovered_features,
                target_aligned,
                feature_names=self.discovered_features.columns.tolist()
            )
            
            # Get feature importance
            self.feature_importance = self.importance_analyzer.get_feature_importance()
            
        # Return results
        results = {
            'history': history,
            'discovered_features': self.discovered_features,
            'feature_importance': self.feature_importance
        }
        
        logger.info(f"Feature discovery completed. Found {self.discovered_features.shape[1]} features.")
        
        return results
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Apply feature extraction to new data.
        
        Args:
            data: Input data (time series data)
            
        Returns:
            DataFrame with extracted features
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor is not fitted. Call fit() first.")
            
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            data_np = data.values
        else:
            data_np = data
            
        # Extract features
        discovered_features = self.feature_extractor.transform(data_np)
        
        # Convert to DataFrame
        if len(discovered_features.shape) == 1:
            feature_df = pd.DataFrame(
                discovered_features.reshape(-1, 1),
                columns=['discovered_feature_0']
            )
        else:
            feature_df = pd.DataFrame(
                discovered_features,
                columns=[f'discovered_feature_{i}' for i in range(discovered_features.shape[1])]
            )
            
        return feature_df
    
    def get_top_features(self, n: int = 5) -> pd.DataFrame:
        """
        Get the top discovered features by importance.
        
        Args:
            n: Number of top features to return
            
        Returns:
            DataFrame with top features
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated. Call fit() with a target.")
            
        top_features = self.feature_importance.head(n)
        
        return top_features
    
    def plot_top_features(self, n: int = 5, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot the top discovered features by importance.
        
        Args:
            n: Number of top features to plot
            figsize: Figure size
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated. Call fit() with a target.")
            
        top_features = self.feature_importance.head(n)
        
        plt.figure(figsize=figsize)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('Top Discovered Features')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        
    def plot_feature_clusters(self, n_clusters: int = 3, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot clusters of discovered features.
        
        Args:
            n_clusters: Number of clusters
            figsize: Figure size
        """
        if self.discovered_features is None:
            raise ValueError("No discovered features. Call fit() first.")
            
        # Perform clustering
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(self.discovered_features)
        
        # Cluster features
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.discovered_features)
        
        # Plot clusters
        plt.figure(figsize=figsize)
        
        for i in range(n_clusters):
            plt.scatter(
                features_2d[clusters == i, 0],
                features_2d[clusters == i, 1],
                label=f'Cluster {i}'
            )
            
        plt.title('Feature Clusters')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def save(self, path: str):
        """
        Save the feature discovery system.
        
        Args:
            path: Path to save the system
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor is not fitted. Call fit() first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save feature extractor
        self.feature_extractor.save_model(f"{path}_extractor.pt")
        
        # Save discovered features and importance if available
        if self.discovered_features is not None:
            self.discovered_features.to_csv(f"{path}_features.csv", index=False)
            
        if self.feature_importance is not None:
            self.feature_importance.to_csv(f"{path}_importance.csv", index=False)
            
        logger.info(f"Feature discovery system saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'NeuralFeatureDiscovery':
        """
        Load a saved feature discovery system.
        
        Args:
            path: Path to the saved system
            
        Returns:
            Loaded feature discovery system
        """
        # Create new instance
        instance = cls()
        
        # Load feature extractor
        instance.feature_extractor = DeepFeatureExtractor.load_model(f"{path}_extractor.pt")
        
        # Load discovered features if available
        try:
            instance.discovered_features = pd.read_csv(f"{path}_features.csv")
        except:
            logger.warning(f"Could not load discovered features from {path}_features.csv")
            
        # Load feature importance if available
        try:
            instance.feature_importance = pd.read_csv(f"{path}_importance.csv")
        except:
            logger.warning(f"Could not load feature importance from {path}_importance.csv")
            
        logger.info(f"Feature discovery system loaded from {path}")
        
        return instance


def main():
    """
    Example usage of the neural feature discovery module.
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Neural Network-based Feature Discovery")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--target", default=None, help="Target column name (optional)")
    parser.add_argument("--extractor", default="temporal", choices=["autoencoder", "temporal", "attention"], help="Feature extractor type")
    parser.add_argument("--window", type=int, default=60, help="Window size for temporal models")
    parser.add_argument("--features", type=int, default=10, help="Number of features to discover")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--output", default="output", help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"Loading data from {args.data}")
        data = pd.read_csv(args.data)
        
        # Extract target if provided
        target = None
        if args.target is not None and args.target in data.columns:
            target = data[args.target]
            data = data.drop(columns=[args.target])
            
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Initialize feature discovery system
        feature_discovery = NeuralFeatureDiscovery(
            extractor_type=args.extractor,
            window_size=args.window,
            latent_dim=args.features,
            num_epochs=args.epochs
        )
        
        # Fit the system
        print(f"Discovering features using {args.extractor} model...")
        results = feature_discovery.fit(data, target)
        
        # Save the system
        feature_discovery.save(os.path.join(args.output, "feature_discovery"))
        
        # Plot top features if target was provided
        if target is not None:
            top_features = feature_discovery.get_top_features(n=5)
            print("\nTop discovered features:")
            print(top_features)
            
            # Save feature importance plot
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title('Top Discovered Features')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output, "top_features.png"))
            
        # Plot reconstruction error
        plt.figure(figsize=(10, 6))
        plt.plot(results['history']['train_loss'], label='Train Loss')
        plt.plot(results['history']['val_loss'], label='Validation Loss')
        plt.title('Feature Extractor Training')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "training_history.png"))
        
        # Save discovered features
        feature_discovery.discovered_features.to_csv(os.path.join(args.output, "discovered_features.csv"), index=False)
        
        print("\nNEURAL FEATURE DISCOVERY MODULE IMPLEMENTATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Discovered {args.features} features from the data.")
        print(f"Results saved to {args.output}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())