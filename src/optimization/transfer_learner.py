#!/usr/bin/env python
"""
Transfer Learning Module

This module implements transfer learning techniques for adapting trading strategies
across different asset classes, leveraging domain adaptation and feature alignment
methods.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Callable, Any, Optional
import logging
from datetime import datetime

class TransferLearner:
    """Implements transfer learning for cross-asset strategy adaptation."""
    
    def __init__(self,
                source_strategy: Callable,
                adaptation_method: str = 'feature_alignment',
                n_components: int = 10,
                similarity_threshold: float = 0.7,
                max_adaptation_window: int = 252):
        """
        Initialize the transfer learner.

        Args:
            source_strategy: Original strategy to transfer
            adaptation_method: Method for domain adaptation
            n_components: Number of components for feature alignment
            similarity_threshold: Minimum similarity for transfer
            max_adaptation_window: Maximum window for adaptation
        """
        self.source_strategy = source_strategy
        self.adaptation_method = adaptation_method
        self.n_components = n_components
        self.similarity_threshold = similarity_threshold
        self.max_adaptation_window = max_adaptation_window
        
        # Initialize components
        self.feature_transformer = None
        self.domain_adapter = None
        self.target_models = {}
        self.feature_alignments = {}
        
        self.logger = logging.getLogger(__name__)
        
    def fit(self,
           source_data: pd.DataFrame,
           target_data: Dict[str, pd.DataFrame],
           source_returns: pd.Series,
           target_returns: Dict[str, pd.Series]):
        """
        Fit transfer learning model.
        
        Args:
            source_data: Data from source asset class
            target_data: Data from target asset classes
            source_returns: Returns from source asset
            target_returns: Returns from target assets
        """
        # Extract features
        source_features = self._extract_features(source_data)
        target_features = {
            asset: self._extract_features(data)
            for asset, data in target_data.items()
        }
        
        if self.adaptation_method == 'feature_alignment':
            self._fit_feature_alignment(
                source_features, target_features,
                source_returns, target_returns
            )
        else:  # domain_adaptation
            self._fit_domain_adaptation(
                source_features, target_features,
                source_returns, target_returns
            )
            
    def transfer(self,
                target_data: pd.DataFrame,
                asset_class: str) -> pd.Series:
        """
        Transfer strategy to new asset.
        
        Args:
            target_data: Data for target asset
            asset_class: Asset class identifier
            
        Returns:
            Adapted strategy predictions
        """
        # Extract features
        features = self._extract_features(target_data)
        
        if self.adaptation_method == 'feature_alignment':
            return self._transfer_aligned(features, asset_class)
        else:
            return self._transfer_adapted(features, asset_class)
            
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract relevant features for transfer learning."""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        returns = data['close'].pct_change()
        features['volatility'] = returns.rolling(21).std()
        features['momentum'] = returns.rolling(63).mean()
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(returns)
        features['ma_ratio'] = (
            data['close'].rolling(20).mean() /
            data['close'].rolling(50).mean()
        )
        
        # Volume-based features if available
        if 'volume' in data.columns:
            features['volume_trend'] = (
                data['volume'].rolling(20).mean() /
                data['volume'].rolling(50).mean()
            )
            
        return features.fillna(0)
    
    def _fit_feature_alignment(self,
                            source_features: pd.DataFrame,
                            target_features: Dict[str, pd.DataFrame],
                            source_returns: pd.Series,
                            target_returns: Dict[str, pd.Series]):
        """Fit feature alignment transfer."""
        # Initialize feature transformer
        self.feature_transformer = PCA(n_components=self.n_components)
        
        # Fit on source features
        source_transformed = pd.DataFrame(
            self.feature_transformer.fit_transform(source_features),
            index=source_features.index
        )
        
        # Transform target features
        target_transformed = {
            asset: pd.DataFrame(
                self.feature_transformer.transform(features),
                index=features.index
            )
            for asset, features in target_features.items()
        }
        
        # Calculate feature alignments
        for asset in target_features.keys():
            alignment = self._calculate_feature_alignment(
                source_transformed,
                target_transformed[asset]
            )
            self.feature_alignments[asset] = alignment
            
            # Train target-specific model if alignment is good
            if alignment > self.similarity_threshold:
                self.target_models[asset] = self._train_target_model(
                    target_transformed[asset],
                    target_returns[asset]
                )
                
    def _fit_domain_adaptation(self,
                            source_features: pd.DataFrame,
                            target_features: Dict[str, pd.DataFrame],
                            source_returns: pd.Series,
                            target_returns: Dict[str, pd.Series]):
        """Fit domain adaptation transfer."""
        # Initialize domain adapter
        self.domain_adapter = RandomForestRegressor(
            n_estimators=100,
            max_depth=3
        )
        
        for asset, features in target_features.items():
            # Calculate domain similarity
            similarity = self._calculate_domain_similarity(
                source_features, features
            )
            
            if similarity > self.similarity_threshold:
                # Combine source and target data
                combined_features = pd.concat([
                    source_features,
                    features
                ])
                
                combined_returns = pd.concat([
                    source_returns,
                    target_returns[asset]
                ])
                
                # Add domain indicator
                domain_indicator = pd.Series(
                    [0] * len(source_features) + [1] * len(features),
                    index=combined_features.index
                )
                
                combined_features['domain'] = domain_indicator
                
                # Train domain-adapted model
                self.target_models[asset] = self._train_adapted_model(
                    combined_features,
                    combined_returns
                )
                
    def _transfer_aligned(self,
                        features: pd.DataFrame,
                        asset_class: str) -> pd.Series:
        """Transfer using feature alignment."""
        if asset_class not in self.target_models:
            self.logger.warning(f"No model available for {asset_class}")
            return pd.Series(0, index=features.index)
            
        # Transform features
        transformed = pd.DataFrame(
            self.feature_transformer.transform(features),
            index=features.index
        )
        
        # Generate predictions
        predictions = self.target_models[asset_class].predict(transformed)
        
        return pd.Series(predictions, index=features.index)
    
    def _transfer_adapted(self,
                        features: pd.DataFrame,
                        asset_class: str) -> pd.Series:
        """Transfer using domain adaptation."""
        if asset_class not in self.target_models:
            self.logger.warning(f"No model available for {asset_class}")
            return pd.Series(0, index=features.index)
            
        # Add domain indicator
        features = features.copy()
        features['domain'] = 1
        
        # Generate predictions
        predictions = self.target_models[asset_class].predict(features)
        
        return pd.Series(predictions, index=features.index)
    
    def _calculate_feature_alignment(self,
                                 source_features: pd.DataFrame,
                                 target_features: pd.DataFrame) -> float:
        """Calculate feature alignment score."""
        # Calculate correlation between feature spaces
        correlation = np.corrcoef(
            source_features.T,
            target_features.T
        )
        
        # Extract cross-correlation block
        n_source = len(source_features.columns)
        cross_corr = correlation[:n_source, n_source:]
        
        return np.mean(np.abs(cross_corr))
    
    def _calculate_domain_similarity(self,
                                 source_features: pd.DataFrame,
                                 target_features: pd.DataFrame) -> float:
        """Calculate domain similarity score."""
        # Calculate distribution similarity using MMD
        mmd = self._maximum_mean_discrepancy(
            source_features.values,
            target_features.values
        )
        
        # Convert to similarity score (0 to 1)
        similarity = np.exp(-mmd)
        
        return similarity
    
    def _maximum_mean_discrepancy(self,
                               source: np.ndarray,
                               target: np.ndarray) -> float:
        """Calculate Maximum Mean Discrepancy."""
        source_kernel = self._rbf_kernel(source, source)
        target_kernel = self._rbf_kernel(target, target)
        cross_kernel = self._rbf_kernel(source, target)
        
        mmd = (np.mean(source_kernel) + np.mean(target_kernel) - 
               2 * np.mean(cross_kernel))
        
        return mmd
    
    def _rbf_kernel(self,
                   x: np.ndarray,
                   y: np.ndarray,
                   gamma: float = 1.0) -> np.ndarray:
        """Calculate RBF kernel."""
        x_norm = np.sum(x**2, axis=1).reshape(-1, 1)
        y_norm = np.sum(y**2, axis=1).reshape(1, -1)
        dist = x_norm + y_norm - 2 * np.dot(x, y.T)
        return np.exp(-gamma * dist)
    
    def _train_target_model(self,
                         features: pd.DataFrame,
                         returns: pd.Series) -> RandomForestRegressor:
        """Train model for target asset."""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=3
        )
        model.fit(features, returns)
        return model
    
    def _train_adapted_model(self,
                          features: pd.DataFrame,
                          returns: pd.Series) -> RandomForestRegressor:
        """Train domain-adapted model."""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=3
        )
        model.fit(features, returns)
        return model
    
    def _calculate_rsi(self, returns: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        
        avg_gain = gains.rolling(window).mean()
        avg_loss = losses.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def get_feature_importance(self, asset_class: str) -> Optional[pd.Series]:
        """Get feature importance for target asset class."""
        if asset_class not in self.target_models:
            return None
            
        model = self.target_models[asset_class]
        if hasattr(model, 'feature_importances_'):
            return pd.Series(
                model.feature_importances_,
                index=self.feature_transformer.get_feature_names_out()
            )
        return None
    
    def plot_transfer_analysis(self, save_path: Optional[str] = None):
        """Plot transfer learning analysis."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot feature alignments
        alignments = pd.Series(self.feature_alignments)
        alignments.plot(kind='bar', ax=ax1)
        ax1.set_title('Feature Alignment Scores')
        ax1.set_xlabel('Asset Class')
        ax1.set_ylabel('Alignment Score')
        ax1.axhline(y=self.similarity_threshold, color='r', linestyle='--')
        
        # Plot model performance comparison
        if self.target_models:
            performance = pd.Series({
                asset: model.score(
                    self.feature_transformer.transform(
                        self._extract_features(data)
                    ),
                    returns
                )
                for asset, (model, data, returns) in self.target_models.items()
            })
            
            performance.plot(kind='bar', ax=ax2)
            ax2.set_title('Model Performance')
            ax2.set_xlabel('Asset Class')
            ax2.set_ylabel('R² Score')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        plt.close()