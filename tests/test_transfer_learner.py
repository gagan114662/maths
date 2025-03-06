#!/usr/bin/env python
"""Unit tests for transfer learning."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.optimization.transfer_learner import TransferLearner

class TestTransferLearner(unittest.TestCase):
    def setUp(self):
        """Set up test data and transfer learner."""
        # Create test data for source asset (equity)
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        equity_prices = 100 * (1 + np.random.normal(0, 0.02, 1000)).cumprod()
        equity_volume = np.random.lognormal(0, 1, 1000) * 1000000
        
        self.source_data = pd.DataFrame({
            'close': equity_prices,
            'volume': equity_volume
        }, index=dates)
        
        # Create test data for target assets (forex and crypto)
        forex_prices = 1.2 * (1 + np.random.normal(0, 0.01, 1000)).cumprod()
        crypto_prices = 10000 * (1 + np.random.normal(0, 0.04, 1000)).cumprod()
        
        self.target_data = {
            'forex': pd.DataFrame({
                'close': forex_prices,
                'volume': np.random.lognormal(0, 1, 1000) * 1000000
            }, index=dates),
            'crypto': pd.DataFrame({
                'close': crypto_prices,
                'volume': np.random.lognormal(0, 1, 1000) * 1000
            }, index=dates)
        }
        
        # Calculate returns
        self.source_returns = self.source_data['close'].pct_change()
        self.target_returns = {
            asset: data['close'].pct_change()
            for asset, data in self.target_data.items()
        }
        
        # Define source strategy
        def momentum_strategy(data):
            returns = data['close'].pct_change()
            return np.sign(returns.rolling(20).mean())
            
        # Initialize transfer learner
        self.learner = TransferLearner(
            source_strategy=momentum_strategy,
            adaptation_method='feature_alignment',
            n_components=5,
            similarity_threshold=0.7
        )
        
    def test_initialization(self):
        """Test transfer learner initialization."""
        self.assertEqual(self.learner.adaptation_method, 'feature_alignment')
        self.assertEqual(self.learner.n_components, 5)
        self.assertEqual(self.learner.similarity_threshold, 0.7)
        
    def test_feature_extraction(self):
        """Test feature extraction."""
        features = self.learner._extract_features(self.source_data)
        
        # Check feature structure
        self.assertIn('volatility', features.columns)
        self.assertIn('momentum', features.columns)
        self.assertIn('rsi', features.columns)
        self.assertIn('ma_ratio', features.columns)
        self.assertIn('volume_trend', features.columns)
        
        # Check feature values
        self.assertTrue(features['volatility'].notna().all())
        self.assertTrue(features['momentum'].notna().all())
        self.assertTrue((features['rsi'] >= 0).all())
        self.assertTrue((features['rsi'] <= 100).all())
        
    def test_feature_alignment(self):
        """Test feature alignment method."""
        # Fit transfer learner
        self.learner.fit(
            self.source_data,
            self.target_data,
            self.source_returns,
            self.target_returns
        )
        
        # Check feature transformer
        self.assertIsNotNone(self.learner.feature_transformer)
        self.assertEqual(
            self.learner.feature_transformer.n_components_,
            self.learner.n_components
        )
        
        # Check alignments
        for asset in self.target_data.keys():
            self.assertIn(asset, self.learner.feature_alignments)
            alignment = self.learner.feature_alignments[asset]
            self.assertTrue(0 <= alignment <= 1)
            
    def test_transfer_prediction(self):
        """Test strategy transfer and prediction."""
        # Fit transfer learner
        self.learner.fit(
            self.source_data,
            self.target_data,
            self.source_returns,
            self.target_returns
        )
        
        # Test transfer to each target asset
        for asset, data in self.target_data.items():
            predictions = self.learner.transfer(data, asset)
            
            # Check predictions
            self.assertEqual(len(predictions), len(data))
            self.assertTrue((-1 <= predictions).all())
            self.assertTrue((predictions <= 1).all())
            
    def test_domain_adaptation(self):
        """Test domain adaptation method."""
        # Initialize with domain adaptation
        learner = TransferLearner(
            source_strategy=self.learner.source_strategy,
            adaptation_method='domain_adaptation'
        )
        
        # Fit and transfer
        learner.fit(
            self.source_data,
            self.target_data,
            self.source_returns,
            self.target_returns
        )
        
        for asset, data in self.target_data.items():
            predictions = learner.transfer(data, asset)
            self.assertEqual(len(predictions), len(data))
            
    def test_feature_importance(self):
        """Test feature importance calculation."""
        # Fit transfer learner
        self.learner.fit(
            self.source_data,
            self.target_data,
            self.source_returns,
            self.target_returns
        )
        
        # Get feature importance for each target asset
        for asset in self.target_data.keys():
            importance = self.learner.get_feature_importance(asset)
            if importance is not None:
                self.assertEqual(len(importance), self.learner.n_components)
                self.assertTrue((importance >= 0).all())
                
    def test_domain_similarity(self):
        """Test domain similarity calculation."""
        source_features = self.learner._extract_features(self.source_data)
        
        for asset, data in self.target_data.items():
            target_features = self.learner._extract_features(data)
            similarity = self.learner._calculate_domain_similarity(
                source_features,
                target_features
            )
            
            self.assertTrue(0 <= similarity <= 1)
            
    def test_error_handling(self):
        """Test error handling."""
        # Test with insufficient data
        short_data = self.source_data.head(10)
        short_returns = self.source_returns.head(10)
        
        with self.assertRaises(ValueError):
            self.learner.fit(
                short_data,
                {'forex': self.target_data['forex'].head(10)},
                short_returns,
                {'forex': self.target_returns['forex'].head(10)}
            )
            
    def test_visualization(self):
        """Test visualization functionality."""
        # Fit transfer learner
        self.learner.fit(
            self.source_data,
            self.target_data,
            self.source_returns,
            self.target_returns
        )
        
        # Test plotting without saving
        self.learner.plot_transfer_analysis()
        
        # Test plotting with saving
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            self.learner.plot_transfer_analysis(save_path=tmp.name)

if __name__ == '__main__':
    unittest.main()