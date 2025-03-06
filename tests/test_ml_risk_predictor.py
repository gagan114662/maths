#!/usr/bin/env python
"""Unit tests for machine learning risk predictor."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.risk_management.ml_risk_predictor import MLRiskPredictor

class TestMLRiskPredictor(unittest.TestCase):
    def setUp(self):
        """Set up test data and predictor."""
        np.random.seed(42)
        
        # Generate test data
        n_days = 1000
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        
        # Create returns with different regimes
        returns = []
        
        # Low volatility period
        returns.extend(np.random.normal(0.0005, 0.005, n_days // 4))
        
        # High volatility period
        returns.extend(np.random.normal(0, 0.02, n_days // 4))
        
        # Trending period
        returns.extend(np.random.normal(0.001, 0.01, n_days // 4))
        
        # Crisis period
        returns.extend(np.random.normal(-0.002, 0.03, n_days // 4))
        
        self.returns = pd.Series(returns, index=dates)
        
        # Initialize predictor
        self.predictor = MLRiskPredictor(
            lookback_window=252,
            forecast_horizon=21,
            feature_windows=[5, 21, 63, 252]
        )
        
    def test_feature_preparation(self):
        """Test feature preparation."""
        features = self.predictor.prepare_features(self.returns)
        
        # Check that features exist
        for window in self.predictor.feature_windows:
            self.assertIn(f'volatility_{window}d', features.columns)
            self.assertIn(f'skewness_{window}d', features.columns)
            self.assertIn(f'kurtosis_{window}d', features.columns)
            self.assertIn(f'cum_return_{window}d', features.columns)
            self.assertIn(f'max_drawdown_{window}d', features.columns)
            self.assertIn(f'up_days_{window}d', features.columns)
            self.assertIn(f'var_95_{window}d', features.columns)
            self.assertIn(f'cvar_95_{window}d', features.columns)
            
        # Check technical features
        self.assertIn('rsi_14', features.columns)
        self.assertIn('volatility_trend', features.columns)
        
        # Check sequential features
        self.assertIn('day_of_week', features.columns)
        self.assertIn('month', features.columns)
        
        # Check for NaN values
        self.assertFalse(features.isnull().any().any())
        
    def test_target_preparation(self):
        """Test target preparation."""
        targets = self.predictor.prepare_targets(self.returns)
        
        # Check that targets exist
        self.assertIn('future_volatility', targets.columns)
        self.assertIn('future_drawdown', targets.columns)
        self.assertIn('future_var_95', targets.columns)
        
        # Check dimensions
        expected_len = len(self.returns) - self.predictor.forecast_horizon
        self.assertEqual(len(targets.dropna()), expected_len)
        
        # Check value ranges
        self.assertTrue((targets['future_volatility'] >= 0).all())
        self.assertTrue((targets['future_drawdown'] >= 0).all())
        self.assertTrue((targets['future_drawdown'] <= 1).all())
        
    def test_model_training(self):
        """Test model training."""
        features = self.predictor.prepare_features(self.returns)
        targets = self.predictor.prepare_targets(self.returns)
        
        # Train models
        self.predictor.train_models(features, targets)
        
        # Check that models are created
        for target in targets.columns:
            self.assertIn(target, self.predictor.models)
            self.assertIn(target, self.predictor.scalers)
            self.assertIn(target, self.predictor.feature_importance)
            
        # Check feature importance
        for target in targets.columns:
            importance = self.predictor.get_feature_importance(target)
            self.assertEqual(len(importance), len(features.columns))
            self.assertTrue((importance >= 0).all())
            
    def test_risk_prediction(self):
        """Test risk metric prediction."""
        features = self.predictor.prepare_features(self.returns)
        targets = self.predictor.prepare_targets(self.returns)
        
        # Train models
        self.predictor.train_models(features, targets)
        
        # Make predictions
        predictions = self.predictor.predict_risk_metrics(features)
        
        # Check predictions
        self.assertEqual(len(predictions.columns), len(targets.columns))
        self.assertEqual(len(predictions), len(features))
        
        # Check value ranges
        self.assertTrue((predictions['future_volatility'] >= 0).all())
        self.assertTrue((predictions['future_drawdown'] >= 0).all())
        self.assertTrue((predictions['future_drawdown'] <= 1).all())
        
    def test_model_persistence(self):
        """Test model saving and loading."""
        import tempfile
        import os
        
        features = self.predictor.prepare_features(self.returns)
        targets = self.predictor.prepare_targets(self.returns)
        
        # Train models
        self.predictor.train_models(features, targets)
        
        # Save models
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            self.predictor.save_models(tmp.name)
            
            # Create new predictor and load models
            new_predictor = MLRiskPredictor()
            new_predictor.load_models(tmp.name)
            
            # Compare predictions
            pred1 = self.predictor.predict_risk_metrics(features)
            pred2 = new_predictor.predict_risk_metrics(features)
            
            pd.testing.assert_frame_equal(pred1, pred2)
            
            # Clean up
            os.unlink(tmp.name)
            
    def test_error_handling(self):
        """Test error handling."""
        # Test insufficient data
        short_returns = pd.Series(
            np.random.normal(0, 0.01, 50)
        )
        
        features = self.predictor.prepare_features(short_returns)
        self.assertTrue(len(features) < len(short_returns))
        
        # Test invalid feature importance request
        with self.assertRaises(ValueError):
            self.predictor.get_feature_importance('invalid_target')

if __name__ == '__main__':
    unittest.main()