#!/usr/bin/env python
"""Unit tests for ensemble learning."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.optimization.ensemble_learner import EnsembleLearner

class TestEnsembleLearner(unittest.TestCase):
    def setUp(self):
        """Set up test data and ensemble learner."""
        # Create test data
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        prices = 100 * (1 + np.random.normal(0, 0.02, 1000)).cumprod()
        
        self.market_data = pd.DataFrame({
            'close': prices,
            'returns': pd.Series(prices).pct_change()
        }, index=dates)
        
        # Create synthetic strategy predictions
        self.strategy_predictions = {
            'trend_following': self._generate_trend_strategy(self.market_data),
            'mean_reversion': self._generate_mean_reversion_strategy(self.market_data),
            'momentum': self._generate_momentum_strategy(self.market_data)
        }
        
        # Create target returns
        self.target_returns = self.market_data['returns']
        
        # Define base strategies
        self.base_strategies = {
            'trend_following': lambda x: x['trend'],
            'mean_reversion': lambda x: x['mean_rev'],
            'momentum': lambda x: x['momentum']
        }
        
        # Initialize ensemble learner
        self.learner = EnsembleLearner(
            base_strategies=self.base_strategies,
            ensemble_method='weighted',
            meta_model='gbm',
            lookback_window=252,
            rebalance_frequency=21
        )
        
    def _generate_trend_strategy(self, data: pd.DataFrame) -> pd.Series:
        """Generate trend following strategy signals."""
        returns = data['returns']
        ma_fast = returns.rolling(20).mean()
        ma_slow = returns.rolling(50).mean()
        return pd.Series(np.where(ma_fast > ma_slow, 1, -1), index=data.index)
        
    def _generate_mean_reversion_strategy(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion strategy signals."""
        returns = data['returns']
        zscore = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
        return pd.Series(np.where(zscore < -1, 1, np.where(zscore > 1, -1, 0)), index=data.index)
        
    def _generate_momentum_strategy(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum strategy signals."""
        returns = data['returns']
        momentum = returns.rolling(63).mean()
        return pd.Series(np.where(momentum > 0, 1, -1), index=data.index)
        
    def test_initialization(self):
        """Test ensemble learner initialization."""
        self.assertEqual(self.learner.ensemble_method, 'weighted')
        self.assertEqual(self.learner.lookback_window, 252)
        self.assertEqual(self.learner.rebalance_frequency, 21)
        self.assertIsNotNone(self.learner.meta_model)
        
    def test_weighted_ensemble(self):
        """Test weighted ensemble method."""
        # Initialize weighted ensemble
        learner = EnsembleLearner(
            self.base_strategies,
            ensemble_method='weighted'
        )
        
        # Fit ensemble
        learner.fit(
            self.market_data,
            self.strategy_predictions,
            self.target_returns
        )
        
        # Check weights
        weights = learner.get_strategy_weights()
        self.assertIsNotNone(weights)
        self.assertEqual(len(weights), len(self.base_strategies))
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        
        # Generate predictions
        predictions = learner.predict(
            self.market_data,
            self.strategy_predictions
        )
        
        self.assertEqual(len(predictions), len(self.market_data))
        self.assertTrue(all(abs(predictions) <= 1.0))
        
    def test_stacking_ensemble(self):
        """Test stacking ensemble method."""
        # Initialize stacking ensemble
        learner = EnsembleLearner(
            self.base_strategies,
            ensemble_method='stacking'
        )
        
        # Fit ensemble
        learner.fit(
            self.market_data,
            self.strategy_predictions,
            self.target_returns
        )
        
        # Check feature importance
        importance = learner.get_feature_importance()
        if importance is not None:
            self.assertGreater(len(importance), 0)
            
        # Generate predictions
        predictions = learner.predict(
            self.market_data,
            self.strategy_predictions
        )
        
        self.assertEqual(len(predictions), len(self.market_data))
        
    def test_dynamic_ensemble(self):
        """Test dynamic ensemble method."""
        # Initialize dynamic ensemble
        learner = EnsembleLearner(
            self.base_strategies,
            ensemble_method='dynamic'
        )
        
        # Fit ensemble
        learner.fit(
            self.market_data,
            self.strategy_predictions,
            self.target_returns
        )
        
        # Generate predictions
        predictions = learner.predict(
            self.market_data,
            self.strategy_predictions
        )
        
        self.assertEqual(len(predictions), len(self.market_data))
        
    def test_meta_features(self):
        """Test meta-feature preparation."""
        features = self.learner._prepare_meta_features(
            self.market_data,
            self.strategy_predictions
        )
        
        # Check feature structure
        self.assertGreater(len(features.columns), len(self.base_strategies))
        self.assertEqual(len(features), len(self.market_data))
        
        # Check feature values
        self.assertTrue(features['volatility'].notna().all())
        self.assertTrue(features['momentum'].notna().all())
        self.assertTrue(features['trend'].notna().all())
        
    def test_regime_detection(self):
        """Test market regime detection."""
        regimes = self.learner._detect_market_regimes(self.market_data)
        
        # Check regime classification
        self.assertEqual(len(regimes), len(self.market_data))
        self.assertTrue(all(r in ['low_vol_uptrend', 'low_vol_downtrend',
                                'medium_vol', 'high_vol'] for r in regimes))
        
    def test_strategy_agreement(self):
        """Test strategy agreement calculation."""
        agreement = self.learner._calculate_strategy_agreement(
            self.strategy_predictions
        )
        
        self.assertEqual(len(agreement), len(self.market_data))
        self.assertTrue(all(agreement >= 0))  # Standard deviation is non-negative
        
    def test_error_handling(self):
        """Test error handling."""
        # Test with insufficient data
        short_data = self.market_data.head(10)
        short_predictions = {
            name: preds.head(10)
            for name, preds in self.strategy_predictions.items()
        }
        
        with self.assertRaises(ValueError):
            self.learner.fit(
                short_data,
                short_predictions,
                self.target_returns.head(10)
            )
            
    def test_performance_comparison(self):
        """Test performance comparison plotting."""
        # Add some performance data
        self.learner.strategy_performance = {
            'trend_following': self.target_returns * self.strategy_predictions['trend_following'],
            'mean_reversion': self.target_returns * self.strategy_predictions['mean_reversion'],
            'ensemble': self.target_returns
        }
        
        # Test plotting without saving
        self.learner.plot_strategy_performance()
        
        # Test plotting with saving
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            self.learner.plot_strategy_performance(save_path=tmp.name)

if __name__ == '__main__':
    unittest.main()