#!/usr/bin/env python
"""Unit tests for regime-dependent position sizing."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.risk_management.regime_position_sizing import RegimePositionSizer, MarketRegime

class TestRegimePositionSizing(unittest.TestCase):
    def setUp(self):
        """Set up test data and position sizer."""
        np.random.seed(42)
        
        # Generate test data
        n_days = 1000
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        
        # Create different market regimes
        low_vol_returns = np.random.normal(0.0005, 0.005, n_days // 4)
        high_vol_returns = np.random.normal(0, 0.02, n_days // 4)
        trend_returns = np.cumsum(np.random.normal(0.001, 0.005, n_days // 4))
        crisis_returns = np.random.normal(-0.002, 0.03, n_days // 4)
        
        self.returns = pd.Series(
            np.concatenate([
                low_vol_returns,
                high_vol_returns,
                trend_returns,
                crisis_returns
            ]),
            index=dates
        )
        
        self.position_sizer = RegimePositionSizer(
            base_risk_fraction=0.02,
            max_position_size=0.20,
            volatility_lookback=63,
            regime_lookback=252
        )
        
        # Portfolio and price data
        self.portfolio_value = 1000000  # $1M portfolio
        self.current_price = 100
        
    def test_regime_detection(self):
        """Test market regime detection."""
        # Test low volatility regime
        low_vol_returns = pd.Series(
            np.random.normal(0.0005, 0.005, 300)
        )
        regime = self.position_sizer.detect_regime(low_vol_returns)
        self.assertEqual(regime, MarketRegime.LOW_VOL)
        
        # Test high volatility regime
        high_vol_returns = pd.Series(
            np.random.normal(0, 0.02, 300)
        )
        regime = self.position_sizer.detect_regime(high_vol_returns)
        self.assertEqual(regime, MarketRegime.HIGH_VOL)
        
        # Test crisis regime
        crisis_returns = pd.Series(
            np.random.normal(-0.02, 0.03, 300)
        )
        regime = self.position_sizer.detect_regime(crisis_returns)
        self.assertEqual(regime, MarketRegime.CRISIS)
        
    def test_position_sizing(self):
        """Test position size calculation."""
        # Calculate position sizes for different regimes
        position_details = self.position_sizer.calculate_position_size(
            self.portfolio_value,
            self.current_price,
            self.returns
        )
        
        # Check that all expected fields are present
        self.assertIn('position_size', position_details)
        self.assertIn('regime', position_details)
        self.assertIn('regime_adjustment', position_details)
        self.assertIn('volatility', position_details)
        self.assertIn('num_units', position_details)
        self.assertIn('value_at_risk', position_details)
        
        # Check position size constraints
        self.assertLessEqual(
            position_details['position_size'],
            self.portfolio_value * self.position_sizer.max_position_size
        )
        
        # Position size should be positive
        self.assertGreater(position_details['position_size'], 0)
        
    def test_regime_adjustments(self):
        """Test regime-specific position size adjustments."""
        base_size = self.position_sizer._calculate_base_position_size(
            self.portfolio_value,
            self.current_price,
            0.15  # 15% volatility
        )
        
        # Test adjustments for each regime
        for regime in MarketRegime:
            adjustment = self.position_sizer._get_regime_adjustment(regime)
            adjusted_size = base_size * adjustment
            
            # Size should be smaller in high vol and crisis regimes
            if regime in [MarketRegime.HIGH_VOL, MarketRegime.CRISIS]:
                self.assertLess(adjusted_size, base_size)
            
            # Size should be larger in low vol regime
            if regime == MarketRegime.LOW_VOL:
                self.assertGreater(adjusted_size, base_size)
                
    def test_correlation_adjustment(self):
        """Test position size adjustment for correlations."""
        # Create test correlation matrix
        assets = ['A', 'B', 'C']
        corr_matrix = pd.DataFrame(
            [[1.0, 0.5, 0.3],
             [0.5, 1.0, 0.2],
             [0.3, 0.2, 1.0]],
            index=assets,
            columns=assets
        )
        
        # Create base position sizes
        base_sizes = {
            'A': 100000,
            'B': 150000,
            'C': 200000
        }
        
        # Test correlation adjustment
        adjusted_sizes = self.position_sizer.adjust_for_correlation(
            base_sizes,
            corr_matrix,
            max_portfolio_var=0.04
        )
        
        # Check that all assets are adjusted
        self.assertEqual(set(adjusted_sizes.keys()), set(assets))
        
        # Adjusted sizes should be smaller due to positive correlations
        for asset in assets:
            self.assertLessEqual(adjusted_sizes[asset], base_sizes[asset])
            
    def test_regime_limits(self):
        """Test regime-specific position limits."""
        for regime in MarketRegime:
            limits = self.position_sizer.get_regime_limits(regime)
            
            # Check that all limit types are present
            self.assertIn('max_position_size', limits)
            self.assertIn('max_leverage', limits)
            self.assertIn('concentration_limit', limits)
            
            # Verify crisis regime has lowest limits
            if regime == MarketRegime.CRISIS:
                self.assertLess(
                    limits['max_position_size'],
                    self.position_sizer.max_position_size
                )
                
    def test_error_handling(self):
        """Test error handling for insufficient data."""
        # Test with insufficient data
        short_returns = pd.Series(
            np.random.normal(0, 0.01, 50)
        )
        
        with self.assertRaises(ValueError):
            self.position_sizer.detect_regime(short_returns)
            
    def test_var_calculation(self):
        """Test Value at Risk calculation for positions."""
        position_size = 100000
        volatility = 0.15  # 15% annualized volatility
        
        for regime in MarketRegime:
            var = self.position_sizer._calculate_position_var(
                position_size,
                volatility,
                regime
            )
            
            # VaR should be positive
            self.assertGreater(var, 0)
            
            # Crisis regime should have highest VaR
            if regime == MarketRegime.CRISIS:
                crisis_var = var
            elif regime == MarketRegime.LOW_VOL:
                low_vol_var = var
                
        # Crisis VaR should be higher than low vol VaR
        self.assertGreater(crisis_var, low_vol_var)

if __name__ == '__main__':
    unittest.main()