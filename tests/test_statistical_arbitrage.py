#!/usr/bin/env python
"""Unit tests for statistical arbitrage strategy."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.strategies.statistical_arbitrage import StatisticalArbitrageStrategy

class TestStatisticalArbitrageStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test data and strategy instance."""
        # Create synthetic price data for testing
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
        np.random.seed(42)
        
        # Create cointegrated price series
        price1 = np.random.randn(len(dates)).cumsum() + 100
        price2 = 2 * price1 + np.random.randn(len(dates)) * 0.5 + 50
        price3 = np.random.randn(len(dates)).cumsum() + 75  # Unrelated series
        
        self.price_data = pd.DataFrame({
            'ASSET1': price1,
            'ASSET2': price2,
            'ASSET3': price3
        }, index=dates)
        
        # Initialize strategy
        self.strategy = StatisticalArbitrageStrategy(
            correlation_threshold=0.7,
            zscore_entry=2.0,
            zscore_exit=0.0,
            lookback_period=60,
            minimum_history=120,
            max_pairs=5
        )
        
    def test_pair_selection(self):
        """Test pair selection logic."""
        pairs = self.strategy.select_pairs(self.price_data)
        
        # Should identify ASSET1 and ASSET2 as a pair
        self.assertEqual(len(pairs), 1)
        self.assertIn(('ASSET1', 'ASSET2'), pairs)
        
        # Verify hedge ratio is stored
        self.assertIn(('ASSET1', 'ASSET2'), self.strategy.hedge_ratios)
        self.assertGreater(self.strategy.hedge_ratios[('ASSET1', 'ASSET2')], 0)
        
    def test_spread_calculation(self):
        """Test spread calculation and normalization."""
        self.strategy.select_pairs(self.price_data)
        spreads = self.strategy.calculate_spreads(self.price_data)
        
        # Verify spread exists for selected pair
        self.assertIn(('ASSET1', 'ASSET2'), spreads)
        
        # Verify z-score properties
        zscore = spreads[('ASSET1', 'ASSET2')]
        self.assertAlmostEqual(zscore.mean(), 0, places=1)
        self.assertAlmostEqual(zscore.std(), 1, places=1)
        
    def test_signal_generation(self):
        """Test trading signal generation."""
        self.strategy.select_pairs(self.price_data)
        self.strategy.calculate_spreads(self.price_data)
        signals = self.strategy.generate_signals()
        
        # Verify signals are within expected range
        for pair, signal in signals.items():
            self.assertIn(signal, [-1, 0, 1])
            
    def test_position_sizing(self):
        """Test position size calculation."""
        portfolio_value = 1000000  # $1M portfolio
        
        self.strategy.select_pairs(self.price_data)
        self.strategy.calculate_spreads(self.price_data)
        positions = self.strategy.calculate_position_sizes(portfolio_value, self.price_data)
        
        # Verify all assets have positions
        self.assertEqual(set(positions.keys()), set(self.price_data.columns))
        
        # Verify dollar neutrality
        last_prices = self.price_data.iloc[-1]
        total_long = sum(pos * price for asset, pos in positions.items() 
                        for price in [last_prices[asset]] if pos > 0)
        total_short = sum(pos * price for asset, pos in positions.items() 
                         for price in [last_prices[asset]] if pos < 0)
        
        self.assertAlmostEqual(total_long + total_short, 0, places=2)
        
    def test_pair_metrics(self):
        """Test pair performance metrics calculation."""
        self.strategy.select_pairs(self.price_data)
        self.strategy.calculate_spreads(self.price_data)
        self.strategy.calculate_position_sizes(1000000, self.price_data)
        self.strategy.update_pair_metrics(self.price_data)
        
        # Verify metrics are calculated for each pair
        for pair in self.strategy.pairs:
            self.assertIn(pair, self.strategy.pair_metrics)
            metrics = self.strategy.pair_metrics[pair]
            
            # Verify required metrics exist
            self.assertIn('mean_reversion_half_life', metrics)
            self.assertIn('spread_volatility', metrics)
            self.assertIn('current_zscore', metrics)
            self.assertIn('days_since_signal', metrics)
            self.assertIn('current_profit_loss', metrics)

if __name__ == '__main__':
    unittest.main()