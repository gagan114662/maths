"""
Test script for dynamic asset allocation.
"""

import unittest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import logging
import matplotlib.pyplot as plt
from pandas.testing import assert_frame_equal, assert_series_equal

from src.allocation.dynamic import DynamicAssetAllocator, RelativeStrengthAllocator, AllocationMethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TestDynamicAssetAllocation(unittest.TestCase):
    """Test case for dynamic asset allocation."""
    
    def setUp(self):
        """Set up test case."""
        # Create synthetic asset data
        self.assets = self._create_synthetic_data()
        
        # Create allocator
        self.allocator = DynamicAssetAllocator(
            default_method=AllocationMethod.EQUAL_WEIGHT,
            lookback_window=252,
            rebalance_frequency=21,
            volatility_window=63,
            max_allocation_pct=0.4,
            min_allocation_pct=0.05
        )
        
        # Add asset data to allocator
        for symbol, data in self.assets.items():
            self.allocator.add_asset_data(
                symbol=symbol,
                price_data=data
            )
            
        # Create relative strength allocator
        self.rs_allocator = RelativeStrengthAllocator(
            benchmark_data=self.assets['SPY'],
            lookback_periods=[21, 63, 126, 252],
            min_rs_score=0.0,
            lookback_window=252,
            rebalance_frequency=21,
            volatility_window=63,
            max_allocation_pct=0.4,
            min_allocation_pct=0.05
        )
        
        # Add asset data to relative strength allocator
        for symbol, data in self.assets.items():
            if symbol != 'SPY':  # Skip benchmark
                self.rs_allocator.add_asset_data(
                    symbol=symbol,
                    price_data=data
                )
    
    def _create_synthetic_data(self):
        """Create synthetic price data for testing."""
        # Parameters
        num_days = 1000
        start_date = datetime.now() - timedelta(days=num_days)
        
        # Base dates
        dates = pd.date_range(start=start_date, periods=num_days, freq='D')
        
        # Create assets dictionary
        assets = {}
        
        # Create random seed for reproducibility
        np.random.seed(42)
        
        # Create a market factor (base signal)
        market_returns = np.random.normal(0.0005, 0.01, num_days)  # ~12.6% annual return
        market_prices = 100 * (1 + market_returns).cumprod()
        
        # Create SPY (market benchmark)
        assets['SPY'] = pd.DataFrame({
            'close': market_prices
        }, index=dates)
        
        # Create 5 stocks with different characteristics
        for i in range(5):
            # Create returns with correlation to market
            market_correlation = 0.5 + (i * 0.1)  # Varying correlations to market
            stock_specific = np.random.normal(0.0002 * (5-i), 0.02, num_days)  # Varying stock-specific returns
            
            # Combine market and stock-specific returns
            returns = market_correlation * market_returns + (1 - market_correlation) * stock_specific
            
            # Add momentum effect for some stocks
            if i < 3:
                # Add momentum effect (stocks that go up tend to continue going up)
                momentum = np.zeros_like(returns)
                for j in range(21, num_days):
                    if sum(returns[j-21:j]) > 0:
                        momentum[j] = 0.001  # Slight positive boost
                returns += momentum
            
            # Create prices
            prices = 100 * (1 + returns).cumprod()
            
            assets[f'STOCK_{i+1}'] = pd.DataFrame({
                'close': prices
            }, index=dates)
            
        # Create 3 bonds with negative correlation to equities
        for i in range(3):
            # Create returns with negative correlation to market
            market_correlation = -0.3 - (i * 0.1)  # Varying negative correlations
            bond_specific = np.random.normal(0.0001, 0.005, num_days)  # Lower volatility
            
            # Combine market and bond-specific returns
            returns = market_correlation * market_returns + bond_specific
            
            # Create prices
            prices = 100 * (1 + returns).cumprod()
            
            assets[f'BOND_{i+1}'] = pd.DataFrame({
                'close': prices
            }, index=dates)
            
        # Create 2 alternative assets with low correlation
        for i in range(2):
            # Create returns with low correlation to market
            market_correlation = 0.1
            alt_specific = np.random.normal(0.0004, 0.015, num_days)
            
            # Create mean-reverting component
            mean_reversion = np.zeros_like(alt_specific)
            for j in range(1, num_days):
                mean_reversion[j] = -0.1 * sum(alt_specific[max(0, j-10):j])
                
            # Combine components
            returns = market_correlation * market_returns + alt_specific + mean_reversion
            
            # Create prices
            prices = 100 * (1 + returns).cumprod()
            
            assets[f'ALT_{i+1}'] = pd.DataFrame({
                'close': prices
            }, index=dates)
            
        return assets
    
    def test_equal_weight_allocation(self):
        """Test equal weight allocation."""
        # Get allocation
        weights = self.allocator.get_allocation(method=AllocationMethod.EQUAL_WEIGHT)
        
        # Check that weights add up to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        
        # Check that all assets have equal weight
        expected_weight = 1.0 / len(self.assets)
        for asset, weight in weights.items():
            self.assertAlmostEqual(weight, expected_weight, places=6)
    
    def test_inverse_volatility_allocation(self):
        """Test inverse volatility allocation."""
        # Get allocation
        weights = self.allocator.get_allocation(method=AllocationMethod.INVERSE_VOLATILITY)
        
        # Check that weights add up to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        
        # Check that bonds have higher weights than stocks (lower volatility)
        bond_weights = [weights[f'BOND_{i+1}'] for i in range(3)]
        stock_weights = [weights[f'STOCK_{i+1}'] for i in range(5)]
        
        avg_bond_weight = sum(bond_weights) / len(bond_weights)
        avg_stock_weight = sum(stock_weights) / len(stock_weights)
        
        self.assertGreater(avg_bond_weight, avg_stock_weight)
    
    def test_momentum_allocation(self):
        """Test momentum allocation."""
        # Get allocation
        weights = self.allocator.get_allocation(method=AllocationMethod.MOMENTUM_WEIGHT)
        
        # Check that weights add up to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        
        # Check that only assets with positive momentum are included
        for asset, weight in weights.items():
            momentum = self.allocator._calculate_momentum(self.assets[asset])
            if momentum <= 0:
                self.assertEqual(weight, 0.0)
    
    def test_relative_strength_allocation(self):
        """Test relative strength allocation."""
        # Get allocation
        weights = self.rs_allocator.get_allocation()
        
        # Check that weights add up to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        
        # Check that SPY is not in weights (it's the benchmark)
        self.assertNotIn('SPY', weights)
        
        # Check that only assets with positive relative strength are included
        rs_rankings = self.rs_allocator.calculate_rs_rankings()
        for asset, weight in weights.items():
            rs_score = rs_rankings.loc[asset, 'rs_score']
            if rs_score <= 0:
                self.assertEqual(weight, 0.0)
    
    def test_get_top_assets(self):
        """Test getting top assets by relative strength."""
        # Get top 3 assets
        top_assets = self.rs_allocator.get_top_assets(n=3)
        
        # Check that we get exactly 3 assets
        self.assertEqual(len(top_assets), 3)
        
        # Check that they are sorted by relative strength
        rs_rankings = self.rs_allocator.calculate_rs_rankings()
        for i in range(len(top_assets) - 1):
            self.assertGreaterEqual(
                rs_rankings.loc[top_assets[i], 'rs_score'],
                rs_rankings.loc[top_assets[i+1], 'rs_score']
            )
    
    def test_allocation_history(self):
        """Test getting allocation history."""
        # Set dates
        start_date = datetime.now() - timedelta(days=100)
        end_date = datetime.now()
        
        # Get allocation history
        allocation_df = self.allocator.get_allocation_history(
            start_date=start_date,
            end_date=end_date,
            method=AllocationMethod.EQUAL_WEIGHT
        )
        
        # Check that the index is a DatetimeIndex
        self.assertIsInstance(allocation_df.index, pd.DatetimeIndex)
        
        # Check that all dates are within range
        self.assertGreaterEqual(allocation_df.index.min(), start_date)
        self.assertLessEqual(allocation_df.index.max(), end_date)
        
        # Check that all allocations sum to 1.0
        row_sums = allocation_df.sum(axis=1)
        for date, total in row_sums.items():
            self.assertAlmostEqual(total, 1.0, places=6)
    
    def test_backtest_allocation(self):
        """Test backtesting an allocation strategy."""
        # Set dates
        start_date = datetime.now() - timedelta(days=500)
        end_date = datetime.now()
        
        # Run backtest with equal weight
        results_equal = self.allocator.backtest_allocation(
            start_date=start_date,
            end_date=end_date,
            method=AllocationMethod.EQUAL_WEIGHT,
            benchmark_data=self.assets['SPY'],
            initial_capital=10000.0
        )
        
        # Check that we have results
        self.assertIn('portfolio_values', results_equal)
        self.assertIn('total_return', results_equal)
        self.assertIn('sharpe_ratio', results_equal)
        
        # Run backtest with inverse volatility
        results_inv_vol = self.allocator.backtest_allocation(
            start_date=start_date,
            end_date=end_date,
            method=AllocationMethod.INVERSE_VOLATILITY,
            benchmark_data=self.assets['SPY'],
            initial_capital=10000.0
        )
        
        # Compare Sharpe ratios (inverse vol should be better)
        sharpe_equal = results_equal['sharpe_ratio']
        sharpe_inv_vol = results_inv_vol['sharpe_ratio']
        
        # This checks that our allocator works as expected - inverse vol
        # should have better Sharpe ratio than equal weight (because bonds
        # have lower volatility and higher weights in inverse vol)
        self.assertGreater(sharpe_inv_vol, sharpe_equal)
    
    def test_backtest_relative_strength(self):
        """Test backtesting relative strength strategy."""
        # Set dates
        start_date = datetime.now() - timedelta(days=500)
        end_date = datetime.now()
        
        # Run backtest with relative strength
        results_rs = self.rs_allocator.backtest_relative_strength(
            start_date=start_date,
            end_date=end_date,
            top_n=3,
            initial_capital=10000.0
        )
        
        # Check that we have results
        self.assertIn('portfolio_values', results_rs)
        self.assertIn('total_return', results_rs)
        self.assertIn('sharpe_ratio', results_rs)
        
        # Run backtest with equal weight
        self.rs_allocator.default_method = AllocationMethod.EQUAL_WEIGHT
        
        results_equal = self.rs_allocator.backtest_allocation(
            start_date=start_date,
            end_date=end_date,
            method=AllocationMethod.EQUAL_WEIGHT,
            benchmark_data=self.assets['SPY'],
            initial_capital=10000.0
        )
        
        # Check that relative strength allocation works differently than equal weight
        # (not necessarily better or worse, just different)
        self.assertNotEqual(
            round(results_rs['total_return'], 4),
            round(results_equal['total_return'], 4)
        )
    
    def test_plot_backtest_results(self):
        """Test plotting backtest results."""
        # Set dates
        start_date = datetime.now() - timedelta(days=500)
        end_date = datetime.now()
        
        # Run backtest
        results = self.allocator.backtest_allocation(
            start_date=start_date,
            end_date=end_date,
            method=AllocationMethod.INVERSE_VOLATILITY,
            benchmark_data=self.assets['SPY'],
            initial_capital=10000.0
        )
        
        # Plot results
        fig = self.allocator.plot_backtest_results(results)
        
        # Check that we got a figure
        self.assertIsInstance(fig, plt.Figure)
        
        # Close figure
        plt.close(fig)

if __name__ == '__main__':
    unittest.main()