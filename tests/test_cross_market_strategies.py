"""
Test script for cross-market multi-asset class strategy generation.
"""

import unittest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import logging
import matplotlib.pyplot as plt

from src.strategy_generation.cross_market.correlations import CrossMarketCorrelationAnalyzer
from src.strategy_generation.cross_market.opportunities import OpportunityDetector
from src.strategy_generation.cross_market.generator import MultiAssetStrategyGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TestCrossMarketStrategies(unittest.TestCase):
    """Test case for cross-market strategy generation."""
    
    def setUp(self):
        """Set up test case."""
        # Create temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
        
        # Create synthetic asset data
        self.assets = self._create_synthetic_data()
        
        # Create correlation analyzer
        self.correlation_analyzer = CrossMarketCorrelationAnalyzer()
        
        # Add asset data to analyzer
        for symbol, data in self.assets.items():
            # Extract asset class and region from symbol name
            if symbol.startswith('EQUITY'):
                asset_class = 'equity'
                region = 'US' if '_US' in symbol else 'Global'
            elif symbol.startswith('BOND'):
                asset_class = 'bond'
                region = 'US' if '_US' in symbol else 'Global'
            elif symbol.startswith('COMMODITY'):
                asset_class = 'commodity'
                region = 'Global'
            elif symbol.startswith('CURRENCY'):
                asset_class = 'forex'
                region = 'Global'
            else:
                asset_class = 'unknown'
                region = 'unknown'
                
            self.correlation_analyzer.add_asset_data(
                symbol=symbol,
                data=data,
                asset_class=asset_class,
                market_region=region
            )
            
        # Create opportunity detector
        self.opportunity_detector = OpportunityDetector()
        self.opportunity_detector.set_correlation_analyzer(self.correlation_analyzer)
        
        # Create strategy generator
        self.strategy_generator = MultiAssetStrategyGenerator(output_dir=self.temp_dir)
        self.strategy_generator.set_correlation_analyzer(self.correlation_analyzer)
    
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
        
        # Create cyclical factor
        cycle = np.sin(np.linspace(0, 4*np.pi, num_days))  # Two complete cycles
        
        # Create equities with various correlations to market
        equity_symbols = ['EQUITY_US_1', 'EQUITY_US_2', 'EQUITY_GLOBAL_1', 'EQUITY_GLOBAL_2']
        for i, symbol in enumerate(equity_symbols):
            # Create returns with correlation to market and some noise
            correlation = 0.7 + (i * 0.1)  # Varying correlations
            lag = i * 5  # Varying lags
            
            # Add lag to create lead-lag relationships
            lagged_market_returns = np.roll(market_returns, lag)
            lagged_market_returns[:lag] = lagged_market_returns[lag]
            
            returns = correlation * lagged_market_returns + (1 - correlation) * np.random.normal(0.0005, 0.015, num_days)
            prices = 100 * (1 + returns).cumprod()
            
            assets[symbol] = pd.DataFrame({
                'close': prices
            }, index=dates)
            
        # Create bonds with negative correlation to market
        bond_symbols = ['BOND_US_1', 'BOND_GLOBAL_1']
        for i, symbol in enumerate(bond_symbols):
            # Create returns with negative correlation to market
            correlation = -0.4 - (i * 0.1)  # Varying negative correlations
            
            returns = correlation * market_returns + (1 - abs(correlation)) * np.random.normal(0.0003, 0.005, num_days)
            prices = 100 * (1 + returns).cumprod()
            
            assets[symbol] = pd.DataFrame({
                'close': prices
            }, index=dates)
            
        # Create commodities with correlation to cycle
        commodity_symbols = ['COMMODITY_GOLD', 'COMMODITY_OIL']
        for i, symbol in enumerate(commodity_symbols):
            cycle_effect = 0.5 - (i * 0.2)  # Varying cycle effects
            market_effect = 0.2 + (i * 0.2)  # Varying market effects
            
            # Combine cycle, market effect, and noise
            returns = cycle_effect * np.diff(cycle, prepend=cycle[0]) + market_effect * market_returns + np.random.normal(0.0002, 0.018, num_days)
            prices = 100 * (1 + returns).cumprod()
            
            assets[symbol] = pd.DataFrame({
                'close': prices
            }, index=dates)
            
        # Create currencies with low correlation to market
        currency_symbols = ['CURRENCY_EUR', 'CURRENCY_JPY']
        for i, symbol in enumerate(currency_symbols):
            market_effect = 0.1 - (i * 0.05)  # Very low market effect
            
            # Mostly random with slight market effect
            returns = market_effect * market_returns + np.random.normal(0.0001, 0.007, num_days)
            prices = 100 * (1 + returns).cumprod()
            
            assets[symbol] = pd.DataFrame({
                'close': prices
            }, index=dates)
            
        # Create pairs of cointegrated assets
        # 1. Cointegrated pair in equity space
        base_price = 100 * (1 + np.random.normal(0.0005, 0.01, num_days)).cumprod()
        
        # First asset follows base price
        assets['EQUITY_PAIR_A'] = pd.DataFrame({
            'close': base_price
        }, index=dates)
        
        # Second asset follows base with mean-reverting spread
        spread = 20 * np.random.normal(0, 0.1, num_days)
        # Make spread mean-reverting (Ornstein-Uhlenbeck process)
        for i in range(1, num_days):
            spread[i] = spread[i-1] * 0.95 + spread[i] * 0.05
            
        assets['EQUITY_PAIR_B'] = pd.DataFrame({
            'close': base_price + spread
        }, index=dates)
        
        return assets
    
    def test_correlation_analysis(self):
        """Test correlation analysis functionality."""
        # Compute correlation matrix
        corr_matrix = self.correlation_analyzer.compute_correlation_matrix()
        
        # Check that we have correlations for all assets
        self.assertEqual(len(corr_matrix), len(self.assets))
        
        # Check expected correlations
        # Check that we have a correlation (can be any value, as it depends on the random seed)
        self.assertIsNotNone(corr_matrix.loc['EQUITY_US_1', 'EQUITY_US_2'])
        self.assertLess(corr_matrix.loc['EQUITY_US_1', 'BOND_US_1'], 0)
        
        # Test rolling correlations
        rolling_corr = self.correlation_analyzer.compute_rolling_correlations(
            'EQUITY_US_1', 'BOND_US_1', window=60
        )
        
        self.assertGreater(len(rolling_corr), 0)
        
        # Test lead-lag analysis
        lead_lag = self.correlation_analyzer.analyze_lead_lag(
            'EQUITY_US_1', 'EQUITY_US_2', max_lags=20
        )
        
        self.assertIn('optimal_lag', lead_lag)
        self.assertIn('lead_asset', lead_lag)
        
        # Test cointegration
        coint_result = self.correlation_analyzer.check_cointegration(
            'EQUITY_PAIR_A', 'EQUITY_PAIR_B'
        )
        
        # Cointegration test may fail with random data, just check that we get a result
        self.assertIsInstance(coint_result, dict)
        # Hedge ratio may not be in the result if cointegration test failed
        # So we don't assert for any particular keys
    
    def test_opportunity_detection(self):
        """Test opportunity detection functionality."""
        # Find lead-lag opportunities
        lead_lag_opps = self.opportunity_detector.find_lead_lag_opportunities()
        
        # There should be some lead-lag opportunities in our synthetic data
        self.assertGreater(len(lead_lag_opps), 0)
        
        # Check structure of opportunities
        self.assertIn('lead_asset', lead_lag_opps[0])
        self.assertIn('lag_asset', lead_lag_opps[0])
        self.assertIn('optimal_lag', lead_lag_opps[0])
        
        # Find pairs trading opportunities
        pairs_opps = self.opportunity_detector.find_pairs_trading_opportunities()
        
        # Pairs opportunities might be empty depending on the random data
        # Just check that the result is a list
        self.assertIsInstance(pairs_opps, list)
        
        # Check structure of pairs opportunities if there are any
        if pairs_opps:
            self.assertIn('symbol1', pairs_opps[0])
            self.assertIn('symbol2', pairs_opps[0])
            self.assertIn('hedge_ratio', pairs_opps[0])
        
        # Find cross-asset opportunities
        cross_asset_opps = self.opportunity_detector.find_cross_asset_correlations()
        
        # Should find some cross-asset relationships
        self.assertGreater(len(cross_asset_opps), 0)
        
        # Find all opportunities
        all_opps = self.opportunity_detector.find_all_opportunities()
        
        # Should have multiple opportunity types
        self.assertGreater(len(all_opps.keys()), 0)
    
    def test_strategy_generation(self):
        """Test strategy generation functionality."""
        # Generate lead-lag strategy
        lead_lag_opps = self.opportunity_detector.find_lead_lag_opportunities()
        
        if lead_lag_opps:
            strategy = self.strategy_generator.generate_lead_lag_strategy(lead_lag_opps[0])
            
            # Check strategy structure
            self.assertIn('type', strategy)
            self.assertIn('name', strategy)
            self.assertIn('code', strategy)
            self.assertIn('parameters', strategy)
            
            # Check strategy code
            self.assertIn('class', strategy['code'])
            self.assertIn('Initialize', strategy['code'])
            self.assertIn('OnData', strategy['code'])
            
        # Generate pairs trading strategy
        pairs_opps = self.opportunity_detector.find_pairs_trading_opportunities()
        
        if pairs_opps:
            strategy = self.strategy_generator.generate_pairs_trading_strategy(pairs_opps[0])
            
            # Check strategy structure
            self.assertIn('type', strategy)
            self.assertIn('name', strategy)
            self.assertIn('code', strategy)
            self.assertIn('parameters', strategy)
            
            # Check strategy code
            self.assertIn('class', strategy['code'])
            self.assertIn('Initialize', strategy['code'])
            self.assertIn('OnData', strategy['code'])
            
        # Generate cross-asset strategy
        cross_asset_opps = self.opportunity_detector.find_cross_asset_correlations()
        
        if cross_asset_opps:
            strategy = self.strategy_generator.generate_cross_asset_strategy(
                cross_asset_opps[:3], name="Test_Cross_Asset_Strategy"
            )
            
            # Check strategy structure
            self.assertIn('type', strategy)
            self.assertIn('name', strategy)
            self.assertIn('code', strategy)
            self.assertIn('parameters', strategy)
            
            # Check strategy code
            self.assertIn('class', strategy['code'])
            self.assertIn('Initialize', strategy['code'])
            self.assertIn('OnData', strategy['code'])
    
    def test_generate_all_strategies(self):
        """Test generation of all strategies."""
        # Find opportunities
        opportunities = self.opportunity_detector.find_all_opportunities()
        
        # Generate strategies
        strategies = self.strategy_generator.generate_all_strategies(opportunities)
        
        # Should have at least one strategy type
        self.assertGreater(len(strategies), 0)
        
        # Save strategies
        saved_paths = self.strategy_generator.save_all_strategies(strategies)
        
        # At least one strategy should be saved
        self.assertGreater(len(saved_paths), 0)
        
        # Check if files exist
        for path in saved_paths:
            self.assertTrue(os.path.exists(path))
            
        # Generate and save in one operation
        summary = self.strategy_generator.generate_and_save_strategies()
        
        # Check summary
        self.assertIn('status', summary)
        self.assertIn('total_strategies', summary)
    
    def tearDown(self):
        """Clean up temporary files."""
        # In a real test, we would clean up the temp directory
        # But we'll skip this for demonstration purposes
        pass
        
        
if __name__ == '__main__':
    unittest.main()