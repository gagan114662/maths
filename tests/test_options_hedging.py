#!/usr/bin/env python
"""Unit tests for options-based hedging strategies."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.risk_management.options_hedging import (
    OptionsHedgeStrategy,
    calculate_hedge_effectiveness
)

class TestOptionsHedging(unittest.TestCase):
    def setUp(self):
        """Set up test data and strategy instance."""
        np.random.seed(42)
        
        # Create test market data
        dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
        self.market_data = pd.DataFrame({
            'SPY': 100 + np.random.normal(0, 1, 252).cumsum(),
            'QQQ': 200 + np.random.normal(0, 1.5, 252).cumsum(),
        }, index=dates)
        
        # Create test options data
        self.options_data = {
            'SPY': self._create_test_options_chain('SPY'),
            'QQQ': self._create_test_options_chain('QQQ')
        }
        
        # Initialize strategy
        self.strategy = OptionsHedgeStrategy(
            max_hedge_cost=0.02,
            min_protection_level=0.95,
            rebalance_threshold=0.1,
            vol_window=63
        )
        
        # Test portfolio
        self.portfolio_value = 1000000  # $1M portfolio
        self.current_positions = {
            'SPY': 1000,  # 1000 shares of SPY
            'QQQ': 500    # 500 shares of QQQ
        }
        
        # Risk metrics
        self.risk_metrics = {
            'portfolio_beta': 1.2,
            'var_99': -0.03,  # 3% daily VaR
            'correlation_with_spy': 0.8
        }
        
    def _create_test_options_chain(self, underlying: str) -> pd.DataFrame:
        """Create synthetic options chain for testing."""
        current_price = self.market_data[underlying].iloc[-1]
        strikes = np.linspace(current_price * 0.8, current_price * 1.2, 10)
        expiries = [datetime.now() + timedelta(days=d) for d in [30, 60, 90]]
        
        options = []
        for strike in strikes:
            for expiry in expiries:
                # Add put option
                options.append({
                    'type': 'put',
                    'strike': strike,
                    'expiration': expiry,
                    'price': self._calculate_synthetic_option_price(
                        current_price, strike, expiry, 'put'
                    ),
                    'delta': self._calculate_synthetic_delta(
                        current_price, strike, expiry, 'put'
                    )
                })
                
                # Add call option
                options.append({
                    'type': 'call',
                    'strike': strike,
                    'expiration': expiry,
                    'price': self._calculate_synthetic_option_price(
                        current_price, strike, expiry, 'call'
                    ),
                    'delta': self._calculate_synthetic_delta(
                        current_price, strike, expiry, 'call'
                    )
                })
                
        return pd.DataFrame(options)
    
    def _calculate_synthetic_option_price(self, 
                                       spot: float,
                                       strike: float,
                                       expiry: datetime,
                                       option_type: str) -> float:
        """Calculate synthetic option price for testing."""
        dte = (expiry - datetime.now()).days / 365
        vol = 0.2  # Assumed volatility
        
        if option_type == 'call':
            return max(0.01, spot - strike + np.sqrt(dte) * vol * spot)
        else:
            return max(0.01, strike - spot + np.sqrt(dte) * vol * spot)
            
    def _calculate_synthetic_delta(self,
                                spot: float,
                                strike: float,
                                expiry: datetime,
                                option_type: str) -> float:
        """Calculate synthetic option delta for testing."""
        if option_type == 'call':
            return 0.5 + 0.5 * np.tanh((spot - strike) / (spot * 0.1))
        else:
            return -0.5 - 0.5 * np.tanh((spot - strike) / (spot * 0.1))
    
    def test_hedge_strategy_design(self):
        """Test hedge strategy design."""
        strategy = self.strategy.design_hedge_strategy(
            self.portfolio_value,
            self.current_positions,
            self.market_data,
            self.options_data,
            self.risk_metrics
        )
        
        # Check strategy structure
        self.assertIn('type', strategy)
        self.assertIn('positions', strategy)
        self.assertIn('annual_cost', strategy)
        self.assertIn('protection_level', strategy)
        
        # Check constraints
        self.assertLessEqual(
            strategy['annual_cost'],
            self.portfolio_value * self.strategy.max_hedge_cost
        )
        self.assertGreaterEqual(
            strategy['protection_level'],
            self.strategy.min_protection_level
        )
        
        # Check positions
        self.assertTrue(len(strategy['positions']) > 0)
        for position in strategy['positions']:
            self.assertIn('type', position)
            self.assertIn('underlying', position)
            self.assertIn('strike', position)
            self.assertIn('expiration', position)
            self.assertIn('contracts', position)
            
    def test_dynamic_hedge_adjustments(self):
        """Test dynamic hedge adjustment calculations."""
        # Create some test hedge positions
        current_hedges = {
            'SPY': {
                'type': 'put',
                'underlying': 'SPY',
                'strike': 100,
                'expiration': datetime.now() + timedelta(days=30),
                'contracts': 10,
                'target_delta': -0.5
            }
        }
        
        adjustments = self.strategy.calculate_dynamic_hedge_adjustments(
            current_hedges,
            self.market_data,
            self.options_data
        )
        
        # Check adjustments structure
        for instrument, adjustment in adjustments.items():
            self.assertIn('current_delta', adjustment)
            self.assertIn('target_delta', adjustment)
            self.assertIn('required_adjustment', adjustment)
            self.assertIn('suggested_trades', adjustment)
            
    def test_hedge_effectiveness(self):
        """Test hedge effectiveness calculation."""
        # Create test hedge positions
        hedge_positions = {
            'SPY_put': {
                'type': 'put',
                'underlying': 'SPY',
                'strike': 100,
                'expiration': datetime.now() + timedelta(days=30),
                'contracts': 10,
                'cost': 1000
            }
        }
        
        effectiveness = calculate_hedge_effectiveness(
            hedge_positions,
            self.market_data
        )
        
        # Check effectiveness metrics
        self.assertIn('beta_reduction', effectiveness)
        self.assertIn('volatility_reduction', effectiveness)
        self.assertIn('tail_risk_reduction', effectiveness)
        self.assertIn('hedge_cost', effectiveness)
        
        # Check metric ranges
        self.assertGreaterEqual(effectiveness['beta_reduction'], 0)
        self.assertGreaterEqual(effectiveness['volatility_reduction'], 0)
        self.assertGreaterEqual(effectiveness['tail_risk_reduction'], 0)
        self.assertGreaterEqual(effectiveness['hedge_cost'], 0)
        
    def test_collar_strategy(self):
        """Test collar strategy design."""
        strategy = self.strategy.design_hedge_strategy(
            self.portfolio_value,
            self.current_positions,
            self.market_data,
            self.options_data,
            self.risk_metrics
        )
        
        if strategy['type'] == 'collar':
            # Check collar-specific properties
            self.assertIn('upside_cap', strategy)
            
            # Verify put and call positions
            put_positions = [p for p in strategy['positions'] if p['type'] == 'put']
            call_positions = [p for p in strategy['positions'] if p['type'] == 'call']
            
            self.assertTrue(len(put_positions) > 0)
            self.assertTrue(len(call_positions) > 0)
            
            # Verify cost reduction
            self.assertLessEqual(
                strategy['annual_cost'],
                self.portfolio_value * self.strategy.max_hedge_cost
            )
            
    def test_error_handling(self):
        """Test error handling."""
        # Test with invalid portfolio value
        with self.assertRaises(ValueError):
            self.strategy.design_hedge_strategy(
                -1000000,
                self.current_positions,
                self.market_data,
                self.options_data,
                self.risk_metrics
            )
            
        # Test with empty options data
        with self.assertRaises(ValueError):
            self.strategy.design_hedge_strategy(
                self.portfolio_value,
                self.current_positions,
                self.market_data,
                {},
                self.risk_metrics
            )

if __name__ == '__main__':
    unittest.main()