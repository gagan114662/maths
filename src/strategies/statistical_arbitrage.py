#!/usr/bin/env python
"""
Statistical Arbitrage Strategy Module

This module implements market-neutral statistical arbitrage strategies with:
- Pair selection using correlation and cointegration analysis
- Spread calculation and signal generation
- Market-neutral position sizing
- Risk management and monitoring
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional

class StatisticalArbitrageStrategy:
    """
    A market-neutral statistical arbitrage strategy implementation.
    """
    
    def __init__(self, 
                 correlation_threshold: float = 0.7,
                 zscore_entry: float = 2.0,
                 zscore_exit: float = 0.0,
                 lookback_period: int = 252,
                 minimum_history: int = 504,
                 max_pairs: int = 20):
        """
        Initialize the statistical arbitrage strategy.

        Args:
            correlation_threshold: Minimum correlation for pair selection
            zscore_entry: Z-score threshold for trade entry
            zscore_exit: Z-score threshold for trade exit
            lookback_period: Period for calculating statistics (trading days)
            minimum_history: Minimum history required for pair selection
            max_pairs: Maximum number of pairs to trade simultaneously
        """
        self.correlation_threshold = correlation_threshold
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.lookback_period = lookback_period
        self.minimum_history = minimum_history
        self.max_pairs = max_pairs
        
        # Runtime state
        self.pairs = []  # Selected pairs for trading
        self.hedge_ratios = {}  # Hedge ratios for each pair
        self.spreads = {}  # Current spreads
        self.positions = {}  # Current positions
        self.pair_metrics = {}  # Performance metrics per pair
        
    def select_pairs(self, price_data: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Select tradeable pairs using correlation and cointegration analysis.
        
        Args:
            price_data: DataFrame of asset prices (columns are assets)
            
        Returns:
            List of selected pairs (tuples of asset names)
        """
        if len(price_data) < self.minimum_history:
            raise ValueError(f"Insufficient history. Need at least {self.minimum_history} data points.")
            
        assets = price_data.columns
        n_assets = len(assets)
        
        # Calculate correlation matrix
        correlations = price_data.corr()
        
        # Find potential pairs based on correlation
        potential_pairs = []
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                if abs(correlations.iloc[i,j]) >= self.correlation_threshold:
                    potential_pairs.append((assets[i], assets[j]))
                    
        # Test pairs for cointegration
        cointegrated_pairs = []
        for asset1, asset2 in potential_pairs:
            # Perform Engle-Granger cointegration test
            score, pvalue, _ = coint(price_data[asset1], price_data[asset2])
            
            if pvalue < 0.05:  # Statistically significant cointegration
                # Calculate hedge ratio using OLS
                hedge_ratio = np.polyfit(price_data[asset1], price_data[asset2], 1)[0]
                
                cointegrated_pairs.append({
                    'assets': (asset1, asset2),
                    'pvalue': pvalue,
                    'hedge_ratio': hedge_ratio,
                    'correlation': correlations.loc[asset1, asset2]
                })
                
        # Sort pairs by strength of cointegration (p-value)
        cointegrated_pairs.sort(key=lambda x: x['pvalue'])
        
        # Select top pairs up to max_pairs
        selected_pairs = [pair['assets'] for pair in cointegrated_pairs[:self.max_pairs]]
        
        # Store hedge ratios for selected pairs
        for pair in cointegrated_pairs[:self.max_pairs]:
            self.hedge_ratios[pair['assets']] = pair['hedge_ratio']
            
        self.pairs = selected_pairs
        return selected_pairs
    
    def calculate_spreads(self, price_data: pd.DataFrame) -> Dict[Tuple[str, str], pd.Series]:
        """
        Calculate normalized spreads for all pairs.
        
        Args:
            price_data: DataFrame of asset prices
            
        Returns:
            Dictionary of normalized spreads for each pair
        """
        spreads = {}
        
        for pair in self.pairs:
            asset1, asset2 = pair
            hedge_ratio = self.hedge_ratios[pair]
            
            # Calculate spread
            spread = price_data[asset2] - hedge_ratio * price_data[asset1]
            
            # Normalize spread
            lookback_spread = spread.rolling(window=self.lookback_period)
            zscore = (spread - lookback_spread.mean()) / lookback_spread.std()
            
            spreads[pair] = zscore
            
        self.spreads = spreads
        return spreads
    
    def generate_signals(self) -> Dict[Tuple[str, str], int]:
        """
        Generate trading signals based on spread z-scores.
        
        Returns:
            Dictionary of trading signals for each pair (-1: short, 0: neutral, 1: long)
        """
        signals = {}
        
        for pair in self.pairs:
            zscore = self.spreads[pair].iloc[-1]  # Most recent z-score
            
            # Generate signal based on z-score thresholds
            if abs(zscore) <= self.zscore_exit:  # Exit positions
                signals[pair] = 0
            elif zscore >= self.zscore_entry:  # Short the spread
                signals[pair] = -1
            elif zscore <= -self.zscore_entry:  # Long the spread
                signals[pair] = 1
            else:  # Maintain current position
                signals[pair] = self.positions.get(pair, 0)
                
        return signals
    
    def calculate_position_sizes(self, portfolio_value: float, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate position sizes maintaining market neutrality.
        
        Args:
            portfolio_value: Current portfolio value
            price_data: DataFrame of asset prices
            
        Returns:
            Dictionary of position sizes for each asset
        """
        signals = self.generate_signals()
        n_active_pairs = sum(1 for signal in signals.values() if signal != 0)
        
        if n_active_pairs == 0:
            return {asset: 0.0 for asset in price_data.columns}
            
        # Allocate equal value to each active pair
        value_per_pair = portfolio_value / (n_active_pairs * 2)  # *2 because each pair needs long and short
        
        positions = {}
        for pair in self.pairs:
            asset1, asset2 = pair
            signal = signals[pair]
            
            if signal == 0:
                positions[asset1] = positions.get(asset1, 0)
                positions[asset2] = positions.get(asset2, 0)
                continue
                
            price1 = price_data[asset1].iloc[-1]
            price2 = price_data[asset2].iloc[-1]
            hedge_ratio = self.hedge_ratios[pair]
            
            # Calculate number of units ensuring dollar-neutral positions
            units1 = value_per_pair / price1
            units2 = value_per_pair / price2
            
            # Adjust units2 by hedge ratio
            units2 = units2 * hedge_ratio
            
            # Apply signal direction
            if signal > 0:  # Long spread
                positions[asset1] = positions.get(asset1, 0) + units1
                positions[asset2] = positions.get(asset2, 0) - units2
            else:  # Short spread
                positions[asset1] = positions.get(asset1, 0) - units1
                positions[asset2] = positions.get(asset2, 0) + units2
                
        self.positions = {pair: np.sign(positions[pair[1]] - positions[pair[0]]) 
                         for pair in self.pairs}
        
        return positions
    
    def update_pair_metrics(self, price_data: pd.DataFrame) -> None:
        """
        Update performance metrics for each pair.
        
        Args:
            price_data: DataFrame of asset prices
        """
        for pair in self.pairs:
            asset1, asset2 = pair
            spread = self.spreads[pair]
            
            metrics = {
                'mean_reversion_half_life': self._calculate_half_life(spread),
                'spread_volatility': spread.std(),
                'current_zscore': spread.iloc[-1],
                'days_since_signal': self._calculate_days_since_signal(pair),
                'current_profit_loss': self._calculate_pair_pnl(
                    pair, price_data[asset1], price_data[asset2]
                )
            }
            
            self.pair_metrics[pair] = metrics
            
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate mean reversion half-life using OLS regression."""
        lagged_spread = spread.shift(1)
        delta = spread - lagged_spread
        lag_matrix = sm.add_constant(lagged_spread.dropna())
        model = sm.OLS(delta.dropna(), lag_matrix)
        results = model.fit()
        half_life = -np.log(2) / results.params[1]
        return half_life if half_life > 0 else np.inf
    
    def _calculate_days_since_signal(self, pair: Tuple[str, str]) -> int:
        """Calculate number of days since last trading signal."""
        current_position = self.positions.get(pair, 0)
        if current_position == 0:
            return 0
            
        # Count days since z-score crossed signal threshold
        zscore = self.spreads[pair]
        if current_position > 0:  # Long spread
            signal_days = (zscore <= -self.zscore_entry).astype(int)
        else:  # Short spread
            signal_days = (zscore >= self.zscore_entry).astype(int)
            
        last_signal = signal_days.values.nonzero()[0][-1]
        return len(signal_days) - last_signal
    
    def _calculate_pair_pnl(self, pair: Tuple[str, str], 
                          price1: pd.Series, price2: pd.Series) -> float:
        """Calculate current P&L for a pair position."""
        position = self.positions.get(pair, 0)
        if position == 0:
            return 0.0
            
        hedge_ratio = self.hedge_ratios[pair]
        price_change1 = price1.pct_change().fillna(0)
        price_change2 = price2.pct_change().fillna(0)
        
        # P&L is based on spread changes
        if position > 0:  # Long spread
            pnl = price_change1 - hedge_ratio * price_change2
        else:  # Short spread
            pnl = hedge_ratio * price_change2 - price_change1
            
        return pnl.sum()