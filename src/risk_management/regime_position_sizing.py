#!/usr/bin/env python
"""
Regime-Dependent Position Sizing Module

This module implements dynamic position sizing algorithms that adapt to different
market regimes, volatility environments, and risk scenarios.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

class MarketRegime(Enum):
    """Market regime classifications."""
    LOW_VOL = "low_volatility"
    HIGH_VOL = "high_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    CRISIS = "crisis"

class RegimePositionSizer:
    """
    Implements position sizing strategies that adapt to market regimes.
    """
    
    def __init__(self, 
                base_risk_fraction: float = 0.02,
                max_position_size: float = 0.20,
                volatility_lookback: int = 63,
                regime_lookback: int = 252):
        """
        Initialize the regime-dependent position sizer.

        Args:
            base_risk_fraction: Base fraction of portfolio to risk per trade
            max_position_size: Maximum position size as fraction of portfolio
            volatility_lookback: Lookback period for volatility calculation
            regime_lookback: Lookback period for regime detection
        """
        self.base_risk_fraction = base_risk_fraction
        self.max_position_size = max_position_size
        self.volatility_lookback = volatility_lookback
        self.regime_lookback = regime_lookback
        self.logger = logging.getLogger(__name__)
        
    def detect_regime(self, 
                     returns: pd.Series,
                     prices: Optional[pd.Series] = None) -> MarketRegime:
        """
        Detect current market regime using returns and price data.
        
        Args:
            returns: Series of asset returns
            prices: Optional series of asset prices
            
        Returns:
            Current market regime classification
        """
        if len(returns) < self.regime_lookback:
            raise ValueError(f"Need at least {self.regime_lookback} observations")
        
        recent_returns = returns.tail(self.regime_lookback)
        recent_vol = returns.tail(self.volatility_lookback).std() * np.sqrt(252)
        
        # Calculate regime indicators
        hurst_exponent = self._calculate_hurst_exponent(returns)
        volatility_ratio = recent_vol / returns.std() * np.sqrt(252)
        
        # Detect crisis regime first (takes precedence)
        if self._is_crisis_regime(returns):
            return MarketRegime.CRISIS
        
        # Classify based on volatility and trend characteristics
        if volatility_ratio > 1.5:
            return MarketRegime.HIGH_VOL
        elif volatility_ratio < 0.75:
            return MarketRegime.LOW_VOL
        elif hurst_exponent > 0.6:  # Strong trend
            return MarketRegime.TRENDING
        elif hurst_exponent < 0.4:  # Mean reversion
            return MarketRegime.MEAN_REVERTING
        else:
            return MarketRegime.LOW_VOL  # Default to low vol regime
        
    def calculate_position_size(self,
                              portfolio_value: float,
                              current_price: float,
                              returns: pd.Series,
                              regime: Optional[MarketRegime] = None,
                              volatility: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate regime-appropriate position size.
        
        Args:
            portfolio_value: Current portfolio value
            current_price: Current asset price
            returns: Historical returns for regime/volatility calculation
            regime: Optional pre-calculated regime
            volatility: Optional pre-calculated volatility
            
        Returns:
            Dictionary with position size details
        """
        # Detect regime if not provided
        if regime is None:
            regime = self.detect_regime(returns)
            
        # Calculate volatility if not provided
        if volatility is None:
            volatility = returns.tail(self.volatility_lookback).std() * np.sqrt(252)
            
        # Get base position size
        base_size = self._calculate_base_position_size(
            portfolio_value, current_price, volatility
        )
        
        # Apply regime-specific adjustments
        regime_adjustment = self._get_regime_adjustment(regime)
        position_size = base_size * regime_adjustment
        
        # Apply constraints
        position_size = min(
            position_size,
            portfolio_value * self.max_position_size
        )
        
        return {
            'position_size': position_size,
            'regime': regime.value,
            'regime_adjustment': regime_adjustment,
            'volatility': volatility,
            'num_units': int(position_size / current_price),
            'value_at_risk': self._calculate_position_var(
                position_size, volatility, regime
            )
        }
    
    def _calculate_base_position_size(self,
                                    portfolio_value: float,
                                    current_price: float,
                                    volatility: float) -> float:
        """Calculate base position size using volatility-adjusted risk."""
        # Use Kelly Criterion with safety fraction
        kelly_fraction = self.base_risk_fraction / volatility
        safe_fraction = kelly_fraction * 0.5  # Half-Kelly for safety
        
        return portfolio_value * safe_fraction
    
    def _get_regime_adjustment(self, regime: MarketRegime) -> float:
        """Get position size adjustment factor for given regime."""
        regime_adjustments = {
            MarketRegime.LOW_VOL: 1.2,      # Increase size in low vol
            MarketRegime.HIGH_VOL: 0.6,     # Reduce size in high vol
            MarketRegime.TRENDING: 1.0,     # Normal size in trends
            MarketRegime.MEAN_REVERTING: 0.8, # Slightly reduce in mean reversion
            MarketRegime.CRISIS: 0.3        # Significantly reduce in crisis
        }
        
        return regime_adjustments.get(regime, 1.0)
    
    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent to detect trends vs mean reversion."""
        # Convert returns to price series
        prices = (1 + returns).cumprod()
        
        # Calculate Hurst exponent using R/S analysis
        lags = range(2, min(len(prices) // 2, 20))
        rs_values = []
        
        for lag in lags:
            # Split into segments
            segments = len(prices) // lag
            rs = []
            
            for i in range(segments):
                segment = prices[i*lag:(i+1)*lag]
                cumdev = (segment - segment.mean()).cumsum()
                r = max(cumdev) - min(cumdev)  # Range
                s = segment.std()              # Standard deviation
                if s > 0:
                    rs.append(r/s)
                    
            rs_values.append(np.mean(rs))
            
        if len(rs_values) > 1:
            hurst = np.polyfit(np.log(lags), np.log(rs_values), 1)[0]
            return min(max(hurst, 0), 1)  # Bound between 0 and 1
        else:
            return 0.5  # Default to random walk
    
    def _is_crisis_regime(self, returns: pd.Series) -> bool:
        """Detect crisis regime using various indicators."""
        recent_returns = returns.tail(self.volatility_lookback)
        
        # Check for extreme volatility
        vol = recent_returns.std() * np.sqrt(252)
        historical_vol = returns.std() * np.sqrt(252)
        
        # Check for extreme drawdown
        cum_returns = (1 + recent_returns).cumprod()
        drawdown = (cum_returns / cum_returns.expanding().max() - 1).min()
        
        # Check for extreme negative skewness
        skew = stats.skew(recent_returns)
        
        # Combined crisis indicators
        is_crisis = (
            (vol > 2.5 * historical_vol) or  # Volatility spike
            (drawdown < -0.15) or            # Large drawdown
            (skew < -2.0)                    # Extreme negative skew
        )
        
        return is_crisis
    
    def _calculate_position_var(self,
                              position_size: float,
                              volatility: float,
                              regime: MarketRegime,
                              confidence_level: float = 0.99) -> float:
        """Calculate Value at Risk for the position."""
        # Adjust VaR calculation based on regime
        regime_var_multipliers = {
            MarketRegime.LOW_VOL: 1.0,
            MarketRegime.HIGH_VOL: 1.5,
            MarketRegime.TRENDING: 1.2,
            MarketRegime.MEAN_REVERTING: 1.3,
            MarketRegime.CRISIS: 2.0
        }
        
        multiplier = regime_var_multipliers.get(regime, 1.0)
        z_score = stats.norm.ppf(confidence_level)
        
        return position_size * volatility * z_score * multiplier / np.sqrt(252)
    
    def adjust_for_correlation(self,
                             base_sizes: Dict[str, float],
                             correlation_matrix: pd.DataFrame,
                             max_portfolio_var: float) -> Dict[str, float]:
        """
        Adjust position sizes accounting for correlations.
        
        Args:
            base_sizes: Dictionary of base position sizes per asset
            correlation_matrix: Correlation matrix of assets
            max_portfolio_var: Maximum acceptable portfolio variance
            
        Returns:
            Dictionary of adjusted position sizes
        """
        assets = list(base_sizes.keys())
        n_assets = len(assets)
        
        if n_assets < 2:
            return base_sizes
            
        # Create position size vector
        pos_sizes = np.array([base_sizes[asset] for asset in assets])
        
        # Calculate portfolio variance
        portfolio_var = pos_sizes.T @ correlation_matrix.loc[assets, assets] @ pos_sizes
        
        if portfolio_var > max_portfolio_var:
            # Scale down positions to meet variance target
            scaling_factor = np.sqrt(max_portfolio_var / portfolio_var)
            pos_sizes *= scaling_factor
            
        return {asset: size for asset, size in zip(assets, pos_sizes)}
    
    def get_regime_limits(self, regime: MarketRegime) -> Dict[str, float]:
        """Get position limits for given regime."""
        base_limits = {
            'max_position_size': self.max_position_size,
            'max_leverage': 2.0,
            'concentration_limit': 0.25
        }
        
        regime_adjustments = {
            MarketRegime.LOW_VOL: {
                'max_position_size': base_limits['max_position_size'] * 1.2,
                'max_leverage': base_limits['max_leverage'] * 1.2,
                'concentration_limit': base_limits['concentration_limit'] * 1.1
            },
            MarketRegime.HIGH_VOL: {
                'max_position_size': base_limits['max_position_size'] * 0.6,
                'max_leverage': base_limits['max_leverage'] * 0.5,
                'concentration_limit': base_limits['concentration_limit'] * 0.7
            },
            MarketRegime.TRENDING: {
                'max_position_size': base_limits['max_position_size'] * 1.0,
                'max_leverage': base_limits['max_leverage'] * 1.0,
                'concentration_limit': base_limits['concentration_limit'] * 1.0
            },
            MarketRegime.MEAN_REVERTING: {
                'max_position_size': base_limits['max_position_size'] * 0.8,
                'max_leverage': base_limits['max_leverage'] * 0.8,
                'concentration_limit': base_limits['concentration_limit'] * 0.9
            },
            MarketRegime.CRISIS: {
                'max_position_size': base_limits['max_position_size'] * 0.3,
                'max_leverage': base_limits['max_leverage'] * 0.3,
                'concentration_limit': base_limits['concentration_limit'] * 0.5
            }
        }
        
        return regime_adjustments.get(regime, base_limits)