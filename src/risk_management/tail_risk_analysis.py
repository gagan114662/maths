#!/usr/bin/env python
"""
Tail Risk Analysis Module

This module implements advanced risk analysis techniques using Extreme Value Theory (EVT)
to better understand and quantify tail risks in trading strategies.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler

class TailRiskAnalyzer:
    """
    Analyzes tail risks using Extreme Value Theory (EVT) techniques.
    """
    
    def __init__(self, 
                threshold_percentile: float = 0.95,
                min_observations: int = 252,
                time_decay_factor: float = 0.94):
        """
        Initialize the tail risk analyzer.

        Args:
            threshold_percentile: Percentile for tail threshold (default: 0.95)
            min_observations: Minimum number of observations needed (default: 252)
            time_decay_factor: Exponential decay factor for time weighting (default: 0.94)
        """
        self.threshold_percentile = threshold_percentile
        self.min_observations = min_observations
        self.time_decay_factor = time_decay_factor
        self.fitted_params = {}
        
    def fit_evt_distribution(self, returns: pd.Series) -> Dict[str, float]:
        """
        Fit Generalized Extreme Value (GEV) distribution to return extremes.
        
        Args:
            returns: Series of asset/strategy returns
            
        Returns:
            Dictionary of fitted parameters
        """
        if len(returns) < self.min_observations:
            raise ValueError(f"Need at least {self.min_observations} observations")
            
        # Calculate extreme negative returns
        negative_returns = -returns  # Convert to losses
        threshold = np.percentile(negative_returns, self.threshold_percentile * 100)
        exceedances = negative_returns[negative_returns > threshold]
        
        if len(exceedances) < 50:  # Need sufficient exceedances for reliable fit
            raise ValueError("Insufficient exceedances for reliable EVT fit")
            
        # Fit Generalized Pareto Distribution to exceedances
        gpd_params = stats.genpareto.fit(exceedances - threshold)
        
        # Store parameters
        self.fitted_params = {
            'threshold': threshold,
            'shape': gpd_params[0],  # ξ (xi) - tail index
            'scale': gpd_params[2],  # β (beta) - scale parameter
            'location': gpd_params[1]  # μ (mu) - location parameter
        }
        
        return self.fitted_params
    
    def estimate_var(self, 
                    confidence_level: float = 0.99, 
                    horizon: int = 1) -> float:
        """
        Estimate Value at Risk using fitted EVT distribution.
        
        Args:
            confidence_level: VaR confidence level (default: 0.99)
            horizon: Time horizon in days (default: 1)
            
        Returns:
            Estimated VaR at specified confidence level
        """
        if not self.fitted_params:
            raise ValueError("Must fit EVT distribution first")
            
        # Extract parameters
        threshold = self.fitted_params['threshold']
        shape = self.fitted_params['shape']
        scale = self.fitted_params['scale']
        
        # Calculate exceedance probability
        p = 1 - confidence_level
        
        # Calculate VaR using GPD formula
        if abs(shape) < 1e-10:  # Shape ≈ 0
            var = threshold + scale * np.log(horizon * p)
        else:
            var = threshold + (scale/shape) * ((horizon * p)**(-shape) - 1)
            
        return var
    
    def estimate_es(self, 
                   confidence_level: float = 0.99, 
                   horizon: int = 1) -> float:
        """
        Estimate Expected Shortfall (Conditional VaR) using fitted EVT distribution.
        
        Args:
            confidence_level: ES confidence level (default: 0.99)
            horizon: Time horizon in days (default: 1)
            
        Returns:
            Estimated ES at specified confidence level
        """
        if not self.fitted_params:
            raise ValueError("Must fit EVT distribution first")
            
        # Extract parameters
        threshold = self.fitted_params['threshold']
        shape = self.fitted_params['shape']
        scale = self.fitted_params['scale']
        
        # Calculate VaR first
        var = self.estimate_var(confidence_level, horizon)
        
        # Calculate ES using GPD formula
        if abs(shape) < 1e-10:  # Shape ≈ 0
            es = var + scale
        else:
            es = (var + scale - shape*threshold)/(1 - shape)
            
        return es
    
    def estimate_tail_dependence(self, returns1: pd.Series, 
                               returns2: pd.Series) -> Dict[str, float]:
        """
        Estimate tail dependence between two return series.
        
        Args:
            returns1: First series of returns
            returns2: Second series of returns
            
        Returns:
            Dictionary containing tail dependence coefficients
        """
        # Ensure equal length
        common_index = returns1.index.intersection(returns2.index)
        returns1 = returns1[common_index]
        returns2 = returns2[common_index]
        
        # Convert to uniform margins using empirical CDF
        u1 = stats.rankdata(returns1) / len(returns1)
        u2 = stats.rankdata(returns2) / len(returns2)
        
        # Calculate tail dependence coefficients
        q = 0.05  # Tail probability
        lower_tail = np.mean((u1 <= q) & (u2 <= q)) / q
        upper_tail = np.mean((u1 >= 1-q) & (u2 >= 1-q)) / q
        
        return {
            'lower_tail_dependence': lower_tail,
            'upper_tail_dependence': upper_tail
        }
    
    def analyze_tail_risk(self, returns: pd.Series) -> Dict[str, float]:
        """
        Perform comprehensive tail risk analysis.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary of tail risk metrics
        """
        # Fit EVT distribution
        self.fit_evt_distribution(returns)
        
        # Calculate various risk metrics
        metrics = {
            'var_99': self.estimate_var(0.99),
            'es_99': self.estimate_es(0.99),
            'var_95': self.estimate_var(0.95),
            'es_95': self.estimate_es(0.95),
            'tail_index': self.fitted_params['shape'],
            'scale': self.fitted_params['scale'],
            'threshold': self.fitted_params['threshold']
        }
        
        # Add tail risk indicators
        metrics.update(self._calculate_tail_risk_indicators(returns))
        
        return metrics
    
    def _calculate_tail_risk_indicators(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate additional tail risk indicators."""
        indicators = {}
        
        # Calculate skewness and kurtosis
        indicators['skewness'] = stats.skew(returns)
        indicators['excess_kurtosis'] = stats.kurtosis(returns)
        
        # Calculate maximum drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding(min_periods=1).max()
        drawdowns = cum_returns/rolling_max - 1
        indicators['max_drawdown'] = drawdowns.min()
        
        # Calculate time under water (below previous peak)
        underwater = (drawdowns < 0).astype(int)
        indicators['avg_time_underwater'] = underwater.mean()
        
        # Calculate Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        indicators['jarque_bera_pvalue'] = jb_pvalue
        
        return indicators
    
    def estimate_stress_var(self, returns: pd.Series, 
                          stress_factor: float = 1.5) -> Dict[str, float]:
        """
        Estimate VaR under stressed market conditions.
        
        Args:
            returns: Series of returns
            stress_factor: Factor to increase volatility (default: 1.5)
            
        Returns:
            Dictionary of stressed risk metrics
        """
        # Calculate normal VaR first
        normal_metrics = self.analyze_tail_risk(returns)
        
        # Apply stress to the distribution parameters
        stressed_params = {
            'threshold': self.fitted_params['threshold'] * stress_factor,
            'scale': self.fitted_params['scale'] * stress_factor,
            'shape': self.fitted_params['shape']  # Shape parameter typically stable
        }
        
        # Store original params
        original_params = self.fitted_params.copy()
        
        # Calculate stressed metrics
        self.fitted_params = stressed_params
        stressed_metrics = {
            'stressed_var_99': self.estimate_var(0.99),
            'stressed_es_99': self.estimate_es(0.99),
            'stress_impact': (self.estimate_var(0.99) / normal_metrics['var_99']) - 1
        }
        
        # Restore original params
        self.fitted_params = original_params
        
        return stressed_metrics

def calculate_conditional_drawdown_risk(returns: pd.Series, 
                                     confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Calculate Conditional Drawdown at Risk (CDaR).
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level for risk calculation
        
    Returns:
        Dictionary of drawdown risk metrics
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.expanding().max()
    
    # Calculate drawdowns
    drawdowns = (cum_returns - running_max) / running_max
    
    # Sort drawdowns to find worst ones
    sorted_drawdowns = np.sort(drawdowns)
    
    # Calculate CDaR
    n = len(drawdowns)
    k = int(n * (1 - confidence_level))
    worst_k_drawdowns = sorted_drawdowns[:k]
    
    cdar = worst_k_drawdowns.mean()
    max_drawdown = drawdowns.min()
    avg_drawdown = drawdowns.mean()
    
    return {
        'cdar': cdar,
        'max_drawdown': max_drawdown,
        'avg_drawdown': avg_drawdown,
        'worst_drawdowns': list(worst_k_drawdowns),
        'confidence_level': confidence_level
    }