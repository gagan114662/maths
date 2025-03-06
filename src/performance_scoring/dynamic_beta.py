"""
Dynamic beta adjustment module.

This module provides sophisticated functionality for dynamic beta management, including:
1. Time-varying beta calculation and forecasting
2. Market regime-aware beta targeting
3. Portfolio optimization to achieve target beta
4. Beta hedging strategies for market-neutral positioning
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BetaMethod(Enum):
    """Methods for calculating beta."""
    STANDARD = "standard"  # Standard regression-based beta
    ROLLING = "rolling"    # Rolling window beta
    KALMAN = "kalman"      # Kalman filter-based beta (time-varying)
    GARCH = "garch"        # GARCH-based conditional beta
    REGIME_CONDITIONAL = "regime_conditional"  # Regime-dependent beta

class BetaTarget(Enum):
    """Target beta strategies."""
    MARKET_NEUTRAL = "market_neutral"  # Beta = 0.0
    LOW_BETA = "low_beta"              # Low beta (0.3 - 0.7)
    BENCHMARK = "benchmark"            # Match benchmark (Beta = 1.0)
    HIGH_BETA = "high_beta"            # High beta for bull markets
    DYNAMIC = "dynamic"                # Dynamically adjusted based on conditions

class DynamicBetaManager:
    """
    Dynamic beta adjustment and optimization manager.
    
    This class handles sophisticated beta calculation, forecasting,
    and portfolio optimization to reach target beta levels
    that adapt to changing market conditions.
    """
    
    def __init__(self, 
                 calculation_method: BetaMethod = BetaMethod.ROLLING,
                 beta_target_strategy: BetaTarget = BetaTarget.MARKET_NEUTRAL,
                 window_size: int = 63,
                 forecast_horizon: int = 21,
                 min_history: int = 126,
                 default_beta: float = 1.0):
        """
        Initialize the dynamic beta manager.
        
        Args:
            calculation_method: Method for calculating beta
            beta_target_strategy: Strategy for targeting beta
            window_size: Window size for rolling beta calculation
            forecast_horizon: Number of days to forecast beta
            min_history: Minimum history required for calculations
            default_beta: Default beta to use when calculation is not possible
        """
        self.calculation_method = calculation_method
        self.beta_target_strategy = beta_target_strategy
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.min_history = min_history
        self.default_beta = default_beta
        
        # Storage for historical and calculated data
        self.strategy_returns = None
        self.benchmark_returns = None
        self.historical_betas = None
        self.forecasted_betas = None
        self.target_betas = None
        self.regime_data = None
        self.current_weights = None
        self.target_weights = None
        self.constraints = {}
        
        logger.info(f"Initialized DynamicBetaManager with {calculation_method.value} method and {beta_target_strategy.value} target strategy")
    
    def set_data(self, 
                strategy_returns: pd.Series,
                benchmark_returns: pd.Series,
                regime_data: pd.Series = None) -> None:
        """
        Set return data for beta calculations.
        
        Args:
            strategy_returns: Series of daily strategy returns
            benchmark_returns: Series of daily benchmark returns
            regime_data: Optional series of market regime labels
        """
        # Check input types
        if not isinstance(strategy_returns, pd.Series):
            strategy_returns = pd.Series(strategy_returns)
        
        if not isinstance(benchmark_returns, pd.Series):
            benchmark_returns = pd.Series(benchmark_returns)
            
        # Align dates
        data = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        # Store aligned data
        self.strategy_returns = data['strategy']
        self.benchmark_returns = data['benchmark']
        
        # Store regime data if provided
        if regime_data is not None:
            if not isinstance(regime_data, pd.Series):
                regime_data = pd.Series(regime_data)
                
            # Align regime data to return dates
            aligned_regime = regime_data.reindex(data.index, method='ffill')
            self.regime_data = aligned_regime
            
        logger.info(f"Set data with {len(data)} observations")
        
        # Reset any previously calculated values
        self.historical_betas = None
        self.forecasted_betas = None
        self.target_betas = None
    
    def calculate_historical_betas(self) -> pd.Series:
        """
        Calculate historical betas using the specified method.
        
        Returns:
            Series of historical beta values
        """
        if self.strategy_returns is None or self.benchmark_returns is None:
            logger.error("Strategy and benchmark returns must be set before calculating betas")
            return None
            
        # Check if we have sufficient data
        if len(self.strategy_returns) < self.min_history:
            logger.warning(f"Insufficient data for beta calculation. Using default beta {self.default_beta}")
            return pd.Series(self.default_beta, index=self.strategy_returns.index)
            
        # Calculate beta based on the specified method
        if self.calculation_method == BetaMethod.STANDARD:
            beta = self._calculate_standard_beta()
            # Create a series with constant beta for all dates
            betas = pd.Series(beta, index=self.strategy_returns.index)
            
        elif self.calculation_method == BetaMethod.ROLLING:
            betas = self._calculate_rolling_beta()
            
        elif self.calculation_method == BetaMethod.KALMAN:
            betas = self._calculate_kalman_beta()
            
        elif self.calculation_method == BetaMethod.GARCH:
            betas = self._calculate_garch_beta()
            
        elif self.calculation_method == BetaMethod.REGIME_CONDITIONAL:
            if self.regime_data is None:
                logger.warning("Regime data not available. Falling back to rolling beta.")
                betas = self._calculate_rolling_beta()
            else:
                betas = self._calculate_regime_conditional_beta()
        else:
            # Default to rolling beta
            logger.warning(f"Unknown beta calculation method: {self.calculation_method}. Using rolling beta.")
            betas = self._calculate_rolling_beta()
            
        # Store historical betas
        self.historical_betas = betas
        
        logger.info(f"Calculated historical betas using {self.calculation_method.value} method")
        return betas
    
    def _calculate_standard_beta(self) -> float:
        """
        Calculate standard regression-based beta.
        
        Returns:
            Beta coefficient
        """
        X = sm.add_constant(self.benchmark_returns)
        model = sm.OLS(self.strategy_returns, X)
        results = model.fit()
        beta = results.params['benchmark']
        return beta
    
    def _calculate_rolling_beta(self) -> pd.Series:
        """
        Calculate rolling beta.
        
        Returns:
            Series of rolling beta values
        """
        # Prepare data for rolling regression
        Y = self.strategy_returns
        X = sm.add_constant(self.benchmark_returns)
        
        # Calculate rolling beta
        rolling_model = RollingOLS(Y, X, window=self.window_size, min_nobs=max(10, self.window_size // 2))
        rolling_results = rolling_model.fit()
        
        # Extract beta series
        betas = rolling_results.params['benchmark']
        
        # Fill NaN values with first valid beta
        betas = betas.fillna(method='bfill').fillna(self.default_beta)
        
        return betas
    
    def _calculate_kalman_beta(self) -> pd.Series:
        """
        Calculate time-varying beta using Kalman filter.
        
        Returns:
            Series of Kalman filter-based beta values
        """
        # This is a simplified implementation of Kalman filter for beta estimation
        # For a full implementation, consider using specialized libraries like pykalman
        
        n = len(self.strategy_returns)
        betas = np.zeros(n)
        
        # Initialize Kalman filter parameters
        beta = self.default_beta  # Initial beta estimate
        p = 1.0                   # Initial covariance
        q = 0.001                 # Process variance
        r = 0.1                   # Measurement variance
        
        for t in range(n):
            # Prediction step
            p = p + q
            
            # Update step
            benchmark_t = self.benchmark_returns.iloc[t]
            strategy_t = self.strategy_returns.iloc[t]
            
            if benchmark_t != 0:  # Avoid division by zero
                k = p * benchmark_t / (benchmark_t**2 * p + r)  # Kalman gain
                beta = beta + k * (strategy_t - beta * benchmark_t)  # Update estimate
                p = (1 - k * benchmark_t) * p  # Update covariance
                
            betas[t] = beta
            
        # Convert to Series
        return pd.Series(betas, index=self.strategy_returns.index)
    
    def _calculate_garch_beta(self) -> pd.Series:
        """
        Calculate GARCH-based conditional beta.
        
        Returns:
            Series of GARCH-based beta values
        """
        # This is a placeholder for GARCH-based beta calculation
        # For a full implementation, consider using specialized libraries like arch
        
        # Attempt to import arch package for GARCH modeling
        try:
            from arch import arch_model
        except ImportError:
            logger.warning("arch package not available. Falling back to rolling beta.")
            return self._calculate_rolling_beta()
        
        # For now, fall back to rolling beta
        logger.warning("GARCH-based beta calculation not implemented. Using rolling beta.")
        return self._calculate_rolling_beta()
    
    def _calculate_regime_conditional_beta(self) -> pd.Series:
        """
        Calculate regime-conditional beta.
        
        Returns:
            Series of regime-conditional beta values
        """
        # Calculate regime-specific betas
        regime_betas = {}
        
        for regime in self.regime_data.unique():
            # Filter data for this regime
            regime_mask = self.regime_data == regime
            regime_strategy = self.strategy_returns[regime_mask]
            regime_benchmark = self.benchmark_returns[regime_mask]
            
            # Skip if insufficient data
            if len(regime_strategy) < 20:
                logger.warning(f"Insufficient data for regime {regime}. Using overall beta.")
                continue
                
            # Calculate beta for this regime
            X = sm.add_constant(regime_benchmark)
            model = sm.OLS(regime_strategy, X)
            results = model.fit()
            regime_betas[regime] = results.params['benchmark']
            
        # If we couldn't calculate betas for any regime, fall back to rolling beta
        if not regime_betas:
            logger.warning("Could not calculate regime-specific betas. Using rolling beta.")
            return self._calculate_rolling_beta()
            
        # Apply regime-specific betas
        betas = pd.Series(index=self.strategy_returns.index)
        
        for date, regime in self.regime_data.items():
            if date in betas.index:
                if regime in regime_betas:
                    betas[date] = regime_betas[regime]
                else:
                    # If we don't have a beta for this regime, use the default
                    betas[date] = self.default_beta
                    
        # Fill any remaining NaN values
        betas = betas.fillna(method='ffill').fillna(method='bfill').fillna(self.default_beta)
        
        return betas
    
    def forecast_beta(self, dates: Optional[pd.DatetimeIndex] = None) -> pd.Series:
        """
        Forecast future beta values.
        
        Args:
            dates: Optional future dates to forecast beta for
            
        Returns:
            Series of forecasted beta values
        """
        # Calculate historical betas if not already done
        if self.historical_betas is None:
            self.calculate_historical_betas()
            
        # If no dates provided, forecast for the next forecast_horizon days
        if dates is None:
            last_date = self.historical_betas.index[-1]
            dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                periods=self.forecast_horizon, freq='B')
                                
        # Simple forecasting method - ARIMA or more sophisticated models could be used
        # For now, we use an exponentially weighted average of recent betas
        recent_betas = self.historical_betas.tail(self.forecast_horizon * 2)
        weights = np.exp(np.linspace(-1, 0, len(recent_betas)))
        weights = weights / weights.sum()
        
        forecasted_beta = np.sum(recent_betas.values * weights)
        
        # Create series with forecasted betas
        forecasted_betas = pd.Series(forecasted_beta, index=dates)
        
        # Store forecasted betas
        self.forecasted_betas = forecasted_betas
        
        logger.info(f"Forecasted beta: {forecasted_beta:.4f} for {len(dates)} future dates")
        return forecasted_betas
    
    def determine_target_beta(self, 
                             current_date: Optional[datetime] = None,
                             forecast_period: int = None) -> float:
        """
        Determine target beta based on the current strategy and market conditions.
        
        Args:
            current_date: Current date (defaults to last date in data)
            forecast_period: Number of days to look ahead (defaults to forecast_horizon)
            
        Returns:
            Target beta value
        """
        # Set default values
        if current_date is None:
            if self.historical_betas is not None:
                current_date = self.historical_betas.index[-1]
            else:
                current_date = self.strategy_returns.index[-1]
                
        if forecast_period is None:
            forecast_period = self.forecast_horizon
            
        # Calculate betas if not already done
        if self.historical_betas is None:
            self.calculate_historical_betas()
            
        # Get current beta
        if current_date in self.historical_betas.index:
            current_beta = self.historical_betas[current_date]
        else:
            # Use the last available beta
            current_beta = self.historical_betas.iloc[-1]
            
        # Determine target beta based on strategy
        if self.beta_target_strategy == BetaTarget.MARKET_NEUTRAL:
            target_beta = 0.0
            
        elif self.beta_target_strategy == BetaTarget.LOW_BETA:
            target_beta = 0.5
            
        elif self.beta_target_strategy == BetaTarget.BENCHMARK:
            target_beta = 1.0
            
        elif self.beta_target_strategy == BetaTarget.HIGH_BETA:
            target_beta = 1.5
            
        elif self.beta_target_strategy == BetaTarget.DYNAMIC:
            # Dynamically adjust target beta based on conditions
            target_beta = self._calculate_dynamic_target_beta(current_date)
        else:
            # Default to current beta
            logger.warning(f"Unknown beta target strategy: {self.beta_target_strategy}. Using current beta.")
            target_beta = current_beta
            
        # Store target beta
        if self.target_betas is None:
            self.target_betas = pd.Series()
            
        self.target_betas[current_date] = target_beta
        
        logger.info(f"Determined target beta: {target_beta:.4f} for {current_date}")
        return target_beta
    
    def _calculate_dynamic_target_beta(self, current_date: datetime) -> float:
        """
        Calculate dynamic target beta based on market conditions.
        
        Args:
            current_date: Current date
            
        Returns:
            Target beta value
        """
        # Check if we have regime data
        if self.regime_data is not None and current_date in self.regime_data.index:
            current_regime = self.regime_data[current_date]
            
            # Adjust beta based on regime
            if str(current_regime).lower() in ['bull', 'uptrend', 'expansion']:
                # Higher beta in bull markets to capture upside
                return 1.2
            elif str(current_regime).lower() in ['bear', 'downtrend', 'contraction']:
                # Lower (or negative) beta in bear markets for protection
                return 0.0
            elif str(current_regime).lower() in ['high_volatility', 'crisis']:
                # Minimal market exposure in high volatility environments
                return -0.3
            elif str(current_regime).lower() in ['low_volatility', 'stable']:
                # Moderate beta in low volatility environments
                return 0.8
            else:
                # Default to market-neutral for unknown regimes
                return 0.0
        
        # If no regime data, use recent market performance and volatility
        recent_market = self.benchmark_returns.tail(self.window_size)
        
        # Calculate annualized return and volatility
        market_return = recent_market.mean() * 252
        market_volatility = recent_market.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming zero risk-free rate for simplicity)
        market_sharpe = market_return / market_volatility if market_volatility > 0 else 0
        
        # Determine target beta based on market conditions
        if market_return > 0 and market_sharpe > 1.0:
            # Strong positive trend with good risk-adjusted returns - higher beta
            return 1.2
        elif market_return > 0 and market_sharpe > 0.5:
            # Positive trend but moderate risk-adjusted returns - neutral to slightly positive beta
            return 0.8
        elif market_return > 0 and market_sharpe <= 0.5:
            # Positive returns but poor risk-adjusted performance - lower beta
            return 0.5
        elif market_return <= 0 and market_volatility > 0.2:
            # Negative returns and high volatility - market-neutral or negative beta
            return -0.2
        else:
            # Default to slight market exposure
            return 0.3
    
    def optimize_portfolio_weights(self, 
                                  current_weights: Dict[str, float],
                                  asset_betas: Dict[str, float],
                                  target_beta: Optional[float] = None,
                                  constraints: Optional[Dict] = None) -> Dict[str, float]:
        """
        Optimize portfolio weights to achieve target beta.
        
        Args:
            current_weights: Current portfolio weights
            asset_betas: Asset betas to benchmark
            target_beta: Target portfolio beta (if None, uses determined target)
            constraints: Optional constraints for optimization
            
        Returns:
            Dictionary of optimized weights
        """
        # Determine target beta if not provided
        if target_beta is None:
            target_beta = self.determine_target_beta()
            
        # Store current weights and constraints
        self.current_weights = current_weights
        if constraints is not None:
            self.constraints = constraints
            
        # Get asset names and prepare vectors
        assets = list(current_weights.keys())
        
        # For benchmark, add it if not in the current portfolio
        benchmark_asset = 'BENCHMARK'
        if benchmark_asset not in assets:
            assets.append(benchmark_asset)
            
        # Make sure all assets have beta values
        for asset in assets:
            if asset not in asset_betas and asset != benchmark_asset:
                logger.warning(f"No beta value for {asset}. Using default beta.")
                asset_betas[asset] = self.default_beta
                
        # Set benchmark beta to 1.0
        asset_betas[benchmark_asset] = 1.0
        
        # Initial weights - including 0 for benchmark if not present
        initial_weights = np.zeros(len(assets))
        for i, asset in enumerate(assets):
            initial_weights[i] = current_weights.get(asset, 0.0)
            
        # Define optimization function to minimize tracking error while targeting beta
        def objective(weights):
            # Calculate portfolio beta
            portfolio_beta = sum(weights[i] * asset_betas.get(assets[i], 0.0) for i in range(len(assets)))
            
            # Target beta constraint
            beta_error = (portfolio_beta - target_beta) ** 2
            
            # Minimize changes from current weights
            weight_changes = sum((weights[i] - current_weights.get(assets[i], 0.0)) ** 2 
                                for i in range(len(assets)) if assets[i] != benchmark_asset)
            
            return beta_error * 100 + weight_changes
            
        # Define constraints
        constraint_functions = []
        
        # Sum of weights = 1 (allowing for short positions in the benchmark)
        constraint_functions.append({
            'type': 'eq',
            'fun': lambda weights: sum(weights[i] for i in range(len(assets)) if assets[i] != benchmark_asset) - 1.0
        })
        
        # Additional constraints from input
        if constraints is not None:
            # Maximum allocation per asset
            if 'max_allocation' in constraints:
                max_alloc = constraints['max_allocation']
                for i, asset in enumerate(assets):
                    if asset != benchmark_asset:  # Don't apply to benchmark
                        constraint_functions.append({
                            'type': 'ineq',
                            'fun': lambda weights, idx=i: max_alloc - weights[idx]
                        })
                        
            # Minimum allocation per asset
            if 'min_allocation' in constraints:
                min_alloc = constraints['min_allocation']
                for i, asset in enumerate(assets):
                    if asset != benchmark_asset:  # Don't apply to benchmark
                        constraint_functions.append({
                            'type': 'ineq',
                            'fun': lambda weights, idx=i: weights[idx] - min_alloc
                        })
                        
            # Maximum benchmark weight
            if 'max_benchmark' in constraints:
                max_bench = constraints['max_benchmark']
                benchmark_idx = assets.index(benchmark_asset)
                constraint_functions.append({
                    'type': 'ineq',
                    'fun': lambda weights: max_bench - abs(weights[benchmark_idx])
                })
                
        # Bounds for weights - allow shorting for the benchmark
        bounds = []
        for asset in assets:
            if asset == benchmark_asset:
                # Allow shorting the benchmark for hedging
                bounds.append((-1.0, 1.0))
            else:
                # Default bounds (adjust based on your requirements)
                min_w = constraints.get('min_allocation', 0.0) if constraints else 0.0
                max_w = constraints.get('max_allocation', 1.0) if constraints else 1.0
                bounds.append((min_w, max_w))
                
        # Run optimization
        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_functions,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimized_weights = {assets[i]: result.x[i] for i in range(len(assets))}
                logger.info(f"Successfully optimized weights for target beta {target_beta:.4f}")
                
                # Calculate resulting portfolio beta
                portfolio_beta = sum(w * asset_betas.get(a, 0.0) for a, w in optimized_weights.items())
                logger.info(f"Resulting portfolio beta: {portfolio_beta:.4f}")
                
                # Store target weights
                self.target_weights = optimized_weights
                
                return optimized_weights
            else:
                logger.warning(f"Optimization failed: {result.message}")
                return current_weights
        except Exception as e:
            logger.error(f"Error in weight optimization: {str(e)}")
            return current_weights
    
    def get_beta_hedging_recommendation(self, 
                                      current_beta: Optional[float] = None,
                                      target_beta: Optional[float] = None) -> Dict:
        """
        Get recommendations for beta hedging.
        
        Args:
            current_beta: Current portfolio beta (if None, uses calculated value)
            target_beta: Target beta (if None, uses determined target)
            
        Returns:
            Dictionary with hedging recommendations
        """
        # Use calculated values if not provided
        if current_beta is None:
            if self.historical_betas is not None:
                current_beta = self.historical_betas.iloc[-1]
            else:
                self.calculate_historical_betas()
                current_beta = self.historical_betas.iloc[-1]
                
        if target_beta is None:
            target_beta = self.determine_target_beta()
            
        # Calculate required beta adjustment
        beta_adjustment = target_beta - current_beta
        
        # Determine hedging action
        if abs(beta_adjustment) < 0.05:
            action = "No adjustment needed"
            hedge_percentage = 0.0
        elif beta_adjustment < 0:
            action = "Short the benchmark to reduce beta"
            hedge_percentage = -beta_adjustment
        else:
            action = "Long the benchmark to increase beta"
            hedge_percentage = beta_adjustment
            
        # Calculate estimated position sizes
        portfolio_value = 1.0  # Normalized to 1.0
        hedge_value = portfolio_value * hedge_percentage
        
        # Prepare recommendation
        recommendation = {
            'current_beta': current_beta,
            'target_beta': target_beta,
            'beta_adjustment': beta_adjustment,
            'action': action,
            'hedge_percentage': hedge_percentage,
            'hedge_value_normalized': hedge_value,
            'hedge_value_description': f"{hedge_percentage:.2%} of portfolio value"
        }
        
        logger.info(f"Beta hedging recommendation: {action} | Adjust by {hedge_percentage:.2%}")
        return recommendation
    
    def generate_beta_report(self, 
                           output_path: Optional[str] = None,
                           include_plots: bool = True) -> Dict:
        """
        Generate a comprehensive beta analysis report.
        
        Args:
            output_path: Path to save report files (None = don't save)
            include_plots: Whether to generate plots
            
        Returns:
            Dictionary with report data
        """
        # Calculate betas if not already done
        if self.historical_betas is None:
            self.calculate_historical_betas()
            
        # Determine target beta
        target_beta = self.determine_target_beta()
        
        # Get current beta
        current_beta = self.historical_betas.iloc[-1]
        
        # Get hedging recommendation
        hedging_recommendation = self.get_beta_hedging_recommendation(current_beta, target_beta)
        
        # Calculate basic statistics
        beta_stats = {
            'mean_beta': self.historical_betas.mean(),
            'median_beta': self.historical_betas.median(),
            'min_beta': self.historical_betas.min(),
            'max_beta': self.historical_betas.max(),
            'std_beta': self.historical_betas.std(),
            'current_beta': current_beta,
            'target_beta': target_beta
        }
        
        # Prepare report data
        report = {
            'beta_stats': beta_stats,
            'hedging_recommendation': hedging_recommendation,
            'calculation_method': self.calculation_method.value,
            'target_strategy': self.beta_target_strategy.value,
            'forecast_horizon': self.forecast_horizon,
            'window_size': self.window_size
        }
        
        # Generate and save plots if requested
        if include_plots:
            plots = {}
            
            # Plot historical betas
            fig_historical = plt.figure(figsize=(12, 6))
            plt.plot(self.historical_betas.index, self.historical_betas, label='Historical Beta')
            
            # Add target beta line
            plt.axhline(y=target_beta, color='r', linestyle='--', label=f'Target Beta ({target_beta:.2f})')
            
            # Add market-neutral line
            plt.axhline(y=0.0, color='g', linestyle=':', label='Market-Neutral (Beta=0)')
            
            # Add benchmark line
            plt.axhline(y=1.0, color='k', linestyle=':', label='Benchmark Beta (Beta=1)')
            
            plt.title('Historical Beta Analysis')
            plt.xlabel('Date')
            plt.ylabel('Beta')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plots['historical_beta'] = fig_historical
            
            # Add forecast if available
            if self.forecasted_betas is not None:
                fig_forecast = plt.figure(figsize=(12, 6))
                
                # Plot historical for context
                plt.plot(self.historical_betas.index[-60:], self.historical_betas.iloc[-60:], 
                        label='Historical Beta')
                
                # Plot forecast
                plt.plot(self.forecasted_betas.index, self.forecasted_betas, 
                        label='Forecasted Beta', color='r', linestyle='--')
                
                # Add target beta line
                plt.axhline(y=target_beta, color='g', linestyle='--', 
                          label=f'Target Beta ({target_beta:.2f})')
                
                plt.title('Beta Forecast')
                plt.xlabel('Date')
                plt.ylabel('Beta')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plots['beta_forecast'] = fig_forecast
                
            # Add regime analysis if available
            if self.regime_data is not None:
                fig_regime = plt.figure(figsize=(12, 6))
                
                # Create a colormap for different regimes
                regimes = self.regime_data.unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(regimes)))
                regime_colors = {regime: colors[i] for i, regime in enumerate(regimes)}
                
                # Plot beta
                plt.plot(self.historical_betas.index, self.historical_betas, label='Beta', color='k')
                
                # Add colored background for regimes
                ylim = plt.ylim()
                for regime in regimes:
                    regime_periods = self.regime_data[self.regime_data == regime].index
                    if len(regime_periods) > 0:
                        # Group consecutive dates
                        breaks = np.where(np.diff(regime_periods) > pd.Timedelta(days=1))[0]
                        start_idx = 0
                        
                        for break_idx in list(breaks) + [len(regime_periods) - 1]:
                            period_start = regime_periods[start_idx]
                            period_end = regime_periods[break_idx]
                            
                            plt.axvspan(period_start, period_end, 
                                      alpha=0.2, color=regime_colors[regime])
                            
                            start_idx = break_idx + 1
                
                # Add regime labels
                handles = [plt.Rectangle((0, 0), 1, 1, fc=regime_colors[r], alpha=0.2) for r in regimes]
                labels = [str(r) for r in regimes]
                plt.legend(handles, labels)
                
                plt.title('Beta by Market Regime')
                plt.xlabel('Date')
                plt.ylabel('Beta')
                plt.grid(True, alpha=0.3)
                
                plots['regime_beta'] = fig_regime
                
            # Save plots if output path provided
            if output_path is not None:
                # Create directory if it doesn't exist
                import os
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                    
                # Save plots
                for name, fig in plots.items():
                    plot_path = os.path.join(output_path, f"{name}.png")
                    fig.savefig(plot_path)
                    plt.close(fig)
                    
                    # Add file paths to report
                    if 'plot_files' not in report:
                        report['plot_files'] = {}
                        
                    report['plot_files'][name] = plot_path
                    logger.info(f"Saved {name} plot to {plot_path}")
            
            # Add plots to report
            report['plots'] = plots
        
        # Save report as JSON if output path provided
        if output_path is not None:
            import json
            import os
            
            report_data = {
                'beta_stats': beta_stats,
                'hedging_recommendation': hedging_recommendation,
                'calculation_method': self.calculation_method.value,
                'target_strategy': self.beta_target_strategy.value,
                'forecast_horizon': self.forecast_horizon,
                'window_size': self.window_size
            }
            
            if 'plot_files' in report:
                report_data['plot_files'] = report['plot_files']
                
            report_path = os.path.join(output_path, "beta_analysis_report.json")
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            report['report_file'] = report_path
            logger.info(f"Saved beta analysis report to {report_path}")
        
        return report

class BenchmarkTracker:
    """
    Enhanced benchmark tracker with dynamic beta adjustment.
    
    This class provides sophisticated benchmark tracking functionality
    with dynamic beta adjustment to achieve target exposure levels.
    """
    
    def __init__(self, 
                benchmark_symbol: str = "SPY",
                tracking_portfolio_size: int = 20,
                min_history: int = 252,
                beta_manager: Optional[DynamicBetaManager] = None):
        """
        Initialize the benchmark tracker.
        
        Args:
            benchmark_symbol: Symbol for the benchmark index
            tracking_portfolio_size: Size of tracking portfolio
            min_history: Minimum history required for calculations
            beta_manager: Optional custom beta manager
        """
        self.benchmark_symbol = benchmark_symbol
        self.tracking_portfolio_size = tracking_portfolio_size
        self.min_history = min_history
        
        # Initialize beta manager if not provided
        if beta_manager is None:
            self.beta_manager = DynamicBetaManager(
                calculation_method=BetaMethod.ROLLING,
                beta_target_strategy=BetaTarget.BENCHMARK  # Default to tracking benchmark
            )
        else:
            self.beta_manager = beta_manager
            
        # Storage for data
        self.benchmark_data = None
        self.constituent_data = {}
        self.tracking_portfolio = {}
        self.portfolio_returns = None
        self.tracking_error = None
        self.portfolio_beta = None
        
        logger.info(f"Initialized BenchmarkTracker for {benchmark_symbol}")
    
    def load_benchmark_data(self, data: pd.DataFrame) -> None:
        """
        Load benchmark data.
        
        Args:
            data: DataFrame with benchmark price data
        """
        self.benchmark_data = data
        
        # Calculate benchmark returns
        if 'returns' not in data.columns:
            if 'close' in data.columns:
                self.benchmark_returns = data['close'].pct_change().dropna()
            else:
                # Use first numeric column
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    self.benchmark_returns = data[numeric_cols[0]].pct_change().dropna()
                else:
                    logger.error("No suitable price data found in benchmark data")
                    return
        else:
            self.benchmark_returns = data['returns']
        
        logger.info(f"Loaded benchmark data with {len(data)} observations")
    
    def add_constituent_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Add benchmark constituent data.
        
        Args:
            symbol: Symbol for constituent
            data: DataFrame with constituent price data
        """
        self.constituent_data[symbol] = data
        
        # Calculate constituent returns
        if 'returns' not in data.columns:
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
            else:
                # Use first numeric column
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    returns = data[numeric_cols[0]].pct_change().dropna()
                else:
                    logger.warning(f"No suitable price data found for {symbol}")
                    return
        else:
            returns = data['returns']
            
        # Store returns
        if not hasattr(self, 'constituent_returns'):
            self.constituent_returns = {}
            
        self.constituent_returns[symbol] = returns
        
        logger.info(f"Added data for {symbol} with {len(data)} observations")
    
    def construct_tracking_portfolio(self, 
                                   method: str = 'optimization',
                                   target_date: Optional[datetime] = None,
                                   constituent_betas: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Construct a portfolio that tracks the benchmark with target beta.
        
        Args:
            method: Method for portfolio construction ('optimization', 'sampling', 'factor')
            target_date: Target date for portfolio construction
            constituent_betas: Optional dictionary of constituent betas
            
        Returns:
            Dictionary mapping constituent symbols to weights
        """
        if not self.constituent_data:
            logger.error("No constituent data available for tracking portfolio construction")
            return {}
            
        if not hasattr(self, 'benchmark_returns') or self.benchmark_returns is None:
            logger.error("Benchmark returns not available")
            return {}
            
        # Set target date to latest date if not provided
        if target_date is None:
            target_date = self.benchmark_returns.index[-1]
            
        # Calculate constituent betas if not provided
        if constituent_betas is None:
            constituent_betas = self._calculate_constituent_betas(target_date)
            
        # Construct portfolio using the specified method
        if method == 'optimization':
            tracking_portfolio = self._construct_optimal_tracking_portfolio(constituent_betas)
        elif method == 'sampling':
            tracking_portfolio = self._construct_sampling_tracking_portfolio(constituent_betas)
        elif method == 'factor':
            tracking_portfolio = self._construct_factor_tracking_portfolio(constituent_betas)
        else:
            logger.warning(f"Unknown portfolio construction method: {method}. Using optimization.")
            tracking_portfolio = self._construct_optimal_tracking_portfolio(constituent_betas)
            
        # Store tracking portfolio
        self.tracking_portfolio = tracking_portfolio
        
        # Calculate tracking portfolio beta
        portfolio_beta = sum(weight * constituent_betas.get(symbol, 1.0) 
                            for symbol, weight in tracking_portfolio.items())
        self.portfolio_beta = portfolio_beta
        
        logger.info(f"Constructed tracking portfolio with {len(tracking_portfolio)} constituents and beta {portfolio_beta:.4f}")
        return tracking_portfolio
    
    def _calculate_constituent_betas(self, target_date: datetime) -> Dict[str, float]:
        """
        Calculate betas for all constituents relative to the benchmark.
        
        Args:
            target_date: Target date for beta calculation
            
        Returns:
            Dictionary mapping constituent symbols to beta values
        """
        # Initialize result dictionary
        constituent_betas = {}
        
        for symbol, returns in self.constituent_returns.items():
            # Align constituent and benchmark returns
            aligned_data = pd.DataFrame({
                'constituent': returns,
                'benchmark': self.benchmark_returns
            }).dropna()
            
            # Skip if insufficient data
            if len(aligned_data) < self.min_history // 2:
                logger.warning(f"Insufficient data for {symbol} beta calculation")
                continue
                
            # Limit to data up to target date
            if target_date is not None:
                aligned_data = aligned_data.loc[:target_date]
                
            # Skip if still insufficient data
            if len(aligned_data) < 30:
                continue
            
            # Calculate beta using regression
            X = sm.add_constant(aligned_data['benchmark'])
            model = sm.OLS(aligned_data['constituent'], X)
            results = model.fit()
            beta = results.params['benchmark']
            
            constituent_betas[symbol] = beta
            
        logger.info(f"Calculated betas for {len(constituent_betas)} constituents")
        return constituent_betas
    
    def _construct_optimal_tracking_portfolio(self, 
                                            constituent_betas: Dict[str, float]) -> Dict[str, float]:
        """
        Construct an optimal tracking portfolio using optimization.
        
        Args:
            constituent_betas: Dictionary of constituent betas
            
        Returns:
            Dictionary mapping constituent symbols to weights
        """
        # Get returns data for optimization
        constituent_returns = {}
        
        for symbol in constituent_betas.keys():
            if symbol in self.constituent_returns:
                constituent_returns[symbol] = self.constituent_returns[symbol]
                
        # Check if we have sufficient data
        if len(constituent_returns) < self.tracking_portfolio_size:
            logger.warning(f"Insufficient constituents for optimization: {len(constituent_returns)} < {self.tracking_portfolio_size}")
            
        # Create returns DataFrame for all constituents
        returns_df = pd.DataFrame(constituent_returns)
        
        # Align with benchmark returns
        aligned_data = pd.concat([returns_df, self.benchmark_returns], axis=1).dropna()
        
        # Get constituent columns
        constituent_cols = [col for col in aligned_data.columns if col != 'benchmark_returns']
        
        # Define optimization variables
        n_constituents = len(constituent_cols)
        
        # Initial weights (equal weighted)
        initial_weights = np.ones(n_constituents) / n_constituents
        
        # Target beta from beta manager
        target_beta = self.beta_manager.determine_target_beta()
        
        # Define optimization objective function
        def objective(weights):
            # Calculate tracking portfolio returns
            portfolio_returns = aligned_data[constituent_cols] @ weights
            
            # Calculate tracking error
            tracking_error = np.std(portfolio_returns - self.benchmark_returns) * np.sqrt(252)
            
            # Calculate portfolio beta
            portfolio_beta = sum(weights[i] * constituent_betas.get(constituent_cols[i], 1.0) 
                               for i in range(n_constituents))
            
            # Beta error
            beta_error = (portfolio_beta - target_beta) ** 2
            
            # Objective: minimize tracking error and beta error
            return tracking_error + 10 * beta_error
            
        # Constraints
        constraints = [
            # Sum of weights = 1
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds (no short selling)
        bounds = [(0, 0.1) for _ in range(n_constituents)]
        
        # Run optimization
        from scipy.optimize import minimize
        
        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                # Create mapping of symbols to optimized weights
                optimized_weights = {constituent_cols[i]: result.x[i] for i in range(n_constituents)}
                
                # Reduce to desired portfolio size
                if len(optimized_weights) > self.tracking_portfolio_size:
                    # Keep top N by weight
                    sorted_weights = sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True)
                    top_constituents = sorted_weights[:self.tracking_portfolio_size]
                    
                    # Re-normalize weights
                    total_weight = sum(w for _, w in top_constituents)
                    optimized_weights = {symbol: weight / total_weight for symbol, weight in top_constituents}
                
                return optimized_weights
            else:
                logger.warning(f"Optimization failed: {result.message}")
                # Fall back to equal-weighted portfolio of constituents with beta closest to target
                return self._construct_sampling_tracking_portfolio(constituent_betas)
        except Exception as e:
            logger.error(f"Error in tracking portfolio optimization: {str(e)}")
            return self._construct_sampling_tracking_portfolio(constituent_betas)
    
    def _construct_sampling_tracking_portfolio(self, 
                                            constituent_betas: Dict[str, float]) -> Dict[str, float]:
        """
        Construct a tracking portfolio by sampling constituents.
        
        Args:
            constituent_betas: Dictionary of constituent betas
            
        Returns:
            Dictionary mapping constituent symbols to weights
        """
        # Get target beta
        target_beta = self.beta_manager.determine_target_beta()
        
        # Calculate beta distance for each constituent
        beta_distances = {symbol: abs(beta - target_beta) for symbol, beta in constituent_betas.items()}
        
        # Sort constituents by beta distance
        sorted_constituents = sorted(beta_distances.items(), key=lambda x: x[1])
        
        # Select top N constituents
        top_constituents = [symbol for symbol, _ in sorted_constituents[:self.tracking_portfolio_size]]
        
        # Equal weight the selected constituents
        equal_weight = 1.0 / len(top_constituents)
        tracking_portfolio = {symbol: equal_weight for symbol in top_constituents}
        
        return tracking_portfolio
    
    def _construct_factor_tracking_portfolio(self, 
                                           constituent_betas: Dict[str, float]) -> Dict[str, float]:
        """
        Construct a tracking portfolio using factor models.
        
        Args:
            constituent_betas: Dictionary of constituent betas
            
        Returns:
            Dictionary mapping constituent symbols to weights
        """
        # This is a placeholder implementation
        # In practice, would use a multi-factor model (e.g., Barra, Axioma)
        logger.warning("Factor-based tracking portfolio construction not fully implemented.")
        
        # For now, fall back to optimization-based approach
        return self._construct_optimal_tracking_portfolio(constituent_betas)
    
    def calculate_tracking_performance(self, 
                                     start_date: Optional[datetime] = None, 
                                     end_date: Optional[datetime] = None) -> Dict:
        """
        Calculate tracking portfolio performance metrics.
        
        Args:
            start_date: Start date for analysis period
            end_date: End date for analysis period
            
        Returns:
            Dictionary with tracking performance metrics
        """
        if not self.tracking_portfolio:
            logger.error("No tracking portfolio available")
            return {}
            
        # Get constituent returns
        tracking_returns = self._calculate_tracking_portfolio_returns()
        
        if tracking_returns is None:
            logger.error("Could not calculate tracking portfolio returns")
            return {}
            
        # Align with benchmark returns
        aligned_data = pd.DataFrame({
            'tracking': tracking_returns,
            'benchmark': self.benchmark_returns
        }).dropna()
        
        # Limit to specified date range
        if start_date is not None or end_date is not None:
            if start_date is not None:
                aligned_data = aligned_data.loc[aligned_data.index >= start_date]
            if end_date is not None:
                aligned_data = aligned_data.loc[aligned_data.index <= end_date]
        
        if aligned_data.empty:
            logger.error("No common date range for tracking and benchmark returns")
            return {}
            
        # Store aligned returns
        self.portfolio_returns = aligned_data['tracking']
        
        # Calculate tracking error
        tracking_error = np.std(aligned_data['tracking'] - aligned_data['benchmark']) * np.sqrt(252)
        self.tracking_error = tracking_error
        
        # Calculate portfolio beta
        X = sm.add_constant(aligned_data['benchmark'])
        model = sm.OLS(aligned_data['tracking'], X)
        results = model.fit()
        beta = results.params['benchmark']
        self.portfolio_beta = beta
        
        # Calculate alpha
        alpha = results.params['const'] * 252  # Annualized
        
        # Calculate R-squared
        r_squared = results.rsquared
        
        # Calculate correlation
        correlation = aligned_data['tracking'].corr(aligned_data['benchmark'])
        
        # Calculate total returns
        tracking_total_return = (1 + aligned_data['tracking']).prod() - 1
        benchmark_total_return = (1 + aligned_data['benchmark']).prod() - 1
        
        # Calculate annualized returns
        years = len(aligned_data) / 252
        if years > 0:
            tracking_annual_return = (1 + tracking_total_return) ** (1 / years) - 1
            benchmark_annual_return = (1 + benchmark_total_return) ** (1 / years) - 1
        else:
            tracking_annual_return = tracking_total_return
            benchmark_annual_return = benchmark_total_return
            
        # Calculate volatility
        tracking_volatility = aligned_data['tracking'].std() * np.sqrt(252)
        benchmark_volatility = aligned_data['benchmark'].std() * np.sqrt(252)
        
        # Calculate information ratio
        information_ratio = (tracking_annual_return - benchmark_annual_return) / tracking_error if tracking_error > 0 else 0
        
        # Calculate active share if we have constituent-level data
        active_share = self._calculate_active_share()
        
        # Compile results
        performance = {
            'tracking_error': tracking_error,
            'beta': beta,
            'alpha': alpha,
            'r_squared': r_squared,
            'correlation': correlation,
            'tracking_total_return': tracking_total_return,
            'benchmark_total_return': benchmark_total_return,
            'tracking_annual_return': tracking_annual_return,
            'benchmark_annual_return': benchmark_annual_return,
            'tracking_volatility': tracking_volatility,
            'benchmark_volatility': benchmark_volatility,
            'information_ratio': information_ratio,
            'active_share': active_share
        }
        
        logger.info(f"Calculated tracking performance: TE={tracking_error:.4f}, Beta={beta:.4f}, IR={information_ratio:.4f}")
        return performance
    
    def _calculate_tracking_portfolio_returns(self) -> Optional[pd.Series]:
        """
        Calculate returns for the tracking portfolio.
        
        Returns:
            Series of tracking portfolio returns
        """
        if not self.tracking_portfolio:
            logger.error("No tracking portfolio available")
            return None
            
        # Get constituent returns
        constituent_returns_data = {}
        
        for symbol, weight in self.tracking_portfolio.items():
            if symbol in self.constituent_returns:
                constituent_returns_data[symbol] = self.constituent_returns[symbol]
                
        if not constituent_returns_data:
            logger.error("No return data available for tracking portfolio constituents")
            return None
            
        # Create returns DataFrame
        returns_df = pd.DataFrame(constituent_returns_data)
        
        # Calculate weighted returns
        weights = [self.tracking_portfolio.get(col, 0.0) for col in returns_df.columns]
        tracking_returns = returns_df.dot(weights)
        
        return tracking_returns
    
    def _calculate_active_share(self) -> float:
        """
        Calculate active share of the tracking portfolio.
        
        Returns:
            Active share (0-1 scale)
        """
        # This is a simplified implementation
        # In practice, would need benchmark constituent weights
        
        # Mock benchmark weights (equal weight for simplicity)
        # In practice, would use actual benchmark constituent weights
        benchmark_constituents = list(self.constituent_data.keys())
        benchmark_weights = {symbol: 1.0 / len(benchmark_constituents) for symbol in benchmark_constituents}
        
        # Calculate active share
        active_share = 0.0
        
        for symbol in set(list(benchmark_weights.keys()) + list(self.tracking_portfolio.keys())):
            benchmark_weight = benchmark_weights.get(symbol, 0.0)
            portfolio_weight = self.tracking_portfolio.get(symbol, 0.0)
            active_share += abs(portfolio_weight - benchmark_weight)
            
        # Active share is half the sum of absolute weight differences
        active_share = active_share / 2.0
        
        return active_share
    
    def rebalance_tracking_portfolio(self, 
                                   target_beta: Optional[float] = None,
                                   target_date: Optional[datetime] = None,
                                   method: str = 'optimization') -> Dict[str, float]:
        """
        Rebalance the tracking portfolio to achieve target beta.
        
        Args:
            target_beta: Target beta (if None, uses beta manager's target)
            target_date: Target date for rebalancing
            method: Method for portfolio construction
            
        Returns:
            Dictionary with rebalanced weights
        """
        # Set target beta
        if target_beta is not None:
            # Override the beta manager's target
            original_target_strategy = self.beta_manager.beta_target_strategy
            self.beta_manager.beta_target_strategy = BetaTarget.CUSTOM
            self.beta_manager.custom_target_beta = target_beta
            
        # Calculate constituent betas
        constituent_betas = self._calculate_constituent_betas(target_date)
        
        # Construct new tracking portfolio
        new_portfolio = self.construct_tracking_portfolio(
            method=method,
            target_date=target_date,
            constituent_betas=constituent_betas
        )
        
        # Reset beta manager's target strategy if it was changed
        if target_beta is not None:
            self.beta_manager.beta_target_strategy = original_target_strategy
            
        logger.info(f"Rebalanced tracking portfolio with {len(new_portfolio)} constituents")
        return new_portfolio
    
    def generate_tracking_report(self, 
                               output_path: Optional[str] = None,
                               include_plots: bool = True) -> Dict:
        """
        Generate a comprehensive benchmark tracking report.
        
        Args:
            output_path: Path to save report files
            include_plots: Whether to generate plots
            
        Returns:
            Dictionary with report data
        """
        # Calculate tracking performance if not done yet
        if not hasattr(self, 'tracking_error') or self.tracking_error is None:
            performance = self.calculate_tracking_performance()
        else:
            performance = {
                'tracking_error': self.tracking_error,
                'beta': self.portfolio_beta
            }
            
        # Get beta recommendation from beta manager
        beta_recommendation = self.beta_manager.get_beta_hedging_recommendation(
            current_beta=self.portfolio_beta,
            target_beta=self.beta_manager.determine_target_beta()
        )
        
        # Prepare report data
        report = {
            'performance': performance,
            'beta_recommendation': beta_recommendation,
            'portfolio': self.tracking_portfolio,
            'benchmark_symbol': self.benchmark_symbol
        }
        
        # Generate plots if requested
        if include_plots and self.portfolio_returns is not None:
            plots = {}
            
            # Performance comparison plot
            fig_performance = plt.figure(figsize=(12, 6))
            
            # Calculate cumulative returns
            portfolio_cumulative = (1 + self.portfolio_returns).cumprod()
            benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
            
            # Align dates
            common_index = portfolio_cumulative.index.intersection(benchmark_cumulative.index)
            portfolio_cumulative = portfolio_cumulative.loc[common_index]
            benchmark_cumulative = benchmark_cumulative.loc[common_index]
            
            # Normalize to start at 1.0
            portfolio_cumulative = portfolio_cumulative / portfolio_cumulative.iloc[0]
            benchmark_cumulative = benchmark_cumulative / benchmark_cumulative.iloc[0]
            
            # Plot cumulative returns
            plt.plot(portfolio_cumulative.index, portfolio_cumulative, 
                   label='Tracking Portfolio', linewidth=2)
            plt.plot(benchmark_cumulative.index, benchmark_cumulative, 
                   label=f'Benchmark ({self.benchmark_symbol})', linewidth=2)
            
            plt.title('Tracking Portfolio vs Benchmark Performance')
            plt.xlabel('Date')
            plt.ylabel('Growth of $1')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plots['performance_comparison'] = fig_performance
            
            # Beta evolution plot
            if hasattr(self.beta_manager, 'historical_betas') and self.beta_manager.historical_betas is not None:
                fig_beta = plt.figure(figsize=(12, 6))
                
                plt.plot(self.beta_manager.historical_betas.index, 
                       self.beta_manager.historical_betas, label='Portfolio Beta', linewidth=2)
                
                # Add target beta line
                target_beta = self.beta_manager.determine_target_beta()
                plt.axhline(y=target_beta, color='r', linestyle='--', 
                          label=f'Target Beta ({target_beta:.2f})')
                
                plt.title('Portfolio Beta Evolution')
                plt.xlabel('Date')
                plt.ylabel('Beta')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plots['beta_evolution'] = fig_beta
            
            # Tracking error plot
            if len(self.portfolio_returns) > 30:
                fig_te = plt.figure(figsize=(12, 6))
                
                # Calculate rolling tracking error
                rolling_te = (self.portfolio_returns - self.benchmark_returns).rolling(window=63).std() * np.sqrt(252)
                
                plt.plot(rolling_te.index, rolling_te, label='Rolling Tracking Error (63-day)', linewidth=2)
                plt.axhline(y=self.tracking_error, color='r', linestyle='--', 
                          label=f'Full-Period TE ({self.tracking_error:.4f})')
                
                plt.title('Tracking Error Evolution')
                plt.xlabel('Date')
                plt.ylabel('Annualized Tracking Error')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plots['tracking_error'] = fig_te
                
            # Portfolio composition
            if self.tracking_portfolio:
                fig_composition = plt.figure(figsize=(10, 6))
                
                # Sort weights for better visualization
                sorted_weights = sorted(self.tracking_portfolio.items(), key=lambda x: x[1], reverse=True)
                symbols = [s for s, _ in sorted_weights]
                weights = [w for _, w in sorted_weights]
                
                plt.bar(symbols, weights)
                plt.xticks(rotation=90)
                plt.title('Tracking Portfolio Composition')
                plt.xlabel('Constituent')
                plt.ylabel('Weight')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                plots['portfolio_composition'] = fig_composition
            
            # Save plots if output path provided
            if output_path is not None:
                # Create directory if it doesn't exist
                import os
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                    
                # Save plots
                for name, fig in plots.items():
                    plot_path = os.path.join(output_path, f"{name}.png")
                    fig.savefig(plot_path)
                    plt.close(fig)
                    
                    # Add file paths to report
                    if 'plot_files' not in report:
                        report['plot_files'] = {}
                        
                    report['plot_files'][name] = plot_path
                    logger.info(f"Saved {name} plot to {plot_path}")
            
            # Add plots to report
            report['plots'] = plots
            
        # Save report as JSON if output path provided
        if output_path is not None:
            import json
            import os
            
            # Create a JSON-serializable version of the report
            json_report = {
                'performance': {k: float(v) if isinstance(v, (float, np.floating)) else v 
                              for k, v in performance.items()},
                'beta_recommendation': beta_recommendation,
                'benchmark_symbol': self.benchmark_symbol,
                'portfolio_size': len(self.tracking_portfolio),
                'analysis_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            if 'plot_files' in report:
                json_report['plot_files'] = report['plot_files']
                
            report_path = os.path.join(output_path, "tracking_report.json")
            
            with open(report_path, 'w') as f:
                json.dump(json_report, f, indent=2)
                
            report['report_file'] = report_path
            logger.info(f"Saved tracking report to {report_path}")
        
        return report