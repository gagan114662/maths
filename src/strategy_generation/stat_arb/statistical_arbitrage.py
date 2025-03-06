"""
Statistical Arbitrage Module for Market-Neutral Performance.

This module provides advanced statistical arbitrage strategies for creating
market-neutral trading portfolios that aim to generate alpha regardless
of overall market direction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import uuid
import json

logger = logging.getLogger(__name__)

class StatisticalArbitrageStrategy:
    """
    Base class for statistical arbitrage strategies.
    
    This class provides core functionality shared by all statistical
    arbitrage strategies, including position sizing, risk management,
    and performance tracking.
    """
    
    def __init__(self, 
                strategy_name: str = None,
                target_volatility: float = 0.10,  # 10% annualized vol
                max_leverage: float = 2.0,
                market_neutrality_target: float = 0.05,  # Maximum acceptable beta to market
                transaction_cost: float = 0.0005):  # 5 bps per trade
        """
        Initialize the statistical arbitrage strategy.
        
        Args:
            strategy_name: Name of the strategy
            target_volatility: Target annualized volatility for the strategy
            max_leverage: Maximum leverage (e.g., 2.0 = 200%)
            market_neutrality_target: Maximum acceptable beta to market
            transaction_cost: Transaction cost per trade (one-way)
        """
        self.strategy_name = strategy_name or f"StatArb_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.market_neutrality_target = market_neutrality_target
        self.transaction_cost = transaction_cost
        
        # Strategy state
        self.current_positions = {}  # Symbol -> position size (as % of capital)
        self.market_exposure = 0.0
        self.position_history = []
        self.performance_history = {}
        self.trade_history = []
        
        # Analysis metrics
        self.current_beta = 0.0
        self.current_volatility = 0.0
        self.total_leverage = 0.0
        
        logger.info(f"Initialized {self.strategy_name} with target vol: {target_volatility}, max leverage: {max_leverage}")
    
    def calculate_position_sizes(self, 
                              raw_signals: Dict[str, float],
                              asset_volatilities: Dict[str, float],
                              asset_correlations: Optional[pd.DataFrame] = None,
                              market_betas: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate position sizes based on signals and risk parameters.
        
        Args:
            raw_signals: Dictionary mapping symbols to raw alpha signals
            asset_volatilities: Dictionary mapping symbols to annualized volatilities
            asset_correlations: Optional correlation matrix between assets
            market_betas: Optional dictionary mapping symbols to market betas
            
        Returns:
            Dictionary mapping symbols to position sizes (as % of capital)
        """
        # Filter out assets with no volatility data
        valid_assets = {k: v for k, v in raw_signals.items() if k in asset_volatilities and asset_volatilities[k] > 0}
        
        if not valid_assets:
            logger.warning("No valid assets with volatility data")
            return {}
            
        # Calculate inverse-volatility weights (initial risk budgeting)
        vol_weights = {}
        total_inv_vol = 0.0
        
        for symbol, signal in valid_assets.items():
            vol = asset_volatilities[symbol]
            # Scale by signal direction and strength
            inv_vol = abs(signal) / vol if vol > 0 else 0
            vol_weights[symbol] = inv_vol
            total_inv_vol += inv_vol
            
        # Normalize weights
        if total_inv_vol > 0:
            norm_weights = {k: v / total_inv_vol for k, v in vol_weights.items()}
        else:
            logger.warning("Zero total inverse volatility")
            return {}
            
        # Adjust for market neutrality if beta data is provided
        if market_betas is not None:
            # Calculate initial market exposure
            market_exposure = sum(norm_weights[s] * signal_direction(raw_signals[s]) * market_betas.get(s, 0)
                               for s in norm_weights if s in market_betas)
            
            if abs(market_exposure) > self.market_neutrality_target:
                # Need to adjust for market neutrality
                logger.info(f"Adjusting for market neutrality: Initial exposure {market_exposure:.3f}")
                
                # Simple approach: find assets to offset the exposure
                offset_candidates = {s: market_betas[s] for s in norm_weights 
                                  if s in market_betas and market_betas[s] * market_exposure > 0}
                
                if offset_candidates:
                    # Sort by highest beta (absolute value)
                    sorted_offsets = sorted(offset_candidates.items(), key=lambda x: -abs(x[1]))
                    
                    # Adjust weights to reduce market exposure
                    for symbol, beta in sorted_offsets:
                        # Reduce weight of this asset to decrease market exposure
                        exposure_reduction = min(norm_weights[symbol], abs(market_exposure / beta) * 0.5)
                        norm_weights[symbol] -= exposure_reduction
                        
                        # Recalculate market exposure
                        market_exposure = sum(norm_weights[s] * signal_direction(raw_signals[s]) * market_betas.get(s, 0)
                                           for s in norm_weights if s in market_betas)
                        
                        if abs(market_exposure) <= self.market_neutrality_target:
                            break
                            
                    # Re-normalize weights
                    total_weight = sum(norm_weights.values())
                    if total_weight > 0:
                        norm_weights = {k: v / total_weight for k, v in norm_weights.items()}
                        
        # Scale to reach target volatility, accounting for correlations if provided
        if asset_correlations is not None:
            # Calculate portfolio variance with correlations
            portfolio_var = 0.0
            symbols = list(norm_weights.keys())
            
            for i, symbol_i in enumerate(symbols):
                for j, symbol_j in enumerate(symbols):
                    if symbol_i in asset_correlations.index and symbol_j in asset_correlations.columns:
                        corr_ij = asset_correlations.loc[symbol_i, symbol_j]
                        vol_i = asset_volatilities[symbol_i]
                        vol_j = asset_volatilities[symbol_j]
                        weight_i = norm_weights[symbol_i] * signal_direction(raw_signals[symbol_i])
                        weight_j = norm_weights[symbol_j] * signal_direction(raw_signals[symbol_j])
                        
                        portfolio_var += weight_i * weight_j * vol_i * vol_j * corr_ij
            
            # Convert to annualized volatility
            portfolio_vol = np.sqrt(portfolio_var)
        else:
            # Simplified approach: assume assets are uncorrelated
            portfolio_var = sum((norm_weights[s] * asset_volatilities[s])**2 for s in norm_weights)
            portfolio_vol = np.sqrt(portfolio_var)
            
        # Calculate volatility scaling factor
        if portfolio_vol > 0:
            vol_scalar = self.target_volatility / portfolio_vol
        else:
            logger.warning("Zero portfolio volatility, cannot scale to target")
            vol_scalar = 1.0
            
        # Apply volatility scaling, honoring max leverage
        total_long = sum(norm_weights[s] for s in norm_weights if raw_signals[s] > 0)
        total_short = sum(norm_weights[s] for s in norm_weights if raw_signals[s] < 0)
        total_exposure = total_long + total_short
        
        leverage_scalar = min(vol_scalar, self.max_leverage / total_exposure) if total_exposure > 0 else 0
        
        # Calculate final position sizes
        position_sizes = {}
        for symbol, weight in norm_weights.items():
            position_sizes[symbol] = weight * leverage_scalar * signal_direction(raw_signals[symbol])
            
        # Store current strategy metrics
        self.current_volatility = portfolio_vol
        self.total_leverage = sum(abs(pos) for pos in position_sizes.values())
        
        if market_betas:
            self.current_beta = sum(position_sizes[s] * market_betas.get(s, 0) 
                                 for s in position_sizes if s in market_betas)
            
        logger.info(f"Calculated positions with leverage: {self.total_leverage:.2f}, beta: {self.current_beta:.3f}")
        
        return position_sizes
    
    def update_positions(self, 
                       new_positions: Dict[str, float],
                       current_prices: Dict[str, float],
                       trade_date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Update strategy positions, accounting for transaction costs.
        
        Args:
            new_positions: Dictionary mapping symbols to new position sizes
            current_prices: Dictionary mapping symbols to current prices
            trade_date: Trade date (defaults to current datetime)
            
        Returns:
            Dictionary mapping symbols to executed trades (as % of capital)
        """
        trade_date = trade_date or datetime.now()
        executed_trades = {}
        
        # Calculate trades required
        for symbol in set(list(self.current_positions.keys()) + list(new_positions.keys())):
            current_size = self.current_positions.get(symbol, 0.0)
            new_size = new_positions.get(symbol, 0.0)
            
            trade_size = new_size - current_size
            
            if abs(trade_size) > 0.001:  # Minimum trade size (0.1% of capital)
                executed_trades[symbol] = trade_size
                
                # Apply transaction costs
                if symbol in current_prices:
                    transaction_cost = abs(trade_size) * self.transaction_cost
                    
                    # Record trade with details
                    self.trade_history.append({
                        'date': trade_date,
                        'symbol': symbol,
                        'trade_size': trade_size,
                        'price': current_prices[symbol],
                        'transaction_cost': transaction_cost
                    })
                    
        # Update current positions
        self.current_positions = {k: v for k, v in new_positions.items() if abs(v) > 0.001}
        
        # Record position snapshot
        position_snapshot = {
            'date': trade_date,
            'positions': self.current_positions.copy(),
            'leverage': self.total_leverage,
            'beta': self.current_beta,
            'volatility': self.current_volatility
        }
        
        self.position_history.append(position_snapshot)
        
        return executed_trades
    
    def calculate_returns(self, 
                        prices: Dict[str, pd.Series],
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.Series:
        """
        Calculate strategy returns based on position history and price data.
        
        Args:
            prices: Dictionary mapping symbols to price series
            start_date: Start date for return calculation
            end_date: End date for return calculation
            
        Returns:
            Series of strategy returns
        """
        if not self.position_history:
            logger.warning("No position history available")
            return pd.Series()
            
        # Convert position history to DataFrame with dates as index
        positions_df = pd.DataFrame([
            {'date': p['date'], **{f"pos_{k}": v for k, v in p['positions'].items()}}
            for p in self.position_history
        ])
        
        if positions_df.empty:
            return pd.Series()
            
        positions_df.set_index('date', inplace=True)
        
        # Filter by date range if specified
        if start_date:
            positions_df = positions_df[positions_df.index >= start_date]
        if end_date:
            positions_df = positions_df[positions_df.index <= end_date]
            
        if positions_df.empty:
            return pd.Series()
            
        # Get all symbols from position history
        symbols = set()
        for p in self.position_history:
            symbols.update(p['positions'].keys())
            
        # Create a unified DataFrame with prices
        price_data = {}
        for symbol in symbols:
            if symbol in prices:
                price_data[symbol] = prices[symbol]
                
        if not price_data:
            logger.warning("No price data available for positions")
            return pd.Series()
            
        price_df = pd.DataFrame(price_data)
        
        # Align dates and calculate returns
        aligned_dates = positions_df.index.intersection(price_df.index)
        positions_df = positions_df.loc[aligned_dates]
        price_df = price_df.loc[aligned_dates]
        
        if len(aligned_dates) < 2:
            logger.warning("Insufficient aligned data for return calculation")
            return pd.Series()
            
        # Calculate price returns
        price_returns = price_df.pct_change().fillna(0)
        
        # Calculate strategy returns for each day
        strategy_returns = pd.Series(0.0, index=aligned_dates)
        
        for symbol in symbols:
            pos_col = f"pos_{symbol}"
            if pos_col in positions_df.columns and symbol in price_returns.columns:
                # Use shifted positions (yesterday's positions) with today's returns
                lagged_positions = positions_df[pos_col].shift(1).fillna(0)
                strategy_returns += lagged_positions * price_returns[symbol]
                
        # Apply transaction costs from trade history
        trade_costs = {}
        for trade in self.trade_history:
            date = trade['date']
            if date in strategy_returns.index:
                if date not in trade_costs:
                    trade_costs[date] = 0.0
                trade_costs[date] += trade['transaction_cost']
                
        for date, cost in trade_costs.items():
            strategy_returns[date] -= cost
            
        # Store performance history
        self.performance_history['returns'] = strategy_returns
        
        if len(strategy_returns) > 0:
            # Calculate cumulative performance
            self.performance_history['cumulative_returns'] = (1 + strategy_returns).cumprod() - 1
            
            # Calculate key metrics
            annualized_return = ((1 + strategy_returns).prod()) ** (252 / len(strategy_returns)) - 1
            annualized_volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(strategy_returns)
            
            self.performance_history['metrics'] = {
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': (1 + strategy_returns).prod() - 1,
                'win_rate': (strategy_returns > 0).mean()
            }
            
        return strategy_returns
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown from returns.
        
        Args:
            returns: Series of returns
            
        Returns:
            Maximum drawdown value (as a positive percentage)
        """
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max - 1)
        max_drawdown = abs(drawdown.min())
        return max_drawdown
    
    def plot_performance(self, 
                       benchmark_returns: Optional[pd.Series] = None,
                       figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot strategy performance.
        
        Args:
            benchmark_returns: Optional benchmark returns for comparison
            figsize: Figure size
            
        Returns:
            Matplotlib figure with performance visualization
        """
        if 'returns' not in self.performance_history:
            logger.warning("No performance history available")
            return None
            
        returns = self.performance_history['returns']
        cum_returns = self.performance_history['cumulative_returns']
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[2, 1, 1], sharex=True)
        
        # Plot cumulative returns
        axes[0].plot(cum_returns.index, cum_returns, label=self.strategy_name)
        
        if benchmark_returns is not None:
            # Align benchmark returns
            aligned_benchmark = benchmark_returns.loc[benchmark_returns.index.intersection(returns.index)]
            if not aligned_benchmark.empty:
                cum_benchmark = (1 + aligned_benchmark).cumprod() - 1
                axes[0].plot(cum_benchmark.index, cum_benchmark, label='Benchmark', alpha=0.7)
                
        axes[0].set_title(f"{self.strategy_name} Performance")
        axes[0].set_ylabel("Cumulative Return")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot drawdowns
        drawdowns = 1 - (1 + cum_returns) / (1 + cum_returns).cummax()
        axes[1].fill_between(drawdowns.index, 0, -drawdowns, color='red', alpha=0.3)
        axes[1].set_title("Drawdowns")
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(True, alpha=0.3)
        
        # Plot rolling metrics
        if len(returns) > 60:  # Need enough data for rolling window
            # 3-month rolling volatility
            rolling_vol = returns.rolling(window=63).std() * np.sqrt(252)
            axes[2].plot(rolling_vol.index, rolling_vol, label='Rolling Volatility (3m)', color='orange')
            
            # Add target volatility line
            axes[2].axhline(y=self.target_volatility, color='red', linestyle='--', 
                          label=f'Target Vol ({self.target_volatility:.1%})')
                          
            # Rolling 3-month Sharpe ratio
            rolling_sharpe = (returns.rolling(window=63).mean() * 252) / (returns.rolling(window=63).std() * np.sqrt(252))
            axes_twin = axes[2].twinx()
            axes_twin.plot(rolling_sharpe.index, rolling_sharpe, label='Rolling Sharpe (3m)', color='green')
            axes_twin.set_ylabel("Sharpe Ratio")
            
            # Add legend
            lines1, labels1 = axes[2].get_legend_handles_labels()
            lines2, labels2 = axes_twin.get_legend_handles_labels()
            axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
        axes[2].set_title("Rolling Metrics")
        axes[2].set_ylabel("Volatility")
        axes[2].set_xlabel("Date")
        axes[2].grid(True, alpha=0.3)
        
        # Add performance metrics as text
        if 'metrics' in self.performance_history:
            metrics = self.performance_history['metrics']
            metrics_text = (
                f"Return: {metrics['annualized_return']:.2%} (ann.)\n"
                f"Volatility: {metrics['annualized_volatility']:.2%} (ann.)\n"
                f"Sharpe: {metrics['sharpe_ratio']:.2f}\n"
                f"Max DD: {metrics['max_drawdown']:.2%}\n"
                f"Win Rate: {metrics['win_rate']:.2%}"
            )
            
            axes[0].text(0.02, 0.05, metrics_text, transform=axes[0].transAxes,
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                      verticalalignment='bottom')
                      
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive strategy report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Dictionary with strategy report data
        """
        if 'metrics' not in self.performance_history:
            logger.warning("No performance metrics available for report")
            return {}
            
        # Calculate exposure statistics
        long_exposure = []
        short_exposure = []
        net_exposure = []
        gross_exposure = []
        
        for snapshot in self.position_history:
            positions = snapshot['positions']
            
            long_pos = sum(pos for pos in positions.values() if pos > 0)
            short_pos = sum(abs(pos) for pos in positions.values() if pos < 0)
            
            long_exposure.append(long_pos)
            short_exposure.append(short_pos)
            net_exposure.append(long_pos - short_pos)
            gross_exposure.append(long_pos + short_pos)
            
        # Calculate position concentration
        concentration = []
        for snapshot in self.position_history:
            positions = snapshot['positions']
            if positions:
                # Herfindahl-Hirschman Index (HHI)
                weights = [abs(p) for p in positions.values()]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    norm_weights = [w / total_weight for w in weights]
                    hhi = sum(w**2 for w in norm_weights)
                    concentration.append(hhi)
                    
        # Generate report
        report = {
            'strategy_name': self.strategy_name,
            'performance_metrics': self.performance_history.get('metrics', {}),
            'exposure_metrics': {
                'mean_long_exposure': np.mean(long_exposure) if long_exposure else 0,
                'mean_short_exposure': np.mean(short_exposure) if short_exposure else 0,
                'mean_net_exposure': np.mean(net_exposure) if net_exposure else 0,
                'mean_gross_exposure': np.mean(gross_exposure) if gross_exposure else 0,
                'max_leverage': max(gross_exposure) if gross_exposure else 0,
                'mean_concentration': np.mean(concentration) if concentration else 0
            },
            'risk_metrics': {
                'mean_volatility': np.mean([s.get('volatility', 0) for s in self.position_history]),
                'mean_beta': np.mean([s.get('beta', 0) for s in self.position_history]),
                'max_drawdown': self.performance_history['metrics']['max_drawdown'],
                'worst_day_return': min(self.performance_history['returns']) if 'returns' in self.performance_history else 0,
                'best_day_return': max(self.performance_history['returns']) if 'returns' in self.performance_history else 0
            },
            'trade_metrics': {
                'total_trades': len(self.trade_history),
                'total_transaction_costs': sum(t['transaction_cost'] for t in self.trade_history),
                'average_trade_size': np.mean([abs(t['trade_size']) for t in self.trade_history]) if self.trade_history else 0
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save report if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Saved strategy report to {output_path}")
            
        return report

class PairsTradingStrategy(StatisticalArbitrageStrategy):
    """
    Pairs Trading Statistical Arbitrage Strategy.
    
    This strategy identifies cointegrated pairs of assets and trades the spread
    between them when it deviates significantly from its historical mean.
    """
    
    def __init__(self, 
                entry_z_score: float = 2.0,
                exit_z_score: float = 0.5,
                stop_loss_z_score: float = 4.0,
                lookback_period: int = 252,
                min_half_life: int = 5,
                max_half_life: int = 100,
                **kwargs):
        """
        Initialize the pairs trading strategy.
        
        Args:
            entry_z_score: Z-score threshold for entering positions
            exit_z_score: Z-score threshold for exiting positions
            stop_loss_z_score: Z-score threshold for stop loss
            lookback_period: Lookback period for calculating spread statistics
            min_half_life: Minimum acceptable half-life for mean reversion
            max_half_life: Maximum acceptable half-life for mean reversion
            **kwargs: Additional arguments for the base class
        """
        # Set defaults for market-neutral strategy
        kwargs.setdefault('market_neutrality_target', 0.05)
        kwargs.setdefault('target_volatility', 0.08)
        kwargs.setdefault('strategy_name', f"PairsTrading_{datetime.now().strftime('%Y%m%d_%H%M')}")
        
        super().__init__(**kwargs)
        
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.stop_loss_z_score = stop_loss_z_score
        self.lookback_period = lookback_period
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        
        # Strategy-specific state
        self.pairs = []  # List of cointegrated pairs
        self.spread_stats = {}  # Statistics for each pair
        self.active_pairs = {}  # Currently active trading pairs
        self.pair_history = []  # History of pairs analysis
        
    def find_cointegrated_pairs(self, 
                             prices: Dict[str, pd.Series],
                             min_correlation: float = 0.5,
                             p_value_threshold: float = 0.05,
                             min_observations: int = 252) -> List[Dict]:
        """
        Find cointegrated pairs suitable for pairs trading.
        
        Args:
            prices: Dictionary mapping symbols to price series
            min_correlation: Minimum absolute correlation between assets
            p_value_threshold: Maximum p-value for cointegration test
            min_observations: Minimum required observations
            
        Returns:
            List of dictionaries with cointegrated pair information
        """
        # Filter series with enough data
        valid_prices = {k: v for k, v in prices.items() if len(v) >= min_observations}
        
        if len(valid_prices) < 2:
            logger.warning("Insufficient data for pairs analysis")
            return []
            
        # Align dates
        price_df = pd.DataFrame(valid_prices)
        price_df = price_df.dropna()
        
        if len(price_df) < min_observations:
            logger.warning(f"Insufficient aligned data: {len(price_df)} < {min_observations}")
            return []
            
        # Calculate correlations
        returns_df = price_df.pct_change().dropna()
        corr_matrix = returns_df.corr()
        
        # Find candidate pairs with sufficient correlation
        candidate_pairs = []
        
        for i, symbol1 in enumerate(price_df.columns):
            for j, symbol2 in enumerate(price_df.columns):
                if i >= j:  # Avoid duplicates and self-pairs
                    continue
                    
                correlation = corr_matrix.loc[symbol1, symbol2]
                
                if abs(correlation) >= min_correlation:
                    candidate_pairs.append((symbol1, symbol2, correlation))
                    
        logger.info(f"Found {len(candidate_pairs)} candidate pairs with min correlation {min_correlation}")
        
        # Test for cointegration
        cointegrated_pairs = []
        
        for symbol1, symbol2, correlation in candidate_pairs:
            # Perform Engle-Granger cointegration test
            y = price_df[symbol1].values
            x = price_df[symbol2].values
            
            result = sm.tsa.stattools.coint(y, x)
            
            p_value = result[1]
            
            if p_value <= p_value_threshold:
                # Calculate hedge ratio
                model = sm.OLS(y, sm.add_constant(x))
                results = model.fit()
                
                hedge_ratio = results.params[1]
                
                # Calculate spread
                spread = price_df[symbol1] - hedge_ratio * price_df[symbol2]
                
                # Test for stationarity of the spread
                adf_result = sm.tsa.stattools.adfuller(spread)
                adf_p_value = adf_result[1]
                
                # Calculate half-life of mean reversion
                spread_lag = spread.shift(1)
                spread_diff = spread - spread_lag
                
                # Remove NaN values
                spread_lag = spread_lag.dropna()
                spread_diff = spread_diff.dropna()
                
                if len(spread_lag) > 0:
                    # Regress spread difference on lagged spread level
                    model = sm.OLS(spread_diff.iloc[1:], sm.add_constant(spread_lag.iloc[1:]))
                    results = model.fit()
                    
                    # Extract mean reversion parameter
                    mean_reversion = results.params[1]
                    
                    # Calculate half-life if mean-reverting
                    if mean_reversion < 0:
                        half_life = int(round(-np.log(2) / mean_reversion))
                    else:
                        half_life = float('inf')
                        
                    # Only include pairs with suitable half-life
                    if self.min_half_life <= half_life <= self.max_half_life:
                        # Calculate spread statistics
                        spread_mean = float(spread.mean())
                        spread_std = float(spread.std())
                        
                        cointegrated_pairs.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': correlation,
                            'hedge_ratio': hedge_ratio,
                            'p_value': p_value,
                            'adf_p_value': adf_p_value,
                            'half_life': half_life,
                            'spread_mean': spread_mean,
                            'spread_std': spread_std,
                            'last_update': datetime.now()
                        })
                        
        # Sort pairs by half-life (lower is better)
        cointegrated_pairs.sort(key=lambda x: x['half_life'])
        
        logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs with suitable half-life")
        
        self.pairs = cointegrated_pairs
        self.pair_history.append({
            'date': datetime.now(),
            'pairs_count': len(cointegrated_pairs),
            'pairs': cointegrated_pairs
        })
        
        return cointegrated_pairs
    
    def update_spread_statistics(self, 
                              prices: Dict[str, pd.Series],
                              recalculate_pairs: bool = False,
                              min_correlation: float = 0.5,
                              p_value_threshold: float = 0.05) -> Dict[str, Dict]:
        """
        Update spread statistics for all pairs.
        
        Args:
            prices: Dictionary mapping symbols to price series
            recalculate_pairs: Whether to recalculate cointegrated pairs
            min_correlation: Minimum correlation for new pairs
            p_value_threshold: Maximum p-value for cointegration test
            
        Returns:
            Dictionary mapping pair IDs to spread statistics
        """
        if recalculate_pairs or not self.pairs:
            self.find_cointegrated_pairs(
                prices=prices,
                min_correlation=min_correlation,
                p_value_threshold=p_value_threshold
            )
            
        if not self.pairs:
            logger.warning("No pairs available")
            return {}
            
        # Update spread statistics for each pair
        spread_stats = {}
        
        for pair in self.pairs:
            symbol1 = pair['symbol1']
            symbol2 = pair['symbol2']
            
            if symbol1 not in prices or symbol2 not in prices:
                continue
                
            # Align dates
            price1 = prices[symbol1]
            price2 = prices[symbol2]
            
            aligned_dates = price1.index.intersection(price2.index)
            
            if len(aligned_dates) < self.lookback_period:
                logger.warning(f"Insufficient aligned data for {symbol1}/{symbol2}")
                continue
                
            # Calculate spread using most recent lookback_period data points
            aligned_price1 = price1.loc[aligned_dates][-self.lookback_period:]
            aligned_price2 = price2.loc[aligned_dates][-self.lookback_period:]
            
            hedge_ratio = pair['hedge_ratio']
            spread = aligned_price1 - hedge_ratio * aligned_price2
            
            # Calculate spread statistics
            spread_mean = float(spread.mean())
            spread_std = float(spread.std())
            
            # Create pair ID
            pair_id = f"{symbol1}_{symbol2}"
            
            # Store stats
            spread_stats[pair_id] = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'hedge_ratio': hedge_ratio,
                'current_spread': float(spread.iloc[-1]),
                'mean': spread_mean,
                'std': spread_std,
                'z_score': (float(spread.iloc[-1]) - spread_mean) / spread_std if spread_std > 0 else 0,
                'half_life': pair['half_life'],
                'last_update': datetime.now()
            }
            
        self.spread_stats = spread_stats
        
        return spread_stats
    
    def generate_trading_signals(self, prices: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Generate trading signals for pairs trading strategy.
        
        Args:
            prices: Dictionary mapping symbols to price series
            
        Returns:
            Dictionary mapping symbols to raw alpha signals
        """
        if not self.spread_stats:
            self.update_spread_statistics(prices)
            
        if not self.spread_stats:
            logger.warning("No spread statistics available")
            return {}
            
        # Generate signals
        raw_signals = {}
        
        for pair_id, stats in self.spread_stats.items():
            symbol1 = stats['symbol1']
            symbol2 = stats['symbol2']
            z_score = stats['z_score']
            hedge_ratio = stats['hedge_ratio']
            
            # Check if pair is already active
            is_active = pair_id in self.active_pairs
            
            if is_active:
                # Check exit conditions
                if abs(z_score) <= self.exit_z_score:
                    # Exit the position - spread has reverted to mean
                    logger.info(f"Exiting pairs position {pair_id}: z-score {z_score:.2f}")
                    self.active_pairs.pop(pair_id)
                    
                    raw_signals[symbol1] = 0
                    raw_signals[symbol2] = 0
                elif abs(z_score) >= self.stop_loss_z_score:
                    # Stop loss - spread has moved too far away
                    logger.info(f"Stop loss for pairs position {pair_id}: z-score {z_score:.2f}")
                    self.active_pairs.pop(pair_id)
                    
                    raw_signals[symbol1] = 0
                    raw_signals[symbol2] = 0
                else:
                    # Position still active - maintain position
                    position = self.active_pairs[pair_id]['position']
                    
                    if position == 'long_spread':
                        # Long symbol1, short symbol2
                        raw_signals[symbol1] = 1.0
                        raw_signals[symbol2] = -hedge_ratio
                    else:
                        # Short symbol1, long symbol2
                        raw_signals[symbol1] = -1.0
                        raw_signals[symbol2] = hedge_ratio
            else:
                # Check entry conditions
                if z_score >= self.entry_z_score:
                    # Short the spread (short symbol1, long symbol2)
                    logger.info(f"Entering SHORT spread position {pair_id}: z-score {z_score:.2f}")
                    self.active_pairs[pair_id] = {
                        'entry_date': datetime.now(),
                        'entry_z_score': z_score,
                        'position': 'short_spread'
                    }
                    
                    raw_signals[symbol1] = -1.0
                    raw_signals[symbol2] = hedge_ratio
                elif z_score <= -self.entry_z_score:
                    # Long the spread (long symbol1, short symbol2)
                    logger.info(f"Entering LONG spread position {pair_id}: z-score {z_score:.2f}")
                    self.active_pairs[pair_id] = {
                        'entry_date': datetime.now(),
                        'entry_z_score': z_score,
                        'position': 'long_spread'
                    }
                    
                    raw_signals[symbol1] = 1.0
                    raw_signals[symbol2] = -hedge_ratio
                    
        return raw_signals
    
    def run_strategy(self, 
                   prices: Dict[str, pd.Series],
                   asset_volatilities: Dict[str, float] = None,
                   market_betas: Dict[str, float] = None,
                   correlation_matrix: pd.DataFrame = None,
                   current_date: datetime = None) -> Dict[str, float]:
        """
        Run a complete strategy update.
        
        Args:
            prices: Dictionary mapping symbols to price series
            asset_volatilities: Dictionary mapping symbols to annualized volatilities
            market_betas: Dictionary mapping symbols to market betas
            correlation_matrix: Correlation matrix between assets
            current_date: Current date for this update
            
        Returns:
            Dictionary mapping symbols to new positions
        """
        # Default to current date if not provided
        current_date = current_date or datetime.now()
        
        # Update spread statistics
        self.update_spread_statistics(prices)
        
        # Generate raw trading signals
        raw_signals = self.generate_trading_signals(prices)
        
        if not raw_signals:
            logger.info("No trading signals generated")
            return {}
            
        # Ensure we have volatilities for all assets
        if asset_volatilities is None:
            # Calculate from price data
            asset_volatilities = {}
            for symbol, price_series in prices.items():
                if symbol in raw_signals and len(price_series) > 60:
                    returns = price_series.pct_change().dropna()
                    vol = returns[-60:].std() * np.sqrt(252)  # 60-day rolling volatility
                    asset_volatilities[symbol] = vol
                    
        # Calculate optimal position sizes
        position_sizes = self.calculate_position_sizes(
            raw_signals=raw_signals,
            asset_volatilities=asset_volatilities,
            asset_correlations=correlation_matrix,
            market_betas=market_betas
        )
        
        # Convert price Series to latest prices
        current_prices = {}
        for symbol, price_series in prices.items():
            if not price_series.empty:
                current_prices[symbol] = price_series.iloc[-1]
                
        # Update positions
        executed_trades = self.update_positions(
            new_positions=position_sizes,
            current_prices=current_prices,
            trade_date=current_date
        )
        
        return position_sizes
    
    def visualize_pair(self, 
                     pair_id: str,
                     prices: Dict[str, pd.Series],
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Visualize a trading pair with spread and z-score.
        
        Args:
            pair_id: Pair ID in format "symbol1_symbol2"
            prices: Dictionary mapping symbols to price series
            figsize: Figure size
            
        Returns:
            Matplotlib figure with pair visualization
        """
        if pair_id not in self.spread_stats:
            logger.warning(f"Pair {pair_id} not found in spread statistics")
            return None
            
        stats = self.spread_stats[pair_id]
        symbol1 = stats['symbol1']
        symbol2 = stats['symbol2']
        
        if symbol1 not in prices or symbol2 not in prices:
            logger.warning(f"Price data not found for {symbol1} or {symbol2}")
            return None
            
        # Align dates
        price1 = prices[symbol1]
        price2 = prices[symbol2]
        
        aligned_dates = price1.index.intersection(price2.index)
        
        if len(aligned_dates) < self.lookback_period:
            logger.warning(f"Insufficient aligned data for {symbol1}/{symbol2}")
            return None
            
        # Get data for the lookback period
        aligned_price1 = price1.loc[aligned_dates][-self.lookback_period:]
        aligned_price2 = price2.loc[aligned_dates][-self.lookback_period:]
        
        # Calculate spread
        hedge_ratio = stats['hedge_ratio']
        spread = aligned_price1 - hedge_ratio * aligned_price2
        
        # Calculate z-score
        z_score = (spread - stats['mean']) / stats['std'] if stats['std'] > 0 else spread * 0
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True, 
                                          gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot price series (normalized)
        norm_price1 = aligned_price1 / aligned_price1.iloc[0]
        norm_price2 = aligned_price2 / aligned_price2.iloc[0]
        
        ax1.plot(aligned_price1.index, norm_price1, label=symbol1)
        ax1.plot(aligned_price2.index, norm_price2, label=symbol2)
        ax1.set_title(f"Normalized Prices: {symbol1} vs {symbol2}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot spread
        ax2.plot(spread.index, spread, color='purple')
        ax2.axhline(y=stats['mean'], color='black', linestyle='-', alpha=0.5)
        ax2.axhline(y=stats['mean'] + self.entry_z_score * stats['std'], 
                  color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=stats['mean'] - self.entry_z_score * stats['std'], 
                  color='green', linestyle='--', alpha=0.5)
        ax2.set_title(f"Spread: {symbol1} - ({hedge_ratio:.4f} × {symbol2})")
        ax2.grid(True, alpha=0.3)
        
        # Plot z-score
        ax3.plot(z_score.index, z_score, color='blue')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axhline(y=self.entry_z_score, color='red', linestyle='--', 
                  label=f'Entry ({self.entry_z_score})', alpha=0.5)
        ax3.axhline(y=-self.entry_z_score, color='green', linestyle='--', alpha=0.5)
        ax3.axhline(y=self.exit_z_score, color='orange', linestyle=':', 
                  label=f'Exit ({self.exit_z_score})', alpha=0.5)
        ax3.axhline(y=-self.exit_z_score, color='orange', linestyle=':', alpha=0.5)
        ax3.set_title("Z-Score")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Mark active pairs
        if pair_id in self.active_pairs:
            position = self.active_pairs[pair_id]['position']
            entry_date = self.active_pairs[pair_id]['entry_date']
            entry_z = self.active_pairs[pair_id]['entry_z_score']
            
            # Find closest date to entry_date
            if entry_date in z_score.index:
                entry_idx = z_score.index.get_loc(entry_date)
            else:
                # Find closest date
                entry_idx = np.abs(z_score.index - pd.Timestamp(entry_date)).argmin()
                
            if entry_idx < len(z_score):
                ax3.scatter(z_score.index[entry_idx], entry_z, 
                          color='red' if position == 'short_spread' else 'green',
                          marker='o', s=100, zorder=5)
                          
                ax3.text(z_score.index[entry_idx], entry_z, 
                       f" Entry ({position})", 
                       verticalalignment='center')
                       
        # Add statistics
        stats_text = (
            f"Half-Life: {stats['half_life']:.1f} days\n"
            f"Current Z-Score: {stats['z_score']:.2f}\n"
            f"Hedge Ratio: {hedge_ratio:.4f}"
        )
        
        ax1.text(0.02, 0.05, stats_text, transform=ax1.transAxes,
              bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
              verticalalignment='bottom')
              
        plt.tight_layout()
        
        return fig

class StatisticalFactorStrategy(StatisticalArbitrageStrategy):
    """
    Statistical Factor Arbitrage Strategy.
    
    This strategy extracts statistical factors from a universe of assets
    and builds long-short portfolios to capture factor returns while
    maintaining market neutrality.
    """
    
    def __init__(self,
                n_factors: int = 5,
                factor_lookback: int = 252,
                rebalance_frequency: int = 21,  # Trading days
                min_asset_count: int = 20,
                use_shrinkage: bool = True,
                **kwargs):
        """
        Initialize the statistical factor strategy.
        
        Args:
            n_factors: Number of statistical factors to extract
            factor_lookback: Lookback period for factor estimation
            rebalance_frequency: How often to rebalance the portfolio (in trading days)
            min_asset_count: Minimum number of assets required
            use_shrinkage: Whether to use covariance shrinkage
            **kwargs: Additional arguments for the base class
        """
        # Set defaults for market-neutral strategy
        kwargs.setdefault('market_neutrality_target', 0.05)
        kwargs.setdefault('strategy_name', f"StatFactorArb_{datetime.now().strftime('%Y%m%d_%H%M')}")
        
        super().__init__(**kwargs)
        
        self.n_factors = n_factors
        self.factor_lookback = factor_lookback
        self.rebalance_frequency = rebalance_frequency
        self.min_asset_count = min_asset_count
        self.use_shrinkage = use_shrinkage
        
        # Strategy-specific state
        self.factor_model = None
        self.factor_exposures = None
        self.factor_returns = None
        self.last_rebalance_date = None
        self.days_since_rebalance = float('inf')
        self.asset_universe = []
        
    def extract_statistical_factors(self, returns: pd.DataFrame) -> Dict:
        """
        Extract statistical factors from return data using PCA.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Dictionary with factor model information
        """
        if returns.shape[0] < self.factor_lookback or returns.shape[1] < self.min_asset_count:
            logger.warning(f"Insufficient data for factor extraction: {returns.shape[0]} x {returns.shape[1]}")
            return None
            
        # Fill missing values (if any)
        returns_filled = returns.fillna(0)
        
        # Calculate return covariance matrix
        if self.use_shrinkage:
            # Use Ledoit-Wolf shrinkage for robustness
            from sklearn.covariance import LedoitWolf
            cov_estimator = LedoitWolf().fit(returns_filled.values)
            cov_matrix = pd.DataFrame(
                cov_estimator.covariance_, 
                index=returns.columns, 
                columns=returns.columns
            )
        else:
            cov_matrix = returns_filled.cov()
            
        # Apply PCA to extract factors
        pca = PCA(n_components=min(self.n_factors, returns.shape[1]))
        pca.fit(returns_filled.values)
        
        # Extract factor returns
        factor_returns_raw = pca.transform(returns_filled.values)
        factor_returns = pd.DataFrame(
            factor_returns_raw,
            index=returns.index,
            columns=[f"Factor_{i+1}" for i in range(pca.n_components_)]
        )
        
        # Extract factor exposures (loadings)
        factor_exposures = pd.DataFrame(
            pca.components_.T,
            index=returns.columns,
            columns=[f"Factor_{i+1}" for i in range(pca.n_components_)]
        )
        
        # Calculate variance explained by each factor
        explained_variance = pca.explained_variance_ratio_
        
        # Store factor model information
        factor_model = {
            'n_factors': pca.n_components_,
            'factor_returns': factor_returns,
            'factor_exposures': factor_exposures,
            'explained_variance': explained_variance,
            'total_explained_variance': sum(explained_variance),
            'cov_matrix': cov_matrix,
            'last_update': datetime.now()
        }
        
        self.factor_model = factor_model
        self.factor_exposures = factor_exposures
        self.factor_returns = factor_returns
        
        logger.info(f"Extracted {pca.n_components_} factors explaining {sum(explained_variance):.2%} of variance")
        
        return factor_model
    
    def forecast_factor_returns(self, lookback_days: int = 63) -> Dict[str, float]:
        """
        Forecast factor returns using recent data.
        
        Args:
            lookback_days: Number of days to use for forecasting
            
        Returns:
            Dictionary mapping factor names to forecasted returns
        """
        if self.factor_model is None or self.factor_returns is None:
            logger.warning("No factor model available")
            return {}
            
        factor_returns = self.factor_returns
        
        if len(factor_returns) < lookback_days:
            logger.warning(f"Insufficient factor return history: {len(factor_returns)} < {lookback_days}")
            lookback_days = len(factor_returns)
            
        if lookback_days < 10:
            logger.warning("Too few observations for reliable forecast")
            return {}
            
        # Use historical mean as forecast (simple approach)
        recent_returns = factor_returns.iloc[-lookback_days:]
        forecasts = recent_returns.mean()
        
        # Adjust by factor momentum
        momentum_1m = recent_returns.iloc[-21:].mean() if len(recent_returns) >= 21 else recent_returns.mean()
        momentum_3m = recent_returns.mean()
        
        # Blend forecasts (more weight to recent momentum)
        blended_forecast = 0.7 * momentum_1m + 0.3 * momentum_3m
        
        # Convert to dictionary
        forecast_dict = {col: float(blended_forecast[col]) for col in blended_forecast.index}
        
        return forecast_dict
    
    def generate_trading_signals(self, 
                              factor_exposures: pd.DataFrame = None,
                              factor_forecasts: Dict[str, float] = None) -> Dict[str, float]:
        """
        Generate trading signals based on factor model.
        
        Args:
            factor_exposures: Asset exposures to factors
            factor_forecasts: Forecasted factor returns
            
        Returns:
            Dictionary mapping symbols to raw alpha signals
        """
        if factor_exposures is None:
            factor_exposures = self.factor_exposures
            
        if factor_exposures is None:
            logger.warning("No factor exposures available")
            return {}
            
        if factor_forecasts is None:
            factor_forecasts = self.forecast_factor_returns()
            
        if not factor_forecasts:
            logger.warning("No factor forecasts available")
            return {}
            
        # Calculate expected returns as dot product of exposures and forecasts
        expected_returns = {}
        
        for symbol, exposures in factor_exposures.iterrows():
            # Calculate weighted sum of exposures * forecasts
            expected_return = 0
            
            for factor, exposure in exposures.items():
                if factor in factor_forecasts:
                    expected_return += exposure * factor_forecasts[factor]
                    
            expected_returns[symbol] = expected_return
            
        # Neutralize the expected returns (zero mean)
        mean_return = np.mean(list(expected_returns.values()))
        neutralized_returns = {k: v - mean_return for k, v in expected_returns.items()}
        
        return neutralized_returns
    
    def select_portfolio(self, 
                       raw_signals: Dict[str, float],
                       asset_volatilities: Dict[str, float] = None,
                       n_long: int = None,
                       n_short: int = None) -> Dict[str, float]:
        """
        Select portfolio constituents from raw signals.
        
        Args:
            raw_signals: Dictionary mapping symbols to raw alpha signals
            asset_volatilities: Dictionary mapping symbols to annualized volatilities
            n_long: Number of long positions (defaults to 1/3 of assets)
            n_short: Number of short positions (defaults to 1/3 of assets)
            
        Returns:
            Dictionary mapping selected symbols to raw alpha signals
        """
        if not raw_signals:
            return {}
            
        # Default to 1/3 of assets for each side
        n_total = len(raw_signals)
        n_long = n_long or max(5, n_total // 3)
        n_short = n_short or max(5, n_total // 3)
        
        # Sort assets by signal strength
        sorted_assets = sorted(raw_signals.items(), key=lambda x: x[1], reverse=True)
        
        # Select top n_long for long positions
        long_assets = sorted_assets[:n_long]
        
        # Select bottom n_short for short positions
        short_assets = sorted_assets[-n_short:]
        
        # Create portfolio
        portfolio = {}
        
        # Add long positions
        for symbol, signal in long_assets:
            if signal > 0:
                portfolio[symbol] = signal
                
        # Add short positions
        for symbol, signal in short_assets:
            if signal < 0:
                portfolio[symbol] = signal
                
        return portfolio
    
    def run_strategy(self,
                   returns: pd.DataFrame,
                   asset_volatilities: Dict[str, float] = None,
                   market_betas: Dict[str, float] = None,
                   correlation_matrix: pd.DataFrame = None,
                   current_date: datetime = None,
                   prices: Dict[str, pd.Series] = None,
                   force_rebalance: bool = False) -> Dict[str, float]:
        """
        Run a complete strategy update.
        
        Args:
            returns: DataFrame of asset returns
            asset_volatilities: Dictionary mapping symbols to annualized volatilities
            market_betas: Dictionary mapping symbols to market betas
            correlation_matrix: Correlation matrix between assets
            current_date: Current date for this update
            prices: Dictionary mapping symbols to price series (for current price)
            force_rebalance: Whether to force a rebalance regardless of schedule
            
        Returns:
            Dictionary mapping symbols to new positions
        """
        # Default to current date if not provided
        current_date = current_date or datetime.now()
        
        # Update days since last rebalance
        if self.last_rebalance_date:
            if isinstance(current_date, pd.Timestamp) and isinstance(self.last_rebalance_date, pd.Timestamp):
                self.days_since_rebalance = (current_date - self.last_rebalance_date).days
            else:
                # Try to convert to datetime objects if needed
                curr_dt = pd.Timestamp(current_date) if not isinstance(current_date, pd.Timestamp) else current_date
                last_dt = pd.Timestamp(self.last_rebalance_date) if not isinstance(self.last_rebalance_date, pd.Timestamp) else self.last_rebalance_date
                self.days_since_rebalance = (curr_dt - last_dt).days
            
        # Check if we need to rebalance
        if not force_rebalance and self.days_since_rebalance < self.rebalance_frequency:
            logger.info(f"Skipping rebalance ({self.days_since_rebalance}/{self.rebalance_frequency} days since last)")
            return self.current_positions
            
        # Update factor model
        if self.factor_model is None or force_rebalance:
            self.extract_statistical_factors(returns)
            
        if self.factor_model is None:
            logger.warning("Factor model generation failed")
            return {}
            
        # Generate raw trading signals
        raw_signals = self.generate_trading_signals()
        
        if not raw_signals:
            logger.warning("No trading signals generated")
            return {}
            
        # Select portfolio constituents
        selected_signals = self.select_portfolio(raw_signals, asset_volatilities)
        
        if not selected_signals:
            logger.warning("Portfolio selection failed")
            return {}
            
        # Ensure we have volatilities for all assets
        if asset_volatilities is None and returns is not None:
            # Calculate from returns data
            asset_volatilities = {}
            for symbol in selected_signals.keys():
                if symbol in returns.columns:
                    vol = returns[symbol].std() * np.sqrt(252)
                    asset_volatilities[symbol] = vol
                    
        # Calculate optimal position sizes
        position_sizes = self.calculate_position_sizes(
            raw_signals=selected_signals,
            asset_volatilities=asset_volatilities,
            asset_correlations=correlation_matrix,
            market_betas=market_betas
        )
        
        # Convert price Series to latest prices (if provided)
        current_prices = {}
        if prices:
            for symbol, price_series in prices.items():
                if not price_series.empty:
                    current_prices[symbol] = price_series.iloc[-1]
                    
        # Update positions
        executed_trades = self.update_positions(
            new_positions=position_sizes,
            current_prices=current_prices,
            trade_date=current_date
        )
        
        # Update rebalance date
        self.last_rebalance_date = current_date
        self.days_since_rebalance = 0
        
        return position_sizes
    
    def visualize_factor_analysis(self, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Visualize factor analysis results.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with factor visualization
        """
        if self.factor_model is None:
            logger.warning("No factor model available")
            return None
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot explained variance
        explained_variance = self.factor_model['explained_variance']
        cum_variance = np.cumsum(explained_variance)
        
        axes[0, 0].bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
        axes[0, 0].plot(range(1, len(explained_variance) + 1), cum_variance, 'r-o')
        axes[0, 0].set_title("Factor Explained Variance")
        axes[0, 0].set_xlabel("Factor")
        axes[0, 0].set_ylabel("Variance Explained (%)")
        axes[0, 0].set_xticks(range(1, len(explained_variance) + 1))
        axes[0, 0].grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Plot recent factor returns
        factor_returns = self.factor_model['factor_returns']
        recent_returns = factor_returns.iloc[-63:] if len(factor_returns) > 63 else factor_returns
        
        # Calculate cumulative returns
        cum_returns = (1 + recent_returns).cumprod() - 1
        
        for col in cum_returns.columns:
            axes[0, 1].plot(cum_returns.index, cum_returns[col], label=col)
            
        axes[0, 1].set_title("Cumulative Factor Returns (Last 3 Months)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot factor exposures
        exposures = self.factor_model['factor_exposures'].copy()
        
        # Limit to top assets by absolute exposure
        top_assets = []
        for col in exposures.columns:
            # Get top 5 assets by absolute exposure
            top_for_factor = exposures[col].abs().nlargest(5).index
            top_assets.extend(top_for_factor)
            
        # Get unique assets
        top_assets = list(dict.fromkeys(top_assets))
        
        # Limit to top 15 assets
        if len(top_assets) > 15:
            top_assets = top_assets[:15]
            
        # Filter exposures to top assets
        top_exposures = exposures.loc[top_assets]
        
        # Create heatmap
        sns.heatmap(top_exposures, cmap='coolwarm', center=0, ax=axes[1, 0], 
                  cbar_kws={'label': 'Exposure'})
        
        axes[1, 0].set_title("Factor Exposures (Top Assets)")
        axes[1, 0].set_ylabel("Asset")
        axes[1, 0].set_xlabel("Factor")
        
        # Plot forecasted factor returns
        forecasts = self.forecast_factor_returns()
        
        if forecasts:
            factors = list(forecasts.keys())
            forecast_values = [forecasts[f] for f in factors]
            
            axes[1, 1].barh(factors, forecast_values, color=['green' if x > 0 else 'red' for x in forecast_values])
            axes[1, 1].set_title("Forecasted Factor Returns")
            axes[1, 1].set_xlabel("Forecasted Return")
            axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        return fig

# Helper functions
def signal_direction(signal: float) -> int:
    """Returns the direction of a signal: 1 for positive, -1 for negative, 0 for zero."""
    if signal > 0:
        return 1
    elif signal < 0:
        return -1
    else:
        return 0