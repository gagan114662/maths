"""
Dynamic asset allocation module.

This module provides dynamic asset allocation strategies that adapt
to changing market conditions and relative performance of assets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from enum import Enum
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AllocationMethod(Enum):
    """Allocation methods supported by the dynamic allocator."""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP_WEIGHT = "market_cap_weight"
    VOLATILITY_WEIGHT = "volatility_weight"
    INVERSE_VOLATILITY = "inverse_volatility"
    MOMENTUM_WEIGHT = "momentum_weight"
    RELATIVE_STRENGTH = "relative_strength"
    ADAPTIVE_WEIGHT = "adaptive_weight"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    CUSTOM = "custom"

class DynamicAssetAllocator:
    """
    Dynamic asset allocation that adapts to changing market conditions.
    
    This class provides various allocation strategies that dynamically
    adjust based on market conditions, asset performance, volatility,
    and other factors.
    """
    
    def __init__(self, 
                default_method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT,
                lookback_window: int = 252,
                rebalance_frequency: int = 21,  # Trading days (roughly monthly)
                volatility_window: int = 63,
                max_allocation_pct: float = 0.4,  # Maximum 40% in any single asset
                min_allocation_pct: float = 0.0):
        """
        Initialize the dynamic asset allocator.
        
        Args:
            default_method: Default allocation method
            lookback_window: Lookback period for calculating metrics (in days)
            rebalance_frequency: How often to rebalance (in days)
            volatility_window: Window for calculating volatility (in days)
            max_allocation_pct: Maximum allocation to any single asset
            min_allocation_pct: Minimum allocation to any single asset
        """
        self.default_method = default_method
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        self.volatility_window = volatility_window
        self.max_allocation_pct = max_allocation_pct
        self.min_allocation_pct = min_allocation_pct
        self.last_rebalance_date = None
        self.current_allocations = {}
        self.asset_data = {}
        self.market_cap_data = {}
        self.asset_metadata = {}
        self.custom_allocation_func = None
        
    def add_asset_data(self, 
                     symbol: str, 
                     price_data: pd.DataFrame,
                     market_cap: Optional[pd.Series] = None,
                     metadata: Optional[Dict] = None) -> None:
        """
        Add asset price data and metadata.
        
        Args:
            symbol: Asset symbol
            price_data: DataFrame with price history
            market_cap: Series with market cap history (for cap-weighted allocations)
            metadata: Additional metadata for the asset
        """
        self.asset_data[symbol] = price_data
        
        if market_cap is not None:
            self.market_cap_data[symbol] = market_cap
            
        if metadata is not None:
            self.asset_metadata[symbol] = metadata
        else:
            self.asset_metadata[symbol] = {}
            
        logger.info(f"Added data for {symbol} with {len(price_data)} data points")
    
    def set_custom_allocation_function(self, func):
        """
        Set a custom allocation function.
        
        Args:
            func: Function that takes a dictionary of asset data and returns
                 a dictionary of asset weights
        """
        self.custom_allocation_func = func
    
    def _get_returns(self, price_data: pd.DataFrame, column: str = None) -> pd.Series:
        """
        Get returns from price data.
        
        Args:
            price_data: DataFrame with price data
            column: Column to use (if None, tries to guess)
            
        Returns:
            Series of returns
        """
        # If returns column exists, use it
        if 'returns' in price_data.columns:
            return price_data['returns']
            
        # If specific column provided, use it
        if column and column in price_data.columns:
            prices = price_data[column]
            return prices.pct_change().dropna()
            
        # Try to find close prices
        if 'close' in price_data.columns:
            prices = price_data['close']
            return prices.pct_change().dropna()
            
        # Use first numeric column
        numeric_cols = price_data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            prices = price_data[numeric_cols[0]]
            return prices.pct_change().dropna()
            
        # No suitable data found
        logger.error(f"No suitable price data found")
        return pd.Series()
    
    def _calculate_volatility(self, 
                            returns: pd.Series, 
                            window: int = None) -> float:
        """
        Calculate asset volatility.
        
        Args:
            returns: Series of returns
            window: Window for calculation (if None, uses volatility_window)
            
        Returns:
            Annualized volatility
        """
        window = window or self.volatility_window
        
        if len(returns) < window:
            if len(returns) == 0:
                return np.nan
            # Use all available data if less than window
            return returns.std() * np.sqrt(252)
            
        # Use rolling window
        return returns.iloc[-window:].std() * np.sqrt(252)
    
    def _calculate_momentum(self, 
                          price_data: pd.DataFrame, 
                          lookback: int = None,
                          column: str = None) -> float:
        """
        Calculate asset momentum.
        
        Args:
            price_data: DataFrame with price data
            lookback: Lookback period (if None, uses lookback_window)
            column: Column to use
            
        Returns:
            Momentum score (higher is stronger)
        """
        lookback = lookback or self.lookback_window
        
        # Get price series
        if column and column in price_data.columns:
            prices = price_data[column]
        elif 'close' in price_data.columns:
            prices = price_data['close']
        else:
            # Use first numeric column
            numeric_cols = price_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                prices = price_data[numeric_cols[0]]
            else:
                logger.error(f"No suitable price data found")
                return 0.0
                
        if len(prices) < lookback:
            if len(prices) < 2:
                return 0.0
            # Use all available data if less than lookback
            return prices.iloc[-1] / prices.iloc[0] - 1
            
        # Calculate momentum as return over lookback period
        return prices.iloc[-1] / prices.iloc[-lookback] - 1
    
    def _calculate_relative_strength(self, 
                                   price_data: pd.DataFrame,
                                   benchmark_data: pd.DataFrame = None,
                                   lookback_periods: List[int] = None,
                                   column: str = None) -> float:
        """
        Calculate relative strength.
        
        Args:
            price_data: DataFrame with price data
            benchmark_data: Benchmark price data (if None, uses equal-weighted average)
            lookback_periods: List of periods to calculate momentum over
            column: Column to use
            
        Returns:
            Relative strength score (higher is stronger)
        """
        # Default lookback periods (1, 3, 6, 12 months in trading days)
        lookback_periods = lookback_periods or [21, 63, 126, 252]
        
        # Get price series
        if column and column in price_data.columns:
            prices = price_data[column]
        elif 'close' in price_data.columns:
            prices = price_data['close']
        else:
            # Use first numeric column
            numeric_cols = price_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                prices = price_data[numeric_cols[0]]
            else:
                logger.error(f"No suitable price data found")
                return 0.0
                
        # If no benchmark, create equal-weighted average of all assets
        if benchmark_data is None:
            # Calculate average price across all assets
            all_prices = []
            for symbol, data in self.asset_data.items():
                if column and column in data.columns:
                    asset_prices = data[column]
                elif 'close' in data.columns:
                    asset_prices = data['close']
                else:
                    # Use first numeric column
                    numeric_cols = data.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        asset_prices = data[numeric_cols[0]]
                    else:
                        continue
                        
                # Normalize to start at 1.0
                asset_prices = asset_prices / asset_prices.iloc[0]
                all_prices.append(asset_prices)
                
            if all_prices:
                # Create DataFrame with all normalized prices
                price_df = pd.concat(all_prices, axis=1)
                # Calculate average price
                benchmark_prices = price_df.mean(axis=1)
            else:
                logger.error("No suitable benchmark data could be created")
                return 0.0
        else:
            # Use provided benchmark data
            if column and column in benchmark_data.columns:
                benchmark_prices = benchmark_data[column]
            elif 'close' in benchmark_data.columns:
                benchmark_prices = benchmark_data['close']
            else:
                # Use first numeric column
                numeric_cols = benchmark_data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    benchmark_prices = benchmark_data[numeric_cols[0]]
                else:
                    logger.error(f"No suitable benchmark price data found")
                    return 0.0
        
        # Calculate relative strength for each period
        rs_scores = []
        
        for period in lookback_periods:
            if len(prices) <= period or len(benchmark_prices) <= period:
                continue
                
            # Calculate price return over period
            price_return = prices.iloc[-1] / prices.iloc[-period] - 1
            benchmark_return = benchmark_prices.iloc[-1] / benchmark_prices.iloc[-period] - 1
            
            # Calculate relative strength
            if benchmark_return != 0:
                relative_strength = price_return / benchmark_return if benchmark_return > 0 else -price_return / benchmark_return
            else:
                # If benchmark return is 0, use price return as relative strength
                relative_strength = price_return
                
            rs_scores.append(relative_strength)
            
        if not rs_scores:
            return 0.0
            
        # Weight more recent periods more heavily
        weights = np.linspace(1, 2, len(rs_scores))
        weights = weights / weights.sum()
        
        # Calculate weighted average relative strength
        weighted_rs = np.sum(np.array(rs_scores) * weights)
        
        return weighted_rs
    
    def _equal_weight_allocation(self) -> Dict[str, float]:
        """
        Calculate equal-weight allocation.
        
        Returns:
            Dictionary mapping asset symbols to allocation weights
        """
        assets = list(self.asset_data.keys())
        n_assets = len(assets)
        
        if n_assets == 0:
            return {}
            
        weight = 1.0 / n_assets
        
        return {asset: weight for asset in assets}
    
    def _market_cap_allocation(self) -> Dict[str, float]:
        """
        Calculate market-cap-weighted allocation.
        
        Returns:
            Dictionary mapping asset symbols to allocation weights
        """
        assets = list(self.asset_data.keys())
        
        if not assets:
            return {}
            
        # Check that we have market cap data
        assets_with_cap = [a for a in assets if a in self.market_cap_data]
        
        if not assets_with_cap:
            logger.warning("No market cap data available, using equal weight")
            return self._equal_weight_allocation()
            
        # Get latest market cap for each asset
        market_caps = {}
        
        for asset in assets_with_cap:
            cap_data = self.market_cap_data[asset]
            if len(cap_data) > 0:
                market_caps[asset] = cap_data.iloc[-1]
            else:
                market_caps[asset] = 0
                
        # Calculate total market cap
        total_cap = sum(market_caps.values())
        
        if total_cap == 0:
            logger.warning("Total market cap is zero, using equal weight")
            return self._equal_weight_allocation()
            
        # Calculate weights
        weights = {asset: cap / total_cap for asset, cap in market_caps.items()}
        
        return weights
    
    def _inverse_volatility_allocation(self) -> Dict[str, float]:
        """
        Calculate inverse-volatility-weighted allocation.
        
        Returns:
            Dictionary mapping asset symbols to allocation weights
        """
        assets = list(self.asset_data.keys())
        
        if not assets:
            return {}
            
        # Calculate volatility for each asset
        volatilities = {}
        
        for asset in assets:
            returns = self._get_returns(self.asset_data[asset])
            vol = self._calculate_volatility(returns)
            volatilities[asset] = vol if not np.isnan(vol) else float('inf')
            
        # Calculate inverse volatilities
        inverse_vol = {}
        
        for asset, vol in volatilities.items():
            if vol > 0 and vol != float('inf'):
                inverse_vol[asset] = 1.0 / vol
            else:
                inverse_vol[asset] = 0.0
                
        # Calculate total inverse volatility
        total_inverse_vol = sum(inverse_vol.values())
        
        if total_inverse_vol == 0:
            logger.warning("Total inverse volatility is zero, using equal weight")
            return self._equal_weight_allocation()
            
        # Calculate weights
        weights = {asset: vol / total_inverse_vol for asset, vol in inverse_vol.items()}
        
        return weights
    
    def _momentum_allocation(self) -> Dict[str, float]:
        """
        Calculate momentum-based allocation.
        
        Returns:
            Dictionary mapping asset symbols to allocation weights
        """
        assets = list(self.asset_data.keys())
        
        if not assets:
            return {}
            
        # Calculate momentum for each asset
        momentum_scores = {}
        
        for asset in assets:
            momentum = self._calculate_momentum(self.asset_data[asset])
            momentum_scores[asset] = momentum
            
        # Only allocate to assets with positive momentum
        positive_momentum = {k: v for k, v in momentum_scores.items() if v > 0}
        
        if not positive_momentum:
            logger.warning("No assets with positive momentum, using equal weight")
            return self._equal_weight_allocation()
            
        # Calculate weights proportional to momentum scores
        total_momentum = sum(positive_momentum.values())
        
        if total_momentum == 0:
            logger.warning("Total momentum is zero, using equal weight for positive momentum assets")
            n_assets = len(positive_momentum)
            return {asset: 1.0 / n_assets for asset in positive_momentum.keys()}
            
        # Calculate weights
        weights = {asset: score / total_momentum for asset, score in positive_momentum.items()}
        
        return weights
    
    def _calculate_relative_strength_allocation(self, 
                                              benchmark_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Calculate allocation based on relative strength.
        
        Args:
            benchmark_data: Benchmark price data (if None, uses equal-weighted average)
            
        Returns:
            Dictionary mapping asset symbols to allocation weights
        """
        assets = list(self.asset_data.keys())
        
        if not assets:
            return {}
            
        # Calculate relative strength for each asset
        rs_scores = {}
        
        for asset in assets:
            rs = self._calculate_relative_strength(self.asset_data[asset], benchmark_data)
            rs_scores[asset] = rs
            
        # Only allocate to assets with positive relative strength
        positive_rs = {k: v for k, v in rs_scores.items() if v > 0}
        
        if not positive_rs:
            logger.warning("No assets with positive relative strength, using inverse volatility weight")
            return self._inverse_volatility_allocation()
            
        # Calculate weights proportional to relative strength scores
        total_rs = sum(positive_rs.values())
        
        if total_rs == 0:
            logger.warning("Total relative strength is zero, using equal weight for positive RS assets")
            n_assets = len(positive_rs)
            return {asset: 1.0 / n_assets for asset in positive_rs.keys()}
            
        # Calculate weights
        weights = {asset: score / total_rs for asset, score in positive_rs.items()}
        
        return weights
    
    def _risk_parity_allocation(self) -> Dict[str, float]:
        """
        Calculate risk parity allocation.
        
        Returns:
            Dictionary mapping asset symbols to allocation weights
        """
        assets = list(self.asset_data.keys())
        
        if not assets:
            return {}
            
        # Calculate returns for all assets
        returns_data = {}
        
        for asset in assets:
            returns = self._get_returns(self.asset_data[asset])
            if len(returns) > 0:
                returns_data[asset] = returns
                
        if not returns_data:
            logger.warning("No return data available, using equal weight")
            return self._equal_weight_allocation()
            
        # Create DataFrame with aligned returns
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 2:
            logger.warning("Insufficient return data for risk parity, using inverse volatility")
            return self._inverse_volatility_allocation()
            
        # Calculate covariance matrix
        cov_matrix = returns_df.cov() * 252  # Annualized
        
        # Calculate marginal risk contributions
        volatilities = np.sqrt(np.diag(cov_matrix))
        
        # Simple risk parity using inverse volatility
        inv_vol = 1.0 / volatilities
        weights = inv_vol / np.sum(inv_vol)
        
        # Convert to dictionary
        weight_dict = {asset: weights[i] for i, asset in enumerate(returns_df.columns)}
        
        return weight_dict
    
    def _apply_allocation_constraints(self, 
                                    weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply allocation constraints to weights.
        
        Args:
            weights: Dictionary mapping assets to weights
            
        Returns:
            Constrained weights
        """
        if not weights:
            return {}
            
        # Apply minimum allocation constraint
        if self.min_allocation_pct > 0:
            weights = {k: max(v, self.min_allocation_pct) for k, v in weights.items()}
            
        # Apply maximum allocation constraint
        if self.max_allocation_pct < 1.0:
            weights = {k: min(v, self.max_allocation_pct) for k, v in weights.items()}
            
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # If total weight is 0, use equal weights
            n_assets = len(weights)
            weights = {k: 1.0 / n_assets for k in weights.keys()}
            
        return weights
    
    def get_allocation(self, 
                     method: AllocationMethod = None,
                     current_date: datetime = None,
                     benchmark_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Get asset allocation based on specified method.
        
        Args:
            method: Allocation method to use (defaults to self.default_method)
            current_date: Current date (used for rebalancing decision)
            benchmark_data: Benchmark price data for relative allocation
            
        Returns:
            Dictionary mapping asset symbols to allocation weights
        """
        method = method or self.default_method
        
        # Determine if we should rebalance
        if current_date and self.last_rebalance_date:
            if isinstance(current_date, str):
                current_date = pd.to_datetime(current_date)
                
            if isinstance(self.last_rebalance_date, str):
                self.last_rebalance_date = pd.to_datetime(self.last_rebalance_date)
                
            days_since_rebalance = (current_date - self.last_rebalance_date).days
            
            if days_since_rebalance < self.rebalance_frequency:
                # Return current allocations if not yet time to rebalance
                return self.current_allocations
                
        # Calculate new allocations based on method
        if method == AllocationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight_allocation()
        elif method == AllocationMethod.MARKET_CAP_WEIGHT:
            weights = self._market_cap_allocation()
        elif method == AllocationMethod.INVERSE_VOLATILITY:
            weights = self._inverse_volatility_allocation()
        elif method == AllocationMethod.MOMENTUM_WEIGHT:
            weights = self._momentum_allocation()
        elif method == AllocationMethod.RELATIVE_STRENGTH:
            weights = self._calculate_relative_strength_allocation(benchmark_data)
        elif method == AllocationMethod.RISK_PARITY:
            weights = self._risk_parity_allocation()
        elif method == AllocationMethod.CUSTOM:
            if self.custom_allocation_func:
                weights = self.custom_allocation_func(self.asset_data)
            else:
                logger.warning("No custom allocation function set, using equal weight")
                weights = self._equal_weight_allocation()
        else:
            # Default to equal weight
            weights = self._equal_weight_allocation()
            
        # Apply constraints
        weights = self._apply_allocation_constraints(weights)
        
        # Update current allocations and rebalance date
        self.current_allocations = weights
        self.last_rebalance_date = current_date if current_date else datetime.now()
        
        return weights
    
    def get_allocation_history(self,
                             start_date: datetime,
                             end_date: datetime,
                             method: AllocationMethod = None,
                             benchmark_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate allocation history over a time period.
        
        Args:
            start_date: Start date
            end_date: End date
            method: Allocation method to use
            benchmark_data: Benchmark price data for relative allocation
            
        Returns:
            DataFrame with allocation weights over time
        """
        method = method or self.default_method
        
        # Convert dates to datetime
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
            
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Filter to business days
        dates = dates[dates.dayofweek < 5]
        
        # Calculate allocations on rebalance dates
        allocations = {}
        last_allocation = None
        
        for date in dates:
            # Determine if this is a rebalance date
            if not last_allocation or (date - last_allocation['date']).days >= self.rebalance_frequency:
                # Calculate new allocation
                weights = self.get_allocation(method=method, current_date=date, benchmark_data=benchmark_data)
                allocations[date] = weights
                last_allocation = {'date': date, 'weights': weights}
            else:
                # Use last allocation
                allocations[date] = last_allocation['weights']
                
        # Convert to DataFrame
        all_assets = set()
        for weights in allocations.values():
            all_assets.update(weights.keys())
            
        allocation_data = {}
        
        for date, weights in allocations.items():
            allocation_data[date] = {asset: weights.get(asset, 0.0) for asset in all_assets}
            
        # Create DataFrame
        df = pd.DataFrame.from_dict(allocation_data, orient='index')
        df.index.name = 'date'
        
        return df
    
    def backtest_allocation(self,
                          start_date: datetime,
                          end_date: datetime,
                          method: AllocationMethod = None,
                          benchmark_data: pd.DataFrame = None,
                          initial_capital: float = 10000.0) -> Dict:
        """
        Backtest an allocation strategy.
        
        Args:
            start_date: Start date
            end_date: End date
            method: Allocation method to use
            benchmark_data: Benchmark price data for relative allocation
            initial_capital: Initial capital
            
        Returns:
            Dictionary with backtest results
        """
        method = method or self.default_method
        
        # Calculate allocation history
        allocation_df = self.get_allocation_history(
            start_date=start_date,
            end_date=end_date,
            method=method,
            benchmark_data=benchmark_data
        )
        
        # Get price data for all assets
        price_data = {}
        
        for asset in allocation_df.columns:
            if asset in self.asset_data:
                if 'close' in self.asset_data[asset].columns:
                    prices = self.asset_data[asset]['close']
                else:
                    # Use first numeric column
                    numeric_cols = self.asset_data[asset].select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        prices = self.asset_data[asset][numeric_cols[0]]
                    else:
                        logger.warning(f"No suitable price data found for {asset}, skipping")
                        continue
                
                price_data[asset] = prices
                
        if not price_data:
            logger.error("No price data available for backtest")
            return {}
            
        # Align dates
        price_df = pd.DataFrame(price_data)
        price_df = price_df.reindex(allocation_df.index, method='ffill')
        
        # Calculate portfolio value
        portfolio_value = initial_capital
        portfolio_values = [portfolio_value]
        dates = [allocation_df.index[0]]
        
        holdings = {asset: 0 for asset in allocation_df.columns}
        
        for i in range(1, len(allocation_df)):
            prev_date = allocation_df.index[i-1]
            curr_date = allocation_df.index[i]
            
            # Update holdings based on price changes
            for asset in holdings:
                if asset in price_data:
                    price_change = price_df.loc[curr_date, asset] / price_df.loc[prev_date, asset]
                    holdings[asset] *= price_change
                    
            # Calculate current portfolio value
            portfolio_value = sum(holdings.values())
            
            # Check if rebalance is needed
            days_since_last_rebalance = (curr_date - dates[-1]).days
            
            if days_since_last_rebalance >= self.rebalance_frequency:
                # Rebalance
                for asset, weight in allocation_df.loc[curr_date].items():
                    if asset in price_data:
                        holdings[asset] = portfolio_value * weight
                        
            portfolio_values.append(portfolio_value)
            dates.append(curr_date)
            
        # Calculate benchmark performance if provided
        benchmark_values = None
        
        if benchmark_data is not None:
            if 'close' in benchmark_data.columns:
                benchmark_prices = benchmark_data['close']
            else:
                # Use first numeric column
                numeric_cols = benchmark_data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    benchmark_prices = benchmark_data[numeric_cols[0]]
                else:
                    logger.warning("No suitable benchmark price data found")
                    benchmark_values = None
                
            if benchmark_prices is not None:
                # Align benchmark prices to backtest dates
                benchmark_prices = benchmark_prices.reindex(dates, method='ffill')
                
                # Calculate benchmark values
                benchmark_initial = benchmark_prices.iloc[0]
                benchmark_values = [initial_capital]
                
                for i in range(1, len(dates)):
                    benchmark_return = benchmark_prices.iloc[i] / benchmark_prices.iloc[i-1]
                    benchmark_values.append(benchmark_values[-1] * benchmark_return)
        
        # Calculate performance metrics
        returns = [portfolio_values[i] / portfolio_values[i-1] - 1 for i in range(1, len(portfolio_values))]
        
        # Calculate metrics
        total_return = portfolio_values[-1] / portfolio_values[0] - 1
        ann_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = ann_return / volatility if volatility > 0 else 0
        
        # Calculate drawdowns
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = [1 - val / peak if peak > 0 else 0 for val, peak in zip(portfolio_values, running_max)]
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Calculate benchmark-relative metrics
        if benchmark_values:
            benchmark_returns = [benchmark_values[i] / benchmark_values[i-1] - 1 for i in range(1, len(benchmark_values))]
            benchmark_total_return = benchmark_values[-1] / benchmark_values[0] - 1
            benchmark_ann_return = (1 + benchmark_total_return) ** (252 / len(benchmark_values)) - 1
            
            # Excess return
            excess_return = ann_return - benchmark_ann_return
            
            # Tracking error
            if len(returns) == len(benchmark_returns):
                tracking_error = np.std([r - b for r, b in zip(returns, benchmark_returns)]) * np.sqrt(252)
                information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            else:
                tracking_error = np.nan
                information_ratio = np.nan
            
            # Beta
            if len(returns) == len(benchmark_returns):
                covariance = np.cov(returns, benchmark_returns)[0][1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
                
                # Alpha
                alpha = ann_return - (0.0 + beta * benchmark_ann_return)  # Assuming risk-free rate of 0
            else:
                beta = np.nan
                alpha = np.nan
        else:
            excess_return = np.nan
            tracking_error = np.nan
            information_ratio = np.nan
            beta = np.nan
            alpha = np.nan
        
        # Compile results
        results = {
            'portfolio_values': portfolio_values,
            'dates': dates,
            'benchmark_values': benchmark_values,
            'total_return': total_return,
            'annualized_return': ann_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'excess_return': excess_return,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'allocation_history': allocation_df
        }
        
        return results
    
    def plot_backtest_results(self, backtest_results: Dict, title: str = None) -> plt.Figure:
        """
        Plot backtest results.
        
        Args:
            backtest_results: Results from backtest_allocation
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if not backtest_results:
            logger.error("No backtest results to plot")
            return None
            
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), 
                                           gridspec_kw={'height_ratios': [3, 1, 2]},
                                           sharex=True)
        
        # Plot portfolio value
        dates = backtest_results['dates']
        portfolio_values = backtest_results['portfolio_values']
        
        ax1.plot(dates, portfolio_values, label='Portfolio', linewidth=2)
        
        # Plot benchmark if available
        if backtest_results['benchmark_values']:
            benchmark_values = backtest_results['benchmark_values']
            ax1.plot(dates, benchmark_values, label='Benchmark', linewidth=2, alpha=0.7)
            
        ax1.set_title('Portfolio Performance' if title is None else title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Value')
        
        # Plot drawdowns
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = [1 - val / peak if peak > 0 else 0 for val, peak in zip(portfolio_values, running_max)]
        
        ax2.fill_between(dates, 0, drawdowns, color='red', alpha=0.3)
        ax2.set_title('Drawdowns')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel('Drawdown')
        ax2.set_ylim(max(max(drawdowns) + 0.05, 0.1), 0)  # Invert y-axis
        
        # Plot allocation over time
        allocation_df = backtest_results['allocation_history']
        ax3.stackplot(allocation_df.index, allocation_df.T, 
                     labels=allocation_df.columns, alpha=0.7)
        ax3.set_title('Asset Allocation')
        ax3.legend(loc='upper left', fontsize='small')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylabel('Weight')
        ax3.set_xlabel('Date')
        
        # Add performance metrics as text box
        metrics_text = "\n".join([
            f"Total Return: {backtest_results['total_return']:.2%}",
            f"Annualized Return: {backtest_results['annualized_return']:.2%}",
            f"Volatility: {backtest_results['volatility']:.2%}",
            f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}",
            f"Max Drawdown: {backtest_results['max_drawdown']:.2%}",
        ])
        
        # Add benchmark metrics if available
        if backtest_results['benchmark_values']:
            benchmark_metrics = "\n".join([
                f"Alpha: {backtest_results['alpha']:.2%}",
                f"Beta: {backtest_results['beta']:.2f}",
                f"Excess Return: {backtest_results['excess_return']:.2%}",
                f"Information Ratio: {backtest_results['information_ratio']:.2f}"
            ])
            metrics_text += "\n\n" + benchmark_metrics
            
        # Add text box
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax1.text(0.02, 0.05, metrics_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='bottom', bbox=props)
        
        plt.tight_layout()
        
        return fig

class RelativeStrengthAllocator(DynamicAssetAllocator):
    """
    Asset allocator based on relative strength.
    
    This class specializes in allocating assets based on their relative
    strength compared to a benchmark or peer group.
    """
    
    def __init__(self, 
                benchmark_data: pd.DataFrame = None,
                lookback_periods: List[int] = None,
                min_rs_score: float = 0.0,
                **kwargs):
        """
        Initialize relative strength allocator.
        
        Args:
            benchmark_data: Benchmark price data
            lookback_periods: List of periods for RS calculation
            min_rs_score: Minimum relative strength score for inclusion
            **kwargs: Additional arguments passed to DynamicAssetAllocator
        """
        super().__init__(default_method=AllocationMethod.RELATIVE_STRENGTH, **kwargs)
        self.benchmark_data = benchmark_data
        self.lookback_periods = lookback_periods or [21, 63, 126, 252]
        self.min_rs_score = min_rs_score
        
    def set_benchmark(self, benchmark_data: pd.DataFrame) -> None:
        """
        Set benchmark data for relative strength calculation.
        
        Args:
            benchmark_data: Benchmark price data
        """
        self.benchmark_data = benchmark_data
        
    def set_lookback_periods(self, periods: List[int]) -> None:
        """
        Set lookback periods for relative strength calculation.
        
        Args:
            periods: List of periods in days
        """
        self.lookback_periods = periods
        
    def calculate_rs_rankings(self) -> pd.DataFrame:
        """
        Calculate relative strength rankings for all assets.
        
        Returns:
            DataFrame with RS scores and rankings
        """
        assets = list(self.asset_data.keys())
        
        if not assets:
            return pd.DataFrame()
            
        # Calculate RS scores for each asset
        rs_scores = {}
        
        for asset in assets:
            rs = self._calculate_relative_strength(
                self.asset_data[asset],
                self.benchmark_data,
                self.lookback_periods
            )
            rs_scores[asset] = rs
            
        # Convert to DataFrame
        rs_df = pd.DataFrame({
            'rs_score': rs_scores
        })
        
        # Add rankings
        rs_df['rs_rank'] = rs_df['rs_score'].rank(ascending=False)
        
        # Sort by ranking
        rs_df = rs_df.sort_values('rs_rank')
        
        return rs_df
    
    def get_top_assets(self, n: int = 10) -> List[str]:
        """
        Get top N assets by relative strength.
        
        Args:
            n: Number of assets to return
            
        Returns:
            List of asset symbols
        """
        rankings = self.calculate_rs_rankings()
        
        if rankings.empty:
            return []
            
        # Filter by minimum RS score
        if self.min_rs_score > 0:
            rankings = rankings[rankings['rs_score'] >= self.min_rs_score]
            
        # Return top N
        return rankings.head(n).index.tolist()
    
    def get_allocation(self, 
                     method: AllocationMethod = None,
                     current_date: datetime = None,
                     top_n: int = None) -> Dict[str, float]:
        """
        Get allocation based on relative strength.
        
        Args:
            method: Allocation method to use (ignored for this class)
            current_date: Current date
            top_n: Number of top assets to include
            
        Returns:
            Dictionary mapping assets to weights
        """
        # Calculate relative strength rankings
        rankings = self.calculate_rs_rankings()
        
        if rankings.empty:
            return {}
            
        # Filter by minimum RS score
        if self.min_rs_score > 0:
            rankings = rankings[rankings['rs_score'] >= self.min_rs_score]
            
        if rankings.empty:
            return {}
            
        # Limit to top N assets
        if top_n and top_n > 0:
            top_assets = rankings.head(top_n).index.tolist()
        else:
            top_assets = rankings.index.tolist()
            
        # Calculate weights inversely proportional to rank
        ranks = rankings.loc[top_assets, 'rs_rank'].values
        inverse_ranks = 1.0 / ranks
        
        # Normalize weights
        total_weight = np.sum(inverse_ranks)
        normalized_weights = inverse_ranks / total_weight
        
        # Create weight dictionary
        weights = {asset: weight for asset, weight in zip(top_assets, normalized_weights)}
        
        # Apply constraints
        weights = self._apply_allocation_constraints(weights)
        
        # Update current allocations and rebalance date
        self.current_allocations = weights
        self.last_rebalance_date = current_date if current_date else datetime.now()
        
        return weights
    
    def backtest_relative_strength(self,
                                 start_date: datetime,
                                 end_date: datetime,
                                 top_n: int = 10,
                                 initial_capital: float = 10000.0) -> Dict:
        """
        Backtest relative strength strategy.
        
        Args:
            start_date: Start date
            end_date: End date
            top_n: Number of top assets to include
            initial_capital: Initial capital
            
        Returns:
            Dictionary with backtest results
        """
        # Create date range
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
            
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Initialize portfolio
        portfolio_value = initial_capital
        portfolio_values = [portfolio_value]
        dates = [date_range[0]]
        
        # Initialize holdings
        holdings = {}
        
        # Track allocations over time
        allocations = {}
        
        # Get price data
        price_data = {}
        
        for asset, data in self.asset_data.items():
            if 'close' in data.columns:
                price_data[asset] = data['close']
            else:
                # Use first numeric column
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    price_data[asset] = data[numeric_cols[0]]
                    
        if not price_data:
            logger.error("No price data available for backtest")
            return {}
            
        # Create price DataFrame
        price_df = pd.DataFrame(price_data)
        
        # Rebalance at start
        current_date = date_range[0]
        weights = self.get_allocation(current_date=current_date, top_n=top_n)
        allocations[current_date] = weights
        
        for asset, weight in weights.items():
            if asset in price_data:
                price = price_df.loc[current_date, asset]
                holdings[asset] = portfolio_value * weight / price
                
        # Run backtest
        last_rebalance_date = current_date
        
        for current_date in date_range[1:]:
            # Skip dates not in price data
            if current_date not in price_df.index:
                continue
                
            # Update portfolio value
            portfolio_value = 0
            
            for asset, shares in holdings.items():
                if current_date in price_df.index and asset in price_df.columns:
                    price = price_df.loc[current_date, asset]
                    value = shares * price
                    portfolio_value += value
                    
            portfolio_values.append(portfolio_value)
            dates.append(current_date)
            
            # Check if rebalance needed
            days_since_rebalance = (current_date - last_rebalance_date).days
            
            if days_since_rebalance >= self.rebalance_frequency:
                # Rebalance
                weights = self.get_allocation(current_date=current_date, top_n=top_n)
                allocations[current_date] = weights
                
                for asset, weight in weights.items():
                    if asset in price_data and current_date in price_df.index:
                        price = price_df.loc[current_date, asset]
                        holdings[asset] = portfolio_value * weight / price
                        
                # Update assets not in new allocation
                for asset in list(holdings.keys()):
                    if asset not in weights:
                        holdings[asset] = 0
                        
                last_rebalance_date = current_date
                
        # Calculate benchmark performance
        benchmark_values = None
        
        if self.benchmark_data is not None:
            if 'close' in self.benchmark_data.columns:
                benchmark_prices = self.benchmark_data['close']
            else:
                # Use first numeric column
                numeric_cols = self.benchmark_data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    benchmark_prices = self.benchmark_data[numeric_cols[0]]
                    
            if benchmark_prices is not None:
                # Align to backtest dates
                benchmark_prices = benchmark_prices.reindex(dates, method='ffill')
                
                # Calculate benchmark values
                benchmark_values = [initial_capital]
                
                for i in range(1, len(dates)):
                    if pd.notna(benchmark_prices.iloc[i-1]) and pd.notna(benchmark_prices.iloc[i]):
                        return_rate = benchmark_prices.iloc[i] / benchmark_prices.iloc[i-1]
                        benchmark_values.append(benchmark_values[-1] * return_rate)
                    else:
                        benchmark_values.append(benchmark_values[-1])
                        
        # Convert allocations to DataFrame
        all_assets = set()
        for weights in allocations.values():
            all_assets.update(weights.keys())
            
        allocation_data = {}
        
        for date, weights in allocations.items():
            allocation_data[date] = {asset: weights.get(asset, 0.0) for asset in all_assets}
            
        allocation_df = pd.DataFrame.from_dict(allocation_data, orient='index')
        
        # Forward fill allocation for all dates
        allocation_df = allocation_df.reindex(dates, method='ffill')
        
        # Calculate performance metrics
        returns = [portfolio_values[i] / portfolio_values[i-1] - 1 for i in range(1, len(portfolio_values))]
        
        total_return = portfolio_values[-1] / portfolio_values[0] - 1
        ann_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = ann_return / volatility if volatility > 0 else 0
        
        # Calculate drawdowns
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = [1 - val / peak if peak > 0 else 0 for val, peak in zip(portfolio_values, running_max)]
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Calculate benchmark-relative metrics
        if benchmark_values:
            benchmark_returns = [benchmark_values[i] / benchmark_values[i-1] - 1 for i in range(1, len(benchmark_values))]
            benchmark_total_return = benchmark_values[-1] / benchmark_values[0] - 1
            benchmark_ann_return = (1 + benchmark_total_return) ** (252 / len(benchmark_values)) - 1
            
            # Excess return
            excess_return = ann_return - benchmark_ann_return
            
            # Tracking error
            if len(returns) == len(benchmark_returns):
                tracking_error = np.std([r - b for r, b in zip(returns, benchmark_returns)]) * np.sqrt(252)
                information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            else:
                tracking_error = np.nan
                information_ratio = np.nan
            
            # Beta
            if len(returns) == len(benchmark_returns):
                covariance = np.cov(returns, benchmark_returns)[0][1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
                
                # Alpha
                alpha = ann_return - (0.0 + beta * benchmark_ann_return)  # Assuming risk-free rate of 0
            else:
                beta = np.nan
                alpha = np.nan
        else:
            excess_return = np.nan
            tracking_error = np.nan
            information_ratio = np.nan
            beta = np.nan
            alpha = np.nan
        
        # Compile results
        results = {
            'portfolio_values': portfolio_values,
            'dates': dates,
            'benchmark_values': benchmark_values,
            'total_return': total_return,
            'annualized_return': ann_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'excess_return': excess_return,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'allocation_history': allocation_df
        }
        
        return results


class EnhancedRelativeStrengthAllocator(RelativeStrengthAllocator):
    """
    Enhanced asset allocator based on relative strength with dynamic adjustments.
    
    This advanced allocator extends the basic RelativeStrengthAllocator with:
    1. Multiple relative strength calculation methods
    2. Regime-aware allocation adjustments
    3. Cross-asset correlation and cluster-based allocation
    4. Dynamic asset selection based on multiple factors
    5. Adaptive weighting schemes based on asset characteristics
    """
    
    class RankingMethod(Enum):
        """Ranking methods for relative strength calculation."""
        SIMPLE = "simple"               # Simple RS calculation
        MOMENTUM_ADJUSTED = "momentum"  # RS adjusted by momentum
        VOLATILITY_WEIGHTED = "volatility_weighted"  # RS weighted by inverse volatility
        MULTI_FACTOR = "multi_factor"   # RS combined with other factors
        
    class WeightingScheme(Enum):
        """Weighting schemes for RS-based allocation."""
        RANK_INVERSE = "rank_inverse"       # Weights inversely proportional to rank
        SCORE_PROPORTIONAL = "score_proportional"  # Weights proportional to RS score
        EQUAL_WEIGHT = "equal_weight"       # Equal weight to all selected assets
        SCORE_SQUARED = "score_squared"     # Weights proportional to squared RS score
        CLUSTER_BASED = "cluster_based"     # Weights based on asset clusters
        VOLATILITY_ADJUSTED = "volatility_adjusted"  # Weights adjusted by volatility
    
    def __init__(self, 
                benchmark_data: pd.DataFrame = None,
                lookback_periods: List[int] = None,
                min_rs_score: float = 0.0,
                ranking_method: RankingMethod = RankingMethod.SIMPLE,
                weighting_scheme: WeightingScheme = WeightingScheme.RANK_INVERSE,
                regime_data: pd.DataFrame = None,
                volatility_window: int = 63,
                correlation_threshold: float = 0.7,
                max_cluster_allocation: float = 0.4,
                turnover_limit: float = None,
                **kwargs):
        """
        Initialize enhanced relative strength allocator.
        
        Args:
            benchmark_data: Benchmark price data
            lookback_periods: List of periods for RS calculation
            min_rs_score: Minimum relative strength score for inclusion
            ranking_method: Method for ranking assets
            weighting_scheme: Scheme for weighting selected assets
            regime_data: Market regime data (optional)
            volatility_window: Window for volatility calculation
            correlation_threshold: Threshold for asset clustering
            max_cluster_allocation: Maximum allocation to any single cluster
            turnover_limit: Maximum allowed turnover (None = no limit)
            **kwargs: Additional arguments passed to RelativeStrengthAllocator
        """
        super().__init__(
            benchmark_data=benchmark_data,
            lookback_periods=lookback_periods,
            min_rs_score=min_rs_score,
            **kwargs
        )
        self.ranking_method = ranking_method
        self.weighting_scheme = weighting_scheme
        self.regime_data = regime_data
        self.volatility_window = volatility_window
        self.correlation_threshold = correlation_threshold
        self.max_cluster_allocation = max_cluster_allocation
        self.turnover_limit = turnover_limit
        self.asset_clusters = {}
        self.rs_history = {}
        self.additional_factors = {}
    
    def add_factor_data(self, factor_name: str, factor_data: Dict[str, float]) -> None:
        """
        Add additional factor data for multi-factor ranking.
        
        Args:
            factor_name: Name of the factor
            factor_data: Dictionary mapping assets to factor values
        """
        self.additional_factors[factor_name] = factor_data
        logger.info(f"Added factor data for '{factor_name}' with {len(factor_data)} assets")
    
    def set_regime_data(self, regime_data: pd.DataFrame) -> None:
        """
        Set market regime data for regime-aware allocation.
        
        Args:
            regime_data: DataFrame with regime classifications
        """
        self.regime_data = regime_data
        logger.info(f"Set regime data with {len(regime_data)} data points")
    
    def get_current_regime(self, current_date: datetime) -> Any:
        """
        Get current market regime for given date.
        
        Args:
            current_date: Current date
            
        Returns:
            Current regime value or None if not available
        """
        if self.regime_data is None or current_date is None:
            return None
            
        # Convert to datetime if string
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
            
        # Find closest date in regime data
        if current_date in self.regime_data.index:
            regime = self.regime_data.loc[current_date]
        else:
            # Find closest date less than or equal to current_date
            prev_dates = self.regime_data.index[self.regime_data.index <= current_date]
            if len(prev_dates) > 0:
                closest_date = prev_dates[-1]
                regime = self.regime_data.loc[closest_date]
            else:
                logger.warning(f"No regime data available for or before {current_date}")
                return None
                
        # Extract regime value
        if isinstance(regime, pd.Series):
            # If multiple columns, use first column
            regime = regime.iloc[0]
            
        return regime
    
    def calculate_momentum_adjusted_rs(self, 
                                      asset: str,
                                      benchmark_data: pd.DataFrame = None) -> float:
        """
        Calculate momentum-adjusted relative strength.
        
        Args:
            asset: Asset symbol
            benchmark_data: Benchmark data
            
        Returns:
            Momentum-adjusted RS score
        """
        # Calculate base RS
        base_rs = self._calculate_relative_strength(
            self.asset_data[asset],
            benchmark_data or self.benchmark_data,
            self.lookback_periods
        )
        
        # Calculate momentum
        momentum = self._calculate_momentum(self.asset_data[asset])
        
        # Adjust RS by momentum
        # If both are positive or both are negative, enhance the signal
        # If they have opposite signs, reduce the signal
        if (base_rs > 0 and momentum > 0) or (base_rs < 0 and momentum < 0):
            adjusted_rs = base_rs * (1 + abs(momentum))
        else:
            adjusted_rs = base_rs * (1 - abs(momentum) * 0.5)
            
        return adjusted_rs
    
    def calculate_volatility_weighted_rs(self, 
                                       asset: str,
                                       benchmark_data: pd.DataFrame = None) -> float:
        """
        Calculate volatility-weighted relative strength.
        
        Args:
            asset: Asset symbol
            benchmark_data: Benchmark data
            
        Returns:
            Volatility-weighted RS score
        """
        # Calculate base RS
        base_rs = self._calculate_relative_strength(
            self.asset_data[asset],
            benchmark_data or self.benchmark_data,
            self.lookback_periods
        )
        
        # Calculate asset volatility
        returns = self._get_returns(self.asset_data[asset])
        volatility = self._calculate_volatility(returns, self.volatility_window)
        
        # Calculate benchmark volatility
        if benchmark_data is not None:
            benchmark_returns = self._get_returns(benchmark_data)
            benchmark_volatility = self._calculate_volatility(benchmark_returns, self.volatility_window)
        else:
            # Use average volatility of all assets
            all_vols = []
            for a in self.asset_data:
                a_returns = self._get_returns(self.asset_data[a])
                a_vol = self._calculate_volatility(a_returns, self.volatility_window)
                if not np.isnan(a_vol):
                    all_vols.append(a_vol)
            
            benchmark_volatility = np.mean(all_vols) if all_vols else 1.0
        
        # Calculate relative volatility
        if benchmark_volatility > 0 and not np.isnan(volatility) and not np.isnan(benchmark_volatility):
            relative_volatility = volatility / benchmark_volatility
        else:
            relative_volatility = 1.0
            
        # Adjust RS by inverse volatility
        # Higher volatility -> lower weight, lower volatility -> higher weight
        volatility_factor = 1.0 / relative_volatility if relative_volatility > 0 else 1.0
        
        # Limit the volatility adjustment factor
        volatility_factor = min(max(volatility_factor, 0.5), 2.0)
        
        weighted_rs = base_rs * volatility_factor
        
        return weighted_rs
    
    def calculate_multi_factor_rs(self, 
                                asset: str,
                                benchmark_data: pd.DataFrame = None) -> float:
        """
        Calculate multi-factor relative strength.
        
        Args:
            asset: Asset symbol
            benchmark_data: Benchmark data
            
        Returns:
            Multi-factor RS score
        """
        # Calculate base RS
        base_rs = self._calculate_relative_strength(
            self.asset_data[asset],
            benchmark_data or self.benchmark_data,
            self.lookback_periods
        )
        
        # If no additional factors, return base RS
        if not self.additional_factors:
            return base_rs
            
        # Combine RS with additional factors
        factors = [base_rs]
        weights = [1.0]  # Base RS has weight 1.0
        
        for factor_name, factor_data in self.additional_factors.items():
            if asset in factor_data:
                factors.append(factor_data[asset])
                
                # Different weights for different factors
                if factor_name.lower() in ['momentum', 'trend']:
                    weights.append(0.8)
                elif factor_name.lower() in ['volatility', 'risk']:
                    weights.append(0.6)
                elif factor_name.lower() in ['value', 'quality', 'fundamental']:
                    weights.append(0.7)
                else:
                    weights.append(0.5)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate weighted average
        multi_factor_rs = sum(f * w for f, w in zip(factors, weights))
        
        return multi_factor_rs
    
    def calculate_enhanced_rs_rankings(self) -> pd.DataFrame:
        """
        Calculate enhanced relative strength rankings using the selected method.
        
        Returns:
            DataFrame with enhanced RS scores and rankings
        """
        assets = list(self.asset_data.keys())
        
        if not assets:
            return pd.DataFrame()
            
        # Calculate RS scores using the selected method
        rs_scores = {}
        
        for asset in assets:
            if self.ranking_method == self.RankingMethod.SIMPLE:
                rs = self._calculate_relative_strength(
                    self.asset_data[asset],
                    self.benchmark_data,
                    self.lookback_periods
                )
            elif self.ranking_method == self.RankingMethod.MOMENTUM_ADJUSTED:
                rs = self.calculate_momentum_adjusted_rs(asset)
            elif self.ranking_method == self.RankingMethod.VOLATILITY_WEIGHTED:
                rs = self.calculate_volatility_weighted_rs(asset)
            elif self.ranking_method == self.RankingMethod.MULTI_FACTOR:
                rs = self.calculate_multi_factor_rs(asset)
            else:
                # Default to simple RS
                rs = self._calculate_relative_strength(
                    self.asset_data[asset],
                    self.benchmark_data,
                    self.lookback_periods
                )
                
            rs_scores[asset] = rs
            
        # Convert to DataFrame
        rs_df = pd.DataFrame({
            'rs_score': rs_scores
        })
        
        # Add rankings
        rs_df['rs_rank'] = rs_df['rs_score'].rank(ascending=False)
        
        # Add additional metrics
        rs_df['volatility'] = [
            self._calculate_volatility(self._get_returns(self.asset_data[asset]), self.volatility_window)
            for asset in rs_df.index
        ]
        
        rs_df['momentum'] = [
            self._calculate_momentum(self.asset_data[asset])
            for asset in rs_df.index
        ]
        
        # Sort by ranking
        rs_df = rs_df.sort_values('rs_rank')
        
        # Store in history
        current_date = datetime.now()
        self.rs_history[current_date] = rs_df
        
        return rs_df
    
    def cluster_assets(self, top_assets: List[str]) -> Dict[int, List[str]]:
        """
        Cluster assets based on correlations to avoid over-concentration.
        
        Args:
            top_assets: List of top-ranked assets
            
        Returns:
            Dictionary mapping cluster IDs to lists of assets
        """
        if len(top_assets) <= 1:
            return {0: top_assets}
            
        # Get returns for selected assets
        returns_data = {}
        
        for asset in top_assets:
            returns = self._get_returns(self.asset_data[asset])
            if len(returns) > 0:
                returns_data[asset] = returns
                
        if not returns_data:
            # No return data, assign all to one cluster
            return {0: top_assets}
            
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Convert to distance matrix (1 - abs(corr))
        distance_matrix = 1 - corr_matrix.abs()
        
        # Simple clustering: connect assets with correlation above threshold
        clusters = {}
        assigned = set()
        cluster_id = 0
        
        for asset in top_assets:
            if asset in assigned:
                continue
                
            # Start a new cluster
            cluster = [asset]
            assigned.add(asset)
            
            # Find correlated assets
            if asset in corr_matrix.index:
                for other_asset in top_assets:
                    if other_asset != asset and other_asset not in assigned and other_asset in corr_matrix.columns:
                        if corr_matrix.loc[asset, other_asset] > self.correlation_threshold:
                            cluster.append(other_asset)
                            assigned.add(other_asset)
            
            clusters[cluster_id] = cluster
            cluster_id += 1
            
        # Assign any remaining assets to their own clusters
        for asset in top_assets:
            if asset not in assigned:
                clusters[cluster_id] = [asset]
                cluster_id += 1
                
        return clusters
    
    def adjust_allocation_for_regime(self, 
                                   weights: Dict[str, float], 
                                   current_date: datetime) -> Dict[str, float]:
        """
        Adjust allocation based on the current market regime.
        
        Args:
            weights: Current asset weights
            current_date: Current date
            
        Returns:
            Adjusted weights
        """
        if self.regime_data is None:
            return weights
            
        # Get current regime
        regime = self.get_current_regime(current_date)
        
        if regime is None:
            return weights
            
        # Calculate asset volatilities
        volatilities = {}
        for asset in weights:
            returns = self._get_returns(self.asset_data[asset])
            vol = self._calculate_volatility(returns, self.volatility_window)
            volatilities[asset] = vol if not np.isnan(vol) else float('inf')
            
        # Adjust weights based on regime
        adjusted_weights = weights.copy()
        
        # Regime-specific adjustments
        if str(regime).lower() in ['bear', 'high_volatility', 'crisis', 'contraction']:
            # In bear/high volatility regimes, reduce allocation to high-volatility assets
            # and increase allocation to low-volatility assets
            
            # Identify high and low volatility assets
            median_vol = np.median(list(volatilities.values()))
            high_vol_assets = [a for a, v in volatilities.items() if v > median_vol * 1.2]
            low_vol_assets = [a for a, v in volatilities.items() if v < median_vol * 0.8]
            
            # Reduce weights for high volatility assets
            for asset in high_vol_assets:
                if asset in adjusted_weights:
                    adjusted_weights[asset] *= 0.8
                    
            # Increase weights for low volatility assets
            extra_weight = sum(weights.get(a, 0) * 0.2 for a in high_vol_assets)
            if low_vol_assets and extra_weight > 0:
                total_low_vol_weight = sum(weights.get(a, 0) for a in low_vol_assets)
                if total_low_vol_weight > 0:
                    for asset in low_vol_assets:
                        if asset in adjusted_weights:
                            adjusted_weights[asset] += extra_weight * (weights[asset] / total_low_vol_weight)
                            
        elif str(regime).lower() in ['bull', 'low_volatility', 'expansion']:
            # In bull/low volatility regimes, increase allocation to high-momentum assets
            
            # Calculate momentum for all assets
            momentums = {}
            for asset in weights:
                momentum = self._calculate_momentum(self.asset_data[asset])
                momentums[asset] = momentum
                
            # Identify high momentum assets
            median_momentum = np.median(list(momentums.values()))
            high_momentum_assets = [a for a, m in momentums.items() if m > median_momentum * 1.2]
            low_momentum_assets = [a for a, m in momentums.items() if m < median_momentum * 0.8]
            
            # Increase weights for high momentum assets
            for asset in high_momentum_assets:
                if asset in adjusted_weights:
                    adjusted_weights[asset] *= 1.2
                    
            # Decrease weights for low momentum assets
            for asset in low_momentum_assets:
                if asset in adjusted_weights:
                    adjusted_weights[asset] *= 0.9
        
        # Normalize weights to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
            
        return adjusted_weights
    
    def calculate_cluster_allocation(self, 
                                   rankings: pd.DataFrame, 
                                   top_n: int = None) -> Dict[str, float]:
        """
        Calculate asset allocation with cluster-based constraints.
        
        Args:
            rankings: DataFrame with RS rankings
            top_n: Number of top assets to include
            
        Returns:
            Dictionary mapping assets to weights
        """
        if rankings.empty:
            return {}
            
        # Filter by minimum RS score
        if self.min_rs_score > 0:
            rankings = rankings[rankings['rs_score'] >= self.min_rs_score]
            
        if rankings.empty:
            return {}
            
        # Limit to top N assets
        if top_n and top_n > 0:
            top_assets = rankings.head(top_n).index.tolist()
        else:
            top_assets = rankings.index.tolist()
            
        # Cluster assets
        clusters = self.cluster_assets(top_assets)
        self.asset_clusters = clusters
        
        # Calculate initial weights within each cluster
        cluster_weights = {}
        asset_weights = {}
        
        for cluster_id, cluster_assets in clusters.items():
            # Calculate weights within cluster
            if self.weighting_scheme == self.WeightingScheme.EQUAL_WEIGHT:
                # Equal weight
                cluster_asset_weights = {asset: 1.0 / len(cluster_assets) for asset in cluster_assets}
            elif self.weighting_scheme == self.WeightingScheme.SCORE_PROPORTIONAL:
                # Weight proportional to RS score
                cluster_scores = {asset: rankings.loc[asset, 'rs_score'] for asset in cluster_assets}
                total_score = sum(max(0, score) for score in cluster_scores.values())
                if total_score > 0:
                    cluster_asset_weights = {asset: max(0, score) / total_score 
                                           for asset, score in cluster_scores.items()}
                else:
                    cluster_asset_weights = {asset: 1.0 / len(cluster_assets) for asset in cluster_assets}
            elif self.weighting_scheme == self.WeightingScheme.SCORE_SQUARED:
                # Weight proportional to squared RS score (emphasizes stronger assets)
                cluster_scores = {asset: rankings.loc[asset, 'rs_score'] for asset in cluster_assets}
                squared_scores = {asset: max(0, score)**2 for asset, score in cluster_scores.items()}
                total_squared = sum(squared_scores.values())
                if total_squared > 0:
                    cluster_asset_weights = {asset: score / total_squared 
                                           for asset, score in squared_scores.items()}
                else:
                    cluster_asset_weights = {asset: 1.0 / len(cluster_assets) for asset in cluster_assets}
            elif self.weighting_scheme == self.WeightingScheme.VOLATILITY_ADJUSTED:
                # Weight inversely proportional to volatility
                cluster_vols = {asset: rankings.loc[asset, 'volatility'] for asset in cluster_assets}
                # Replace any invalid values
                cluster_vols = {asset: vol if not np.isnan(vol) and vol > 0 else float('inf') 
                               for asset, vol in cluster_vols.items()}
                
                inv_vols = {asset: 1.0 / vol if vol > 0 and vol != float('inf') else 0.0 
                           for asset, vol in cluster_vols.items()}
                
                total_inv_vol = sum(inv_vols.values())
                if total_inv_vol > 0:
                    cluster_asset_weights = {asset: inv_vol / total_inv_vol 
                                           for asset, inv_vol in inv_vols.items()}
                else:
                    cluster_asset_weights = {asset: 1.0 / len(cluster_assets) for asset in cluster_assets}
            else:
                # Default to rank inverse
                ranks = {asset: rankings.loc[asset, 'rs_rank'] for asset in cluster_assets}
                inverse_ranks = {asset: 1.0 / rank if rank > 0 else 0.0 for asset, rank in ranks.items()}
                total_inverse = sum(inverse_ranks.values())
                if total_inverse > 0:
                    cluster_asset_weights = {asset: inv / total_inverse 
                                           for asset, inv in inverse_ranks.items()}
                else:
                    cluster_asset_weights = {asset: 1.0 / len(cluster_assets) for asset in cluster_assets}
                    
            # Store weights
            asset_weights.update(cluster_asset_weights)
            
            # Calculate cluster weight (sum of asset weights)
            cluster_weights[cluster_id] = sum(cluster_asset_weights.values())
        
        # Allocate across clusters
        # Default: weight proportional to cluster size and average RS
        cluster_sizes = {cid: len(assets) for cid, assets in clusters.items()}
        cluster_avg_rs = {}
        
        for cluster_id, cluster_assets in clusters.items():
            cluster_rs = [rankings.loc[asset, 'rs_score'] for asset in cluster_assets]
            cluster_avg_rs[cluster_id] = sum(cluster_rs) / len(cluster_rs) if cluster_rs else 0
            
        # Combine size and RS factors
        cluster_factors = {cid: size * (1 + max(0, avg_rs)) 
                         for cid, size, avg_rs in zip(
                             cluster_weights.keys(), 
                             cluster_sizes.values(), 
                             cluster_avg_rs.values()
                         )}
        
        total_factor = sum(cluster_factors.values())
        if total_factor > 0:
            norm_cluster_weights = {cid: factor / total_factor 
                                  for cid, factor in cluster_factors.items()}
        else:
            # Equal weight
            n_clusters = len(clusters)
            norm_cluster_weights = {cid: 1.0 / n_clusters for cid in clusters}
            
        # Apply max cluster constraint
        if self.max_cluster_allocation < 1.0:
            # Check if any cluster exceeds max allocation
            excess_weight = 0
            constrained_clusters = []
            
            for cid, weight in norm_cluster_weights.items():
                if weight > self.max_cluster_allocation:
                    excess_weight += weight - self.max_cluster_allocation
                    constrained_clusters.append(cid)
                    
            if excess_weight > 0:
                # Reduce constrained clusters to max allocation
                for cid in constrained_clusters:
                    norm_cluster_weights[cid] = self.max_cluster_allocation
                    
                # Redistribute excess to other clusters
                unconstrained = [cid for cid in norm_cluster_weights if cid not in constrained_clusters]
                if unconstrained:
                    total_unconstr_weight = sum(norm_cluster_weights[cid] for cid in unconstrained)
                    for cid in unconstrained:
                        if total_unconstr_weight > 0:
                            norm_cluster_weights[cid] += excess_weight * (norm_cluster_weights[cid] / total_unconstr_weight)
                        else:
                            norm_cluster_weights[cid] += excess_weight / len(unconstrained)
        
        # Calculate final asset weights
        final_weights = {}
        for cluster_id, cluster_assets in clusters.items():
            cluster_weight = norm_cluster_weights[cluster_id]
            for asset in cluster_assets:
                within_cluster_weight = asset_weights[asset] / cluster_weights[cluster_id]
                final_weights[asset] = cluster_weight * within_cluster_weight
                
        return final_weights
    
    def limit_turnover(self, 
                     new_weights: Dict[str, float], 
                     current_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Limit turnover to reduce transaction costs.
        
        Args:
            new_weights: New asset weights
            current_weights: Current asset weights (None = use self.current_allocations)
            
        Returns:
            Adjusted weights with limited turnover
        """
        if self.turnover_limit is None or self.turnover_limit >= 1.0:
            return new_weights
            
        current_weights = current_weights or self.current_allocations
        
        if not current_weights:
            return new_weights
            
        # Calculate turnover for each asset
        all_assets = set(new_weights.keys()) | set(current_weights.keys())
        turnover_by_asset = {}
        
        for asset in all_assets:
            current = current_weights.get(asset, 0.0)
            new = new_weights.get(asset, 0.0)
            turnover_by_asset[asset] = abs(new - current)
            
        # Calculate total turnover
        total_turnover = sum(turnover_by_asset.values()) / 2  # Divide by 2 because each trade affects two assets
        
        if total_turnover <= self.turnover_limit:
            return new_weights
            
        # Scale down changes to limit turnover
        scale_factor = self.turnover_limit / total_turnover
        
        # Adjust weights
        adjusted_weights = {}
        for asset in all_assets:
            current = current_weights.get(asset, 0.0)
            new = new_weights.get(asset, 0.0)
            # Scale the change by the factor
            change = (new - current) * scale_factor
            adjusted_weights[asset] = current + change
            
        # Normalize weights to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
            
        return adjusted_weights
    
    def get_allocation(self, 
                     method: AllocationMethod = None,
                     current_date: datetime = None,
                     top_n: int = None) -> Dict[str, float]:
        """
        Get enhanced allocation based on relative strength with dynamic adjustments.
        
        Args:
            method: Allocation method (ignored for this class)
            current_date: Current date
            top_n: Number of top assets to include
            
        Returns:
            Dictionary mapping assets to weights
        """
        # Calculate enhanced RS rankings
        rankings = self.calculate_enhanced_rs_rankings()
        
        if rankings.empty:
            return {}
        
        # Calculate base weights with cluster constraints
        weights = self.calculate_cluster_allocation(rankings, top_n)
        
        # Adjust for market regime
        if self.regime_data is not None and current_date is not None:
            weights = self.adjust_allocation_for_regime(weights, current_date)
            
        # Apply turnover limits
        if self.turnover_limit is not None and self.current_allocations:
            weights = self.limit_turnover(weights)
            
        # Apply general constraints
        weights = self._apply_allocation_constraints(weights)
        
        # Update current allocations and rebalance date
        self.current_allocations = weights
        self.last_rebalance_date = current_date if current_date else datetime.now()
        
        return weights
    
    def analyze_performance_attribution(self, 
                                      backtest_results: Dict,
                                      detailed: bool = False) -> Dict:
        """
        Analyze the contribution of different factors to performance.
        
        Args:
            backtest_results: Results from backtest_relative_strength
            detailed: Whether to include detailed breakdown
            
        Returns:
            Dictionary with performance attribution analysis
        """
        if not backtest_results or 'allocation_history' not in backtest_results:
            return {}
            
        allocation_df = backtest_results['allocation_history']
        dates = backtest_results['dates']
        
        # Get price data for all assets
        price_data = {}
        for asset in allocation_df.columns:
            if asset in self.asset_data:
                if 'close' in self.asset_data[asset].columns:
                    prices = self.asset_data[asset]['close']
                else:
                    # Use first numeric column
                    numeric_cols = self.asset_data[asset].select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        prices = self.asset_data[asset][numeric_cols[0]]
                    else:
                        continue
                
                # Align to backtest dates
                prices = prices.reindex(dates, method='ffill')
                price_data[asset] = prices
                
        if not price_data:
            return {}
            
        # Calculate returns for each asset
        returns_data = {}
        for asset, prices in price_data.items():
            returns = prices.pct_change().fillna(0)
            returns_data[asset] = returns
            
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data, index=dates)
        
        # Calculate contribution to return for each asset and date
        contribution_df = pd.DataFrame(index=dates, columns=allocation_df.columns, data=0.0)
        
        # Shift allocations forward by 1 day (yesterday's allocation affects today's return)
        shifted_allocations = allocation_df.shift(1).fillna(0)
        
        for asset in allocation_df.columns:
            if asset in returns_df.columns:
                contribution_df[asset] = shifted_allocations[asset] * returns_df[asset]
                
        # Calculate total contribution per date
        contribution_df['total'] = contribution_df.sum(axis=1)
        
        # Calculate cumulative contribution
        cumulative_contribution = contribution_df.cumsum()
        
        # Calculate attribution by asset
        total_contribution = contribution_df.sum()
        attribution_by_asset = total_contribution / total_contribution['total'] if total_contribution['total'] != 0 else total_contribution * 0
        
        # If regime data is available, calculate attribution by regime
        regime_attribution = {}
        if self.regime_data is not None:
            # Align regime data to backtest dates
            regime_series = pd.Series(index=dates, data=None)
            
            for date in dates:
                regime_series[date] = self.get_current_regime(date)
                
            # Group by regime
            if not regime_series.empty:
                regimes = regime_series.dropna().unique()
                for regime in regimes:
                    regime_dates = regime_series[regime_series == regime].index
                    if len(regime_dates) > 0:
                        regime_contrib = contribution_df.loc[regime_dates, 'total'].sum()
                        regime_attribution[regime] = regime_contrib
                        
                # Normalize to percentages
                total_contrib = sum(regime_attribution.values())
                if total_contrib != 0:
                    regime_attribution = {k: v / total_contrib for k, v in regime_attribution.items()}
        
        # Compile results
        results = {
            'total_return': backtest_results['total_return'],
            'attribution_by_asset': attribution_by_asset.drop('total').to_dict(),
            'cumulative_contribution': cumulative_contribution,
            'regime_attribution': regime_attribution
        }
        
        # Add detailed breakdown if requested
        if detailed:
            results['contribution_by_date'] = contribution_df
            
            # Calculate attribution by cluster
            if hasattr(self, 'asset_clusters') and self.asset_clusters:
                cluster_attribution = {}
                
                for cluster_id, cluster_assets in self.asset_clusters.items():
                    # Filter to assets in this cluster
                    cluster_assets_in_data = [a for a in cluster_assets if a in attribution_by_asset.index]
                    if cluster_assets_in_data:
                        cluster_contrib = attribution_by_asset[cluster_assets_in_data].sum()
                        cluster_attribution[f"Cluster {cluster_id}"] = cluster_contrib
                        
                results['cluster_attribution'] = cluster_attribution
                
            # Calculate attribution by factor if using multi-factor ranking
            if self.ranking_method == self.RankingMethod.MULTI_FACTOR and self.additional_factors:
                # We cannot directly attribute to factors, but we can show correlation
                # between factor values and asset contribution
                factor_correlation = {}
                
                for factor_name, factor_data in self.additional_factors.items():
                    factor_values = []
                    asset_contributions = []
                    
                    for asset, contribution in attribution_by_asset.items():
                        if asset in factor_data and asset != 'total':
                            factor_values.append(factor_data[asset])
                            asset_contributions.append(contribution)
                            
                    if factor_values and asset_contributions:
                        correlation = np.corrcoef(factor_values, asset_contributions)[0, 1]
                        factor_correlation[factor_name] = correlation
                        
                results['factor_correlation'] = factor_correlation
                
        return results