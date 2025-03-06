"""
Cross-market correlation analysis module.

This module provides tools to analyze correlations between different asset classes
and markets to identify potential cross-market opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from statsmodels.tsa.stattools import grangercausalitytests, coint

logger = logging.getLogger(__name__)

class CrossMarketCorrelationAnalyzer:
    """
    Analyzes correlations and relationships between different markets and asset classes.
    
    This class provides functionality to identify correlations, lead-lag relationships,
    cointegration, and other statistical relationships between different markets and
    asset classes that can be leveraged for trading strategies.
    """
    
    def __init__(self, default_window: int = 60, min_history: int = 252):
        """
        Initialize the correlation analyzer.
        
        Args:
            default_window: Default rolling window size for correlation analysis
            min_history: Minimum history required for meaningful analysis
        """
        self.default_window = default_window
        self.min_history = min_history
        self.asset_data = {}
        self.asset_classes = {}
        self.market_regions = {}
        
    def add_asset_data(self, 
                     symbol: str, 
                     data: pd.DataFrame,
                     asset_class: str = None,
                     market_region: str = None) -> None:
        """
        Add price data for an asset to the analyzer.
        
        Args:
            symbol: Symbol or identifier for the asset
            data: DataFrame containing price/returns data
            asset_class: Asset class (e.g., 'equity', 'bond', 'commodity', 'forex')
            market_region: Market region (e.g., 'US', 'Europe', 'Asia')
        """
        self.asset_data[symbol] = data
        
        if asset_class:
            self.asset_classes[symbol] = asset_class
            
        if market_region:
            self.market_regions[symbol] = market_region
            
        logger.info(f"Added data for {symbol} ({asset_class or 'unknown class'}, {market_region or 'unknown region'})")
    
    def compute_correlation_matrix(self, 
                                 symbols: List[str] = None,
                                 method: str = 'pearson',
                                 return_type: bool = True) -> pd.DataFrame:
        """
        Compute correlation matrix between multiple assets.
        
        Args:
            symbols: List of symbols to include (all if None)
            method: Correlation method ('pearson' or 'spearman')
            return_type: Whether to use returns (True) or prices (False)
            
        Returns:
            DataFrame containing the correlation matrix
        """
        symbols = symbols or list(self.asset_data.keys())
        
        # Create a DataFrame with aligned data
        aligned_data = {}
        
        for symbol in symbols:
            if symbol not in self.asset_data:
                logger.warning(f"No data found for symbol {symbol}")
                continue
                
            # Get price or return data
            if return_type:
                # Convert to returns if needed
                if 'returns' in self.asset_data[symbol].columns:
                    series = self.asset_data[symbol]['returns']
                elif 'close' in self.asset_data[symbol].columns:
                    series = self.asset_data[symbol]['close'].pct_change().dropna()
                else:
                    # Use first numeric column
                    numeric_cols = self.asset_data[symbol].select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        series = self.asset_data[symbol][numeric_cols[0]].pct_change().dropna()
                    else:
                        logger.warning(f"No suitable numeric data found for {symbol}")
                        continue
            else:
                # Use price data
                if 'close' in self.asset_data[symbol].columns:
                    series = self.asset_data[symbol]['close']
                else:
                    # Use first numeric column
                    numeric_cols = self.asset_data[symbol].select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        series = self.asset_data[symbol][numeric_cols[0]]
                    else:
                        logger.warning(f"No suitable numeric data found for {symbol}")
                        continue
                        
            aligned_data[symbol] = series
            
        # Create DataFrame with aligned dates
        df = pd.DataFrame(aligned_data)
        df = df.dropna()
        
        if len(df) < self.min_history:
            logger.warning(f"Limited data for correlation analysis: only {len(df)} common data points")
            
        # Compute correlation matrix
        if method == 'spearman':
            corr_matrix = df.corr(method='spearman')
        else:
            corr_matrix = df.corr(method='pearson')
            
        return corr_matrix
    
    def compute_rolling_correlations(self,
                                   symbol1: str,
                                   symbol2: str,
                                   window: int = None,
                                   method: str = 'pearson',
                                   return_type: bool = True) -> pd.Series:
        """
        Compute rolling correlation between two assets.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            window: Rolling window size (uses default_window if None)
            method: Correlation method ('pearson' or 'spearman')
            return_type: Whether to use returns (True) or prices (False)
            
        Returns:
            Series containing rolling correlations
        """
        window = window or self.default_window
        
        if symbol1 not in self.asset_data or symbol2 not in self.asset_data:
            logger.error(f"Data not found for one or both symbols: {symbol1}, {symbol2}")
            return pd.Series()
            
        # Get data for both symbols
        data1 = self.asset_data[symbol1]
        data2 = self.asset_data[symbol2]
        
        # Extract price or return series
        if return_type:
            # Convert to returns if needed
            if 'returns' in data1.columns:
                series1 = data1['returns']
            elif 'close' in data1.columns:
                series1 = data1['close'].pct_change().dropna()
            else:
                # Use first numeric column
                numeric_cols = data1.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    series1 = data1[numeric_cols[0]].pct_change().dropna()
                else:
                    logger.error(f"No suitable numeric data found for {symbol1}")
                    return pd.Series()
                    
            if 'returns' in data2.columns:
                series2 = data2['returns']
            elif 'close' in data2.columns:
                series2 = data2['close'].pct_change().dropna()
            else:
                # Use first numeric column
                numeric_cols = data2.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    series2 = data2[numeric_cols[0]].pct_change().dropna()
                else:
                    logger.error(f"No suitable numeric data found for {symbol2}")
                    return pd.Series()
        else:
            # Use price data
            if 'close' in data1.columns:
                series1 = data1['close']
            else:
                # Use first numeric column
                numeric_cols = data1.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    series1 = data1[numeric_cols[0]]
                else:
                    logger.error(f"No suitable numeric data found for {symbol1}")
                    return pd.Series()
                    
            if 'close' in data2.columns:
                series2 = data2['close']
            else:
                # Use first numeric column
                numeric_cols = data2.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    series2 = data2[numeric_cols[0]]
                else:
                    logger.error(f"No suitable numeric data found for {symbol2}")
                    return pd.Series()
                
        # Align dates
        df = pd.DataFrame({symbol1: series1, symbol2: series2})
        df = df.dropna()
        
        if len(df) < window:
            logger.warning(f"Insufficient data for rolling correlation: {len(df)} vs window {window}")
            return pd.Series()
            
        # Compute rolling correlation
        if method == 'spearman':
            rolling_corr = df[symbol1].rolling(window=window).corr(df[symbol2], method='spearman')
        else:
            rolling_corr = df[symbol1].rolling(window=window).corr(df[symbol2])
            
        return rolling_corr

    def analyze_lead_lag(self, 
                        symbol1: str,
                        symbol2: str,
                        max_lags: int = 10,
                        return_type: bool = True) -> Dict:
        """
        Analyze lead-lag relationship between two assets using cross-correlation.
        
        Args:
            symbol1: First symbol (potential leading asset)
            symbol2: Second symbol (potential lagging asset)
            max_lags: Maximum number of lags to test
            return_type: Whether to use returns (True) or prices (False)
            
        Returns:
            Dictionary with lead-lag analysis results
        """
        if symbol1 not in self.asset_data or symbol2 not in self.asset_data:
            logger.error(f"Data not found for one or both symbols: {symbol1}, {symbol2}")
            return {}
            
        # Get data for both symbols
        data1 = self.asset_data[symbol1]
        data2 = self.asset_data[symbol2]
        
        # Extract price or return series
        if return_type:
            # Extract returns
            if 'returns' in data1.columns:
                series1 = data1['returns']
            elif 'close' in data1.columns:
                series1 = data1['close'].pct_change().dropna()
            else:
                # Use first numeric column
                numeric_cols = data1.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    series1 = data1[numeric_cols[0]].pct_change().dropna()
                else:
                    logger.error(f"No suitable numeric data found for {symbol1}")
                    return {}
                    
            if 'returns' in data2.columns:
                series2 = data2['returns']
            elif 'close' in data2.columns:
                series2 = data2['close'].pct_change().dropna()
            else:
                # Use first numeric column
                numeric_cols = data2.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    series2 = data2[numeric_cols[0]].pct_change().dropna()
                else:
                    logger.error(f"No suitable numeric data found for {symbol2}")
                    return {}
        else:
            # Use price data
            if 'close' in data1.columns:
                series1 = data1['close']
            else:
                # Use first numeric column
                numeric_cols = data1.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    series1 = data1[numeric_cols[0]]
                else:
                    logger.error(f"No suitable numeric data found for {symbol1}")
                    return {}
                    
            if 'close' in data2.columns:
                series2 = data2['close']
            else:
                # Use first numeric column
                numeric_cols = data2.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    series2 = data2[numeric_cols[0]]
                else:
                    logger.error(f"No suitable numeric data found for {symbol2}")
                    return {}
                
        # Align dates
        df = pd.DataFrame({symbol1: series1, symbol2: series2})
        df = df.dropna()
        
        if len(df) < self.min_history:
            logger.warning(f"Limited data for lead-lag analysis: only {len(df)} common data points")
            
        # Compute cross-correlation for different lags
        lag_corrs = []
        for lag in range(-max_lags, max_lags + 1):
            if lag < 0:
                # symbol2 leads symbol1
                s1 = df[symbol1].iloc[abs(lag):]
                s2 = df[symbol2].iloc[:len(s1)]
            elif lag > 0:
                # symbol1 leads symbol2
                s2 = df[symbol2].iloc[lag:]
                s1 = df[symbol1].iloc[:len(s2)]
            else:
                # Contemporaneous
                s1 = df[symbol1]
                s2 = df[symbol2]
                
            corr, p_value = pearsonr(s1, s2)
            lag_corrs.append({'lag': lag, 'correlation': corr, 'p_value': p_value})
            
        # Find max correlation and its lag
        max_corr_info = max(lag_corrs, key=lambda x: abs(x['correlation']))
        
        # Determine lead-lag relationship
        max_lag = max_corr_info['lag']
        if max_lag < 0:
            lead_asset = symbol2
            lag_asset = symbol1
            relationship = f"{symbol2} leads {symbol1} by {abs(max_lag)} days"
        elif max_lag > 0:
            lead_asset = symbol1
            lag_asset = symbol2
            relationship = f"{symbol1} leads {symbol2} by {max_lag} days"
        else:
            lead_asset = None
            lag_asset = None
            relationship = "No clear lead-lag relationship (contemporaneous)"
            
        # Run Granger causality test
        granger_results = {}
        try:
            granger_test = grangercausalitytests(df[[symbol1, symbol2]], maxlag=max_lags, verbose=False)
            
            # Extract F-test p-values for each lag
            granger_p_values = []
            for lag, result in granger_test.items():
                # Get p-value for Wald F-test
                p_value = result[0]['ssr_ftest'][1]
                granger_p_values.append({'lag': lag, 'p_value': p_value})
                
            # Find minimum p-value and its lag
            min_p_value_info = min(granger_p_values, key=lambda x: x['p_value'])
            
            granger_results = {
                '1_causes_2': granger_test[min_p_value_info['lag']][0]['ssr_ftest'][1] < 0.05,
                'optimal_lag': min_p_value_info['lag'],
                'p_value': min_p_value_info['p_value']
            }
        except Exception as e:
            logger.warning(f"Error running Granger causality test: {str(e)}")
            
        # Compile results
        results = {
            'max_correlation': max_corr_info['correlation'],
            'optimal_lag': max_lag,
            'lead_asset': lead_asset,
            'lag_asset': lag_asset,
            'relationship': relationship,
            'significance': max_corr_info['p_value'] < 0.05,
            'lag_correlations': lag_corrs,
            'granger_causality': granger_results
        }
        
        return results
    
    def check_cointegration(self,
                          symbol1: str,
                          symbol2: str,
                          return_type: bool = False) -> Dict:
        """
        Test for cointegration between two assets.
        
        Cointegration indicates a potential mean-reverting relationship
        that can be exploited for pairs trading strategies.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            return_type: False for prices (default), True for returns
            
        Returns:
            Dictionary with cointegration test results
        """
        if symbol1 not in self.asset_data or symbol2 not in self.asset_data:
            logger.error(f"Data not found for one or both symbols: {symbol1}, {symbol2}")
            return {}
            
        # For cointegration, we need price series not returns
        if return_type:
            logger.warning("Using returns for cointegration test is not recommended; switching to prices")
            return_type = False
            
        # Get data for both symbols
        data1 = self.asset_data[symbol1]
        data2 = self.asset_data[symbol2]
        
        # Extract price series
        if 'close' in data1.columns:
            series1 = data1['close']
        else:
            # Use first numeric column
            numeric_cols = data1.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                series1 = data1[numeric_cols[0]]
            else:
                logger.error(f"No suitable numeric data found for {symbol1}")
                return {}
                
        if 'close' in data2.columns:
            series2 = data2['close']
        else:
            # Use first numeric column
            numeric_cols = data2.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                series2 = data2[numeric_cols[0]]
            else:
                logger.error(f"No suitable numeric data found for {symbol2}")
                return {}
                
        # Align dates
        df = pd.DataFrame({symbol1: series1, symbol2: series2})
        df = df.dropna()
        
        if len(df) < self.min_history:
            logger.warning(f"Limited data for cointegration test: only {len(df)} common data points")
            
        # Run cointegration test
        try:
            # Perform Engle-Granger cointegration test
            coint_result = coint(df[symbol1], df[symbol2])
            
            # Test statistic, p-value, and critical values
            test_stat, p_value, critical_values = coint_result
            
            # Determine if cointegrated based on p-value
            is_cointegrated = p_value < 0.05
            
            # Calculate optimal hedge ratio using OLS
            y = df[symbol1]
            x = df[symbol2]
            x = pd.DataFrame(x)
            x = x.assign(const=1)
            
            import statsmodels.api as sm
            model = sm.OLS(y, x)
            results = model.fit()
            hedge_ratio = results.params[0]
            
            # Calculate spread
            spread = df[symbol1] - hedge_ratio * df[symbol2]
            
            # Calculate spread statistics
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            # Calculate half-life of mean reversion
            from scipy import stats
            
            spread_lag = spread.shift(1)
            spread_lag.iloc[0] = spread_lag.iloc[1]
            spread_ret = spread - spread_lag
            spread_ret.iloc[0] = spread_ret.iloc[1]
            spread_lag2 = sm.add_constant(spread_lag)
            
            model = sm.OLS(spread_ret, spread_lag2)
            res = model.fit()
            
            half_life = -np.log(2) / res.params[1] if res.params[1] < 0 else np.nan
            
            # Compile results
            results = {
                'is_cointegrated': is_cointegrated,
                'p_value': p_value,
                'test_statistic': test_stat,
                'critical_values': critical_values,
                'hedge_ratio': hedge_ratio,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'half_life': half_life,
                'pairs_trading_potential': is_cointegrated and not np.isnan(half_life) and half_life > 1 and half_life < 252
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error running cointegration test: {str(e)}")
            return {}
    
    def find_correlated_pairs(self, 
                            symbols: List[str] = None,
                            min_correlation: float = 0.7,
                            max_correlation: float = 1.0,
                            require_diff_asset_class: bool = False,
                            method: str = 'pearson') -> List[Dict]:
        """
        Find pairs of assets with correlation above a threshold.
        
        Args:
            symbols: List of symbols to consider (all if None)
            min_correlation: Minimum absolute correlation
            max_correlation: Maximum absolute correlation (to exclude perfect correlation)
            require_diff_asset_class: Whether to require different asset classes
            method: Correlation method ('pearson' or 'spearman')
            
        Returns:
            List of dictionaries with correlated pairs
        """
        symbols = symbols or list(self.asset_data.keys())
        
        # Compute correlation matrix
        corr_matrix = self.compute_correlation_matrix(symbols, method=method)
        
        # Find pairs with correlation above threshold
        pairs = []
        
        for i, symbol1 in enumerate(corr_matrix.columns):
            for j, symbol2 in enumerate(corr_matrix.columns):
                if i >= j:  # Avoid duplicates and self-correlations
                    continue
                    
                correlation = corr_matrix.loc[symbol1, symbol2]
                abs_corr = abs(correlation)
                
                if abs_corr >= min_correlation and abs_corr <= max_correlation:
                    # Check if asset classes are different (if required)
                    if require_diff_asset_class:
                        class1 = self.asset_classes.get(symbol1)
                        class2 = self.asset_classes.get(symbol2)
                        
                        if not class1 or not class2 or class1 == class2:
                            continue
                    
                    pairs.append({
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'correlation': correlation,
                        'asset_class1': self.asset_classes.get(symbol1, 'unknown'),
                        'asset_class2': self.asset_classes.get(symbol2, 'unknown'),
                        'market_region1': self.market_regions.get(symbol1, 'unknown'),
                        'market_region2': self.market_regions.get(symbol2, 'unknown')
                    })
        
        # Sort by absolute correlation (highest first)
        pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return pairs
    
    def find_potential_pairs_trades(self, 
                                  symbols: List[str] = None,
                                  min_correlation: float = 0.7,
                                  require_cointegration: bool = True) -> List[Dict]:
        """
        Find pairs of assets suitable for pairs trading.
        
        Args:
            symbols: List of symbols to consider (all if None)
            min_correlation: Minimum absolute correlation
            require_cointegration: Whether to require cointegration
            
        Returns:
            List of dictionaries with potential pairs trades
        """
        symbols = symbols or list(self.asset_data.keys())
        
        # First find correlated pairs
        correlated_pairs = self.find_correlated_pairs(
            symbols=symbols,
            min_correlation=min_correlation
        )
        
        # Analyze each pair for pairs trading potential
        pairs_trades = []
        
        for pair in correlated_pairs:
            symbol1 = pair['symbol1']
            symbol2 = pair['symbol2']
            
            # Check for cointegration if required
            if require_cointegration:
                coint_results = self.check_cointegration(symbol1, symbol2)
                
                if not coint_results.get('is_cointegrated', False):
                    continue
                    
                # Add cointegration info to pair data
                pair.update({
                    'hedge_ratio': coint_results.get('hedge_ratio'),
                    'half_life': coint_results.get('half_life'),
                    'spread_mean': coint_results.get('spread_mean'),
                    'spread_std': coint_results.get('spread_std')
                })
            
            # Add to potential pairs trades
            pairs_trades.append(pair)
        
        # Sort by half-life (lower is better for trading)
        if require_cointegration:
            pairs_trades.sort(key=lambda x: x.get('half_life', float('inf')))
        
        return pairs_trades
    
    def find_cross_asset_opportunities(self, 
                                     asset_classes: List[str] = None,
                                     market_regions: List[str] = None,
                                     min_correlation: float = 0.5,
                                     max_lags: int = 10) -> List[Dict]:
        """
        Find cross-asset trading opportunities.
        
        Args:
            asset_classes: Asset classes to consider (all if None)
            market_regions: Market regions to consider (all if None)
            min_correlation: Minimum absolute correlation
            max_lags: Maximum lags for lead-lag analysis
            
        Returns:
            List of dictionaries with cross-asset opportunities
        """
        # Filter symbols based on asset classes and market regions
        symbols = []
        
        for symbol, data in self.asset_data.items():
            # Check asset class
            if asset_classes and symbol in self.asset_classes:
                if self.asset_classes[symbol] not in asset_classes:
                    continue
                    
            # Check market region
            if market_regions and symbol in self.market_regions:
                if self.market_regions[symbol] not in market_regions:
                    continue
                    
            symbols.append(symbol)
            
        # Find correlated pairs across different asset classes
        cross_asset_pairs = self.find_correlated_pairs(
            symbols=symbols,
            min_correlation=min_correlation,
            require_diff_asset_class=True
        )
        
        # Analyze lead-lag relationships
        opportunities = []
        
        for pair in cross_asset_pairs:
            symbol1 = pair['symbol1']
            symbol2 = pair['symbol2']
            
            # Check lead-lag relationship
            lead_lag = self.analyze_lead_lag(symbol1, symbol2, max_lags=max_lags)
            
            if lead_lag.get('optimal_lag', 0) != 0 and lead_lag.get('significance', False):
                # Add lead-lag info to pair data
                pair.update({
                    'lead_asset': lead_lag.get('lead_asset'),
                    'lag_asset': lead_lag.get('lag_asset'),
                    'optimal_lag': lead_lag.get('optimal_lag'),
                    'max_correlation': lead_lag.get('max_correlation'),
                    'relationship': lead_lag.get('relationship')
                })
                
                # Add to opportunities
                opportunities.append(pair)
        
        # Sort by absolute correlation (highest first)
        opportunities.sort(key=lambda x: abs(x.get('max_correlation', 0)), reverse=True)
        
        return opportunities
    
    def visualize_correlation_matrix(self, 
                                   symbols: List[str] = None,
                                   cmap: str = 'coolwarm',
                                   figsize: Tuple[int, int] = (10, 8),
                                   method: str = 'pearson') -> plt.Figure:
        """
        Visualize correlation matrix between assets.
        
        Args:
            symbols: List of symbols to include (all if None)
            cmap: Colormap for heatmap
            figsize: Figure size
            method: Correlation method ('pearson' or 'spearman')
            
        Returns:
            Matplotlib figure
        """
        # Compute correlation matrix
        corr_matrix = self.compute_correlation_matrix(symbols, method=method)
        
        if corr_matrix.empty:
            logger.error("Empty correlation matrix, cannot visualize")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            ax=ax,
            fmt='.2f'
        )
        
        # Add title and labels
        title = f"Asset Correlation Matrix ({method.capitalize()})"
        
        if symbols and len(symbols) <= 10:
            title += f" - {', '.join(symbols)}"
            
        ax.set_title(title)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_rolling_correlation(self,
                                    symbol1: str,
                                    symbol2: str,
                                    window: int = None,
                                    figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Visualize rolling correlation between two assets.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            window: Rolling window size (uses default_window if None)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Compute rolling correlation
        rolling_corr = self.compute_rolling_correlations(symbol1, symbol2, window=window)
        
        if rolling_corr.empty:
            logger.error("Empty rolling correlation, cannot visualize")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot rolling correlation
        ax.plot(rolling_corr.index, rolling_corr, label=f"{window}-day Rolling Correlation")
        
        # Add horizontal lines at 0, 0.5, and -0.5
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5)
        
        # Add title and labels
        ax.set_title(f"Rolling Correlation: {symbol1} vs {symbol2}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Correlation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Formatting
        ax.set_ylim(-1.05, 1.05)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_lead_lag_relationship(self,
                                      symbol1: str,
                                      symbol2: str,
                                      max_lags: int = 10,
                                      figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Visualize lead-lag relationship between two assets.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            max_lags: Maximum number of lags to analyze
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Analyze lead-lag relationship
        lead_lag = self.analyze_lead_lag(symbol1, symbol2, max_lags=max_lags)
        
        if not lead_lag:
            logger.error("Lead-lag analysis failed, cannot visualize")
            return None
            
        # Extract lag correlations
        lag_corrs = lead_lag.get('lag_correlations', [])
        
        if not lag_corrs:
            logger.error("No lag correlation data, cannot visualize")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract lags and correlations
        lags = [item['lag'] for item in lag_corrs]
        correlations = [item['correlation'] for item in lag_corrs]
        p_values = [item['p_value'] for item in lag_corrs]
        
        # Plot correlations
        ax.bar(
            lags, 
            correlations, 
            alpha=0.7,
            color=['green' if c > 0 else 'red' for c in correlations]
        )
        
        # Add markers for significant correlations
        for i, (lag, corr, p) in enumerate(zip(lags, correlations, p_values)):
            if p < 0.05:
                ax.plot(lag, corr, 'ko', markersize=8, alpha=0.6)
                
        # Add vertical line at lag 0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add annotations for lead-lag relationship
        if lead_lag.get('optimal_lag', 0) != 0:
            optimal_lag = lead_lag['optimal_lag']
            relationship = lead_lag['relationship']
            
            # Add text annotation
            ax.text(
                0.05, 0.95,
                relationship,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # Highlight optimal lag
            ax.axvline(x=optimal_lag, color='blue', linestyle='--', alpha=0.7)
            
        # Add title and labels
        ax.set_title(f"Lead-Lag Relationship: {symbol1} vs {symbol2}")
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Correlation")
        
        # Set x-axis ticks
        ax.set_xticks(lags)
        
        # Add note about lag interpretation
        ax.text(
            0.05, 0.05,
            f"Negative lag: {symbol2} leads {symbol1}\nPositive lag: {symbol1} leads {symbol2}",
            transform=ax.transAxes,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        return fig
    
    def visualize_pairs_trading_opportunity(self,
                                          symbol1: str,
                                          symbol2: str,
                                          lookback: int = 252,
                                          figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Visualize a potential pairs trading opportunity.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            lookback: Number of days to look back
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Check cointegration
        coint_results = self.check_cointegration(symbol1, symbol2)
        
        if not coint_results or not coint_results.get('is_cointegrated', False):
            logger.warning(f"Assets {symbol1} and {symbol2} are not cointegrated")
            
        # Extract price data
        if symbol1 not in self.asset_data or symbol2 not in self.asset_data:
            logger.error(f"Data not found for one or both symbols: {symbol1}, {symbol2}")
            return None
            
        data1 = self.asset_data[symbol1]
        data2 = self.asset_data[symbol2]
        
        # Extract price series
        if 'close' in data1.columns:
            series1 = data1['close']
        else:
            # Use first numeric column
            numeric_cols = data1.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                series1 = data1[numeric_cols[0]]
            else:
                logger.error(f"No suitable numeric data found for {symbol1}")
                return None
                
        if 'close' in data2.columns:
            series2 = data2['close']
        else:
            # Use first numeric column
            numeric_cols = data2.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                series2 = data2[numeric_cols[0]]
            else:
                logger.error(f"No suitable numeric data found for {symbol2}")
                return None
                
        # Align dates
        df = pd.DataFrame({symbol1: series1, symbol2: series2})
        df = df.dropna()
        
        # Limit to lookback period
        if len(df) > lookback:
            df = df.iloc[-lookback:]
            
        # Calculate hedge ratio
        hedge_ratio = coint_results.get('hedge_ratio', 1.0)
        
        # Calculate spread
        spread = df[symbol1] - hedge_ratio * df[symbol2]
        
        # Calculate z-score
        mean = spread.mean()
        std = spread.std()
        z_score = (spread - mean) / std
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, height_ratios=[2, 1, 1], sharex=True)
        
        # Plot normalized price series
        norm1 = df[symbol1] / df[symbol1].iloc[0]
        norm2 = df[symbol2] / df[symbol2].iloc[0]
        
        ax1.plot(df.index, norm1, label=symbol1)
        ax1.plot(df.index, norm2, label=symbol2)
        ax1.set_title(f"Normalized Prices: {symbol1} vs {symbol2}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot spread
        ax2.plot(df.index, spread, color='purple')
        ax2.axhline(y=mean, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=mean + 2*std, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=mean - 2*std, color='green', linestyle='--', alpha=0.5)
        ax2.set_title(f"Spread: {symbol1} - ({hedge_ratio:.4f} * {symbol2})")
        ax2.grid(True, alpha=0.3)
        
        # Plot z-score
        ax3.plot(df.index, z_score, color='blue')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=-2, color='green', linestyle='--', alpha=0.5)
        ax3.set_title("Z-Score")
        ax3.set_xlabel("Date")
        ax3.grid(True, alpha=0.3)
        
        # Add annotations
        if coint_results:
            half_life = coint_results.get('half_life', np.nan)
            p_value = coint_results.get('p_value', np.nan)
            
            info_text = (
                f"Cointegration p-value: {p_value:.4f}\n"
                f"Hedge ratio: {hedge_ratio:.4f}\n"
                f"Half-life: {half_life:.1f} days"
            )
            
            ax1.text(
                0.02, 0.05,
                info_text,
                transform=ax1.transAxes,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
        plt.tight_layout()
        
        return fig