#!/usr/bin/env python
"""
Benchmark Relative Performance Module

This module provides tools for:
1. Calculating relative performance metrics against market benchmarks
2. Dynamic beta adjustment for market-neutral strategies
3. Sector-specific benchmark comparisons
4. Outperformance attribution analysis
5. Strategy alpha decomposition

Key features:
- Benchmark beta calculation 
- Custom benchmark construction
- Time-varying beta measurement
- Regime-specific performance analysis
- Attribution of returns to market factors
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

# Import the new BenchmarkRelativeScorer implementation
from src.performance_scoring.benchmark_relative import BenchmarkRelativeScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark_performance.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class BenchmarkPerformanceAnalyzer:
    """Class for benchmark-relative performance analysis."""

    def __init__(self, benchmark_data=None, benchmark_symbol="SPY"):
        """
        Initialize the benchmark performance analyzer.

        Args:
            benchmark_data (DataFrame, optional): Benchmark price data
            benchmark_symbol (str): Symbol for benchmark (default: "SPY")
        """
        self.benchmark_data = benchmark_data
        self.benchmark_symbol = benchmark_symbol
        self.benchmark_returns = None
        self.strategy_returns = None
        self.rolling_beta = None
        self.rolling_alpha = None
        self.benchmark_sectors = {}
        self.factor_betas = {}
        
    def load_benchmark_data(self, filepath, column='close'):
        """
        Load benchmark data from file.

        Args:
            filepath (str): Path to benchmark data file
            column (str): Column to use for price data (default: 'close')

        Returns:
            DataFrame: Benchmark data
        """
        try:
            data = pd.read_csv(filepath, parse_dates=True, index_col=0)
            logger.info(f"Loaded benchmark data from {filepath} with {len(data)} rows")
            
            # Store the benchmark data
            self.benchmark_data = data
            
            # Calculate benchmark returns if price column exists
            if column in data.columns:
                self.benchmark_returns = data[column].pct_change().dropna()
                logger.info(f"Calculated benchmark returns from {column} column")
            else:
                logger.warning(f"Column {column} not found in benchmark data")
                
            return data
        except Exception as e:
            logger.error(f"Error loading benchmark data: {e}")
            return None
    
    def load_sector_benchmarks(self, sector_data_dict):
        """
        Load sector benchmark data.

        Args:
            sector_data_dict (dict): Dictionary of sector name -> data filepath pairs

        Returns:
            dict: Sector benchmark data
        """
        sector_benchmarks = {}
        
        for sector, filepath in sector_data_dict.items():
            try:
                data = pd.read_csv(filepath, parse_dates=True, index_col=0)
                # Calculate returns
                if 'close' in data.columns:
                    returns = data['close'].pct_change().dropna()
                    sector_benchmarks[sector] = returns
                    logger.info(f"Loaded {sector} benchmark data with {len(data)} rows")
                else:
                    logger.warning(f"Column 'close' not found in {sector} benchmark data")
            except Exception as e:
                logger.error(f"Error loading {sector} benchmark data: {e}")
        
        self.benchmark_sectors = sector_benchmarks
        return sector_benchmarks
    
    def set_strategy_returns(self, strategy_returns):
        """
        Set strategy returns for analysis.

        Args:
            strategy_returns (Series): Strategy returns series

        Returns:
            None
        """
        self.strategy_returns = strategy_returns
        logger.info(f"Set strategy returns with {len(strategy_returns)} observations")
    
    def calculate_relative_performance(self, strategy_returns=None, benchmark_returns=None, 
                                      window=252, min_periods=63):
        """
        Calculate relative performance metrics against benchmark.

        Args:
            strategy_returns (Series, optional): Strategy returns
            benchmark_returns (Series, optional): Benchmark returns
            window (int): Rolling window size for beta calculation (default: 252 days)
            min_periods (int): Minimum periods for rolling calculation (default: 63 days)

        Returns:
            dict: Relative performance metrics
        """
        # Use provided data or stored data
        s_returns = strategy_returns if strategy_returns is not None else self.strategy_returns
        b_returns = benchmark_returns if benchmark_returns is not None else self.benchmark_returns
        
        if s_returns is None or b_returns is None:
            logger.error("Strategy returns and/or benchmark returns not available")
            return {}
        
        # Align dates
        common_data = pd.DataFrame({
            'strategy': s_returns,
            'benchmark': b_returns
        }).dropna()
        
        if common_data.empty:
            logger.error("No common dates between strategy and benchmark returns")
            return {}
        
        # Calculate time-varying beta using rolling regression
        X = sm.add_constant(common_data['benchmark'])
        model = RollingOLS(common_data['strategy'], X, window=window, min_nobs=min_periods)
        rolling_res = model.fit()
        
        # Store rolling beta and alpha
        self.rolling_beta = rolling_res.params['benchmark']
        self.rolling_alpha = rolling_res.params['const'] * 252  # Annualized
        
        # Calculate full-period metrics
        X = sm.add_constant(common_data['benchmark'])
        model = sm.OLS(common_data['strategy'], X)
        results = model.fit()
        
        # Calculate performance metrics
        beta = results.params['benchmark']
        alpha = results.params['const'] * 252  # Annualized
        r_squared = results.rsquared
        alpha_tstat = results.tvalues['const']
        alpha_pvalue = results.pvalues['const']
        
        # Calculate excess return (strategy return - risk-free rate - beta * [benchmark return - risk-free rate])
        # Here we use 0 as the risk-free rate for simplicity
        excess_returns = common_data['strategy'] - beta * common_data['benchmark']
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Calculate tracking error
        tracking_error = np.std(common_data['strategy'] - common_data['benchmark']) * np.sqrt(252)
        
        # Calculate up/down capture ratios
        up_market = common_data[common_data['benchmark'] > 0]
        down_market = common_data[common_data['benchmark'] < 0]
        
        up_capture = (up_market['strategy'].mean() / up_market['benchmark'].mean()) if not up_market.empty else 0
        down_capture = (down_market['strategy'].mean() / down_market['benchmark'].mean()) if not down_market.empty else 0
        
        # Calculate outperformance statistics
        outperformance = common_data['strategy'] - common_data['benchmark']
        outperformance_mean = outperformance.mean() * 252  # Annualized
        outperformance_std = outperformance.std() * np.sqrt(252)
        outperformance_sharpe = outperformance_mean / outperformance_std if outperformance_std > 0 else 0
        
        # Calculate cumulative returns
        cumulative_strategy = (1 + common_data['strategy']).cumprod()
        cumulative_benchmark = (1 + common_data['benchmark']).cumprod()
        
        # Calculate relative strength
        relative_strength = cumulative_strategy / cumulative_benchmark
        
        # Store metrics
        metrics = {
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'alpha_tstat': alpha_tstat,
            'alpha_pvalue': alpha_pvalue,
            'up_capture': up_capture,
            'down_capture': down_capture,
            'outperformance_mean': outperformance_mean,
            'outperformance_std': outperformance_std,
            'outperformance_sharpe': outperformance_sharpe
        }
        
        # Check if alpha is statistically significant
        metrics['alpha_significant'] = alpha_pvalue < 0.05
        
        # Calculate cumulative outperformance
        metrics['cumulative_outperformance'] = relative_strength.iloc[-1] - 1.0
        
        # Calculate average outperformance per unit of risk
        metrics['outperformance_per_risk'] = outperformance_mean / tracking_error if tracking_error > 0 else 0
        
        # Additional market condition analysis
        bull_market = common_data[common_data['benchmark'] > common_data['benchmark'].rolling(window=window).mean()]
        bear_market = common_data[common_data['benchmark'] < common_data['benchmark'].rolling(window=window).mean()]
        
        metrics['bull_market_outperformance'] = (bull_market['strategy'] - bull_market['benchmark']).mean() * 252
        metrics['bear_market_outperformance'] = (bear_market['strategy'] - bear_market['benchmark']).mean() * 252
        
        # Calculate consistency of outperformance
        rolling_outperformance = (common_data['strategy'] - common_data['benchmark']).rolling(window=21).mean()
        metrics['positive_outperformance_percentage'] = (rolling_outperformance > 0).mean() * 100
        
        logger.info("Calculated benchmark-relative performance metrics")
        return metrics

    def multi_factor_attribution(self, strategy_returns=None, factor_returns_dict=None):
        """
        Perform multi-factor attribution analysis.

        Args:
            strategy_returns (Series, optional): Strategy returns
            factor_returns_dict (dict): Dictionary of factor name -> returns pairs

        Returns:
            dict: Factor attribution results
        """
        if factor_returns_dict is None or len(factor_returns_dict) == 0:
            logger.error("No factor returns provided for attribution analysis")
            return {}
        
        # Use provided data or stored data
        s_returns = strategy_returns if strategy_returns is not None else self.strategy_returns
        
        if s_returns is None:
            logger.error("Strategy returns not available for attribution analysis")
            return {}
        
        # Combine strategy returns with factor returns
        data = pd.DataFrame({'strategy': s_returns})
        
        for factor, returns in factor_returns_dict.items():
            data[factor] = returns
        
        # Drop rows with missing values
        data = data.dropna()
        
        if data.empty:
            logger.error("No common dates between strategy returns and factor returns")
            return {}
        
        # Prepare dependent and independent variables
        y = data['strategy']
        X = data.drop('strategy', axis=1)
        X = sm.add_constant(X)
        
        # Run regression
        model = sm.OLS(y, X)
        results = model.fit()
        
        # Extract factor loadings (betas)
        factor_betas = {}
        for factor in factor_returns_dict.keys():
            factor_betas[factor] = {
                'beta': results.params[factor],
                'tstat': results.tvalues[factor],
                'pvalue': results.pvalues[factor],
                'significant': results.pvalues[factor] < 0.05
            }
        
        # Extract model statistics
        model_stats = {
            'alpha': results.params['const'] * 252,  # Annualized
            'alpha_tstat': results.tvalues['const'],
            'alpha_pvalue': results.pvalues['const'],
            'alpha_significant': results.pvalues['const'] < 0.05,
            'r_squared': results.rsquared,
            'adj_r_squared': results.rsquared_adj
        }
        
        # Calculate factor contributions
        factor_contributions = {}
        total_explained_return = 0
        
        for factor, returns in factor_returns_dict.items():
            # Calculate average factor return
            avg_factor_return = data[factor].mean() * 252  # Annualized
            
            # Calculate factor contribution
            factor_contribution = factor_betas[factor]['beta'] * avg_factor_return
            factor_contributions[factor] = factor_contribution
            total_explained_return += factor_contribution
        
        # Calculate unexplained return (alpha)
        avg_strategy_return = data['strategy'].mean() * 252  # Annualized
        unexplained_return = avg_strategy_return - total_explained_return
        
        # Store factor betas
        self.factor_betas = factor_betas
        
        # Prepare attribution results
        attribution = {
            'factor_betas': factor_betas,
            'factor_contributions': factor_contributions,
            'model_stats': model_stats,
            'total_explained_return': total_explained_return,
            'total_unexplained_return': unexplained_return,
            'percentage_explained': total_explained_return / avg_strategy_return if avg_strategy_return != 0 else 0
        }
        
        logger.info("Performed multi-factor attribution analysis")
        return attribution
    
    def sector_attribution(self, strategy_returns=None):
        """
        Perform sector attribution analysis.

        Args:
            strategy_returns (Series, optional): Strategy returns

        Returns:
            dict: Sector attribution results
        """
        if not self.benchmark_sectors:
            logger.error("No sector benchmarks available for attribution analysis")
            return {}
        
        # Use provided data or stored data
        s_returns = strategy_returns if strategy_returns is not None else self.strategy_returns
        
        if s_returns is None:
            logger.error("Strategy returns not available for sector attribution")
            return {}
        
        # Use sector benchmarks as factors
        return self.multi_factor_attribution(s_returns, self.benchmark_sectors)
    
    def active_beta_optimization(self, target_beta=0.0, strategy_returns=None, benchmark_returns=None,
                               window=63, lookback=252):
        """
        Calculate target weights for dynamic beta adjustment.

        Args:
            target_beta (float): Target portfolio beta (default: 0.0 for market-neutral)
            strategy_returns (Series, optional): Strategy returns
            benchmark_returns (Series, optional): Benchmark returns
            window (int): Window for beta estimation (default: 63 days)
            lookback (int): Lookback period for optimization (default: 252 days)

        Returns:
            dict: Target portfolio weights
        """
        # Use provided data or stored data
        s_returns = strategy_returns if strategy_returns is not None else self.strategy_returns
        b_returns = benchmark_returns if benchmark_returns is not None else self.benchmark_returns
        
        if s_returns is None or b_returns is None:
            logger.error("Strategy returns and/or benchmark returns not available")
            return {}
        
        # Align dates
        common_data = pd.DataFrame({
            'strategy': s_returns,
            'benchmark': b_returns
        }).dropna()
        
        # Ensure sufficient data
        if len(common_data) < lookback:
            logger.warning(f"Insufficient data for beta optimization. Using all {len(common_data)} available data points.")
            lookback = len(common_data)
        
        # Use most recent data for optimization
        recent_data = common_data.tail(lookback)
        
        # Calculate rolling beta
        rolling_beta = np.zeros(len(recent_data) - window + 1)
        
        for i in range(len(rolling_beta)):
            window_data = recent_data.iloc[i:i+window]
            # Calculate beta using regression
            X = sm.add_constant(window_data['benchmark'])
            model = sm.OLS(window_data['strategy'], X)
            results = model.fit()
            rolling_beta[i] = results.params['benchmark']
        
        # Calculate current beta
        current_beta = rolling_beta[-1]
        
        # Calculate required adjustment to reach target beta
        beta_adjustment = target_beta - current_beta
        
        # Calculate target weights
        if current_beta > target_beta:
            # Need to hedge with short benchmark position
            strategy_weight = 1.0
            benchmark_weight = beta_adjustment
        else:
            # Need to increase beta with long benchmark position
            strategy_weight = 1.0
            benchmark_weight = beta_adjustment
        
        # Return target weights
        weights = {
            'strategy_weight': strategy_weight,
            'benchmark_weight': benchmark_weight,
            'current_beta': current_beta,
            'target_beta': target_beta,
            'beta_adjustment': beta_adjustment
        }
        
        logger.info(f"Calculated target weights for beta optimization. Current beta: {current_beta:.2f}, Target beta: {target_beta:.2f}")
        return weights
    
    def market_regime_performance(self, strategy_returns=None, benchmark_returns=None,
                                regimes=None, regime_data=None):
        """
        Analyze performance across different market regimes.

        Args:
            strategy_returns (Series, optional): Strategy returns
            benchmark_returns (Series, optional): Benchmark returns
            regimes (Series): Market regime labels
            regime_data (dict): Regime metadata

        Returns:
            dict: Performance metrics by regime
        """
        # Use provided data or stored data
        s_returns = strategy_returns if strategy_returns is not None else self.strategy_returns
        b_returns = benchmark_returns if benchmark_returns is not None else self.benchmark_returns
        
        if s_returns is None or b_returns is None:
            logger.error("Strategy returns and/or benchmark returns not available")
            return {}
        
        if regimes is None:
            logger.error("No regime data provided for analysis")
            return {}
        
        # Align dates
        common_data = pd.DataFrame({
            'strategy': s_returns,
            'benchmark': b_returns,
            'regime': regimes
        }).dropna()
        
        if common_data.empty:
            logger.error("No common dates between strategy, benchmark returns and regime data")
            return {}
        
        # Initialize results
        regime_performance = {}
        
        # Analyze performance for each regime
        for regime in common_data['regime'].unique():
            regime_name = regime_data.get(regime, f"Regime {regime}") if regime_data else f"Regime {regime}"
            regime_data = common_data[common_data['regime'] == regime]
            
            # Skip if insufficient data
            if len(regime_data) < 20:
                logger.warning(f"Insufficient data for {regime_name} (n={len(regime_data)})")
                continue
            
            # Calculate relative performance metrics
            try:
                metrics = self.calculate_relative_performance(
                    regime_data['strategy'],
                    regime_data['benchmark']
                )
                
                # Calculate additional statistics
                strategy_cumulative = (1 + regime_data['strategy']).cumprod().iloc[-1] - 1
                benchmark_cumulative = (1 + regime_data['benchmark']).cumprod().iloc[-1] - 1
                annualized_strategy = (1 + strategy_cumulative) ** (252 / len(regime_data)) - 1
                annualized_benchmark = (1 + benchmark_cumulative) ** (252 / len(regime_data)) - 1
                
                # Add to metrics
                metrics['period_length'] = len(regime_data)
                metrics['strategy_return'] = strategy_cumulative
                metrics['benchmark_return'] = benchmark_cumulative
                metrics['annualized_strategy_return'] = annualized_strategy
                metrics['annualized_benchmark_return'] = annualized_benchmark
                metrics['outperformance'] = strategy_cumulative - benchmark_cumulative
                metrics['annualized_outperformance'] = annualized_strategy - annualized_benchmark
                
                regime_performance[regime_name] = metrics
                logger.info(f"Calculated performance metrics for {regime_name} ({len(regime_data)} observations)")
                
            except Exception as e:
                logger.error(f"Error calculating performance for {regime_name}: {e}")
        
        return regime_performance
    
    def plot_relative_performance(self, strategy_returns=None, benchmark_returns=None, title=None, figsize=(12, 8)):
        """
        Plot relative performance against benchmark.

        Args:
            strategy_returns (Series, optional): Strategy returns
            benchmark_returns (Series, optional): Benchmark returns
            title (str): Plot title
            figsize (tuple): Figure size (width, height)

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Use provided data or stored data
        s_returns = strategy_returns if strategy_returns is not None else self.strategy_returns
        b_returns = benchmark_returns if benchmark_returns is not None else self.benchmark_returns
        
        if s_returns is None or b_returns is None:
            logger.error("Strategy returns and/or benchmark returns not available")
            return None
        
        # Align dates
        common_data = pd.DataFrame({
            'strategy': s_returns,
            'benchmark': b_returns
        }).dropna()
        
        if common_data.empty:
            logger.error("No common dates between strategy and benchmark returns")
            return None
        
        # Calculate cumulative returns
        cumulative_strategy = (1 + common_data['strategy']).cumprod()
        cumulative_benchmark = (1 + common_data['benchmark']).cumprod()
        
        # Calculate relative performance
        relative_performance = cumulative_strategy / cumulative_benchmark
        
        # Calculate drawdowns
        strategy_peak = cumulative_strategy.expanding().max()
        benchmark_peak = cumulative_benchmark.expanding().max()
        strategy_drawdown = (cumulative_strategy / strategy_peak) - 1
        benchmark_drawdown = (cumulative_benchmark / benchmark_peak) - 1
        
        # Create plot with 2 rows and 2 columns
        fig, axes = plt.subplots(2, 2, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot cumulative returns
        axes[0, 0].plot(cumulative_strategy, label=f'Strategy', color='blue')
        axes[0, 0].plot(cumulative_benchmark, label=f'Benchmark ({self.benchmark_symbol})', color='red')
        axes[0, 0].set_title('Cumulative Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylabel('Growth of $1')
        
        # Plot relative performance
        axes[0, 1].plot(relative_performance, color='green')
        axes[0, 1].axhline(y=1.0, color='gray', linestyle='--')
        axes[0, 1].set_title('Relative Performance (Strategy / Benchmark)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylabel('Ratio')
        
        # Plot drawdowns
        axes[1, 0].fill_between(strategy_drawdown.index, 0, strategy_drawdown, color='blue', alpha=0.3)
        axes[1, 0].fill_between(benchmark_drawdown.index, 0, benchmark_drawdown, color='red', alpha=0.3)
        axes[1, 0].set_title('Drawdowns')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylabel('Drawdown')
        
        # Plot rolling beta if available
        if self.rolling_beta is not None:
            axes[1, 1].plot(self.rolling_beta.index, self.rolling_beta, color='purple')
            axes[1, 1].axhline(y=1.0, color='gray', linestyle='--')
            axes[1, 1].set_title('Rolling Beta')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylabel('Beta')
        else:
            # Calculate rolling correlation instead
            rolling_corr = common_data['strategy'].rolling(window=63).corr(common_data['benchmark'])
            axes[1, 1].plot(rolling_corr.index, rolling_corr, color='purple')
            axes[1, 1].axhline(y=0.0, color='gray', linestyle='--')
            axes[1, 1].set_title('Rolling Correlation')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylabel('Correlation')
        
        # Set main title if provided
        if title:
            fig.suptitle(title, fontsize=14)
        
        # Format x-axis dates
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_factor_contributions(self, attribution_results, title="Factor Return Attribution", figsize=(10, 6)):
        """
        Plot factor contributions from attribution analysis.

        Args:
            attribution_results (dict): Results from multi_factor_attribution
            title (str): Plot title
            figsize (tuple): Figure size (width, height)

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if 'factor_contributions' not in attribution_results:
            logger.error("No factor contributions data in attribution results")
            return None
        
        # Extract factor contributions
        factor_contributions = attribution_results['factor_contributions']
        unexplained_return = attribution_results.get('total_unexplained_return', 0)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Contribution': list(factor_contributions.values()) + [unexplained_return],
            'Factor': list(factor_contributions.keys()) + ['Alpha (Unexplained)']
        })
        
        # Sort by absolute contribution
        df['Abs_Contribution'] = df['Contribution'].abs()
        df = df.sort_values('Abs_Contribution', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        colors = ['green' if x > 0 else 'red' for x in df['Contribution']]
        bars = ax.barh(df['Factor'], df['Contribution'], color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.0005 if width > 0 else width - 0.003
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2%}',
                   va='center')
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel('Contribution to Return (Annualized)')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add grid
        ax.grid(axis='x', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def enhanced_benchmark_analysis(self, strategy_returns=None, benchmark_returns=None):
        """
        Perform enhanced benchmark-relative performance analysis using the new BenchmarkRelativeScorer.
        
        This method provides more sophisticated benchmark-relative metrics beyond traditional
        alpha/beta analysis, including outperformance consistency, up/down capture ratios,
        and temporal analysis of excess returns.
        
        Args:
            strategy_returns (Series, optional): Strategy returns
            benchmark_returns (Series, optional): Benchmark returns
            
        Returns:
            dict: Enhanced benchmark analysis report
        """
        # Use provided data or stored data
        s_returns = strategy_returns if strategy_returns is not None else self.strategy_returns
        b_returns = benchmark_returns if benchmark_returns is not None else self.benchmark_returns
        
        if s_returns is None or b_returns is None:
            logger.error("Strategy returns and/or benchmark returns not available")
            return {}
        
        try:
            # Create benchmark data in the format expected by BenchmarkRelativeScorer
            benchmark_data = {
                self.benchmark_symbol: pd.DataFrame({
                    'returns': b_returns
                }, index=b_returns.index)
            }
            
            # Create the enhanced scorer
            scorer = BenchmarkRelativeScorer(
                benchmark_data=benchmark_data,
                default_benchmark=self.benchmark_symbol
            )
            
            # Generate comprehensive report
            enhanced_report = scorer.generate_outperformance_report(
                s_returns, 
                "Strategy",
                benchmark_symbols=[self.benchmark_symbol]
            )
            
            logger.info("Generated enhanced benchmark-relative analysis")
            return enhanced_report
            
        except Exception as e:
            logger.error(f"Error in enhanced benchmark analysis: {str(e)}")
            return {}
    
    def generate_performance_report(self, output_dir=None):
        """
        Generate comprehensive benchmark-relative performance report.

        Args:
            output_dir (str, optional): Directory to save the report

        Returns:
            dict: Report data and file paths
        """
        if self.strategy_returns is None or self.benchmark_returns is None:
            logger.error("Strategy returns and/or benchmark returns not available")
            return {}
        
        # Create output directory if specified
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calculate relative performance metrics
        performance_metrics = self.calculate_relative_performance()
        
        # Generate visualizations
        try:
            relative_perf_fig = self.plot_relative_performance(
                title=f"Strategy vs {self.benchmark_symbol} Performance"
            )
            
            # Save plot if output directory is specified
            if output_dir is not None and relative_perf_fig is not None:
                plot_path = os.path.join(output_dir, "relative_performance.png")
                relative_perf_fig.savefig(plot_path)
                plt.close(relative_perf_fig)
            else:
                plot_path = None
                
        except Exception as e:
            logger.error(f"Error generating performance visualization: {e}")
            relative_perf_fig = None
            plot_path = None
        
        # Compile report data
        report = {
            'performance_metrics': performance_metrics,
            'visualization_path': plot_path,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'benchmark_symbol': self.benchmark_symbol
        }
        
        # Format outperformance summary
        if 'outperformance_mean' in performance_metrics:
            outperformance = performance_metrics['outperformance_mean']
            report['outperformance_summary'] = {
                'annualized_outperformance': outperformance,
                'outperformance_significant': performance_metrics.get('alpha_significant', False),
                'outperforms_benchmark': outperformance > 0,
                'outperformance_percentage': f"{outperformance * 100:.2f}%"
            }
        
        # Save report as JSON if output directory is specified
        if output_dir is not None:
            try:
                report_path = os.path.join(output_dir, "benchmark_performance_report.json")
                with open(report_path, 'w') as f:
                    # Convert numpy types to native Python types for JSON serialization
                    json_report = {}
                    for key, value in report.items():
                        if isinstance(value, dict):
                            json_report[key] = {k: float(v) if isinstance(v, np.floating) else v 
                                          for k, v in value.items()}
                        else:
                            json_report[key] = value
                    
                    json.dump(json_report, f, indent=2)
                
                report['report_path'] = report_path
                logger.info(f"Performance report saved to {report_path}")
            except Exception as e:
                logger.error(f"Error saving performance report: {e}")
        
        return report

def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Relative Performance Analysis")
    parser.add_argument("--strategy", required=True, help="Path to strategy returns CSV file")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark data CSV file")
    parser.add_argument("--symbol", default="SPY", help="Benchmark symbol (default: SPY)")
    parser.add_argument("--window", type=int, default=252, help="Window size for rolling metrics")
    parser.add_argument("--output", default="performance_output", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Load strategy returns
    try:
        strategy_data = pd.read_csv(args.strategy, parse_dates=True, index_col=0)
        if 'returns' in strategy_data.columns:
            strategy_returns = strategy_data['returns']
        elif 'return' in strategy_data.columns:
            strategy_returns = strategy_data['return']
        else:
            logger.warning("Returns column not found. Using first column as returns.")
            strategy_returns = strategy_data.iloc[:, 0]
        
        logger.info(f"Loaded strategy returns from {args.strategy} with {len(strategy_returns)} observations")
    except Exception as e:
        logger.error(f"Error loading strategy returns: {e}")
        return 1
    
    # Create analyzer and load benchmark data
    analyzer = BenchmarkPerformanceAnalyzer(benchmark_symbol=args.symbol)
    analyzer.load_benchmark_data(args.benchmark)
    
    # Set strategy returns
    analyzer.set_strategy_returns(strategy_returns)
    
    # Create output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Generate and save report
    report = analyzer.generate_performance_report(args.output)
    
    # Also generate enhanced analysis
    enhanced_report = analyzer.enhanced_benchmark_analysis()
    
    # Save enhanced report
    if enhanced_report and args.output:
        enhanced_report_path = os.path.join(args.output, "enhanced_benchmark_analysis.json")
        try:
            with open(enhanced_report_path, 'w') as f:
                # Convert complex types to serializable format
                json_enhanced_report = {}
                for key, value in enhanced_report.items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries
                        json_enhanced_report[key] = {}
                        for k, v in value.items():
                            if isinstance(v, dict):
                                # Convert timestamps in time_analysis data
                                processed_dict = {}
                                for date_key, date_value in v.items():
                                    if isinstance(date_key, pd.Timestamp):
                                        processed_dict[date_key.isoformat()] = date_value
                                    else:
                                        processed_dict[str(date_key)] = date_value
                                json_enhanced_report[key][k] = processed_dict
                            else:
                                json_enhanced_report[key][k] = float(v) if isinstance(v, (np.floating, float)) else v
                    else:
                        json_enhanced_report[key] = value
                
                json.dump(json_enhanced_report, f, indent=2)
                
            print(f"Enhanced benchmark analysis saved to {enhanced_report_path}")
        except Exception as e:
            logger.error(f"Error saving enhanced report: {e}")
    
    # Print summary
    if 'outperformance_summary' in report:
        summary = report['outperformance_summary']
        print("\nBenchmark Relative Performance:")
        print(f"Strategy {'outperforms' if summary['outperforms_benchmark'] else 'underperforms'} the {args.symbol} benchmark by {summary['outperformance_percentage']}")
        print(f"Statistical significance: {'Yes' if summary['outperformance_significant'] else 'No'}")
    
    if 'performance_metrics' in report:
        metrics = report['performance_metrics']
        print("\nKey Metrics:")
        print(f"Alpha: {metrics.get('alpha', 0):.4f}")
        print(f"Beta: {metrics.get('beta', 0):.4f}")
        print(f"Information Ratio: {metrics.get('information_ratio', 0):.4f}")
        print(f"Up/Down Capture: {metrics.get('up_capture', 0):.2f} / {metrics.get('down_capture', 0):.2f}")
    
    # Print enhanced metrics if available
    if enhanced_report and 'benchmarks' in enhanced_report and args.symbol in enhanced_report['benchmarks']:
        enhanced_metrics = enhanced_report['benchmarks'][args.symbol]
        print("\nEnhanced Benchmark Analysis Metrics:")
        if 'excess_return' in enhanced_metrics:
            print(f"Excess Return: {enhanced_metrics['excess_return']*100:.2f}%")
        if 'outperformance_consistency' in enhanced_metrics:
            print(f"Outperformance Consistency: {enhanced_metrics['outperformance_consistency']*100:.2f}%")
        if 'win_rate_vs_benchmark' in enhanced_metrics:
            print(f"Win Rate vs Benchmark: {enhanced_metrics['win_rate_vs_benchmark']*100:.2f}%")
    
    print(f"\nReport and visualizations saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())