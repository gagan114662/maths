#!/usr/bin/env python3
"""
Market Regime Stress Testing Example

This script demonstrates how to use the enhanced market regime stress testing 
functionality to evaluate strategy performance across different market regimes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
from datetime import datetime
import yfinance as yf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import stress testing module
from src.risk_management.stress_testing import (
    run_regime_stress_test, 
    generate_regime_report,
    RegimeType,
    DetectionMethod
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_sample_strategy(params=None):
    """
    Create a sample trading strategy for testing.
    
    Args:
        params: Optional dictionary of strategy parameters
        
    Returns:
        Strategy function that takes market data and returns strategy returns
    """
    # Default parameters
    if params is None:
        params = {
            'fast_ma': 20,
            'slow_ma': 50,
            'entry_threshold': 0.01,
            'exit_threshold': -0.01,
            'stop_loss': 0.05,
            'take_profit': 0.10
        }
    
    def moving_average_strategy(market_data):
        """
        Simple moving average crossover strategy.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Series of strategy returns
        """
        # Ensure market data has required columns
        if 'Close' not in market_data.columns:
            logger.error("Market data must contain 'Close' column")
            return None
        
        # Calculate moving averages
        fast_ma = market_data['Close'].rolling(params['fast_ma']).mean()
        slow_ma = market_data['Close'].rolling(params['slow_ma']).mean()
        
        # Calculate signals
        signal = pd.Series(0, index=market_data.index)
        signal[(fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))] = 1  # Buy signal
        signal[(fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))] = -1  # Sell signal
        
        # Generate positions (1 = long, 0 = flat, -1 = short)
        position = signal.copy()
        position = position.replace(to_replace=0, method='ffill')
        position = position.fillna(0)
        
        # Calculate strategy returns
        strategy_returns = position.shift(1) * market_data['Close'].pct_change()
        strategy_returns = strategy_returns.fillna(0)
        
        return strategy_returns
    
    return moving_average_strategy

def create_rsi_strategy(params=None):
    """
    Create an RSI-based mean reversion strategy.
    
    Args:
        params: Optional dictionary of strategy parameters
        
    Returns:
        Strategy function that takes market data and returns strategy returns
    """
    # Default parameters
    if params is None:
        params = {
            'rsi_period': 14,
            'overbought': 70,
            'oversold': 30,
            'stop_loss': 0.05
        }
    
    def rsi_strategy(market_data):
        """
        RSI mean reversion strategy.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Series of strategy returns
        """
        # Ensure market data has required columns
        if 'Close' not in market_data.columns:
            logger.error("Market data must contain 'Close' column")
            return None
        
        # Calculate RSI
        delta = market_data['Close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        
        ma_up = up.rolling(params['rsi_period']).mean()
        ma_down = down.rolling(params['rsi_period']).mean()
        
        rsi = 100 - (100 / (1 + ma_up / ma_down))
        
        # Calculate signals
        signal = pd.Series(0, index=market_data.index)
        signal[rsi < params['oversold']] = 1  # Buy when oversold
        signal[rsi > params['overbought']] = -1  # Sell when overbought
        
        # Generate positions (1 = long, 0 = flat, -1 = short if we want to implement short selling)
        position = signal.copy()
        position = position.replace(to_replace=0, method='ffill')
        position = position.fillna(0)
        
        # Calculate strategy returns
        strategy_returns = position.shift(1) * market_data['Close'].pct_change()
        strategy_returns = strategy_returns.fillna(0)
        
        return strategy_returns
    
    return rsi_strategy

def run_example():
    """Run the regime stress testing example."""
    logger.info("Starting market regime stress test example")
    
    # Create output directory
    output_dir = 'output/regime_stress_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Download sample market data (using S&P 500 ETF)
    logger.info("Downloading market data...")
    market_data = yf.download('SPY', start='2015-01-01', end='2023-01-01')
    
    # Download benchmark data (using Russell 2000 ETF for comparison)
    logger.info("Downloading benchmark data...")
    benchmark_data = yf.download('IWM', start='2015-01-01', end='2023-01-01')
    
    # Generate regime report
    logger.info("Generating market regime report...")
    regime_report = generate_regime_report(
        market_data=market_data,
        method='hmm',
        n_regimes=4,
        output_dir=os.path.join(output_dir, 'regime_report')
    )
    
    # Log regime information
    if 'regime_dates' in regime_report:
        logger.info("Detected regimes:")
        for regime, info in sorted(regime_report['regime_dates'].items()):
            logger.info(f"  Regime {regime}: {info['total_days']} days")
    
    # Create sample strategies
    moving_avg_strategy = create_sample_strategy()
    rsi_strategy = create_rsi_strategy()
    
    # Run stress test for Moving Average strategy
    logger.info("Running stress test for Moving Average strategy...")
    ma_results = run_regime_stress_test(
        strategy_func=moving_avg_strategy,
        market_data=market_data,
        benchmark_data=benchmark_data,
        method='hmm',
        n_regimes=4,
        output_dir=os.path.join(output_dir, 'moving_avg_strategy')
    )
    
    # Summarize results for Moving Average strategy
    logger.info("Moving Average Strategy Results:")
    if 'overall_performance' in ma_results:
        perf = ma_results['overall_performance']
        logger.info(f"  Annualized Return: {perf['annualized_return']*100:.2f}%")
        logger.info(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {perf['max_drawdown']*100:.2f}%")
    
    if 'regime_performance' in ma_results:
        logger.info("  Performance by Regime:")
        for regime, perf in ma_results['regime_performance'].items():
            logger.info(f"    Regime {regime}: Sharpe={perf['sharpe_ratio']:.2f}, Return={perf['annualized_return']*100:.2f}%")
    
    # Run stress test for RSI strategy
    logger.info("Running stress test for RSI strategy...")
    rsi_results = run_regime_stress_test(
        strategy_func=rsi_strategy,
        market_data=market_data,
        benchmark_data=benchmark_data,
        method='hmm',
        n_regimes=4,
        output_dir=os.path.join(output_dir, 'rsi_strategy')
    )
    
    # Summarize results for RSI strategy
    logger.info("RSI Strategy Results:")
    if 'overall_performance' in rsi_results:
        perf = rsi_results['overall_performance']
        logger.info(f"  Annualized Return: {perf['annualized_return']*100:.2f}%")
        logger.info(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {perf['max_drawdown']*100:.2f}%")
    
    if 'regime_performance' in rsi_results:
        logger.info("  Performance by Regime:")
        for regime, perf in rsi_results['regime_performance'].items():
            logger.info(f"    Regime {regime}: Sharpe={perf['sharpe_ratio']:.2f}, Return={perf['annualized_return']*100:.2f}%")
    
    # Compare strategies across regimes
    logger.info("Comparing strategies across regimes...")
    
    # Create DataFrame for comparison
    regimes = sorted(list(set(list(ma_results['regime_performance'].keys()) + list(rsi_results['regime_performance'].keys()))))
    
    comparison_data = []
    for regime in regimes:
        ma_sharpe = ma_results['regime_performance'].get(regime, {}).get('sharpe_ratio', float('nan'))
        rsi_sharpe = rsi_results['regime_performance'].get(regime, {}).get('sharpe_ratio', float('nan'))
        
        comparison_data.append({
            'Regime': regime,
            'Moving_Avg_Sharpe': ma_sharpe,
            'RSI_Sharpe': rsi_sharpe,
            'Better_Strategy': 'Moving Avg' if ma_sharpe > rsi_sharpe else 'RSI' if rsi_sharpe > ma_sharpe else 'Equal'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison
    logger.info("\nStrategy Comparison by Regime:")
    logger.info(f"\n{comparison_df}")
    
    # Save comparison to CSV
    comparison_path = os.path.join(output_dir, 'strategy_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Saved strategy comparison to {comparison_path}")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot Sharpe ratios by regime
    bar_width = 0.35
    x = np.arange(len(regimes))
    
    ma_sharpes = [ma_results['regime_performance'].get(r, {}).get('sharpe_ratio', 0) for r in regimes]
    rsi_sharpes = [rsi_results['regime_performance'].get(r, {}).get('sharpe_ratio', 0) for r in regimes]
    
    plt.bar(x - bar_width/2, ma_sharpes, bar_width, label='Moving Avg Strategy', color='#1f77b4')
    plt.bar(x + bar_width/2, rsi_sharpes, bar_width, label='RSI Strategy', color='#ff7f0e')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xticks(x, regimes)
    plt.xlabel('Market Regime')
    plt.ylabel('Sharpe Ratio')
    plt.title('Strategy Comparison Across Market Regimes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    comparison_plot_path = os.path.join(output_dir, 'strategy_comparison_plot.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved strategy comparison plot to {comparison_plot_path}")
    
    logger.info("Market regime stress test example completed successfully!")

if __name__ == "__main__":
    run_example()