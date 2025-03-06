#!/usr/bin/env python3
"""
Market Regime Demonstration Script

This script demonstrates the market regime detection capability by:
1. Downloading historical market data
2. Detecting different market regimes using the MarketRegimeDetector
3. Visualizing the regimes with price data
4. Showing how strategy parameters adapt based on regimes
5. Generating a QuantConnect algorithm that integrates regime detection
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import logging
import json
import yfinance as yf
from src.market_regime_detector import MarketRegimeDetector
from src.regime_aware_strategy_adapter import RegimeAwareStrategyAdapter
from quant_connect_adapter import MarketRegimeDetectionAdapter, QuantConnectStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_market_data(symbol='SPY', start_date='2010-01-01', end_date=None):
    """
    Download historical market data using yfinance.
    
    Args:
        symbol (str): Stock symbol to download data for
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format (defaults to today)
        
    Returns:
        pd.DataFrame: Historical market data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    logger.info(f"Downloading market data for {symbol} from {start_date} to {end_date}...")
    
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        
        # Check if data was successfully downloaded
        if data.empty:
            logger.error(f"No data downloaded for {symbol}")
            return None
            
        logger.info(f"Downloaded {len(data)} data points for {symbol}")
        return data
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        return None

def detect_market_regimes(data, method='hmm', n_regimes=4, visualize=True, output_dir='visualizations'):
    """
    Detect market regimes using the MarketRegimeDetector.
    
    Args:
        data (pd.DataFrame): Historical market data
        method (str): Method for regime detection ('hmm', 'gmm', 'kmeans')
        n_regimes (int): Number of regimes to detect
        visualize (bool): Whether to visualize the regimes
        output_dir (str): Directory to save visualizations
        
    Returns:
        tuple: (detector, results) - Trained detector and detection results
    """
    logger.info(f"Detecting market regimes using {method} method with {n_regimes} regimes...")
    
    # Create detector
    detector = MarketRegimeDetector(method=method, n_regimes=n_regimes)
    
    # Train the detector
    results = detector.train(data)
    
    # Print detection results
    logger.info(f"Detected {len(detector.regime_history)} regime periods")
    logger.info(f"Current regime: {results['current_regime']} - {detector.REGIME_LABELS.get(results['current_regime'], 'Unknown')}")
    
    # Print regime counts
    for regime, count in results['regime_counts'].items():
        logger.info(f"Regime {regime} ({detector.REGIME_LABELS.get(regime, 'Unknown')}): {count} observations")
    
    # Print regime transitions
    transition_matrix = detector.get_regime_transition_matrix()
    logger.info("Regime Transition Matrix:")
    logger.info("\n" + str(transition_matrix))
    
    # Calculate regime statistics
    regime_stats = detector.get_regime_returns(data)
    
    logger.info("Regime Statistics:")
    for regime, stats in regime_stats.items():
        logger.info(f"\nRegime {regime} ({stats['label']}):")
        logger.info(f"  Count: {stats['count']}")
        logger.info(f"  Annualized Return: {stats['annualized_return']*100:.2f}%")
        logger.info(f"  Annualized Volatility: {stats['annualized_volatility']*100:.2f}%")
        logger.info(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        logger.info(f"  Win Rate: {stats['positive_returns']*100:.2f}%")
    
    # Visualize regimes
    if visualize:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot regimes
        output_path = os.path.join(output_dir, f"market_regimes_{method}_{n_regimes}.png")
        detector.plot_regimes(data, output_path=output_path)
        logger.info(f"Saved regime visualization to {output_path}")
        
        # Plot regime statistics
        plot_regime_statistics(regime_stats, output_dir, method, n_regimes)
    
    return detector, results

def plot_regime_statistics(regime_stats, output_dir, method, n_regimes):
    """
    Plot statistics for each regime.
    
    Args:
        regime_stats (dict): Regime statistics
        output_dir (str): Directory to save visualizations
        method (str): Method used for regime detection
        n_regimes (int): Number of regimes
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Market Regime Statistics ({method.upper()}, {n_regimes} regimes)', fontsize=16)
    
    # Prepare data for plotting
    regimes = [f"Regime {r}" if 'label' not in regime_stats[r] else regime_stats[r]['label'] for r in regime_stats]
    returns = [regime_stats[r]['annualized_return'] * 100 for r in regime_stats]
    volatilities = [regime_stats[r]['annualized_volatility'] * 100 for r in regime_stats]
    sharpe_ratios = [regime_stats[r]['sharpe_ratio'] for r in regime_stats]
    win_rates = [regime_stats[r]['positive_returns'] * 100 for r in regime_stats]
    
    # Set color palette
    colors = sns.color_palette('viridis', len(regimes))
    
    # Plot returns
    sns.barplot(x=regimes, y=returns, ax=axes[0, 0], palette=colors)
    axes[0, 0].set_title('Annualized Return (%)')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot volatilities
    sns.barplot(x=regimes, y=volatilities, ax=axes[0, 1], palette=colors)
    axes[0, 1].set_title('Annualized Volatility (%)')
    axes[0, 1].set_ylabel('Volatility (%)')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
    
    # Plot Sharpe ratios
    sns.barplot(x=regimes, y=sharpe_ratios, ax=axes[1, 0], palette=colors)
    axes[1, 0].set_title('Sharpe Ratio')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot win rates
    sns.barplot(x=regimes, y=win_rates, ax=axes[1, 1], palette=colors)
    axes[1, 1].set_title('Win Rate (%)')
    axes[1, 1].set_ylabel('Win Rate (%)')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    output_path = os.path.join(output_dir, f"regime_statistics_{method}_{n_regimes}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved regime statistics to {output_path}")
    
    plt.close()

def adapt_strategy_to_regimes(strategy_path, data, detector=None, output_dir='adapted_strategies'):
    """
    Adapt a strategy to market regimes.
    
    Args:
        strategy_path (str): Path to strategy JSON file
        data (pd.DataFrame): Historical market data
        detector (MarketRegimeDetector): Trained detector (if None, a new one will be created)
        output_dir (str): Directory to save adapted strategies
        
    Returns:
        dict: Adapted strategy data
    """
    logger.info(f"Adapting strategy from {strategy_path} to market regimes...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create adapter
    adapter = RegimeAwareStrategyAdapter(
        strategy_path=strategy_path,
        regime_detector=detector,
        regime_config_path=os.path.join(output_dir, 'regime_configs.json')
    )
    
    # Detect regime (or use provided detector)
    regime_results = adapter.detect_regime(data)
    
    # Get adapted strategy
    adapted_strategy = adapter.get_regime_adapted_strategy(data)
    
    # Save adapted strategy
    output_path = adapter.save_adapted_strategy(
        os.path.join(output_dir, f"adapted_strategy_{regime_results['regime_label'].replace('/', '_')}.json")
    )
    
    logger.info(f"Current regime: {regime_results['regime_label']} (ID: {regime_results['current_regime']})")
    logger.info(f"Saved adapted strategy to {output_path}")
    
    # Visualize regime impact
    visualization_path = os.path.join(output_dir, f"regime_impact_{regime_results['regime_label'].replace('/', '_')}.png")
    adapter.visualize_regime_impact(data, visualization_path)
    logger.info(f"Saved regime impact visualization to {visualization_path}")
    
    return adapted_strategy

def generate_quantconnect_regime_algorithm(strategy_path, output_dir='qc_algorithms', 
                                          method='hmm', n_regimes=4, start_date=None, end_date=None):
    """
    Generate a QuantConnect algorithm with market regime detection.
    
    Args:
        strategy_path (str): Path to strategy JSON file
        output_dir (str): Directory to save generated algorithm
        method (str): Method for regime detection ('hmm', 'kmeans')
        n_regimes (int): Number of regimes to detect
        start_date (str): Backtest start date (YYYY-MM-DD)
        end_date (str): Backtest end date (YYYY-MM-DD)
        
    Returns:
        str: Path to generated algorithm file
    """
    logger.info(f"Generating QuantConnect algorithm with {method} regime detection...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate algorithm
    qc_file = QuantConnectStrategy.generate_regime_aware_strategy(
        strategy_path, 
        output_dir, 
        start_date, 
        end_date, 
        100000,  # Initial cash
        n_regimes,
        method
    )
    
    logger.info(f"Generated regime-aware QuantConnect algorithm: {qc_file}")
    
    return qc_file

def create_sample_strategy(output_dir='strategies'):
    """
    Create a sample strategy JSON file for demonstration.
    
    Args:
        output_dir (str): Directory to save the strategy
        
    Returns:
        str: Path to the created strategy file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a sample momentum/RSI strategy
    strategy = {
        "strategy": {
            "Strategy Name": "Momentum RSI Strategy",
            "Description": "A momentum strategy with RSI filter",
            "Universe": "S&P 500 constituents",
            "Entry Rules": [
                "Price above 50-day moving average",
                "RSI between 40 and 65",
                "Volume above 20-day average"
            ],
            "Exit Rules": [
                "Price below 50-day moving average",
                "RSI above 70 or below 30"
            ],
            "Risk Management": {
                "Stop Loss": "2 ATR",
                "Position Sizing": "Risk 1% per trade",
                "Max Positions": 10
            }
        },
        "parameters": {
            "rsi_period": 14,
            "ma_period": 50,
            "entry_threshold": 40,
            "exit_threshold": 70,
            "stop_loss_atr_multiple": 2.0,
            "risk_per_trade": 0.01
        }
    }
    
    # Save strategy to file
    output_path = os.path.join(output_dir, "momentum_rsi_strategy.json")
    with open(output_path, 'w') as f:
        json.dump(strategy, f, indent=2)
    
    logger.info(f"Created sample strategy: {output_path}")
    
    return output_path

def main():
    """Main function to demonstrate market regime detection."""
    parser = argparse.ArgumentParser(description="Demonstrate market regime detection")
    parser.add_argument("--symbol", default="SPY", help="Symbol to download data for")
    parser.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--method", default="hmm", choices=["hmm", "gmm", "kmeans"], help="Regime detection method")
    parser.add_argument("--regimes", type=int, default=4, help="Number of regimes to detect")
    parser.add_argument("--strategy", default=None, help="Path to strategy JSON file (if not provided, a sample will be created)")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false", help="Disable visualizations")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download market data
    data = download_market_data(args.symbol, args.start, args.end)
    if data is None:
        return 1
    
    # Save market data
    data_path = os.path.join(args.output_dir, f"{args.symbol}_data.csv")
    data.to_csv(data_path)
    logger.info(f"Saved market data to {data_path}")
    
    # Detect market regimes
    visualizations_dir = os.path.join(args.output_dir, "visualizations")
    detector, results = detect_market_regimes(
        data, 
        method=args.method, 
        n_regimes=args.regimes, 
        visualize=args.visualize,
        output_dir=visualizations_dir
    )
    
    # Use provided strategy or create a sample one
    strategy_path = args.strategy
    if strategy_path is None:
        strategies_dir = os.path.join(args.output_dir, "strategies")
        strategy_path = create_sample_strategy(strategies_dir)
    
    # Adapt strategy to regimes
    adapted_strategies_dir = os.path.join(args.output_dir, "adapted_strategies")
    adapted_strategy = adapt_strategy_to_regimes(
        strategy_path, 
        data, 
        detector,
        adapted_strategies_dir
    )
    
    # Generate QuantConnect algorithm
    qc_dir = os.path.join(args.output_dir, "qc_algorithms")
    qc_file = generate_quantconnect_regime_algorithm(
        strategy_path,
        qc_dir,
        args.method,
        args.regimes,
        args.start,
        args.end
    )
    
    logger.info("\nMarket Regime Demonstration completed successfully!")
    logger.info(f"All outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()