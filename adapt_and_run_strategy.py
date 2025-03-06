#!/usr/bin/env python3
"""
Script to adapt a trading strategy to the current market regime and run it on QuantConnect.

This script:
1. Detects the current market regime using the MarketRegimeDetector
2. Adapts a trading strategy to the detected regime using RegimeAwareStrategyAdapter
3. Converts the adapted strategy to QuantConnect format using QuantConnectAdapter
4. Optionally uploads the strategy to QuantConnect for backtesting or live trading
"""

import os
import sys
import argparse
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
from src.market_regime_detector import MarketRegimeDetector
from src.regime_aware_strategy_adapter import RegimeAwareStrategyAdapter
from quant_connect_adapter import QuantConnectAdapter, QuantConnectStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("regime_strategy_adapter.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Adapt and run a trading strategy based on market regime")
    
    # Strategy input
    parser.add_argument("--strategy", required=True, help="Path to the strategy JSON file")
    
    # Market regime detection parameters
    parser.add_argument("--method", default="hmm", choices=["hmm", "gmm", "kmeans", "hierarchical"], 
                        help="Market regime detection method")
    parser.add_argument("--n-regimes", type=int, default=4, help="Number of market regimes to detect")
    parser.add_argument("--symbol", default="SPY", help="Symbol to use for market regime detection")
    parser.add_argument("--start", default="2018-01-01", help="Start date for data download (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date for data download (YYYY-MM-DD)")
    
    # Output paths
    parser.add_argument("--output-dir", default="output", help="Output directory for all generated files")
    parser.add_argument("--regime-config", help="Path to optional regime configuration file")
    parser.add_argument("--visualize", action="store_true", help="Visualize market regimes and strategy performance")
    
    # QuantConnect parameters
    parser.add_argument("--qc-backtest-start", help="Start date for QuantConnect backtest (YYYY-MM-DD)")
    parser.add_argument("--qc-backtest-end", help="End date for QuantConnect backtest (YYYY-MM-DD)")
    parser.add_argument("--qc-cash", type=int, default=100000, help="Initial cash for QuantConnect backtest")
    parser.add_argument("--qc-live", action="store_true", help="Prepare strategy for live trading on QuantConnect")
    
    return parser.parse_args()

def ensure_output_directory(output_dir):
    """Ensure output directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different outputs
    os.makedirs(os.path.join(output_dir, "strategies"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "qc_algorithms"), exist_ok=True)
    
    return output_dir

def download_market_data(symbol, start_date, end_date=None):
    """Download market data for regime detection."""
    logger.info(f"Downloading market data for {symbol} from {start_date} to {end_date or 'now'}")
    
    # Use current date if end_date is not provided
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Download data using yfinance
        data = yf.download(symbol, start=start_date, end=end_date)
        
        if data.empty:
            logger.error(f"No data returned for {symbol}")
            raise ValueError(f"No data returned for {symbol}")
        
        logger.info(f"Downloaded {len(data)} data points for {symbol}")
        return data
    
    except Exception as e:
        logger.error(f"Error downloading market data: {str(e)}")
        raise

def detect_market_regime(market_data, method, n_regimes):
    """Detect the current market regime using the MarketRegimeDetector."""
    logger.info(f"Detecting market regime using {method.upper()} with {n_regimes} regimes")
    
    try:
        # Create the regime detector
        detector = MarketRegimeDetector(method=method, n_regimes=n_regimes)
        
        # Detect regime
        regime_results = detector.load_or_train(market_data)
        
        # Log results
        current_regime = regime_results["current_regime"]
        regime_label = regime_results["regime_label"]
        logger.info(f"Detected market regime: {regime_label} (ID: {current_regime})")
        
        return detector, regime_results
    
    except Exception as e:
        logger.error(f"Error detecting market regime: {str(e)}")
        raise

def adapt_strategy_to_regime(strategy_path, regime_detector, market_data, regime_config_path=None):
    """Adapt a trading strategy to the current market regime."""
    logger.info(f"Adapting strategy from {strategy_path} to detected market regime")
    
    try:
        # Create the strategy adapter
        adapter = RegimeAwareStrategyAdapter(
            strategy_path=strategy_path,
            regime_detector=regime_detector,
            regime_config_path=regime_config_path
        )
        
        # Get the adapted strategy
        adapted_strategy = adapter.get_regime_adapted_strategy(market_data)
        
        # Log details about adaptation
        current_regime = adapter.current_regime
        regime_label = adapter.regime_label
        strategy_name = adapted_strategy.get("strategy", {}).get("Strategy Name", "Unknown Strategy")
        
        logger.info(f"Adapted '{strategy_name}' to regime {regime_label} (ID: {current_regime})")
        
        return adapter, adapted_strategy
    
    except Exception as e:
        logger.error(f"Error adapting strategy to current regime: {str(e)}")
        raise

def convert_to_quantconnect(adapted_strategy, output_dir, qc_start_date=None, qc_end_date=None, qc_cash=100000):
    """Convert the adapted strategy to QuantConnect format."""
    logger.info("Converting adapted strategy to QuantConnect format")
    
    try:
        # Save the adapted strategy to a temporary file
        strategy_name = adapted_strategy.get("strategy", {}).get("Strategy Name", "Unknown Strategy")
        strategy_name = strategy_name.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        temp_strategy_path = os.path.join(output_dir, "strategies", f"{strategy_name}_adapted_{timestamp}.json")
        
        with open(temp_strategy_path, 'w') as f:
            json.dump(adapted_strategy, f, indent=2)
        
        # Convert to QuantConnect format
        qc_output_dir = os.path.join(output_dir, "qc_algorithms")
        qc_algorithm_path = QuantConnectStrategy.generate_momentum_rsi_volatility(
            temp_strategy_path, 
            qc_output_dir, 
            qc_start_date, 
            qc_end_date, 
            qc_cash
        )
        
        logger.info(f"Generated QuantConnect algorithm: {qc_algorithm_path}")
        
        return qc_algorithm_path
    
    except Exception as e:
        logger.error(f"Error converting strategy to QuantConnect format: {str(e)}")
        raise

def visualize_results(adapter, market_data, output_dir):
    """Visualize market regimes and strategy performance."""
    logger.info("Generating visualizations")
    
    try:
        # Generate visualization of regime impact on strategy performance
        strategy_name = adapter.strategy_data.get("strategy", {}).get("Strategy Name", "Unknown Strategy")
        strategy_name = strategy_name.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create visualization paths
        regime_vis_path = os.path.join(output_dir, "visualizations", f"{strategy_name}_regimes_{timestamp}.png")
        performance_vis_path = os.path.join(output_dir, "visualizations", f"{strategy_name}_performance_{timestamp}.png")
        
        # Generate regime visualization
        adapter.regime_detector.plot_regimes(market_data, output_path=regime_vis_path)
        
        # Generate performance visualization
        adapter.visualize_regime_impact(market_data, output_path=performance_vis_path)
        
        logger.info(f"Visualizations saved to {regime_vis_path} and {performance_vis_path}")
        
        return {
            "regime_visualization": regime_vis_path,
            "performance_visualization": performance_vis_path
        }
    
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        logger.info("Continuing without visualizations")
        return {}

def prepare_for_quantconnect_live(qc_algorithm_path):
    """Prepare the algorithm for live trading on QuantConnect."""
    logger.info("Preparing algorithm for live trading on QuantConnect")
    
    try:
        # Read the algorithm file
        with open(qc_algorithm_path, 'r') as f:
            algorithm_code = f.read()
        
        # Modify the algorithm for live trading
        # This is a simplified example - actual implementation would depend on QuantConnect API
        live_algorithm_code = algorithm_code.replace(
            "def Initialize(self):",
            "def Initialize(self):\n        self.SetBrokerageModel(BrokerageName.QuantConnect, AccountType.Margin)"
        )
        
        # Save the modified algorithm
        live_algorithm_path = qc_algorithm_path.replace(".py", "_live.py")
        with open(live_algorithm_path, 'w') as f:
            f.write(live_algorithm_code)
        
        logger.info(f"Live trading algorithm saved to {live_algorithm_path}")
        
        return live_algorithm_path
    
    except Exception as e:
        logger.error(f"Error preparing algorithm for live trading: {str(e)}")
        raise

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Ensure output directory exists
        output_dir = ensure_output_directory(args.output_dir)
        
        # Download market data for regime detection
        market_data = download_market_data(args.symbol, args.start, args.end)
        
        # Detect market regime
        regime_detector, regime_results = detect_market_regime(
            market_data, 
            args.method, 
            args.n_regimes
        )
        
        # Adapt strategy to regime
        adapter, adapted_strategy = adapt_strategy_to_regime(
            args.strategy, 
            regime_detector, 
            market_data, 
            args.regime_config
        )
        
        # Save the adapted strategy
        strategy_name = adapted_strategy.get("strategy", {}).get("Strategy Name", "Unknown Strategy")
        strategy_name = strategy_name.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        adapted_strategy_path = os.path.join(
            output_dir, 
            "strategies", 
            f"{strategy_name}_regime_{adapter.current_regime}_{timestamp}.json"
        )
        
        with open(adapted_strategy_path, 'w') as f:
            json.dump(adapted_strategy, f, indent=2)
        
        logger.info(f"Saved adapted strategy to {adapted_strategy_path}")
        
        # Convert to QuantConnect format
        qc_algorithm_path = convert_to_quantconnect(
            adapted_strategy, 
            output_dir, 
            args.qc_backtest_start, 
            args.qc_backtest_end, 
            args.qc_cash
        )
        
        # Prepare for live trading if requested
        if args.qc_live:
            live_algorithm_path = prepare_for_quantconnect_live(qc_algorithm_path)
            logger.info(f"Algorithm prepared for live trading: {live_algorithm_path}")
        
        # Visualize results if requested
        if args.visualize:
            visualizations = visualize_results(adapter, market_data, output_dir)
            
            # Log paths to visualizations
            for vis_type, vis_path in visualizations.items():
                logger.info(f"{vis_type.replace('_', ' ').title()}: {vis_path}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info(f"SUMMARY:")
        logger.info(f"Market Regime: {adapter.regime_label} (ID: {adapter.current_regime})")
        logger.info(f"Adapted Strategy: {adapted_strategy_path}")
        logger.info(f"QuantConnect Algorithm: {qc_algorithm_path}")
        if args.qc_live:
            logger.info(f"Live Trading Algorithm: {live_algorithm_path}")
        if args.visualize and visualizations:
            logger.info(f"Visualizations: {', '.join(visualizations.values())}")
        logger.info("="*80 + "\n")
        
        print(f"\nStrategy successfully adapted to market regime: {adapter.regime_label}")
        print(f"QuantConnect algorithm generated: {os.path.basename(qc_algorithm_path)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.exception("Detailed error:")
        return 1

if __name__ == "__main__":
    sys.exit(main())