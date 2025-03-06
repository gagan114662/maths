#!/usr/bin/env python3
"""
Prepare data for DeepSeek R1 GOD MODE.

This script prepares data for the different market regimes:
- Extreme: High volatility, significant price changes
- Fall: Consistent downtrend
- Fluctuation: Sideways or range-bound markets
- Rise: Consistent uptrend

It uses local data instead of EastMoney API and creates datasets
that can be used for GOD MODE strategy development.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/data_preparation.log")
    ]
)
logger = logging.getLogger(__name__)

# Import market classification
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data.market_classification import MarketClassifier, get_regime_data

def prepare_regime_data(args):
    """
    Prepare data for different market regimes.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with dataset information
    """
    logger.info(f"Preparing data for market regimes: {args.regimes}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize market classifier
    classifier = MarketClassifier()
    
    # Get market statistics
    market_stats = classifier.get_market_statistics(args.market)
    logger.info(f"Market regime statistics: {market_stats}")
    
    # Process each regime
    datasets = {}
    
    for regime in args.regimes:
        logger.info(f"Processing {regime} regime for {args.market} market")
        
        # Get regime data
        regime_data = get_regime_data(
            market=args.market,
            regime=regime,
            timeframe=args.timeframe,
            sample_size=args.sample_size
        )
        
        # Save symbols
        symbols = regime_data["symbols"]
        logger.info(f"Found {len(symbols)} symbols for {regime} regime")
        
        # Create regime directory
        regime_dir = output_dir / regime
        regime_dir.mkdir(exist_ok=True)
        
        # Save symbol list
        symbol_file = regime_dir / "symbols.json"
        with open(symbol_file, "w") as f:
            json.dump(symbols, f, indent=2)
        
        # Save metadata
        metadata = {
            "regime": regime,
            "market": args.market,
            "timeframe": args.timeframe,
            "sample_size": len(symbols),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market_stats": market_stats
        }
        
        metadata_file = regime_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Record dataset information
        datasets[regime] = {
            "symbols": symbols,
            "symbol_file": str(symbol_file),
            "metadata_file": str(metadata_file),
            "data_dir": str(regime_dir)
        }
    
    # Save summary
    summary = {
        "market": args.market,
        "timeframe": args.timeframe,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "regimes": args.regimes,
        "datasets": datasets,
        "market_stats": market_stats
    }
    
    summary_file = output_dir / "god_mode_data_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Data preparation complete. Summary saved to {summary_file}")
    return summary

def main():
    """Main function for preparing GOD MODE data."""
    parser = argparse.ArgumentParser(description="Prepare data for DeepSeek R1 GOD MODE")
    
    parser.add_argument("--market", type=str, default="us_equities",
                        choices=["us_equities", "crypto", "forex"],
                        help="Market to process")
    parser.add_argument("--timeframe", type=str, default="daily",
                        choices=["intraday", "daily", "weekly", "monthly"],
                        help="Timeframe for data")
    parser.add_argument("--regimes", nargs="+", 
                        default=["extreme", "fall", "fluctuation", "rise"],
                        help="Market regimes to process")
    parser.add_argument("--sample-size", type=int, default=30,
                        help="Number of symbols to include in each regime")
    parser.add_argument("--output-dir", type=str, default="data/god_mode",
                        help="Output directory for prepared data")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Prepare data for each regime
        summary = prepare_regime_data(args)
        
        # Print summary
        print("\nGOD MODE Data Preparation Complete!")
        print(f"Market: {args.market}")
        print(f"Timeframe: {args.timeframe}")
        print("Regimes processed:")
        
        for regime, dataset in summary["datasets"].items():
            symbols_count = len(dataset["symbols"])
            print(f"  - {regime.capitalize()}: {symbols_count} symbols")
        
        print(f"\nData saved to: {args.output_dir}")
        print(f"Summary file: {os.path.join(args.output_dir, 'god_mode_data_summary.json')}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())