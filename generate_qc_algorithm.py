#!/usr/bin/env python
"""
Script to generate QuantConnect algorithm files from strategy JSON files or directly from the system.

This script integrates with the strategy generation system to:
1. Take generated strategies and convert them to QuantConnect format
2. Generate optimized strategies directly for QuantConnect
3. Track performance metrics of strategies on QuantConnect

Usage:
    python generate_qc_algorithm.py --strategy path/to/strategy.json --output qc_output/
    python generate_qc_algorithm.py --auto --num 3  # Generate 3 new optimized strategies
"""

import os
import sys
import json
import datetime
import logging
import argparse
import pandas as pd
import numpy as np
import random
from pathlib import Path
from glob import glob

# Import the QuantConnect adapter
from quant_connect_adapter import QuantConnectAdapter, QuantConnectStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qc_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class QuantConnectGenerator:
    """Class to generate QuantConnect algorithms from existing strategies or create new ones."""
    
    def __init__(self, output_dir="qc_algorithms"):
        """
        Initialize the generator.
        
        Args:
            output_dir (str): Directory to save generated algorithms
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def generate_from_json(self, strategy_path, start_date=None, end_date=None, cash=100000):
        """
        Generate a QuantConnect algorithm from a strategy JSON file.
        
        Args:
            strategy_path (str): Path to strategy JSON file
            start_date (str): Backtest start date (YYYY-MM-DD)
            end_date (str): Backtest end date (YYYY-MM-DD)
            cash (int): Initial capital
            
        Returns:
            str: Path to generated algorithm file
        """
        logger.info(f"Generating QuantConnect algorithm from {strategy_path}")
        
        qc_file = QuantConnectStrategy.generate_momentum_rsi_volatility(
            strategy_path, 
            self.output_dir, 
            start_date, 
            end_date, 
            cash
        )
        
        logger.info(f"Generated QuantConnect algorithm: {qc_file}")
        return qc_file
    
    def generate_from_latest(self, num_strategies=1, start_date=None, end_date=None, cash=100000):
        """
        Generate QuantConnect algorithms from the latest strategy JSON files.
        
        Args:
            num_strategies (int): Number of strategies to generate
            start_date (str): Backtest start date (YYYY-MM-DD)
            end_date (str): Backtest end date (YYYY-MM-DD)
            cash (int): Initial capital
            
        Returns:
            list: Paths to generated algorithm files
        """
        strategy_dir = os.path.join(os.getcwd(), 'generated_strategies')
        json_files = sorted(glob(os.path.join(strategy_dir, '*.json')), key=os.path.getmtime, reverse=True)
        
        if not json_files:
            logger.error("No strategy JSON files found in generated_strategies directory")
            return []
        
        # Take the latest n strategies
        json_files = json_files[:num_strategies]
        
        generated_files = []
        for json_file in json_files:
            qc_file = self.generate_from_json(json_file, start_date, end_date, cash)
            generated_files.append(qc_file)
        
        return generated_files
    
    def generate_auto(self, num_strategies=1, start_date=None, end_date=None, cash=100000):
        """
        Automatically generate optimized strategies and convert to QuantConnect algorithms.
        
        Args:
            num_strategies (int): Number of strategies to generate
            start_date (str): Backtest start date (YYYY-MM-DD)
            end_date (str): Backtest end date (YYYY-MM-DD)
            cash (int): Initial capital
            
        Returns:
            list: Paths to generated algorithm files
        """
        from strategy_generator import generate_strategy
        
        generated_files = []
        for i in range(num_strategies):
            # Generate a new strategy
            strategy_name = f"QC_Optimized_Strategy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Call the strategy generator
            try:
                strategy_path = generate_strategy(strategy_name)
                
                # Convert to QuantConnect algorithm
                qc_file = self.generate_from_json(strategy_path, start_date, end_date, cash)
                generated_files.append(qc_file)
                
            except Exception as e:
                logger.error(f"Error generating strategy {i+1}: {e}")
                continue
        
        return generated_files

def optimize_parameters(strategy_path, output_path=None):
    """
    Optimize strategy parameters for QuantConnect.
    
    Args:
        strategy_path (str): Path to strategy JSON file
        output_path (str): Path to save optimized strategy JSON
        
    Returns:
        str: Path to optimized strategy JSON
    """
    # Load strategy
    with open(strategy_path, 'r') as f:
        strategy_data = json.load(f)
    
    # Perform optimization (simplified example)
    # In a real implementation, this would use historical data and parameter search
    optimized_strategy = strategy_data.copy()
    
    # Modify parameters based on optimization
    # This is a placeholder for actual optimization logic
    if 'strategy' in optimized_strategy:
        # Example parameter modifications
        entry_rules = optimized_strategy['strategy'].get('Entry Rules', [])
        for i, rule in enumerate(entry_rules):
            if 'RSI' in rule.get('Rule', ''):
                # Optimize RSI parameters
                entry_rules[i]['Rule'] = entry_rules[i]['Rule'].replace('RSI must be above 50', 'RSI must be above 55')
    
    # Save optimized strategy
    if output_path is None:
        strategy_name = os.path.basename(strategy_path)
        output_path = os.path.join(os.path.dirname(strategy_path), f"optimized_{strategy_name}")
    
    with open(output_path, 'w') as f:
        json.dump(optimized_strategy, f, indent=2)
    
    return output_path

def main():
    """Main function to parse arguments and generate algorithms."""
    parser = argparse.ArgumentParser(description="Generate QuantConnect algorithms from strategies")
    
    # Strategy source options
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--strategy", help="Path to strategy JSON file")
    source_group.add_argument("--latest", action="store_true", help="Use latest generated strategies")
    source_group.add_argument("--auto", action="store_true", help="Generate new optimized strategies")
    
    # Common options
    parser.add_argument("--output", default="qc_algorithms", help="Output directory for generated algorithms")
    parser.add_argument("--num", type=int, default=1, help="Number of strategies to generate (for --latest and --auto)")
    parser.add_argument("--start", default=None, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=int, default=100000, help="Initial capital")
    parser.add_argument("--optimize", action="store_true", help="Optimize strategy parameters before generation")
    
    args = parser.parse_args()
    
    # Create generator
    generator = QuantConnectGenerator(args.output)
    
    # Generate algorithms based on source
    if args.strategy:
        strategy_path = args.strategy
        
        # Optimize if requested
        if args.optimize:
            strategy_path = optimize_parameters(strategy_path)
        
        # Generate algorithm
        qc_file = generator.generate_from_json(strategy_path, args.start, args.end, args.cash)
        print(f"Generated QuantConnect algorithm: {qc_file}")
        
    elif args.latest:
        # Generate from latest strategies
        qc_files = generator.generate_from_latest(args.num, args.start, args.end, args.cash)
        for qc_file in qc_files:
            print(f"Generated QuantConnect algorithm: {qc_file}")
        
    elif args.auto:
        # Generate new optimized strategies
        qc_files = generator.generate_auto(args.num, args.start, args.end, args.cash)
        for qc_file in qc_files:
            print(f"Generated QuantConnect algorithm: {qc_file}")
    
if __name__ == "__main__":
    main()