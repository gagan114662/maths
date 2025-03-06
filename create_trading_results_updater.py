#!/usr/bin/env python3
"""
Script to specifically update Trading Results tab in Google Sheets.
"""

import os
import sys
import json
import logging
import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "trading_results_update.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingResultsUpdater")

# Import necessary modules
from src.utils.google_sheet_integration import GoogleSheetIntegration

def update_trading_results():
    """Update the Trading Results tab with the latest backtest results."""
    logger.info("Starting Trading Results update...")
    
    # Load completed backtests
    completed_file = os.path.join("mathematricks", "systems", "backtests_queue", "completed_backtests.json")
    try:
        with open(completed_file, 'r') as f:
            completed_backtests = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading completed backtests: {e}")
        return False
    
    if not completed_backtests:
        logger.info("No completed backtests found")
        return False
    
    # Initialize Google Sheets integration
    logger.info("Initializing Google Sheets integration...")
    sheets = GoogleSheetIntegration()
    if not sheets.initialize():
        logger.error("Failed to initialize Google Sheets integration")
        return False
    
    # Get trading results worksheet - access via the worksheets dictionary
    try:
        worksheet = sheets.worksheets.get("Trading Results")
        if not worksheet:
            logger.error("Trading Results worksheet not found")
            return False
    except Exception as e:
        logger.error(f"Error getting Trading Results worksheet: {e}")
        return False
    
    # Count successfully updated entries
    updated_count = 0
    
    # Update each backtest in the Trading Results tab
    for backtest_result in completed_backtests:
        try:
            backtest = backtest_result['backtest']
            results = backtest_result['results']
            
            # Extract performance metrics
            strategy_name = backtest.get('backtest_name', 'Unknown Strategy')
            timestamp = datetime.datetime.now()
            
            # Check if already processed
            if backtest_result.get("trading_results_updated", False):
                logger.info(f"Strategy {strategy_name} already updated in Trading Results, skipping...")
                continue
                
            # Build the row data directly
            row_data = [
                strategy_name,  # Strategy
                timestamp.strftime("%Y-%m-%d"),  # Date
                f"{results.get('performance', {}).get('annualized_return', 0)*100:.2f}%",  # CAGR
                f"{results.get('performance', {}).get('sharpe_ratio', 0):.2f}",  # Sharpe
                f"{results.get('performance', {}).get('max_drawdown', 0)*100:.2f}%",  # Drawdown
                f"{results.get('trades', {}).get('win_rate', 0)*100:.2f}%",  # Win Rate
                str(results.get('trades', {}).get('total_trades', 0)),  # Trades
                f"{results.get('trades', {}).get('average_trade', 0)*100:.2f}%",  # Avg Return
                f"{results.get('trades', {}).get('profit_factor', 1.0):.2f}",  # Profit Factor
                "Real Market Data (mathematricks)",  # Data Source
                "Completed",  # Status
                f"Backtest with real market data using the mathematricks framework"  # Notes
            ]
            
            # Append to Trading Results
            worksheet.append_row(row_data)
            logger.info(f"Added strategy {strategy_name} to Trading Results")
            
            # Mark as updated
            backtest_result["trading_results_updated"] = True
            updated_count += 1
            
        except Exception as e:
            logger.error(f"Error updating Trading Results for {backtest.get('backtest_name', 'Unknown')}: {e}")
    
    # Save updated status to completed_backtests.json if any were updated
    if updated_count > 0:
        with open(completed_file, 'w') as f:
            json.dump(completed_backtests, f, indent=2)
        logger.info(f"Updated {updated_count} strategies in Trading Results")
    
    return updated_count > 0

if __name__ == "__main__":
    update_trading_results()