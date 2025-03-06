#\!/usr/bin/env python3
"""
Update Google Sheets with strategy performance (minimal version).
"""
import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.utils.google_sheet_integration import GoogleSheetIntegration

def main():
    """Update Google Sheets with strategy performance."""
    try:
        # Load the strategy file
        strategy_path = "generated_strategies/Momentum_RSI_Volatility_20250304_102400.json"
        logger.info(f"Loading strategy from {strategy_path}")
        
        with open(strategy_path, 'r') as f:
            data = json.load(f)
        
        strategy = data['strategy']
        results = data['results']
        
        # Format strategy data for Google Sheets
        strategy_data = {
            "strategy_name": strategy["Strategy Name"],
            "cagr": results["performance"]["annualized_return"],
            "sharpe_ratio": results["performance"]["sharpe_ratio"],
            "max_drawdown": abs(results["performance"]["max_drawdown"]),
            "avg_profit": results["trades"]["average_trade"],
            "win_rate": results["trades"]["win_rate"],
            "trades_count": results["trades"]["total_trades"],
            "start_date": "2023-01-01",  # Example date
            "end_date": "2024-01-01",    # Example date
            "description": strategy["Edge"],
            "universe": strategy["Universe"],
            "timeframe": strategy["Timeframe"]
        }
        
        # Initialize Google Sheets integration
        logger.info("Initializing Google Sheets integration...")
        sheets = GoogleSheetIntegration()
        
        # Initialize the connection
        init_result = sheets.initialize()
        logger.info(f"Google Sheets initialization result: {init_result}")
        
        if init_result:
            # Update performance data only
            logger.info("Updating strategy performance data...")
            update_result = sheets.update_strategy_performance(strategy_data)
            logger.info(f"Strategy performance update result: {update_result}")
            logger.info("Google Sheets update completed")
        else:
            logger.error("Google Sheets initialization failed")
        
    except Exception as e:
        logger.error(f"Error updating Google Sheets: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
