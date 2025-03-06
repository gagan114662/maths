#\!/usr/bin/env python3
"""
Script to update Google Sheets with completed backtest results.
"""
import os
import sys
import json
import logging
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "sheets_update.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SheetsUpdater")

# Import necessary modules
try:
    from src.utils.google_sheet_integration import GoogleSheetIntegration
    sheets_available = True
except ImportError:
    logger.warning("Google Sheets integration not available")
    sheets_available = False

def update_sheets():
    """Update Google Sheets with latest backtest results"""
    if not sheets_available:
        logger.error("Google Sheets integration not available - cannot update")
        return False
    
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
    
    # Get unprocessed backtests (where processed = False or not present)
    unprocessed = [b for b in completed_backtests if not b.get("processed", False)]
    
    if not unprocessed:
        logger.info("No unprocessed backtests found")
        return False
    
    logger.info(f"Found {len(unprocessed)} unprocessed backtest results")
    
    # Initialize Google Sheets integration
    logger.info("Initializing Google Sheets integration...")
    sheets = GoogleSheetIntegration()
    if not sheets.initialize():
        logger.error("Failed to initialize Google Sheets integration")
        return False
    
    # Process each unprocessed backtest
    updated_count = 0
    
    for backtest_result in unprocessed:
        try:
            backtest = backtest_result['backtest']
            results = backtest_result['results']
            
            strategy_name = backtest.get('backtest_name', 'Unknown Strategy')
            logger.info(f"Processing results for: {strategy_name}")
            
            # Log feedback
            feedback = {
                "agent_name": "Mathematricks Backtester",
                "feedback": f"Completed backtest for {strategy_name} using mathematricks framework with real market data. "+
                           f"Results: CAGR: {results.get('performance', {}).get('annualized_return', 0):.2f}, "+
                           f"Sharpe: {results.get('performance', {}).get('sharpe_ratio', 0):.2f}, "+
                           f"Win Rate: {results.get('trades', {}).get('win_rate', 0):.2f}",
                "feedback_type": "SUCCESS"
            }
            
            try:
                sheets.update_ai_feedback([{
                    "agent": feedback["agent_name"],
                    "feedback": feedback["feedback"],
                    "type": feedback["feedback_type"],
                    "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }])
                logger.info("Added feedback to AI Feedback tab")
            except Exception as e:
                logger.error(f"Error updating AI Feedback: {e}")
            
            # Create strategy model
            strategy_model = {
                "name": backtest.get('backtest_name'),
                "description": f"Strategy using {backtest.get('strategy_name')} algorithm with mathematricks framework",
                "universe": backtest.get('universe'),
                "performance": {
                    "cagr": results.get('performance', {}).get('annualized_return', 0),
                    "sharpe_ratio": results.get('performance', {}).get('sharpe_ratio', 0),
                    "max_drawdown": results.get('performance', {}).get('max_drawdown', 0),
                    "volatility": results.get('performance', {}).get('volatility', 0)
                },
                "trades": {
                    "total_trades": results.get('trades', {}).get('total_trades', 0),
                    "win_rate": results.get('trades', {}).get('win_rate', 0),
                    "average_trade": results.get('trades', {}).get('average_trade', 0),
                    "profit_factor": results.get('trades', {}).get('profit_factor', 0)
                }
            }
            
            # Update performance
            try:
                sheets.update_strategy_performance([strategy_model])
                logger.info("Updated strategy performance")
            except Exception as e:
                logger.error(f"Error updating strategy performance: {e}")
            
            # Mark as processed
            backtest_result["processed"] = True
            backtest_result["processed_at"] = datetime.datetime.now().isoformat()
            updated_count += 1
            
        except Exception as e:
            logger.error(f"Error processing backtest result: {e}")
    
    # Save updated backtests
    if updated_count > 0:
        with open(completed_file, 'w') as f:
            json.dump(completed_backtests, f, indent=2)
        logger.info(f"Updated {updated_count} backtest results in Google Sheets")
        return True
    
    return False

if __name__ == "__main__":
    update_sheets()
