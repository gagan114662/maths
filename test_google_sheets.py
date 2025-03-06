#\!/usr/bin/env python3
"""
Test script for Google Sheets integration.
This script tests the enhanced Google Sheets integration with all required tabs.
"""
import asyncio
import logging
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.utils.google_sheet_integration import GoogleSheetIntegration

def main():
    """Test enhanced Google Sheets integration with all required tabs."""
    try:
        # Initialize Google Sheets integration
        logger.info("Initializing Google Sheets integration...")
        sheets = GoogleSheetIntegration()
        
        # Initialize the connection
        init_result = sheets.initialize()
        logger.info(f"Google Sheets initialization result: {init_result}")
        
        if init_result:
            # Test AI Feedback tab
            logger.info("Testing AI Feedback tab...")
            test_ai_feedback(sheets)
            time.sleep(2)  # Avoid rate limiting
            
            # Test Strategy Performance
            logger.info("Testing Strategy Performance (Backtest Results)...")
            strategy_data = test_strategy_performance(sheets)
            time.sleep(2)  # Avoid rate limiting
            
            # Test Signals tab
            logger.info("Testing Signals tab...")
            test_signals(sheets, strategy_data)
            time.sleep(2)  # Avoid rate limiting
            
            # Test Hypotheses tab
            logger.info("Testing Hypotheses tab...")
            test_hypotheses(sheets, strategy_data)
            time.sleep(2)  # Avoid rate limiting
            
            # Test Trade Data
            logger.info("Testing Trade Data...")
            test_trade_data(sheets, strategy_data)
            
            logger.info("Google Sheets integration test completed successfully")
            
        else:
            logger.error("Google Sheets initialization failed")
        
    except Exception as e:
        logger.error(f"Error in Google Sheets test: {str(e)}", exc_info=True)

def test_ai_feedback(sheets):
    """Test AI Feedback tab updating."""
    # Create sample agent feedback
    agent_feedback = {
        'agent_name': 'Generation Agent',
        'message': 'Created new momentum strategy based on market regime analysis',
        'context': 'Market regime: Bullish',
        'decision': 'Generate momentum strategy',
        'reasoning': 'Momentum strategies perform well in bullish markets with strong trends',
        'action': 'Generated strategy code',
        'result': 'Strategy successfully created'
    }
    
    # Update AI Feedback tab
    result = sheets.log_agent_interaction(**agent_feedback)
    logger.info(f"AI Feedback update result: {result}")
    
    # Add another entry with minimal information
    result = sheets.log_agent_interaction(
        agent_name='Backtesting Agent',
        message='Executed backtest on historical data',
        result='Backtest completed with Sharpe ratio 1.35'
    )
    logger.info(f"Minimal AI Feedback update result: {result}")
    
    return result

def test_strategy_performance(sheets):
    """Test Strategy Performance (Backtest Results) tab updating."""
    # Create sample strategy data
    strategy_data = {
        "strategy_name": "Enhanced Test Strategy",
        "cagr": 0.28,
        "sharpe_ratio": 1.35,
        "max_drawdown": 0.15,
        "avg_profit": 0.009,
        "win_rate": 0.62,
        "trades_count": 152,
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "description": "Enhanced test strategy created via automated script",
        "universe": "S&P 500",
        "timeframe": "daily"
    }
    
    # Update performance data
    update_result = sheets.update_strategy_performance(strategy_data)
    logger.info(f"Strategy performance update result: {update_result}")
    
    return strategy_data

def test_signals(sheets, strategy_data):
    """Test Signals tab updating."""
    # The signals are automatically updated with strategy_performance
    # But we can directly test the update_signals method
    
    strategy_id = f"strategy_test_{int(time.time())}"
    targets_text = "✅ CAGR: 28.00% (Target: ≥25.0%)\n✅ Sharpe: 1.35 (Target: ≥1.0)\n✅ Drawdown: 15.00% (Target: ≤20.0%)\n✅ Avg Profit/Trade: 0.90% (Target: ≥0.75%)"
    
    result = sheets.update_signals(strategy_id, strategy_data, targets_text)
    logger.info(f"Direct Signals update result: {result}")
    
    return result

def test_hypotheses(sheets, strategy_data):
    """Test Hypotheses tab updating."""
    # The hypotheses are automatically updated with strategy_performance
    # But we can directly test the update_hypotheses method
    
    strategy_id = f"strategy_test_{int(time.time())}"
    meets_all_targets = True
    
    result = sheets.update_hypotheses(strategy_id, strategy_data, meets_all_targets)
    logger.info(f"Direct Hypotheses update result: {result}")
    
    return result

def test_trade_data(sheets, strategy_data):
    """Test Trade Data updating."""
    # Create sample trade data
    trades = [
        {
            "Entry Date": "2023-01-15",
            "Exit Date": "2023-02-01",
            "Symbol": "AAPL",
            "Direction": "LONG",
            "Entry Price": 142.53,
            "Exit Price": 150.82,
            "PnL": 8.29,
            "PnL %": 5.82
        },
        {
            "Entry Date": "2023-02-10",
            "Exit Date": "2023-03-05",
            "Symbol": "MSFT",
            "Direction": "LONG",
            "Entry Price": 263.10,
            "Exit Price": 280.32,
            "PnL": 17.22,
            "PnL %": 6.54
        },
        {
            "Entry Date": "2023-03-15",
            "Exit Date": "2023-04-01",
            "Symbol": "AMZN",
            "Direction": "SHORT",
            "Entry Price": 98.71,
            "Exit Price": 92.43,
            "PnL": 6.28,
            "PnL %": 6.36
        },
        {
            "Entry Date": "2023-04-10",
            "Exit Date": "2023-05-05",
            "Symbol": "GOOGL",
            "Direction": "LONG",
            "Entry Price": 104.22,
            "Exit Price": 111.75,
            "PnL": 7.53,
            "PnL %": 7.22
        },
        {
            "Entry Date": "2023-05-15",
            "Exit Date": "2023-06-01",
            "Symbol": "TSLA",
            "Direction": "SHORT",
            "Entry Price": 180.11,
            "Exit Price": 168.54,
            "PnL": 11.57,
            "PnL %": 6.42
        }
    ]
    
    trades_df = pd.DataFrame(trades)
    
    # Update trades data
    result = sheets.update_strategy_trades(strategy_data["strategy_name"], trades_df)
    logger.info(f"Trade data update result: {result}")
    
    return result

if __name__ == "__main__":
    main()