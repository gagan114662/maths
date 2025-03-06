#!/usr/bin/env python3
import os
import sys
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/queue_checker.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import GoogleSheetIntegration
from src.utils.google_sheet_integration import GoogleSheetIntegration

def check_backtest_queue():
    # Check the backtest queue and prepare strategies for testing
    # Initialize GoogleSheetIntegration
    gs = GoogleSheetIntegration()
    if not gs.initialize():
        logger.error("Failed to initialize Google Sheets integration")
        return False
    
    try:
        # Get the Backtest Queue worksheet
        worksheet = gs.worksheets.get('Backtest Queue')
        if not worksheet:
            logger.error("Backtest Queue worksheet not found")
            return False
        
        # Get all records
        queue_items = worksheet.get_all_records()
        
        # Filter for pending items
        pending_items = [item for item in queue_items if item.get('Status', '').lower() == 'pending']
        
        if not pending_items:
            logger.info("No pending items in backtest queue")
            return True
        
        logger.info(f"Found {len(pending_items)} pending items in backtest queue")
        
        # Process each pending item
        for i, item in enumerate(pending_items):
            queue_id = item.get('ID', f"queue_{i}")
            strategy_name = item.get('Strategy Name', f"Strategy_{i}")
            strategy_type = item.get('Type', 'Initial')
            market = item.get('Market', 'US Equities')
            
            # Parse parameters
            parameters_str = item.get('Parameters', '{}')
            try:
                parameters = eval(parameters_str)
            except:
                parameters = {}
            
            # Create strategy configuration
            strategy_config = {
                "name": strategy_name,
                "strategy_type": strategy_type,
                "market": market,
                "parameters": parameters
            }
            
            # Save configuration to file
            config_dir = os.path.join(parent_dir, "backtest_queue")
            os.makedirs(config_dir, exist_ok=True)
            
            config_file = os.path.join(config_dir, f"{queue_id}.json")
            with open(config_file, 'w') as f:
                json.dump(strategy_config, f, indent=4)
            
            # Update status to "Processing"
            row_idx = queue_items.index(item) + 2  # +2 for header row and 0-indexing
            worksheet.update_cell(row_idx, 4, "Processing")
            
            logger.info(f"Prepared strategy {strategy_name} (ID: {queue_id}) for backtesting")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking backtest queue: {str(e)}")
        return False

if __name__ == "__main__":
    check_backtest_queue()
