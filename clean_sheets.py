#!/usr/bin/env python3
"""
Script to clean unused Google Sheets tabs
"""
import logging
from src.utils.google_sheet_integration import GoogleSheetIntegration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Clean unused Google Sheets tabs."""
    logger.info("Initializing Google Sheets connection")
    gs = GoogleSheetIntegration()
    if not gs.initialize():
        logger.error("Failed to initialize Google Sheets")
        return
    
    # Define sheets to clear/delete
    sheets_to_clear = [
        'Trades_Enhanced Test Strategy',
        'Trades_Mean Reversion',
        'Old Tests'
    ]
    
    # Clear sheets
    for name in sheets_to_clear:
        worksheet = gs.worksheets.get(name)
        if worksheet:
            worksheet.clear()
            logger.info(f"Cleared {name}")
            
    logger.info("Sheets cleanup completed")

if __name__ == "__main__":
    main()