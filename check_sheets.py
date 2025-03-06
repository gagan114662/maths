#\!/usr/bin/env python3
"""Script to check Google Sheet tabs and their content."""

import logging
import os
import sys
from src.utils.google_sheet_integration import GoogleSheetIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SheetChecker")

def main():
    """Check Google Sheet tabs and their content."""
    logger.info("Initializing Google Sheets integration...")
    sheets = GoogleSheetIntegration()
    if not sheets.initialize():
        logger.error("Failed to initialize Google Sheets integration")
        return

    logger.info("Getting worksheet information...")
    
    # Get all available worksheets
    available_worksheets = sheets.worksheets.values()
    worksheet_names = [ws.title for ws in available_worksheets]
    logger.info(f"Available worksheets: {worksheet_names}")
    
    # Check which tabs have content
    for name in worksheet_names:
        try:
            worksheet = sheets.get_worksheet(name)
            if worksheet:
                cell_count = worksheet.cell_count
                # Get a sample of the first few rows if available
                rows = worksheet.get_values(start='A1', end='E5')
                row_count = len(rows) if rows else 0
                
                logger.info(f"Worksheet '{name}': {cell_count} cells, {row_count} sample rows")
                if rows and row_count > 0:
                    logger.info(f"  Headers: {rows[0] if len(rows[0]) > 0 else 'Empty'}")
            else:
                logger.info(f"Worksheet '{name}': Failed to retrieve")
        except Exception as e:
            logger.error(f"Error checking worksheet '{name}': {e}")
    
    # Identify tabs that should be deleted (those without useful content)
    tabs_to_consider_deletion = []
    
    for name in worksheet_names:
        if name in ['MARKET_RESEARCH', 'STRATEGY_FRAMEWORK', 'MATH_ANALYSIS', 
                    'CODE_GENERATION', 'PARAMETER_OPTIMIZATION']:
            tabs_to_consider_deletion.append(name)
    
    if tabs_to_consider_deletion:
        logger.info(f"Tabs that could be deleted (review first): {tabs_to_consider_deletion}")
    else:
        logger.info("No tabs identified for potential deletion")
    
    # Check if Trading Results is being updated
    try:
        trading_results = sheets.get_worksheet("Trading Results")
        if trading_results:
            values = trading_results.get_all_values()
            if values:
                logger.info(f"Trading Results tab has {len(values)} rows of data")
                logger.info(f"First row: {values[0] if values else 'None'}")
                logger.info(f"Last row: {values[-1] if values and len(values) > 1 else 'None'}")
            else:
                logger.info("Trading Results tab exists but has no data")
    except Exception as e:
        logger.error(f"Error checking Trading Results tab: {e}")

if __name__ == "__main__":
    main()
