# GOD MODE setup script
from src.utils.google_sheet_integration import GoogleSheetIntegration
import gspread
import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def setup_god_mode():
    sheets = GoogleSheetIntegration()
    if sheets.initialize():
        # Remove all existing summary sheets and create a fresh one
        try:
            for title in ['Summary', 'Summary ', 'System Activity Log']:
                try:
                    worksheet = sheets.sheet.worksheet(title)
                    sheets.sheet.del_worksheet(worksheet)
                    logger.info(f'Deleted existing worksheet: {title}')
                except gspread.exceptions.WorksheetNotFound:
                    pass
        except Exception as e:
            logger.info(f'Error cleaning up sheets: {str(e)}')
        
        # Create a fresh Activity Log sheet
        activity_log = sheets.sheet.add_worksheet('System Activity Log', rows=1000, cols=8)
        sheets.worksheets['System Activity Log'] = activity_log  # Store with correct name
        sheets.worksheets['Summary'] = activity_log  # Also use this as our Summary sheet
        
        # Add headers with better formatting
        headers = ['Timestamp', 'Action', 'Component', 'Status', 'Details', 'Performance', 'Next Steps', 'Notes']
        activity_log.append_row(headers)
        # Make header row bold and frozen
        activity_log.format('A1:H1', {'textFormat': {'bold': True}})
        activity_log.freeze(rows=1)
        # Set column widths for better readability
        activity_log.set_column_width(0, 180)  # Timestamp
        activity_log.set_column_width(1, 150)  # Action
        return True
    return False

if __name__ == "__main__":
    setup_god_mode()
