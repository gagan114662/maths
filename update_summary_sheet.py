#!/usr/bin/env python3
"""
Update the Summary sheet in Google Sheets to track system activity.
"""
import logging
import logging.config
import sys
import yaml
import argparse
from datetime import datetime

# Configure advanced logging with config file
try:
    with open('config/logging.yaml', 'r') as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
except Exception as e:
    # Fallback to basic logging if loading config fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    print(f"Warning: Could not load logging config: {e}")

logger = logging.getLogger(__name__)

from src.utils.google_sheet_integration import GoogleSheetIntegration

def main():
    """Update the Summary sheet with system activity."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Update Google Sheets Summary sheet with system activity")
    parser.add_argument("--action", type=str, required=True, help="Type of action performed (e.g., 'Backtest', 'Strategy Generation')")
    parser.add_argument("--component", type=str, required=True, help="System component involved (e.g., 'God Mode', 'Autopilot')")
    parser.add_argument("--status", type=str, required=True, help="Status of the action (e.g., 'Completed', 'Failed', 'In Progress')")
    parser.add_argument("--details", type=str, default="", help="Details about the action")
    parser.add_argument("--performance", type=str, default="", help="Performance metrics if applicable")
    parser.add_argument("--next-steps", type=str, default="", help="Next steps to be taken")
    parser.add_argument("--notes", type=str, default="", help="Additional notes")
    
    args = parser.parse_args()
    
    try:
        # Initialize Google Sheets integration
        logger.info("Initializing Google Sheets integration...")
        sheets = GoogleSheetIntegration()
        
        # Initialize the connection
        init_result = sheets.initialize()
        logger.info(f"Google Sheets initialization result: {init_result}")
        
        if init_result:
            # For compatibility, try to update both Summary and System Activity Log
            summary_result = False
            
            # Try System Activity Log first
            try:
                activity_log = sheets.sheet.worksheet('System Activity Log')
                if activity_log:
                    # Add a row to the activity log
                    import datetime
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    activity_log.append_row([
                        now,
                        args.action,
                        args.component,
                        args.status,
                        args.details,
                        args.performance,
                        args.next_steps,
                        args.notes
                    ])
                    summary_result = True
                    logger.info(f"Added log entry to System Activity Log for '{args.action}'")
            except Exception as e:
                logger.warning(f"Could not update System Activity Log: {str(e)}")
                
            # Fall back to Summary if needed
            if not summary_result:
                summary_result = sheets.update_summary(
                    action=args.action,
                    component=args.component,
                    status=args.status,
                    details=args.details,
                    performance=args.performance,
                    next_steps=args.next_steps,
                    notes=args.notes
                )
            
            if summary_result:
                logger.info("Summary sheet updated successfully")
                return 0
            else:
                logger.error("Failed to update Summary sheet")
                return 1
        else:
            logger.error("Google Sheets initialization failed")
            return 1
            
    except Exception as e:
        logger.error(f"Error updating Summary sheet: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())