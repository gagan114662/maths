# Activity logger for GOD MODE
import logging
import os
import json
import time
import threading
import queue
from datetime import datetime
import sys

# Set up logging
logger = logging.getLogger(__name__)

# Global variables
_message_queue = queue.Queue()
_worker_thread = None
_worker_running = False
_worksheet = None
_last_row = 2  # Start after header row

def initialize():
    """Initialize the activity logger"""
    global _worker_running, _worker_thread, _worksheet
    
    try:
        # Get Google Sheet integration
        try:
            from src.utils.google_sheet_integration import GoogleSheetIntegration
            sheets = GoogleSheetIntegration()
            if sheets.initialize():
                _worksheet = sheets.worksheets.get('System Activity Log')
                if _worksheet:
                    logger.info("Activity logger connected to Google Sheets")
        except Exception as sheet_error:
            logger.warning(f"Could not connect to Google Sheets: {str(sheet_error)}")
            
        # Start the worker thread
        _worker_running = True
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
        _worker_thread.start()
        
        logger.info("Activity logger initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize activity logger: {str(e)}")
        return False

def log_activity(action, component, status, details='', performance='', next_steps='', notes=''):
    """Log a system activity"""
    try:
        # Log to console
        print(f"[{component}] {action}: {status}")
        
        # Add to queue for background processing if sheet is available
        if _worker_running:
            _message_queue.put([action, component, status, details, performance, next_steps, notes])
    except Exception as e:
        logger.error(f"Error logging activity: {str(e)}")

def _worker_loop():
    """Background thread to process messages"""
    global _worker_running, _last_row
    
    while _worker_running:
        try:
            # Process all pending messages
            messages = []
            try:
                while True:
                    messages.append(_message_queue.get_nowait())
            except queue.Empty:
                pass
            
            # If we have messages and a worksheet, append them
            if messages and _worksheet:
                try:
                    # Add timestamp to each message
                    rows = []
                    for msg in messages:
                        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        rows.append([now] + msg)
                    
                    # Append all rows at once
                    _worksheet.append_rows(rows)
                    
                except Exception as sheet_error:
                    logger.error(f"Error writing to sheet: {str(sheet_error)}")
            
            # Sleep to avoid hammering APIs
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in activity logger worker: {str(e)}")
            time.sleep(5)  # Longer sleep on error

def shutdown():
    """Shut down the activity logger"""
    global _worker_running
    
    logger.info("Shutting down activity logger")
    _worker_running = False
    
    # Wait for worker thread to finish
    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=5)
