#\!/usr/bin/env python3
"""
Autonomous backtester that continuously generates strategies,
processes the backtest queue, and updates Google Sheets.
"""
import os
import sys
import time
import random
import logging
import subprocess
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "autopilot.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutopilotBacktester")

def run_command(command, timeout=None):
    """Run a shell command and log the output"""
    try:
        logger.info(f"Running command: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            logger.info(f"Command succeeded: {command}")
            if result.stdout:
                logger.debug(f"Command output: {result.stdout}")
            return True
        else:
            logger.error(f"Command failed with error code {result.returncode}: {command}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {command}")
        return False
    except Exception as e:
        logger.error(f"Error running command {command}: {e}")
        return False

def ensure_queue_processor_running():
    """Make sure the queue processor is running"""
    # Check if queue processor is running
    result = subprocess.run("ps aux | grep queue_processor.py | grep -v grep", shell=True, capture_output=True, text=True)
    
    if result.returncode \!= 0:
        logger.info("Queue processor not running, starting it...")
        # Start the queue processor
        run_command(""
                    "nohup python3 mathematricks/systems/backtests_queue/queue_processor.py > "
                    "logs/backtest_processor.log 2>&1 &")
        return True
    else:
        logger.debug("Queue processor is already running")
        return False

def generate_strategies():
    """Generate new strategies and add them to the queue"""
    return run_command("cd /mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp && "
                     "python3 strategy_generator.py")

def update_google_sheets():
    """Update Google Sheets with completed backtest results"""
    return run_command("cd /mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp && "
                     "python3 update_sheets_with_results.py")

def check_queue_status():
    """Check the status of the backtest queue"""
    try:
        queue_file = "/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp/mathematricks/systems/backtests_queue/queue.json"
        completed_file = "/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp/mathematricks/systems/backtests_queue/completed_backtests.json"
        
        import json
        with open(queue_file, 'r') as f:
            queue = json.load(f)
        
        with open(completed_file, 'r') as f:
            completed = json.load(f)
        
        pending_count = len([item for item in queue if item.get('status') == 'pending'])
        processing_count = len([item for item in queue if item.get('status') == 'processing'])
        completed_count = len(completed)
        
        logger.info(f"Queue status: {pending_count} pending, {processing_count} processing, {completed_count} completed")
        
        return pending_count, processing_count, completed_count
    except Exception as e:
        logger.error(f"Error checking queue status: {e}")
        return 0, 0, 0

def run_autopilot():
    """Run the autopilot system continuously"""
    logger.info("Starting autopilot backtester")
    
    # Initialize
    ensure_queue_processor_running()
    
    # Initial queue check
    pending, processing, completed = check_queue_status()
    
    # Initial strategy generation if queue is empty
    if pending + processing == 0:
        logger.info("Queue is empty, generating initial strategies")
        generate_strategies()
    
    # Main loop
    while True:
        try:
            # Make sure queue processor is running
            processor_started = ensure_queue_processor_running()
            if processor_started:
                logger.info("Restarted queue processor")
                time.sleep(10)  # Give it time to start
            
            # Check queue status
            pending, processing, completed = check_queue_status()
            
            # If queue is getting low, generate more strategies
            if pending < 2 and random.random() < 0.7:  # 70% chance to generate when queue is low
                logger.info("Queue is getting low, generating more strategies")
                generate_strategies()
            
            # Update Google Sheets with completed results
            if completed > 0:
                logger.info("Updating Google Sheets with completed results")
                update_google_sheets()
            
            # Wait before next cycle - randomize to avoid patterns
            wait_time = random.uniform(300, 600)  # 5-10 minutes
            logger.info(f"Sleeping for {wait_time:.1f} seconds before next cycle")
            time.sleep(wait_time)
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected, exiting")
            break
        except Exception as e:
            logger.error(f"Error in autopilot loop: {e}")
            time.sleep(60)  # Wait a minute before trying again

if __name__ == "__main__":
    run_autopilot()
