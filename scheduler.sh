#\!/bin/bash
# Scheduler script to regularly run backtest processing and Google Sheets updates

# Set up environment
MAIN_DIR="/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp"
LOG_DIR="$MAIN_DIR/logs"
QUEUE_DIR="$MAIN_DIR/mathematricks/systems/backtests_queue"

# Create directories if needed
mkdir -p "$LOG_DIR"
mkdir -p "$QUEUE_DIR"

# Function to generate random strategy and add to queue
generate_strategy() {
  python3 -c "
import os
import sys
import json
import random
import datetime
from pathlib import Path

# Strategy types
STRATEGY_TYPES = [
    'Momentum', 'Mean Reversion', 'Trend Following', 'Statistical Arbitrage',
    'Volatility', 'Breakout', 'RSI', 'MACD', 'Dual Moving Average', 'Triple Indicator',
    'Adaptive', 'Dynamic', 'Smart', 'Intelligent', 'Optimized'
]

# Generate unique name
strategy_type1 = random.choice(STRATEGY_TYPES)
strategy_type2 = random.choice(STRATEGY_TYPES)
modifiers = ['Enhanced', 'Advanced', 'Balanced', 'Premium', 'Core', 'Alpha', 'Prime']
strategy_name = f'{random.choice(modifiers)} {strategy_type1} {strategy_type2}'

# Create entry
entry = {
    'backtest_name': strategy_name,
    'strategy_name': 'MomentumRSIStrategy',  # Using existing strategy class
    'universe': 'S&P 500 constituents',
    'start_date': '2024-03-05',
    'end_date': '2025-03-05',
    'status': 'pending',
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

# Load queue
queue_file = '$QUEUE_DIR/queue.json'
with open(queue_file, 'r') as f:
    queue = json.load(f)

# Add entry
queue.append(entry)

# Save queue
with open(queue_file, 'w') as f:
    json.dump(queue, f, indent=2)

print(f'Added {strategy_name} to queue')
"
}

# Function to process queue
process_queue() {
  echo "Processing backtest queue..."
  cd "$MAIN_DIR" && python3 mathematricks/systems/backtests_queue/queue_processor.py --once
}

# Function to update Google Sheets
update_sheets() {
  echo "Updating Google Sheets..."
  cd "$MAIN_DIR" && python3 update_google_sheets_with_backtest.py
  cd "$MAIN_DIR" && python3 create_trading_results_updater.py
}

# Main loop
echo "Starting intelligent scheduler at $(date)"
echo "This will automatically generate strategies, process backtests, and update results."

while true; do
  # Check if queue is empty
  QUEUE_COUNT=$(jq length "$QUEUE_DIR/queue.json")
  
  if [ "$QUEUE_COUNT" -lt 2 ]; then
    echo "Queue has fewer than 2 entries, generating random strategy..."
    generate_strategy
  fi
  
  # Process queue
  process_queue
  
  # Wait before updating sheets to ensure processing completes
  sleep 5
  
  # Update sheets
  update_sheets
  
  # Wait random time between 2-5 minutes before next cycle
  WAIT_TIME=$((120 + RANDOM % 180))
  echo "Waiting $WAIT_TIME seconds before next cycle..."
  sleep $WAIT_TIME
done
