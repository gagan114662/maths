#\!/bin/bash
# Script to run the backtest queue processor as a dedicated service

# Set up directories
MAIN_DIR="/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp"
MATH_DIR="$MAIN_DIR/mathematricks"
LOG_DIR="$MAIN_DIR/logs"
QUEUE_DIR="$MATH_DIR/systems/backtests_queue"

# Make sure log directory exists
mkdir -p "$LOG_DIR"

# Create an empty completed_backtests.json if it doesn't exist
if [ \! -f "$QUEUE_DIR/completed_backtests.json" ]; then
    echo "[]" > "$QUEUE_DIR/completed_backtests.json"
fi

# Run the processor
echo "Starting backtest queue processor..."
cd "$MATH_DIR" && python3 systems/backtests_queue/queue_processor.py > "$LOG_DIR/backtest_processor.log" 2>&1 &

# Get the process ID
PID=$\!
echo "Backtest processor started with PID: $PID"
echo $PID > "$MAIN_DIR/backtest_processor.pid"

echo "Backtest processor is now running in the background."
echo "Check logs at: $LOG_DIR/backtest_processor.log"
