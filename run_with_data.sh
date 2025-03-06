#!/bin/bash
# Custom script to run with specific data path

# Display GOD MODE banner
echo "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️"
echo "⚡️                                          ⚡️"
echo "⚡️       CUSTOM DATA PATH GOD MODE         ⚡️"
echo "⚡️                                          ⚡️"
echo "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Ensure the Python environment is active
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "fintsb_env" ]; then
    source fintsb_env/bin/activate
fi

# Get timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="logs/custom_run_${timestamp}.log"

# Set data path environment variable
export MATHEMATRICKS_DATA_PATH="/mnt/VANDAN_DISK/gagan_stuff/data"
echo "Using data path: $MATHEMATRICKS_DATA_PATH"

# Run with specific symbols and limited scope
python run_agent_system.py \
  --god-mode \
  --use-simple-memory \
  --llm-provider ollama \
  --llm-model deepseek-r1 \
  --market us_equities \
  --plan-name "Tech Stock Strategy" \
  --goal "Develop a momentum strategy for AAPL with a target CAGR of 25%" \
  --time-horizon daily \
  --min-sharpe 1.2 \
  --max-drawdown 0.2 \
  --min-win-rate 0.55 \
  "$@" 2>&1 | tee -a "$logfile"

echo ""
echo "⚡️ Custom run completed. Log saved to: $logfile ⚡️"