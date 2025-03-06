#!/bin/bash
# Run the AI Co-Scientist with DeepSeek R1 in GOD MODE

# Display GOD MODE banner
echo "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️"
echo "⚡️                                          ⚡️"
echo "⚡️       DEEPSEEK R1 GOD MODE ENABLED      ⚡️"
echo "⚡️                                          ⚡️"
echo "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️"
echo ""
echo "Unleashing full capabilities of DeepSeek R1..."
echo ""

# Make sure the script is executable
chmod +x run_agent_system.py

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
logfile="logs/deepseek_run_${timestamp}.log"

# Create output directory
output_dir="output_${timestamp}"
mkdir -p $output_dir

# Run directly with GOD MODE and log to file
python run_agent_system.py --god-mode --use-simple-memory --llm-provider ollama --llm-model deepseek-r1 "$@" 2>&1 | tee -a "$logfile"

# Create a symlink to the latest log
ln -sf "$logfile" logs/deepseek_latest.log

echo ""
echo "⚡️ GOD MODE session completed. Log saved to: $logfile ⚡️"