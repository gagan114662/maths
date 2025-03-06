#!/bin/bash
# Run the AI Co-Scientist system with Ollama and DeepSeek R1

# Make sure Ollama is running
if ! pgrep -x "ollama" > /dev/null
then
    echo "Ollama is not running. Starting Ollama..."
    ollama serve &
    sleep 5  # Wait for Ollama to start
fi

# Check if DeepSeek R1 is available
if ! ollama list | grep -q "deepseek-r1"
then
    echo "DeepSeek R1 model not found. Pulling model..."
    ollama pull deepseek-r1
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source /mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp/venv/bin/activate

# Set log file
LOG_FILE="ollama_run.log"
echo "Starting run with DeepSeek R1 at $(date)" | tee -a "$LOG_FILE"

# Run the system with Ollama and DeepSeek R1
echo "Running the full production system with DeepSeek R1..."
python run_with_ollama.py --interactive "$@" | tee -a "$LOG_FILE"

echo "Run completed at $(date)" | tee -a "$LOG_FILE"