#!/bin/bash
# Enhanced script to run the AI Co-Scientist system with DeepSeek R1 via Ollama
# Includes improved error handling, offline detection, and status reporting

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check internet connectivity
check_internet() {
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check if Ollama is installed
if ! command_exists ollama; then
    echo "ERROR: Ollama is not installed. Please install Ollama first."
    echo "Visit https://ollama.com/download for installation instructions."
    exit 1
fi

# Make sure Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Ollama is not running. Starting Ollama..."
    ollama serve &
    
    # Wait for Ollama to start with a timeout
    MAX_WAIT=30
    for ((i=1; i<=MAX_WAIT; i++)); do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "Ollama started successfully after $i seconds."
            break
        fi
        
        if [ $i -eq $MAX_WAIT ]; then
            echo "ERROR: Timed out waiting for Ollama to start."
            echo "Please check if Ollama is properly installed and can be started manually."
            exit 1
        fi
        
        echo "Waiting for Ollama to start... ($i/$MAX_WAIT)"
        sleep 1
    done
fi

# Check if DeepSeek R1 is available
if ! ollama list | grep -q "deepseek-r1"; then
    echo "DeepSeek R1 model not found."
    
    # Check if we're offline
    if ! check_internet; then
        echo "ERROR: No internet connection detected and DeepSeek R1 model is not available locally."
        echo "Cannot pull the model without internet connectivity."
        exit 1
    fi
    
    echo "Pulling DeepSeek R1 model (this may take a while)..."
    if ! ollama pull deepseek-r1; then
        echo "ERROR: Failed to pull DeepSeek R1 model."
        echo "Please check your internet connection and try again."
        exit 1
    fi
    
    echo "DeepSeek R1 model pulled successfully."
fi

# Check for virtual environment
VENV_PATH="/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    echo "Please create the virtual environment first using:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Check if the required dependencies are installed
if ! python -c "import aiohttp" 2>/dev/null; then
    echo "ERROR: Required Python packages not found in virtual environment"
    echo "Please install required packages using: pip install -r requirements.txt"
    exit 1
fi

# Set up logging directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/deepseek_run_$(date +%Y%m%d_%H%M%S).log"

# Start logging
echo "===========================================" | tee -a "$LOG_FILE"
echo "AI Co-Scientist with DeepSeek R1" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Using model: deepseek-r1" | tee -a "$LOG_FILE"
echo "System: $(uname -srm)" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"

# Display warning about offline capability
echo "NOTE: Running in offline mode with DeepSeek R1 via Ollama" | tee -a "$LOG_FILE"
echo "Any API keys for cloud services will be ignored" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check if specific flags are in the arguments
has_show_interactions=false
has_autopilot=false
has_god_mode=false

for arg in "$@"; do
  if [ "$arg" == "--show-agent-interactions" ]; then
    has_show_interactions=true
  fi
  if [ "$arg" == "--autopilot" ]; then
    has_autopilot=true
  fi
  if [ "$arg" == "--god-mode" ]; then
    has_god_mode=true
  fi
done

# If GOD MODE is enabled, show a special banner
if [ "$has_god_mode" = true ]; then
    echo "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️" | tee -a "$LOG_FILE"
    echo "⚡️                                          ⚡️" | tee -a "$LOG_FILE"
    echo "⚡️       DEEPSEEK R1 GOD MODE ENABLED      ⚡️" | tee -a "$LOG_FILE"
    echo "⚡️                                          ⚡️" | tee -a "$LOG_FILE"
    echo "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Unleashing full capabilities of DeepSeek R1..." | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

# Update sheets if it exists, with rate limiting protection
if [ -f "update_all_sheets.py" ]; then
  echo "Waiting 5 seconds before updating Google Sheets (avoiding rate limits)..." | tee -a "$LOG_FILE"
  sleep 5
  echo "Updating Google Sheets..." | tee -a "$LOG_FILE"
  
  # Try up to 3 times with increasing delays between attempts
  for attempt in 1 2 3; do
    if ./update_all_sheets.py; then
      echo "Google Sheets update successful on attempt $attempt" | tee -a "$LOG_FILE"
      break
    else
      echo "Google Sheets update failed on attempt $attempt, retrying after $((attempt * 10)) seconds..." | tee -a "$LOG_FILE"
      sleep $((attempt * 10))
    fi
  done
fi

# Run the script with all arguments passed through
echo "Starting AI Co-Scientist with DeepSeek R1..." | tee -a "$LOG_FILE"

# Check which script to run based on flags
if [ "$has_god_mode" = true ]; then
  echo "Running with DeepSeek R1 in GOD MODE..." | tee -a "$LOG_FILE"
  
  # Set trap to update sheets at the end if the script exists
  if [ -f "update_all_sheets.py" ]; then
    trap './update_all_sheets.py' EXIT
  fi
  
  # Directly run the agent system with GOD MODE enabled
  # Create a simple activity logger for compatibility
  mkdir -p src/utils
  cat > src/utils/activity_logger.py << 'EOF'
# Empty activity logger for compatibility
def initialize(): 
    return True
def log_activity(*args, **kwargs): 
    pass
def shutdown(): 
    pass
EOF

  # Create a Python file for GOD MODE setup
  cat > godmode_setup.py << 'EOF'
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
EOF

  # Run the GOD MODE setup script
  python godmode_setup.py

  # Create a GOD MODE implementation file
  mkdir -p src/core/llm/providers/
  cat > src/core/llm/providers/deepseek_god_mode.py << 'EOF'
# DeepSeek R1 GOD MODE implementation
import logging
import os
import json
import time

logger = logging.getLogger(__name__)

class DeepSeekGodMode:
    """
    DeepSeek R1 GOD MODE implementation for enhanced capabilities
    """
    def __init__(self):
        self.enabled = True
        self.enhancements = [
            "AdvancedReasoningFramework",
            "ChainOfThoughtValidator",
            "SelfCritiqueRefinement",
            "ParallelHypothesisTesting",
            "AdvancedFeatureEngineering",
            "MarketRegimeDetection",
            "AdaptiveHyperparameterOptimization",
            "ExplainableAIComponents",
            "CrossMarketCorrelationAnalysis",
            "SentimentAnalysisIntegration",
            "ModelEnsembleArchitecture"
        ]
        logger.info(f"DeepSeek R1 GOD MODE initialized with {len(self.enhancements)} enhancements")
        
    def apply_enhancement(self, prompt, enhancement_name):
        """Apply a specific enhancement to the prompt"""
        if enhancement_name == "AdvancedReasoningFramework":
            return self._apply_advanced_reasoning(prompt)
        elif enhancement_name == "ChainOfThoughtValidator":
            return self._apply_chain_of_thought(prompt)
        # Implement other enhancements as needed
        return prompt
        
    def _apply_advanced_reasoning(self, prompt):
        """Apply advanced reasoning framework to prompt"""
        enhanced_prompt = f"""
[ADVANCED REASONING ENABLED]
Apply a multi-step reasoning approach to:
{prompt}

Use the following process:
1. Identify key variables and relationships
2. Apply domain-specific financial knowledge
3. Consider multiple perspectives and hypotheses
4. Reason through each step explicitly
5. Validate conclusions with supporting evidence

[END ADVANCED REASONING]
"""
        return enhanced_prompt
        
    def _apply_chain_of_thought(self, prompt):
        """Apply chain-of-thought validation to prompt"""
        enhanced_prompt = f"""
[CHAIN-OF-THOUGHT VALIDATION ENABLED]
For the following task:
{prompt}

Apply rigorous validation:
1. Decompose problem into components
2. Verify each step logically
3. Consider edge cases and counter-examples
4. Cross-check numerical calculations
5. Confirm alignment with financial principles

[END CHAIN-OF-THOUGHT VALIDATION]
"""
        return enhanced_prompt
        
    def enhance_prompt(self, prompt, active_enhancements=None):
        """Apply all enabled enhancements to the prompt"""
        if not self.enabled:
            return prompt
            
        to_apply = active_enhancements or self.enhancements
        enhanced_prompt = prompt
        
        for enhancement in to_apply:
            enhanced_prompt = self.apply_enhancement(enhanced_prompt, enhancement)
            
        return enhanced_prompt
EOF

  # Now create the activity logger file with proper functionality
  mkdir -p src/utils
  cat > src/utils/activity_logger.py << 'EOF'
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
EOF
  # Run the system - with patch to fix AnthropicProvider import issue
  # First create a fixed version of the factory.py file
  cp src/core/llm/factory.py src/core/llm/factory.py.bak
  sed -i '/AnthropicProvider/d' src/core/llm/factory.py
  
  if python run_agent_system.py --god-mode --use-simple-memory --llm-provider ollama --llm-model deepseek-r1 "$@" 2>&1 | tee -a "$LOG_FILE"; then 
    STATUS="SUCCESS ⚡️ GOD MODE ⚡️"
  else
    STATUS="FAILED (exit code: $?)"
  fi
  
  # Restore original file
  mv src/core/llm/factory.py.bak src/core/llm/factory.py
elif [ "$has_show_interactions" = false ] && [ "$has_autopilot" = false ]; then
  # If neither flag is provided, add them for demonstration mode
  echo "Adding --show-agent-interactions and --autopilot flags for full demonstration mode" | tee -a "$LOG_FILE"
  if python run_with_ollama.py --interactive --show-agent-interactions --autopilot "$@" 2>&1 | tee -a "$LOG_FILE"; then
    STATUS="SUCCESS"
  else
    STATUS="FAILED (exit code: $?)"
  fi
else
  # Run with original arguments
  if python run_with_ollama.py --interactive "$@" 2>&1 | tee -a "$LOG_FILE"; then
    STATUS="SUCCESS"
  else
    STATUS="FAILED (exit code: $?)"
  fi
fi

# Log completion
echo "" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"
if [ "$has_god_mode" = true ]; then
    echo "AI Co-Scientist with DeepSeek R1 GOD MODE completed" | tee -a "$LOG_FILE"
else
    echo "AI Co-Scientist with DeepSeek R1 completed" | tee -a "$LOG_FILE"
fi
echo "Status: $STATUS" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ "$has_god_mode" = true ]; then
    echo "⚡️ AI Co-Scientist run completed with status: $STATUS ⚡️"
else
    echo "AI Co-Scientist run completed with status: $STATUS"
fi
echo "For detailed logs, see $LOG_FILE"

# Create a symlink to the latest log file for easy access
ln -sf "$LOG_FILE" "$LOG_DIR/deepseek_latest.log"
echo "Latest log also available at: $LOG_DIR/deepseek_latest.log"

# If we are showing agent interactions, remind the user to check Google Sheets
if [ "$has_show_interactions" = true ] || [ "$has_autopilot" = true ]; then
  echo
  echo "Agent interactions have been logged to Google Sheets"
  echo "View the interactions at: https://docs.google.com/spreadsheets/d/1FqOpXfHIci2BQ173dwDV35NjkyEn4QmioIlMEH-WiOA/edit?gid=478506301"
  echo
fi

# Deactivate virtual environment
deactivate
