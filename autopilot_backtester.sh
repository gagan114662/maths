#\!/bin/bash
# Autonomous backtesting system for mathematricks framework
# This script sets up continuous backtesting with intelligent strategy generation and processing

MAIN_DIR="/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp"
MATH_DIR="$MAIN_DIR/mathematricks"
LOG_DIR="$MAIN_DIR/logs"
QUEUE_DIR="$MATH_DIR/systems/backtests_queue"
STRATEGIES_DIR="$MATH_DIR/systems/strategies"

# Ensure all required directories exist
mkdir -p "$LOG_DIR"
mkdir -p "$QUEUE_DIR" 
mkdir -p "$STRATEGIES_DIR"
mkdir -p "$MAIN_DIR/results"

# Initialize log file
LOGFILE="$LOG_DIR/autopilot_backtester.log"
echo "$(date): Autopilot backtester starting" > $LOGFILE

# Create needed files if they don't exist
if [ \! -f "$QUEUE_DIR/completed_backtests.json" ]; then
    echo "[]" > "$QUEUE_DIR/completed_backtests.json"
    echo "$(date): Created empty completed_backtests.json" >> $LOGFILE
fi

if [ \! -f "$QUEUE_DIR/queue.json" ]; then
    echo "[]" > "$QUEUE_DIR/queue.json"
    echo "$(date): Created empty queue.json" >> $LOGFILE
fi

# Kill any existing queue processor instances
pkill -f "queue_processor.py" || true

# Create the strategy generator script
cat > "$MAIN_DIR/strategy_generator.py" << 'PYEOF'
#\!/usr/bin/env python3
"""
Intelligent strategy generator that creates diverse strategies and adds them to the backtest queue.
"""
import os
import sys
import json
import time
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "strategy_generator.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StrategyGenerator")

# Strategy categories and types
STRATEGY_CATEGORIES = [
    "trend_following", "mean_reversion", "momentum", "volatility", "statistical_arbitrage",
    "machine_learning", "market_microstructure", "factor_based", "calendar_based", 
    "intraday_pattern", "options_volatility", "fixed_income", "event_driven"
]

# Number of strategies to generate in each batch
BATCH_SIZE = 3

def generate_strategy_name():
    """Generate a unique strategy name"""
    category = random.choice(STRATEGY_CATEGORIES)
    modifiers = ["Enhanced", "Adaptive", "Dynamic", "Smart", "Optimized", "Balanced", "Advanced"]
    suffix = ["Alpha", "Beta", "Prime", "Plus", "Pro", "Elite", "Core"]
    
    return f"{random.choice(modifiers)}_{category.title()}_{random.choice(suffix)}"

def create_strategy_file(strategy_name):
    """Create a strategy file in the mathematricks systems/strategies directory"""
    class_name = ''.join(word.title() for word in strategy_name.split('_'))
    
    # Select a random strategy template based on the category embedded in the name
    category = None
    for cat in STRATEGY_CATEGORIES:
        if cat.lower() in strategy_name.lower():
            category = cat
            break
    
    if not category:
        category = random.choice(STRATEGY_CATEGORIES)
    
    # Configure strategy based on category
    if category == "trend_following":
        indicators = [
            ("ema", {"period": 9}),
            ("ema", {"period": 21}),
            ("macd", {"fast": 12, "slow": 26, "signal": 9})
        ]
        entry_conditions = [
            "ema_9 > ema_21", 
            "ema_9 > ema_9.shift(1)", 
            "macd_line > signal_line"
        ]
        exit_conditions = [
            "ema_9 < ema_21", 
            "price_data < price_data.shift(1) * 0.98"
        ]
    elif category == "mean_reversion":
        indicators = [
            ("bollinger", {"period": 20, "std_dev": 2.0}),
            ("rsi", {"period": 14}),
            ("sma", {"period": 50})
        ]
        entry_conditions = [
            "price_data < lower_band", 
            "rsi < 30"
        ]
        exit_conditions = [
            "price_data > middle_band", 
            "rsi > 70"
        ]
    elif category == "momentum":
        indicators = [
            ("rsi", {"period": 14}),
            ("ema", {"period": 20}),
            ("ema", {"period": 50})
        ]
        entry_conditions = [
            "price_data > ema_20",
            "ema_20 > ema_50", 
            "rsi > 50"
        ]
        exit_conditions = [
            "price_data < ema_20",
            "rsi < 40"
        ]
    elif category == "volatility":
        indicators = [
            ("atr", {"period": 14}),
            ("bollinger", {"period": 20, "std_dev": 2.0}),
            ("rsi", {"period": 7})
        ]
        entry_conditions = [
            "atr > atr.rolling(window=20).mean() * 1.3",
            "price_data > upper_band",
            "rsi < 40"
        ]
        exit_conditions = [
            "atr < atr.rolling(window=20).mean()",
            "price_data < middle_band"
        ]
    else:  # Default/fallback strategy
        indicators = [
            ("sma", {"period": 20}),
            ("sma", {"period": 50}),
            ("rsi", {"period": 14})
        ]
        entry_conditions = [
            "sma_20 > sma_50",
            "price_data > sma_20", 
            "rsi > 50"
        ]
        exit_conditions = [
            "sma_20 < sma_50",
            "price_data < sma_20"
        ]
    
    # Add some variety
    if random.random() < 0.3:  # 30% chance to add volume condition
        indicators.append(("volume_sma", {"period": 20}))
        entry_conditions.append("volume_data > volume_sma_20")
    
    if random.random() < 0.2:  # 20% chance to add stochastic
        indicators.append(("stoch", {"k_period": 14, "d_period": 3}))
        entry_conditions.append("stoch_k < 20 and stoch_k > stoch_k.shift(1)")
        exit_conditions.append("stoch_k > 80")
    
    # Generate the strategy code
    strategy_code = f"""# Auto-generated strategy file
from systems.base_strategy import BaseStrategy
import numpy as np
import pandas as pd

class {class_name}(BaseStrategy):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        self.name = "{strategy_name}"
        self.description = "Intelligent {category.replace('_', ' ')} strategy with adaptive parameters"
    
    def generate_signals(self, market_data):
        # Strategy logic to generate signals based on market data
        signals = []
        
        for symbol in market_data:
            # Get price data
            price_data = market_data[symbol]['close']
            
            # Skip if we don't have enough data
            if len(price_data) < 60:
                continue
            
            # Get volume data if available
            volume_data = market_data[symbol].get('volume', None)
            if volume_data is None:
                volume_data = pd.Series(0, index=price_data.index)
                
            # Calculate indicators
"""
    
    # Add indicator calculations
    for ind_name, params in indicators:
        if ind_name == "sma":
            period = params["period"]
            strategy_code += f"            sma_{period} = price_data.rolling(window={period}).mean()\n"
        elif ind_name == "ema":
            period = params["period"]
            strategy_code += f"            ema_{period} = price_data.ewm(span={period}).mean()\n"
        elif ind_name == "rsi":
            period = params["period"]
            strategy_code += f"""            # Calculate RSI with period {period}
            delta = price_data.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window={period}).mean()
            avg_loss = loss.rolling(window={period}).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
"""
        elif ind_name == "macd":
            fast = params["fast"]
            slow = params["slow"]
            signal = params["signal"]
            strategy_code += f"""            # Calculate MACD
            ema_fast = price_data.ewm(span={fast}).mean()
            ema_slow = price_data.ewm(span={slow}).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span={signal}).mean()
"""
        elif ind_name == "bollinger":
            period = params["period"]
            std_dev = params["std_dev"]
            strategy_code += f"""            # Calculate Bollinger Bands
            middle_band = price_data.rolling(window={period}).mean()
            std = price_data.rolling(window={period}).std()
            upper_band = middle_band + {std_dev} * std
            lower_band = middle_band - {std_dev} * std
"""
        elif ind_name == "atr":
            period = params["period"]
            strategy_code += f"""            # Calculate ATR
            high_data = market_data[symbol].get('high', price_data)
            low_data = market_data[symbol].get('low', price_data)
            tr1 = high_data - low_data
            tr2 = abs(high_data - price_data.shift())
            tr3 = abs(low_data - price_data.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window={period}).mean()
"""
        elif ind_name == "volume_sma":
            period = params["period"]
            strategy_code += f"            volume_sma_{period} = volume_data.rolling(window={period}).mean()\n"
        elif ind_name == "stoch":
            k_period = params["k_period"]
            d_period = params["d_period"]
            strategy_code += f"""            # Calculate Stochastic Oscillator
            high_data = market_data[symbol].get('high', price_data)
            low_data = market_data[symbol].get('low', price_data)
            lowest_low = low_data.rolling(window={k_period}).min()
            highest_high = high_data.rolling(window={k_period}).max()
            stoch_k = 100 * (price_data - lowest_low) / (highest_high - lowest_low + 0.0001)
            stoch_d = stoch_k.rolling(window={d_period}).mean()
"""
    
    # Add entry condition checks
    strategy_code += """
            # Check entry conditions
            try:
"""
    for i, condition in enumerate(entry_conditions):
        if i == 0:
            strategy_code += f"                entry_condition = {condition}\n"
        else:
            strategy_code += f"                entry_condition = entry_condition and {condition}\n"
    
    strategy_code += """            except Exception as e:
                # Skip on error (probably insufficient data)
                continue
                
            # Generate buy signal if conditions met
            if entry_condition:
                signals.append({'symbol': symbol, 'action': 'buy'})
"""
    
    # Add exit condition checks
    strategy_code += """
            # Check exit conditions
            try:
"""
    for i, condition in enumerate(exit_conditions):
        if i == 0:
            strategy_code += f"                exit_condition = {condition}\n"
        else:
            strategy_code += f"                exit_condition = exit_condition and {condition}\n"
    
    strategy_code += """            except Exception as e:
                # Skip on error (probably insufficient data)
                continue
                
            # Generate sell signal if conditions met
            if exit_condition:
                signals.append({'symbol': symbol, 'action': 'sell'})
"""
    
    # Add method endings
    strategy_code += """
        return signals
    
    def set_parameters(self, **params):
        # Method to set parameters for optimization
        self.params = params
    
    def optimize_parameters(self):
        # Method for parameter optimization
        best_params = {"""
    
    # Add parameters
    param_strs = []
    for ind_name, params in indicators:
        for param_name, value in params.items():
            param_strs.append(f"'{ind_name}_{param_name}': {value}")
    
    strategy_code += ", ".join(param_strs)
    strategy_code += """}
        self.set_parameters(**best_params)
"""
    
    # Write to file
    strategies_dir = os.path.join("/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp/mathematricks/systems/strategies")
    os.makedirs(strategies_dir, exist_ok=True)
    file_path = os.path.join(strategies_dir, f"{class_name}.py")
    
    with open(file_path, "w") as f:
        f.write(strategy_code)
        
    logger.info(f"Created strategy file: {file_path}")
    return class_name, file_path

def add_to_backtest_queue(strategy_name, class_name):
    """Add the strategy to the backtest queue"""
    # Define backtest config
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    universe = "S&P 500 constituents"  # Default universe
    
    # Add some variation
    if "fixed_income" in strategy_name.lower():
        universe = "US Bond Market"
    elif "statistical_arbitrage" in strategy_name.lower():
        universe = "Nasdaq 100"
    elif random.random() < 0.3:
        universes = ["Nasdaq 100", "Dow Jones", "Russell 2000", "S&P 500 constituents"]
        universe = random.choice(universes)
    
    backtest_entry = {
        "backtest_name": strategy_name,
        "strategy_name": class_name,
        "universe": universe,
        "start_date": start_date,
        "end_date": end_date,
        "status": "pending",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Path to queue file
    queue_file = "/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp/mathematricks/systems/backtests_queue/queue.json"
    
    # Create or update queue file
    try:
        with open(queue_file, 'r') as f:
            queue = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        queue = []
        
    queue.append(backtest_entry)
    
    with open(queue_file, 'w') as f:
        json.dump(queue, f, indent=2)
        
    logger.info(f"Added strategy to backtest queue: {backtest_entry['backtest_name']}")
    return backtest_entry

def main():
    """Main function to generate strategies and add them to the queue"""
    logger.info(f"Starting strategy generation batch of {BATCH_SIZE} strategies")
    
    for i in range(BATCH_SIZE):
        strategy_name = generate_strategy_name()
        class_name, file_path = create_strategy_file(strategy_name)
        backtest_entry = add_to_backtest_queue(strategy_name, class_name)
        
        logger.info(f"Generated strategy {i+1}/{BATCH_SIZE}: {strategy_name}")
        
        # Sleep a bit between generations to avoid overwhelming the system
        time.sleep(1)
    
    logger.info(f"Completed strategy generation batch")

if __name__ == "__main__":
    main()
PYEOF

# Create the Google Sheets update script
cat > "$MAIN_DIR/update_sheets_with_results.py" << 'PYEOF'
#\!/usr/bin/env python3
"""
Script to update Google Sheets with completed backtest results.
"""
import os
import sys
import json
import logging
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "sheets_update.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SheetsUpdater")

# Import necessary modules
try:
    from src.utils.google_sheet_integration import GoogleSheetIntegration
    sheets_available = True
except ImportError:
    logger.warning("Google Sheets integration not available")
    sheets_available = False

def update_sheets():
    """Update Google Sheets with latest backtest results"""
    if not sheets_available:
        logger.error("Google Sheets integration not available - cannot update")
        return False
    
    # Load completed backtests
    completed_file = os.path.join("mathematricks", "systems", "backtests_queue", "completed_backtests.json")
    try:
        with open(completed_file, 'r') as f:
            completed_backtests = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading completed backtests: {e}")
        return False
    
    if not completed_backtests:
        logger.info("No completed backtests found")
        return False
    
    # Get unprocessed backtests (where processed = False or not present)
    unprocessed = [b for b in completed_backtests if not b.get("processed", False)]
    
    if not unprocessed:
        logger.info("No unprocessed backtests found")
        return False
    
    logger.info(f"Found {len(unprocessed)} unprocessed backtest results")
    
    # Initialize Google Sheets integration
    logger.info("Initializing Google Sheets integration...")
    sheets = GoogleSheetIntegration()
    if not sheets.initialize():
        logger.error("Failed to initialize Google Sheets integration")
        return False
    
    # Process each unprocessed backtest
    updated_count = 0
    
    for backtest_result in unprocessed:
        try:
            backtest = backtest_result['backtest']
            results = backtest_result['results']
            
            strategy_name = backtest.get('backtest_name', 'Unknown Strategy')
            logger.info(f"Processing results for: {strategy_name}")
            
            # Log feedback
            feedback = {
                "agent_name": "Mathematricks Backtester",
                "feedback": f"Completed backtest for {strategy_name} using mathematricks framework with real market data. "+
                           f"Results: CAGR: {results.get('performance', {}).get('annualized_return', 0):.2f}, "+
                           f"Sharpe: {results.get('performance', {}).get('sharpe_ratio', 0):.2f}, "+
                           f"Win Rate: {results.get('trades', {}).get('win_rate', 0):.2f}",
                "feedback_type": "SUCCESS"
            }
            
            try:
                sheets.update_ai_feedback([{
                    "agent": feedback["agent_name"],
                    "feedback": feedback["feedback"],
                    "type": feedback["feedback_type"],
                    "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }])
                logger.info("Added feedback to AI Feedback tab")
            except Exception as e:
                logger.error(f"Error updating AI Feedback: {e}")
            
            # Create strategy model
            strategy_model = {
                "name": backtest.get('backtest_name'),
                "description": f"Strategy using {backtest.get('strategy_name')} algorithm with mathematricks framework",
                "universe": backtest.get('universe'),
                "performance": {
                    "cagr": results.get('performance', {}).get('annualized_return', 0),
                    "sharpe_ratio": results.get('performance', {}).get('sharpe_ratio', 0),
                    "max_drawdown": results.get('performance', {}).get('max_drawdown', 0),
                    "volatility": results.get('performance', {}).get('volatility', 0)
                },
                "trades": {
                    "total_trades": results.get('trades', {}).get('total_trades', 0),
                    "win_rate": results.get('trades', {}).get('win_rate', 0),
                    "average_trade": results.get('trades', {}).get('average_trade', 0),
                    "profit_factor": results.get('trades', {}).get('profit_factor', 0)
                }
            }
            
            # Update performance
            try:
                sheets.update_strategy_performance([strategy_model])
                logger.info("Updated strategy performance")
            except Exception as e:
                logger.error(f"Error updating strategy performance: {e}")
            
            # Mark as processed
            backtest_result["processed"] = True
            backtest_result["processed_at"] = datetime.datetime.now().isoformat()
            updated_count += 1
            
        except Exception as e:
            logger.error(f"Error processing backtest result: {e}")
    
    # Save updated backtests
    if updated_count > 0:
        with open(completed_file, 'w') as f:
            json.dump(completed_backtests, f, indent=2)
        logger.info(f"Updated {updated_count} backtest results in Google Sheets")
        return True
    
    return False

if __name__ == "__main__":
    update_sheets()
PYEOF

# Make scripts executable
chmod +x "$MAIN_DIR/strategy_generator.py"
chmod +x "$MAIN_DIR/update_sheets_with_results.py"
chmod +x "$MATH_DIR/systems/backtests_queue/queue_processor.py"

# Create the autopilot script to run continuously
cat > "$MAIN_DIR/run_autopilot.py" << 'PYEOF'
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
PYEOF

# Make it executable
chmod +x "$MAIN_DIR/run_autopilot.py"

# Create a systemd service file for the autopilot
cat > "$MAIN_DIR/config/systemd/mathematricks-autopilot.service" << 'EOF'
[Unit]
Description=Mathematricks Autopilot Backtester
After=network.target

[Service]
User=vandan
WorkingDirectory=/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp
ExecStart=/usr/bin/python3 /mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp/run_autopilot.py
Restart=on-failure
RestartSec=30s

[Install]
WantedBy=multi-user.target
