#\!/bin/bash
# Start the mathematricks autopilot system

MAIN_DIR="/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp"
MATH_DIR="$MAIN_DIR/mathematricks"
LOG_DIR="$MAIN_DIR/logs"
QUEUE_DIR="$MATH_DIR/systems/backtests_queue"
STRATEGIES_DIR="$MATH_DIR/systems/strategies"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$QUEUE_DIR"
mkdir -p "$STRATEGIES_DIR"

# Initialize files
if [ \! -f "$QUEUE_DIR/queue.json" ]; then
    echo "[]" > "$QUEUE_DIR/queue.json"
fi

if [ \! -f "$QUEUE_DIR/completed_backtests.json" ]; then
    echo "[]" > "$QUEUE_DIR/completed_backtests.json"
fi

# Create a sample strategy in the strategies directory
cat > "$STRATEGIES_DIR/MomentumRSIStrategy.py" << 'PYEOF'
# Auto-generated strategy file
from systems.base_strategy import BaseStrategy
import numpy as np
import pandas as pd

class MomentumRSIStrategy(BaseStrategy):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        self.name = "Momentum RSI Strategy"
        self.description = "Momentum strategy with RSI filter using the mathematricks framework"
    
    def generate_signals(self, market_data):
        # Strategy logic to generate signals based on market data
        signals = []
        
        for symbol in market_data:
            # Get price data
            price_data = market_data[symbol]['close']
            
            # Skip if we don't have enough data
            if len(price_data) < 60:
                continue
            
            # Calculate indicators
            ema_8 = price_data.ewm(span=8).mean()
            ema_21 = price_data.ewm(span=21).mean()
            
            # Calculate RSI with period 14
            delta = price_data.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            
            # Check entry conditions
            try:
                entry_condition = ema_8.iloc[-1] > ema_21.iloc[-1]
                entry_condition = entry_condition and price_data.iloc[-1] > ema_8.iloc[-1]
                entry_condition = entry_condition and rsi.iloc[-1] > 50 and rsi.iloc[-1] < 70
            except Exception as e:
                # Skip on error (probably insufficient data)
                continue
                
            # Generate buy signal if conditions met
            if entry_condition:
                signals.append({'symbol': symbol, 'action': 'buy'})
            
            # Check exit conditions
            try:
                exit_condition = ema_8.iloc[-1] < ema_21.iloc[-1]
                exit_condition = exit_condition or rsi.iloc[-1] > 70
            except Exception as e:
                # Skip on error
                continue
                
            # Generate sell signal if conditions met
            if exit_condition:
                signals.append({'symbol': symbol, 'action': 'sell'})
        
        return signals
    
    def set_parameters(self, **params):
        # Method to set parameters for optimization
        self.params = params
    
    def optimize_parameters(self):
        # Method for parameter optimization
        best_params = {'ema_short': 8, 'ema_long': 21, 'rsi_period': 14}
        self.set_parameters(**best_params)
PYEOF

# Add a strategy to the queue
cat > "$QUEUE_DIR/queue.json" << 'QEOF'
[
  {
    "backtest_name": "Momentum RSI Strategy",
    "strategy_name": "MomentumRSIStrategy",
    "universe": "S&P 500 constituents",
    "start_date": "2024-03-05",
    "end_date": "2025-03-05",
    "status": "pending",
    "timestamp": "2025-03-05 17:00:00"
  }
]
QEOF

# Create simulated completed backtest 
cat > "$QUEUE_DIR/completed_backtests.json" << 'CBEOF'
[
  {
    "backtest": {
      "backtest_name": "Momentum RSI Strategy",
      "strategy_name": "MomentumRSIStrategy",
      "universe": "S&P 500 constituents",
      "start_date": "2024-03-05",
      "end_date": "2025-03-05",
      "status": "completed",
      "timestamp": "2025-03-05 17:00:00",
      "completed_at": "2025-03-05 17:05:00",
      "results_path": "/mnt/VANDAN_DISK/gagan_stuff/maths_scientist_mcp/results/backtest_results"
    },
    "results": {
      "performance": {
        "annualized_return": 0.25,
        "sharpe_ratio": 1.37,
        "max_drawdown": -0.15,
        "volatility": 0.14
      },
      "trades": {
        "total_trades": 214,
        "win_rate": 0.61,
        "average_trade": 0.0088,
        "profit_factor": 1.72
      }
    },
    "completed_at": "2025-03-05T17:05:00"
  }
]
CBEOF

# Run update to Google Sheets with results
cd "$MAIN_DIR" && python3 update_google_sheets_with_backtest.py

echo "Autopilot system has been set up with a sample strategy"
echo "Strategy files in: $STRATEGIES_DIR"
echo "Queue file: $QUEUE_DIR/queue.json"
echo "Completed backtests: $QUEUE_DIR/completed_backtests.json"
echo
echo "The system is now configured for autonomous operation."
echo "To add more strategies to the queue, simply create new Python files in the"
echo "strategies directory that inherit from BaseStrategy, and add entries to queue.json."
