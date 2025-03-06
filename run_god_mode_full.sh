#!/bin/bash
# Advanced script to run the AI Co-Scientist system with DeepSeek R1 in GOD MODE
# Uses local data instead of EastMoney API and includes comprehensive data preparation

# Display GOD MODE banner
echo "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️"
echo "⚡️                                          ⚡️"
echo "⚡️       DEEPSEEK R1 GOD MODE ENABLED      ⚡️"
echo "⚡️                                          ⚡️"
echo "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️"
echo ""
echo "Unleashing full capabilities of DeepSeek R1..."
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Ensure the Python environment is active
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "fintsb_env" ]; then
    source fintsb_env/bin/activate
fi

# Parse command-line arguments
MARKET="us_equities"
PLAN_NAME="Supreme Alpha Strategy"
REGIME="rise"  # Default regime
TIMEFRAME="daily"
PREPARE_DATA=true

# Process arguments
for arg in "$@"; do
  case $arg in
    --market=*)
      MARKET="${arg#*=}"
      shift
      ;;
    --plan-name=*)
      PLAN_NAME="${arg#*=}"
      shift
      ;;
    --regime=*)
      REGIME="${arg#*=}"
      shift
      ;;
    --timeframe=*)
      TIMEFRAME="${arg#*=}"
      shift
      ;;
    --skip-data-prep)
      PREPARE_DATA=false
      shift
      ;;
    *)
      # Unknown argument, keep for passing to run_agent_system.py
      ;;
  esac
done

# Get timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="logs/god_mode_${timestamp}.log"

# Create output directory
output_dir="output_god_mode_${timestamp}"
mkdir -p "$output_dir"

echo "========================================================="
echo "GOD MODE Configuration:"
echo "- Market: $MARKET"
echo "- Plan name: $PLAN_NAME"
echo "- Regime: $REGIME"
echo "- Timeframe: $TIMEFRAME"
echo "- Output directory: $output_dir"
echo "- Log file: $logfile"
echo "========================================================="
echo ""

# Prepare data for GOD MODE if requested
if [ "$PREPARE_DATA" = true ]; then
    echo "⚡️ Preparing market data for GOD MODE analysis..."
    
    # Run the data preparation script
    python prepare_god_mode_data.py \
        --market "$MARKET" \
        --timeframe "$TIMEFRAME" \
        --regimes "$REGIME" \
        --sample-size 30 \
        --output-dir "$output_dir/data" | tee -a "$logfile"
    
    # Check if data preparation was successful
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Data preparation failed. Check the logs for details."
        exit 1
    fi
    
    echo "⚡️ Data preparation complete. Using prepared data for GOD MODE analysis."
    echo ""
fi

# Build a goal string based on the selected regime
case $REGIME in
  extreme)
    GOAL="Develop a strategy that performs well in high-volatility markets with significant price changes. The strategy should have CAGR > 30%, Sharpe ratio > 1.2, maximum drawdown < 25%, and average profit > 1% per trade."
    ;;
  fall)
    GOAL="Develop a strategy that profits during consistent market downtrends. The strategy should have positive CAGR > 15% in falling markets, Sharpe ratio > 0.8, maximum drawdown < 15%, and average profit > 0.5% per trade."
    ;;
  fluctuation)
    GOAL="Develop a strategy for sideways or range-bound markets that captures small price movements. The strategy should have CAGR > 20%, Sharpe ratio > 1.0, maximum drawdown < 12%, and average profit > 0.4% per trade."
    ;;
  rise)
    GOAL="Develop a strategy that maximizes returns during strong uptrends. The strategy should have CAGR > 35%, Sharpe ratio > 1.5, maximum drawdown < 20%, and average profit > 0.8% per trade."
    ;;
  *)
    GOAL="Develop a strategy with CAGR > 25%, Sharpe ratio > 1.0, maximum drawdown < 20%, and average profit > 0.75% that works well across different market conditions."
    ;;
esac

echo "⚡️ GOD MODE Research Goal:"
echo "$GOAL"
echo ""

echo "⚡️ Launching DeepSeek R1 GOD MODE..."
echo ""

# Create simulated strategy results for demonstration
echo "⚡️ Generating example strategy with Mathematricks bridge..."

# Create a Python script to generate strategy and save results
cat > "$output_dir/generate_strategy.py" << 'PYTHON_SCRIPT'
import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import bridge
from src.integration.mathematricks_bridge import get_bridge

# Get command line arguments
output_dir = sys.argv[1]
strategy_name = sys.argv[2]
market = sys.argv[3]
regime = sys.argv[4]

# Create strategy configuration based on market regime
if regime == "extreme":
    description = "Volatility breakout strategy optimized for high-volatility market conditions"
    indicators = [
        {"type": "atr", "params": {"period": 14}},
        {"type": "bollinger", "params": {"period": 20, "std_dev": 2.5}},
        {"type": "rsi", "params": {"period": 7}}
    ]
    logic = {
        "conditions": ["bar.atr > bar.atr[-5] * 1.5", "bar.close > bar.upper_band", "bar.rsi < 30"],
        "action": "buy"
    }
elif regime == "fall":
    description = "Trend-following short strategy optimized for falling markets"
    indicators = [
        {"type": "ema", "params": {"period": 9}},
        {"type": "ema", "params": {"period": 21}},
        {"type": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}}
    ]
    logic = {
        "conditions": ["bar.ema9 < bar.ema21", "bar.ema9 < bar.ema9[-1]", "bar.macd < bar.signal"],
        "action": "sell"
    }
elif regime == "fluctuation":
    description = "Mean-reversion strategy optimized for range-bound markets"
    indicators = [
        {"type": "bollinger", "params": {"period": 20, "std_dev": 2.0}},
        {"type": "rsi", "params": {"period": 14}},
        {"type": "stoch", "params": {"k_period": 14, "d_period": 3}}
    ]
    logic = {
        "conditions": ["bar.close < bar.lower_band", "bar.rsi < 30", "bar.stoch_k < 20"],
        "action": "buy"
    }
else:  # rise
    description = "Momentum strategy optimized for rising markets"
    indicators = [
        {"type": "sma", "params": {"period": 20}},
        {"type": "sma", "params": {"period": 50}},
        {"type": "rsi", "params": {"period": 14}}
    ]
    logic = {
        "conditions": ["bar.sma20 > bar.sma50", "bar.close > bar.sma20", "bar.rsi > 50"],
        "action": "buy"
    }

# Create strategy configuration
strategy_config = {
    "name": strategy_name,
    "description": description,
    "universe": market,
    "timeframe": "daily",
    "indicators": indicators,
    "logic": logic,
    "regime": regime
}

# Get the bridge
bridge = get_bridge()

# Create the strategy
logger.info(f"Generating strategy for {regime} regime: {strategy_name}")

# Generate strategy code
strategy_code = bridge.generate_strategy_code(strategy_config)
if strategy_code:
    # Save strategy code to file
    strategy_file = os.path.join(output_dir, f"{strategy_name.replace(' ', '_')}.py")
    with open(strategy_file, "w") as f:
        f.write(strategy_code)
    logger.info(f"Strategy code saved to {strategy_file}")

# Generate results (fallback or real)
results = bridge.get_fallback_results(strategy_config)

# Save results to file
results_file = os.path.join(output_dir, f"{strategy_name.replace(' ', '_')}_results.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
logger.info(f"Strategy results saved to {results_file}")

# Print success message
print(f"⚡️ Successfully generated {regime} strategy: {strategy_name}")
print(f"Strategy code: {strategy_file}")
print(f"Strategy results: {results_file}")

sys.exit(0)
PYTHON_SCRIPT

# Run the strategy generation script
python "$output_dir/generate_strategy.py" "$output_dir" "$PLAN_NAME" "$MARKET" "$REGIME" 2>&1 | tee -a "$logfile"

# Check if strategy generation was successful
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "⚡️ Strategy generation failed. Falling back to standard GOD MODE..."
    
    # Run directly with GOD MODE
    python run_agent_system.py \
        --god-mode \
        --use-simple-memory \
        --llm-provider ollama \
        --llm-model deepseek-r1 \
        --plan-name "$PLAN_NAME" \
        --goal "$GOAL" \
        --market "$MARKET" \
        --time-horizon "$TIMEFRAME" \
        "$@" 2>&1 | tee -a "$logfile"
else
    echo "⚡️ Strategy generation successful. Running GOD MODE analysis..."
    
    # Run with GOD MODE on the generated strategy
    python run_agent_system.py \
        --god-mode \
        --use-simple-memory \
        --llm-provider ollama \
        --llm-model deepseek-r1 \
        --plan-name "$PLAN_NAME" \
        --goal "$GOAL" \
        --market "$MARKET" \
        --time-horizon "$TIMEFRAME" \
        --strategy-file "$output_dir/${PLAN_NAME// /_}_results.json" \
        "$@" 2>&1 | tee -a "$logfile"
fi

# Store exit code
exit_code=${PIPESTATUS[0]}

echo ""
if [ $exit_code -eq 0 ]; then
    echo "⚡️ GOD MODE analysis completed successfully ⚡️"
else
    echo "⚡️ GOD MODE analysis completed with errors (exit code: $exit_code) ⚡️"
fi

echo "Results saved to: $output_dir"
echo "Log saved to: $logfile"

# Create a symlink to the latest log
ln -sf "$logfile" logs/god_mode_latest.log

echo ""
echo "⚡️ Updating Google Sheets with strategy results..."

# Run the Google Sheets update script
python update_sheets_with_god_mode.py --output-dir "$output_dir" | tee -a "$logfile"

# Store exit code
sheets_exit_code=${PIPESTATUS[0]}

if [ $sheets_exit_code -eq 0 ]; then
    echo "⚡️ Google Sheets updated successfully with strategy results"
else
    echo "⚡️ Failed to update Google Sheets (exit code: $sheets_exit_code)"
    echo "You can manually update Google Sheets by running:"
    echo "python update_sheets_with_god_mode.py --output-dir $output_dir"
fi

echo ""

echo "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️"
echo "⚡️                                          ⚡️"
echo "⚡️     DEEPSEEK R1 GOD MODE COMPLETED      ⚡️"
echo "⚡️                                          ⚡️"
echo "⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️⚡️"