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

# Generate results using real strategy results instead of fallback
results = bridge.get_strategy_results(strategy_config)

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
