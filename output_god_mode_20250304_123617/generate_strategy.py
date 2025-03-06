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

# Define strategy categories
import random

# Define strategy categories
strategy_categories = [
    "trend_following", "mean_reversion", "momentum", "volatility", "statistical_arbitrage",
    "machine_learning", "market_microstructure", "factor_based", "calendar_based", 
    "intraday_pattern", "options_volatility", "fixed_income", "event_driven"
]

# Select a category based on the regime and randomization
if regime == "extreme":
    primary_category = "volatility"
    secondary_category = "trend_following"
elif regime == "fall":
    primary_category = "trend_following"
    secondary_category = "statistical_arbitrage"
elif regime == "fluctuation":
    primary_category = "mean_reversion"
    secondary_category = "calendar_based"
else:  # rise
    primary_category = "momentum"
    secondary_category = "factor_based"

# Add some randomness - occasionally use the secondary category or a completely random one
chance = random.random()
if chance < 0.2:  # 20% chance to use secondary
    selected_category = secondary_category
elif chance < 0.3:  # 10% chance to use random category
    selected_category = random.choice(strategy_categories)
else:  # 70% chance to use primary category
    selected_category = primary_category

logger.info(f"Selected strategy category: {selected_category} for {regime} regime")

# Select from multiple strategy options within the chosen category
if selected_category == "volatility":
    strategy_options = [
        {
            "description": "Volatility breakout strategy optimized for high-volatility market conditions",
            "indicators": [
                {"type": "atr", "params": {"period": 14}},
                {"type": "bollinger", "params": {"period": 20, "std_dev": 2.5}},
                {"type": "rsi", "params": {"period": 7}}
            ],
            "logic": {
                "conditions": ["bar.atr > bar.atr[-5] * 1.5", "bar.close > bar.upper_band", "bar.rsi < 30"],
                "action": "buy"
            }
        },
        {
            "description": "Mean reversion strategy that capitalizes on volatility spikes",
            "indicators": [
                {"type": "atr", "params": {"period": 10}},
                {"type": "bollinger", "params": {"period": 20, "std_dev": 2.0}},
                {"type": "rsi", "params": {"period": 14}}
            ],
            "logic": {
                "conditions": ["bar.close < bar.lower_band", "bar.atr > bar.atr[-20] * 1.2", "bar.rsi < 30"],
                "action": "buy"
            }
        },
        {
            "description": "Strategy based on volatility surface modeling",
            "indicators": [
                {"type": "atr", "params": {"period": 5}},
                {"type": "bollinger", "params": {"period": 10, "std_dev": 1.5}},
                {"type": "rsi", "params": {"period": 7}}
            ],
            "logic": {
                "conditions": ["bar.atr > bar.atr[-20] * 1.3", "bar.close.diff() > 0", "bar.rsi < 40"],
                "action": "buy"
            }
        }
    ]
elif selected_category == "trend_following":
    strategy_options = [
        {
            "description": "Trend-following strategy based on moving average crossovers",
            "indicators": [
                {"type": "ema", "params": {"period": 9}},
                {"type": "ema", "params": {"period": 21}},
                {"type": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}}
            ],
            "logic": {
                "conditions": ["bar.ema9 > bar.ema21", "bar.ema9 > bar.ema9[-1]", "bar.macd > bar.signal"],
                "action": "buy"
            }
        },
        {
            "description": "Channel breakout strategy that identifies breakouts from price channels",
            "indicators": [
                {"type": "sma", "params": {"period": 20}},
                {"type": "bollinger", "params": {"period": 20, "std_dev": 2.0}}
            ],
            "logic": {
                "conditions": ["bar.close > bar.upper_band", "bar.close[-1] <= bar.upper_band[-1]", "bar.close > bar.sma20"],
                "action": "buy"
            }
        },
        {
            "description": "Trend-following short strategy optimized for falling markets",
            "indicators": [
                {"type": "ema", "params": {"period": 9}},
                {"type": "ema", "params": {"period": 21}},
                {"type": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}}
            ],
            "logic": {
                "conditions": ["bar.ema9 < bar.ema21", "bar.ema9 < bar.ema9[-1]", "bar.macd < bar.signal"],
                "action": "sell"
            }
        }
    ]
elif selected_category == "mean_reversion":
    strategy_options = [
        {
            "description": "Mean-reversion strategy optimized for range-bound markets",
            "indicators": [
                {"type": "bollinger", "params": {"period": 20, "std_dev": 2.0}},
                {"type": "rsi", "params": {"period": 14}},
                {"type": "stoch", "params": {"k_period": 14, "d_period": 3}}
            ],
            "logic": {
                "conditions": ["bar.close < bar.lower_band", "bar.rsi < 30", "bar.stoch_k < 20"],
                "action": "buy"
            }
        },
        {
            "description": "Statistical approach to mean reversion using z-scores",
            "indicators": [
                {"type": "sma", "params": {"period": 50}},
                {"type": "bollinger", "params": {"period": 20, "std_dev": 2.0}},
                {"type": "rsi", "params": {"period": 7}}
            ],
            "logic": {
                "conditions": ["(bar.close - bar.sma50) / bar.std50 < -2", "bar.rsi < 30"],
                "action": "buy"
            }
        },
        {
            "description": "RSI divergence strategy for range-bound markets",
            "indicators": [
                {"type": "rsi", "params": {"period": 7}},
                {"type": "sma", "params": {"period": 20}}
            ],
            "logic": {
                "conditions": ["bar.rsi < 30", "bar.close < bar.close[-1]", "bar.rsi > bar.rsi[-1]"],
                "action": "buy"
            }
        }
    ]
elif selected_category == "momentum":
    strategy_options = [
        {
            "description": "Momentum strategy optimized for rising markets",
            "indicators": [
                {"type": "sma", "params": {"period": 20}},
                {"type": "sma", "params": {"period": 50}},
                {"type": "rsi", "params": {"period": 14}}
            ],
            "logic": {
                "conditions": ["bar.sma20 > bar.sma50", "bar.close > bar.sma20", "bar.rsi > 50"],
                "action": "buy"
            }
        },
        {
            "description": "Triple momentum system combining price, volume, and volatility",
            "indicators": [
                {"type": "ema", "params": {"period": 8}},
                {"type": "ema", "params": {"period": 21}},
                {"type": "rsi", "params": {"period": 10}}
            ],
            "logic": {
                "conditions": ["bar.close > bar.ema8", "bar.close > bar.ema21", "bar.rsi > 50 and bar.rsi < 70"],
                "action": "buy"
            }
        },
        {
            "description": "Strategy focusing on assets with the strongest performance momentum",
            "indicators": [
                {"type": "rsi", "params": {"period": 14}},
                {"type": "ema", "params": {"period": 10}},
                {"type": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}}
            ],
            "logic": {
                "conditions": ["bar.rsi > 60", "bar.close > bar.ema10", "bar.macd > bar.signal"],
                "action": "buy"
            }
        }
    ]
elif selected_category == "statistical_arbitrage":
    strategy_options = [
        {
            "description": "Statistical arbitrage strategy focusing on correlated pairs",
            "indicators": [
                {"type": "ema", "params": {"period": 20}},
                {"type": "bollinger", "params": {"period": 20, "std_dev": 2.0}}
            ],
            "logic": {
                "conditions": ["bar.spread < bar.lower_band", "bar.spread > bar.spread[-1]"],
                "action": "buy"
            }
        },
        {
            "description": "ETF arbitrage exploiting price discrepancies",
            "indicators": [
                {"type": "sma", "params": {"period": 5}},
                {"type": "rsi", "params": {"period": 14}}
            ],
            "logic": {
                "conditions": ["bar.premium < -0.01", "bar.rsi < 40"],
                "action": "buy"
            }
        },
        {
            "description": "Index arbitrage strategy based on futures-cash basis",
            "indicators": [
                {"type": "bollinger", "params": {"period": 20, "std_dev": 1.5}},
                {"type": "rsi", "params": {"period": 5}}
            ],
            "logic": {
                "conditions": ["bar.basis < bar.lower_band", "bar.rsi < 30"],
                "action": "buy"
            }
        }
    ]
elif selected_category == "factor_based":
    strategy_options = [
        {
            "description": "Multi-factor strategy combining quality and momentum",
            "indicators": [
                {"type": "sma", "params": {"period": 50}},
                {"type": "rsi", "params": {"period": 14}}
            ],
            "logic": {
                "conditions": ["bar.quality_score > 0.8", "bar.close > bar.sma50", "bar.rsi > 50"],
                "action": "buy"
            }
        },
        {
            "description": "Low volatility strategy with momentum filter",
            "indicators": [
                {"type": "atr", "params": {"period": 20}},
                {"type": "sma", "params": {"period": 50}}
            ],
            "logic": {
                "conditions": ["bar.vol_rank < 0.3", "bar.close > bar.sma50"],
                "action": "buy"
            }
        },
        {
            "description": "Value factor strategy with technical confirmation",
            "indicators": [
                {"type": "ema", "params": {"period": 20}},
                {"type": "rsi", "params": {"period": 14}}
            ],
            "logic": {
                "conditions": ["bar.value_score > 0.7", "bar.close > bar.ema20", "bar.rsi > 40"],
                "action": "buy"
            }
        }
    ]
elif selected_category == "calendar_based":
    strategy_options = [
        {
            "description": "Turn-of-month strategy exploiting monthly effects",
            "indicators": [
                {"type": "sma", "params": {"period": 10}},
                {"type": "rsi", "params": {"period": 14}}
            ],
            "logic": {
                "conditions": ["bar.is_month_end", "bar.close > bar.sma10"],
                "action": "buy"
            }
        },
        {
            "description": "Day-of-week pattern strategy",
            "indicators": [
                {"type": "ema", "params": {"period": 5}},
                {"type": "rsi", "params": {"period": 7}}
            ],
            "logic": {
                "conditions": ["bar.is_monday", "bar.close > bar.ema5", "bar.rsi < 60"],
                "action": "buy"
            }
        },
        {
            "description": "Seasonal patterns trading strategy",
            "indicators": [
                {"type": "sma", "params": {"period": 20}},
                {"type": "bollinger", "params": {"period": 20, "std_dev": 2.0}}
            ],
            "logic": {
                "conditions": ["bar.is_favorable_season", "bar.close > bar.sma20", "bar.close < bar.upper_band"],
                "action": "buy"
            }
        }
    ]
elif selected_category == "intraday_pattern":
    strategy_options = [
        {
            "description": "Opening range breakout strategy",
            "indicators": [
                {"type": "sma", "params": {"period": 5}},
                {"type": "atr", "params": {"period": 14}}
            ],
            "logic": {
                "conditions": ["bar.close > bar.opening_high", "bar.close > bar.sma5", "bar.time < '12:00'"],
                "action": "buy"
            }
        },
        {
            "description": "Intraday momentum reversal strategy",
            "indicators": [
                {"type": "ema", "params": {"period": 10}},
                {"type": "rsi", "params": {"period": 7}}
            ],
            "logic": {
                "conditions": ["bar.close < bar.ema10", "bar.rsi < 30", "bar.time > '14:00'"],
                "action": "buy"
            }
        },
        {
            "description": "End-of-day reversal pattern strategy",
            "indicators": [
                {"type": "bollinger", "params": {"period": 20, "std_dev": 2.0}},
                {"type": "rsi", "params": {"period": 14}}
            ],
            "logic": {
                "conditions": ["bar.close < bar.lower_band", "bar.time > '15:00'", "bar.rsi < 40"],
                "action": "buy"
            }
        }
    ]
else:  # Default or any other category
    strategy_options = [
        {
            "description": "Adaptive trend following strategy",
            "indicators": [
                {"type": "sma", "params": {"period": 20}},
                {"type": "sma", "params": {"period": 50}},
                {"type": "atr", "params": {"period": 14}}
            ],
            "logic": {
                "conditions": ["bar.sma20 > bar.sma50", "bar.close > bar.sma20", "bar.atr > bar.atr[-5]"],
                "action": "buy"
            }
        },
        {
            "description": "Volatility-adjusted momentum strategy",
            "indicators": [
                {"type": "ema", "params": {"period": 10}},
                {"type": "atr", "params": {"period": 14}},
                {"type": "rsi", "params": {"period": 14}}
            ],
            "logic": {
                "conditions": ["bar.close > bar.ema10", "bar.close / bar.atr > 2", "bar.rsi > 50"],
                "action": "buy"
            }
        },
        {
            "description": "Multi-timeframe confirmation strategy",
            "indicators": [
                {"type": "sma", "params": {"period": 10}},
                {"type": "sma", "params": {"period": 50}},
                {"type": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}}
            ],
            "logic": {
                "conditions": ["bar.close > bar.sma10", "bar.sma10 > bar.sma50", "bar.macd > bar.signal"],
                "action": "buy"
            }
        }
    ]

# Randomly select one of the strategies from the chosen category
selected_strategy = random.choice(strategy_options)
description = selected_strategy["description"]
indicators = selected_strategy["indicators"]
logic = selected_strategy["logic"]

logger.info(f"Selected strategy: {description}")

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
