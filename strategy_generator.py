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
