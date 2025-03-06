"""
Configuration settings for the trading system.
"""
import yaml
from pathlib import Path
import os

def load_config(config_path=None):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    # Default configuration
    default_config = {
        "risk_free_rate": RISK_FREE_RATE,
        "target_metrics": TARGET_METRICS,
        "transaction_costs": TRANSACTION_COSTS,
        "market_constraints": MARKET_CONSTRAINTS,
        "agent_weights": DEFAULT_AGENT_WEIGHTS
    }
    
    # If config path is provided, load and merge with defaults
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge configurations
                if user_config:
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                            # Deep merge for nested dictionaries
                            default_config[key].update(value)
                        else:
                            # Direct replacement for non-dict values
                            default_config[key] = value
        except Exception as e:
            print(f"Error loading config file {config_path}: {str(e)}")
            
    return default_config

# Data paths
BASE_DATA_PATH = Path("/mnt/VANDAN_DISK/gagan_stuff/data")
IBKR_DATA_PATH = BASE_DATA_PATH / "ibkr"
KRAKEN_DATA_PATH = BASE_DATA_PATH / "kraken"
STOCK_SYMBOLS_PATH = BASE_DATA_PATH / "stocksymbolslists"

# FinTSB paths
FINTSB_PATH = Path("FinTSB")
MATHEMATRICKS_PATH = Path("mathematricks")

# Trading parameters
RISK_FREE_RATE = 0.05  # 5% as specified in requirements
TARGET_METRICS = {
    "cagr": 0.25,  # 25% minimum CAGR
    "sharpe_ratio": 1.0,  # Minimum Sharpe ratio (with 5% risk-free rate)
    "max_drawdown": 0.20,  # Maximum 20% drawdown allowed
    "avg_profit": 0.0075  # 0.75% minimum average profit
}

# Agent weights configuration
DEFAULT_AGENT_WEIGHTS = {
    "volatility": 1.0,
    "signal_extraction": 1.0,
    "generative": 1.0,
    "backtesting": 1.0,
    "risk_assessment": 1.0,
    "evolution": 1.0,
    "meta_review": 1.0
}

# Transaction costs and constraints
TRANSACTION_COSTS = {
    "commission": 0.001,  # 0.1% commission
    "slippage": 0.001    # 0.1% estimated slippage
}

# Market constraints
MARKET_CONSTRAINTS = {
    "limit_up": 0.10,    # 10% upper limit
    "limit_down": -0.10, # 10% lower limit
    "min_volume": 100000 # Minimum daily volume requirement
}