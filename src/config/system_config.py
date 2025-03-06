"""
System configuration module for the autonomous trading strategy agent.
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Base system paths
BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"

# Ensure directories exist
for directory in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Performance targets
PERFORMANCE_TARGETS = {
    "cagr": 0.25,           # 25% Compound Annual Growth Rate
    "sharpe_ratio": 1.0,    # Sharpe ratio with 5% risk-free rate
    "max_drawdown": 0.20,   # Maximum drawdown of 20%
    "avg_profit": 0.0075,   # Average profit of 0.75%
}

# LLM Configuration
LLM_CONFIG = {
    "provider": "ollama",
    "model": "deepseek-r1",
    "temperature": 0.2,
    "max_tokens": 4000,
    "retry_attempts": 3,
    "backoff_factor": 1.5,
    "api_url": "http://localhost:11434/api",
    # DeepSeek R1 specific config
    "deepseek_config": {
        "top_k": 40,        # Number of tokens to consider for top-k sampling
        "top_p": 0.9,       # Probability threshold for nucleus sampling
        "repeat_penalty": 1.1,  # Penalty for repeating tokens
        "mirostat": 0,      # Use mirostat sampling (0: disabled, 1: mirostat, 2: mirostat 2.0)
        "num_ctx": 8192     # Maximum context length (8k tokens)
    }
}

# Market data configuration
MARKET_DATA_CONFIG = {
    "data_sources": [
        {"name": "ibkr", "type": "historical", "enabled": True},
        {"name": "fintsb", "type": "benchmark", "enabled": True},
    ],
    "timeframes": ["1d", "1h", "15m", "5m", "1m"],
    "default_window": "252d",  # Default lookback window (252 trading days)
}

# Agent configuration
AGENT_CONFIG = {
    "supervisor": {
        "enabled": True,
        "memory_limit": 10000,
        "coordination_interval": 60,  # seconds
    },
    "generation": {
        "enabled": True,
        "strategies_per_run": 5,
        "exploration_weight": 0.7,
    },
    "backtest": {
        "enabled": True,
        "transaction_cost": 0.0015,  # 15 bps transaction cost
        "slippage": 0.0010,         # 10 bps slippage
    },
    "risk": {
        "enabled": True,
        "max_position_size": 0.10,  # 10% max position size
        "max_portfolio_volatility": 0.15,  # 15% annualized volatility target
    },
    "ranking": {
        "enabled": True,
        "tournament_size": 10,
        "elo_k_factor": 32,
    },
    "evolution": {
        "enabled": True,
        "mutation_rate": 0.2,
        "crossover_rate": 0.7,
        "generation_size": 10,
    },
    "meta_review": {
        "enabled": True,
        "review_frequency": 10,  # Review after every 10 strategies
    },
}

# Safety configuration
SAFETY_CONFIG = {
    "market_manipulation_checks": True,
    "position_limits": True,
    "trading_frequency_limits": True,
    "price_impact_monitoring": True,
    "max_concentration": 0.20,  # 20% maximum concentration in a single asset
    "max_leverage": 1.5,  # 1.5x maximum leverage
}

# Memory system configuration
MEMORY_CONFIG = {
    "short_term_capacity": 100,   # Number of recent events to keep
    "long_term_retention": 1000,  # Number of important events to retain
    "importance_threshold": 0.7,  # Threshold for transferring to long-term memory
}

# Tool configuration
TOOL_CONFIG = {
    "market_data": {
        "enabled": True,
        "cache_expiry": 300,  # seconds
    },
    "backtesting": {
        "enabled": True,
        "default_period": "1y",
    },
    "optimization": {
        "enabled": True,
        "max_iterations": 100,
    },
}

# Default trading parameters
DEFAULT_TRADING_PARAMS = {
    "universe": "sp500",  # Default trading universe
    "position_sizing": "equal_weight",  # Default position sizing method
    "rebalance_frequency": "daily",     # Default rebalancing frequency
    "trading_hours": "regular",         # Regular market hours only
}

def get_full_config() -> Dict[str, Any]:
    """Return the complete system configuration as a dictionary."""
    return {
        "performance_targets": PERFORMANCE_TARGETS,
        "llm": LLM_CONFIG,
        "market_data": MARKET_DATA_CONFIG,
        "agents": AGENT_CONFIG,
        "safety": SAFETY_CONFIG,
        "memory": MEMORY_CONFIG,
        "tools": TOOL_CONFIG,
        "trading": DEFAULT_TRADING_PARAMS,
        "paths": {
            "base_dir": str(BASE_DIR),
            "data_dir": str(DATA_DIR),
            "output_dir": str(OUTPUT_DIR),
            "model_dir": str(MODEL_DIR),
        }
    }