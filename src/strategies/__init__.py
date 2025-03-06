"""
Strategies module containing evaluation and trading strategy implementations.
"""
from .evaluator import StrategyEvaluator

__all__ = [
    'StrategyEvaluator',
]

# Version control
__version__ = '0.1.0'

# Default strategy parameters
DEFAULT_STRATEGY_PARAMS = {
    'max_position_size': 0.1,  # 10% of portfolio
    'risk_free_rate': 0.05,    # 5% annual
    'transaction_cost': 0.001  # 0.1% per trade
}