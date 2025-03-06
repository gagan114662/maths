"""
Cross-market multi-asset strategy generation module.

This package contains tools for generating trading strategies that operate
across multiple asset classes and leverage cross-market opportunities.
"""

from .generator import MultiAssetStrategyGenerator
from .correlations import CrossMarketCorrelationAnalyzer
from .opportunities import OpportunityDetector

__all__ = [
    "MultiAssetStrategyGenerator",
    "CrossMarketCorrelationAnalyzer",
    "OpportunityDetector"
]