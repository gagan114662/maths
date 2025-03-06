"""
Performance scoring modules for strategy evaluation.

This package contains various scoring mechanisms to evaluate trading strategies
with a focus on market outperformance and relative performance.
"""

from .benchmark_relative import BenchmarkRelativeScorer

__all__ = ["BenchmarkRelativeScorer"]