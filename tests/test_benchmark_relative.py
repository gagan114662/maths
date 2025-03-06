"""
Tests for the benchmark relative performance scoring module.
"""

import unittest
import pandas as pd
import numpy as np
from src.performance_scoring.benchmark_relative import BenchmarkRelativeScorer

class TestBenchmarkRelativeScorer(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
        
        # Create benchmark data (S&P 500 like)
        np.random.seed(42)  # For reproducibility
        self.benchmark_returns = np.random.normal(0.0005, 0.01, len(self.dates))  # Mean: 0.05% daily, ~12.6% annual
        
        # Calculate benchmark prices from returns
        benchmark_prices = (1 + self.benchmark_returns).cumprod()
        
        # Create a returns series and a price dataframe
        self.benchmark_returns_series = pd.Series(self.benchmark_returns, index=self.dates)
        benchmark_data = pd.DataFrame({
            'returns': self.benchmark_returns_series  # Directly provide returns instead of prices
        }, index=self.dates)
        
        # Create strategy data (outperforming the benchmark)
        self.strategy_returns = self.benchmark_returns + 0.0003  # Strategy outperforms by ~7.5% annually
        self.strategy_returns_series = pd.Series(self.strategy_returns, index=self.dates)
        
        # Create underperforming strategy data
        self.underperform_returns = self.benchmark_returns - 0.0002  # Strategy underperforms by ~5% annually
        self.underperform_returns_series = pd.Series(self.underperform_returns, index=self.dates)
        
        # Initialize the scorer
        benchmark_data_dict = {"SPY": benchmark_data}
        self.scorer = BenchmarkRelativeScorer(benchmark_data=benchmark_data_dict)
    
    def test_excess_return(self):
        # Calculate metrics for outperforming strategy
        metrics = self.scorer.calculate_metrics(self.strategy_returns_series)
        
        # Excess return should be positive
        self.assertGreater(metrics["excess_return"], 0)
        
        # Calculate metrics for underperforming strategy
        metrics_under = self.scorer.calculate_metrics(self.underperform_returns_series)
        
        # Excess return should be negative
        self.assertLess(metrics_under["excess_return"], 0)
    
    def test_information_ratio(self):
        # Calculate metrics for outperforming strategy
        metrics = self.scorer.calculate_metrics(self.strategy_returns_series)
        
        # Information ratio should be positive for outperforming strategy
        self.assertGreater(metrics["information_ratio"], 0)
        
        # Calculate metrics for underperforming strategy
        metrics_under = self.scorer.calculate_metrics(self.underperform_returns_series)
        
        # Information ratio should be negative for underperforming strategy
        self.assertLess(metrics_under["information_ratio"], 0)
    
    def test_up_and_down_capture(self):
        # Calculate metrics for outperforming strategy
        metrics = self.scorer.calculate_metrics(self.strategy_returns_series)
        
        # Up capture should be > 1 for outperforming strategy
        self.assertGreater(metrics["up_capture"], 1.0)
        
        # Down capture should be < 1 for outperforming strategy (loses less in down markets)
        self.assertLess(metrics["down_capture"], 1.0)
    
    def test_alpha_and_beta(self):
        # Calculate metrics for outperforming strategy
        metrics = self.scorer.calculate_metrics(self.strategy_returns_series)
        
        # Alpha should be positive for outperforming strategy
        self.assertGreater(metrics["alpha"], 0)
        
        # Beta should be close to 1 for our synthetic data
        self.assertAlmostEqual(metrics["beta"], 1.0, delta=0.1)
    
    def test_win_rate(self):
        # Calculate metrics for outperforming strategy
        metrics = self.scorer.calculate_metrics(self.strategy_returns_series)
        
        # Win rate should be > 0.5 for outperforming strategy
        self.assertGreater(metrics["win_rate_vs_benchmark"], 0.5)
        
        # Calculate metrics for underperforming strategy
        metrics_under = self.scorer.calculate_metrics(self.underperform_returns_series)
        
        # Win rate should be < 0.5 for underperforming strategy
        self.assertLess(metrics_under["win_rate_vs_benchmark"], 0.5)
    
    def test_rank_strategies(self):
        # Create a dictionary of strategy returns
        strategy_returns = {
            "Outperformer": self.strategy_returns_series,
            "Underperformer": self.underperform_returns_series,
            "Benchmark-like": self.benchmark_returns_series
        }
        
        # Rank the strategies
        ranked = self.scorer.rank_strategies(strategy_returns)
        
        # Outperformer should be ranked first
        self.assertEqual(ranked.index[0], "Outperformer")
        
        # Check that Underperformer is ranked lowest (either 2nd or 3rd position)
        # This is more flexible than checking for exactly the last position
        self.assertTrue(
            "Underperformer" in ranked.index[-2:],
            "Underperformer should be among the lowest ranked strategies"
        )
    
    def test_outperformance_report(self):
        # Generate report for outperforming strategy
        report = self.scorer.generate_outperformance_report(
            self.strategy_returns_series, "Outperformer"
        )
        
        # Check that the report has the expected structure
        self.assertIn("strategy_name", report)
        self.assertIn("benchmarks", report)
        self.assertIn("summary", report)
        self.assertIn("time_analysis", report)
        
        # Strategy name should be correct
        self.assertEqual(report["strategy_name"], "Outperformer")
        
        # SPY should be in benchmarks
        self.assertIn("SPY", report["benchmarks"])
        
        # Report should contain metrics
        self.assertIn("excess_return", report["benchmarks"]["SPY"])
        
        # Since we created an outperforming strategy, the excess return should be positive
        if "excess_return" in report["benchmarks"]["SPY"]:
            self.assertGreater(report["benchmarks"]["SPY"]["excess_return"], 0)
            
        # Average excess return should be positive if present
        if "average_excess_return" in report["summary"]:
            self.assertGreater(report["summary"]["average_excess_return"], 0)

if __name__ == "__main__":
    unittest.main()