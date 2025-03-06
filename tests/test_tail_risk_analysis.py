#!/usr/bin/env python
"""Unit tests for tail risk analysis."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.risk_management.tail_risk_analysis import TailRiskAnalyzer, calculate_conditional_drawdown_risk

class TestTailRiskAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Generate test return data with fat tails
        n_days = 1000
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        
        # Generate normal returns
        normal_returns = np.random.normal(0, 0.01, n_days)
        
        # Add some extreme events to create fat tails
        extreme_events = np.random.choice(n_days, size=int(n_days*0.05))  # 5% extreme events
        normal_returns[extreme_events] *= 3  # Make these returns 3x larger
        
        self.returns = pd.Series(normal_returns, index=dates)
        self.analyzer = TailRiskAnalyzer(threshold_percentile=0.95)
        
    def test_evt_distribution_fitting(self):
        """Test fitting of EVT distribution."""
        params = self.analyzer.fit_evt_distribution(self.returns)
        
        # Check that parameters exist and have reasonable values
        self.assertIn('threshold', params)
        self.assertIn('shape', params)
        self.assertIn('scale', params)
        
        # Shape parameter should typically be positive for financial returns
        self.assertGreater(params['shape'], 0)
        
        # Scale parameter should be positive
        self.assertGreater(params['scale'], 0)
        
    def test_var_estimation(self):
        """Test Value at Risk estimation."""
        self.analyzer.fit_evt_distribution(self.returns)
        
        # Test VaR at different confidence levels
        var_99 = self.analyzer.estimate_var(confidence_level=0.99)
        var_95 = self.analyzer.estimate_var(confidence_level=0.95)
        
        # 99% VaR should be larger than 95% VaR
        self.assertGreater(var_99, var_95)
        
        # VaR should be positive (for losses)
        self.assertGreater(var_99, 0)
        self.assertGreater(var_95, 0)
        
    def test_expected_shortfall(self):
        """Test Expected Shortfall estimation."""
        self.analyzer.fit_evt_distribution(self.returns)
        
        # Test ES at different confidence levels
        es_99 = self.analyzer.estimate_es(confidence_level=0.99)
        es_95 = self.analyzer.estimate_es(confidence_level=0.95)
        var_99 = self.analyzer.estimate_var(confidence_level=0.99)
        
        # ES should be larger than VaR
        self.assertGreater(es_99, var_99)
        
        # 99% ES should be larger than 95% ES
        self.assertGreater(es_99, es_95)
        
    def test_tail_dependence(self):
        """Test tail dependence estimation."""
        # Create correlated return series
        returns2 = 0.7 * self.returns + 0.3 * pd.Series(
            np.random.normal(0, 0.01, len(self.returns)), 
            index=self.returns.index
        )
        
        dependence = self.analyzer.estimate_tail_dependence(self.returns, returns2)
        
        # Check that dependence coefficients are between 0 and 1
        self.assertGreaterEqual(dependence['lower_tail_dependence'], 0)
        self.assertLessEqual(dependence['lower_tail_dependence'], 1)
        self.assertGreaterEqual(dependence['upper_tail_dependence'], 0)
        self.assertLessEqual(dependence['upper_tail_dependence'], 1)
        
    def test_comprehensive_analysis(self):
        """Test comprehensive tail risk analysis."""
        metrics = self.analyzer.analyze_tail_risk(self.returns)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'var_99', 'es_99', 'var_95', 'es_95', 'tail_index',
            'scale', 'threshold', 'skewness', 'excess_kurtosis',
            'max_drawdown', 'avg_time_underwater', 'jarque_bera_pvalue'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            
        # Check relationships between metrics
        self.assertGreater(metrics['es_99'], metrics['var_99'])
        self.assertGreater(metrics['var_99'], metrics['var_95'])
        
    def test_stressed_var(self):
        """Test stressed VaR calculation."""
        stressed_metrics = self.analyzer.estimate_stress_var(
            self.returns, stress_factor=1.5
        )
        
        normal_metrics = self.analyzer.analyze_tail_risk(self.returns)
        
        # Stressed VaR should be higher than normal VaR
        self.assertGreater(
            stressed_metrics['stressed_var_99'],
            normal_metrics['var_99']
        )
        
        # Stress impact should be positive
        self.assertGreater(stressed_metrics['stress_impact'], 0)
        
    def test_conditional_drawdown_risk(self):
        """Test Conditional Drawdown at Risk calculation."""
        drawdown_metrics = calculate_conditional_drawdown_risk(
            self.returns, confidence_level=0.95
        )
        
        # Check that all metrics are present
        self.assertIn('cdar', drawdown_metrics)
        self.assertIn('max_drawdown', drawdown_metrics)
        self.assertIn('avg_drawdown', drawdown_metrics)
        self.assertIn('worst_drawdowns', drawdown_metrics)
        
        # CDaR should be larger than average drawdown
        self.assertLess(drawdown_metrics['cdar'], drawdown_metrics['avg_drawdown'])
        
        # Max drawdown should be the worst
        self.assertLessEqual(drawdown_metrics['cdar'], abs(drawdown_metrics['max_drawdown']))
        
    def test_error_handling(self):
        """Test error handling for insufficient data."""
        short_returns = pd.Series(np.random.normal(0, 0.01, 50))
        
        # Should raise error for insufficient data
        with self.assertRaises(ValueError):
            self.analyzer.fit_evt_distribution(short_returns)
            
        # Should raise error if trying to estimate VaR before fitting
        analyzer = TailRiskAnalyzer()
        with self.assertRaises(ValueError):
            analyzer.estimate_var()

if __name__ == '__main__':
    unittest.main()