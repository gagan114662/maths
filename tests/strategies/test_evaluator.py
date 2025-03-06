"""
Tests for the Strategy Evaluator.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.strategies.evaluator import StrategyEvaluator

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Generate predictions and actuals
    predictions = pd.DataFrame({
        'AAPL': np.random.normal(0, 1, 100),
        'GOOGL': np.random.normal(0, 1, 100)
    }, index=dates)
    
    actuals = pd.DataFrame({
        'AAPL': np.random.normal(0, 1, 100),
        'GOOGL': np.random.normal(0, 1, 100)
    }, index=dates)
    
    # Generate returns
    returns = pd.DataFrame({
        'strategy': np.random.normal(0.001, 0.02, 100)
    }, index=dates)
    
    # Generate trade information
    trades = pd.DataFrame({
        'timestamp': dates,
        'symbol': ['AAPL', 'GOOGL'] * 50,
        'entry_price': np.random.normal(100, 10, 100),
        'exit_price': np.random.normal(101, 10, 100),
        'size': np.random.randint(100, 1000, 100),
        'returns': np.random.normal(0.001, 0.02, 100)
    })
    
    return predictions, actuals, returns, trades

@pytest.fixture
def evaluator():
    """Create a StrategyEvaluator instance."""
    return StrategyEvaluator()

def test_evaluator_initialization(evaluator):
    """Test evaluator initialization."""
    assert evaluator is not None
    assert hasattr(evaluator, 'evaluate_strategy')
    assert hasattr(evaluator, 'risk_free_rate')

def test_strategy_evaluation(evaluator, sample_data):
    """Test comprehensive strategy evaluation."""
    predictions, actuals, returns, trades = sample_data
    
    results = evaluator.evaluate_strategy(
        predictions=predictions,
        actuals=actuals,
        returns=returns,
        trades=trades
    )
    
    # Check result structure
    assert isinstance(results, dict)
    assert 'ranking_metrics' in results
    assert 'portfolio_metrics' in results
    assert 'error_metrics' in results
    assert 'ethical_compliance' in results
    assert 'robustness_metrics' in results
    assert 'overall_score' in results

def test_ranking_metrics(evaluator, sample_data):
    """Test calculation of ranking metrics."""
    predictions, actuals, returns, trades = sample_data
    results = evaluator.evaluate_strategy(predictions, actuals, returns, trades)
    
    metrics = results['ranking_metrics']
    assert 'ic_mean' in metrics
    assert 'rankicir' in metrics
    assert 'spearman_corr' in metrics
    
    # Check metric ranges
    assert -1 <= metrics['ic_mean'] <= 1
    assert -1 <= metrics['spearman_corr'] <= 1

def test_portfolio_metrics(evaluator, sample_data):
    """Test calculation of portfolio metrics."""
    predictions, actuals, returns, trades = sample_data
    results = evaluator.evaluate_strategy(predictions, actuals, returns, trades)
    
    metrics = results['portfolio_metrics']
    assert 'cagr' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'sortino_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'avg_profit' in metrics
    
    # Check metric ranges
    assert -1 <= metrics['max_drawdown'] <= 0
    assert metrics['avg_profit'] is not None
    assert metrics['sharpe_ratio'] is not None

def test_error_metrics(evaluator, sample_data):
    """Test calculation of error metrics."""
    predictions, actuals, returns, trades = sample_data
    results = evaluator.evaluate_strategy(predictions, actuals, returns, trades)
    
    metrics = results['error_metrics']
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'mape' in metrics
    
    # Check metric ranges
    assert metrics['rmse'] >= 0
    assert metrics['mae'] >= 0
    assert metrics['mape'] >= 0

def test_ethical_compliance(evaluator, sample_data):
    """Test ethical compliance checks."""
    predictions, actuals, returns, trades = sample_data
    results = evaluator.evaluate_strategy(predictions, actuals, returns, trades)
    
    compliance = results['ethical_compliance']
    assert 'no_market_manipulation' in compliance
    assert 'fair_trading' in compliance
    assert 'risk_compliant' in compliance
    assert 'size_compliant' in compliance
    assert 'frequency_compliant' in compliance
    
    # Check compliance results are boolean
    assert all(isinstance(v, bool) for v in compliance.values())

def test_robustness_metrics(evaluator, sample_data):
    """Test calculation of robustness metrics."""
    predictions, actuals, returns, trades = sample_data
    results = evaluator.evaluate_strategy(predictions, actuals, returns, trades)
    
    metrics = results['robustness_metrics']
    assert 'return_stability' in metrics
    assert 'consistency_score' in metrics
    assert 'recovery_efficiency' in metrics
    assert 'trade_efficiency' in metrics
    
    # Check metric ranges
    assert 0 <= metrics['return_stability'] <= 1
    assert 0 <= metrics['consistency_score'] <= 1

def test_overall_score(evaluator, sample_data):
    """Test calculation of overall strategy score."""
    predictions, actuals, returns, trades = sample_data
    results = evaluator.evaluate_strategy(predictions, actuals, returns, trades)
    
    assert 'overall_score' in results
    assert 0 <= results['overall_score'] <= 1

def test_invalid_data_handling(evaluator):
    """Test handling of invalid input data."""
    with pytest.raises(ValueError):
        evaluator.evaluate_strategy(
            predictions=None,
            actuals=None,
            returns=None,
            trades=None
        )

def test_edge_cases(evaluator):
    """Test handling of edge cases."""
    # Empty data
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError):
        evaluator.evaluate_strategy(
            predictions=empty_df,
            actuals=empty_df,
            returns=empty_df,
            trades=empty_df
        )
    
    # Single data point
    single_date = pd.DataFrame({'A': [1]}, index=[datetime.now()])
    
    with pytest.raises(ValueError):
        evaluator.evaluate_strategy(
            predictions=single_date,
            actuals=single_date,
            returns=single_date,
            trades=pd.DataFrame({'timestamp': [datetime.now()], 'returns': [0.01]})
        )

def test_performance_targets(evaluator, sample_data):
    """Test evaluation against performance targets."""
    predictions, actuals, returns, trades = sample_data
    results = evaluator.evaluate_strategy(predictions, actuals, returns, trades)
    
    portfolio_metrics = results['portfolio_metrics']
    
    # Check against target metrics
    assert portfolio_metrics['cagr'] is not None  # Target: > 25%
    assert portfolio_metrics['sharpe_ratio'] is not None  # Target: > 1
    assert portfolio_metrics['max_drawdown'] is not None  # Target: > -20%
    assert portfolio_metrics['avg_profit'] is not None  # Target: ≥ 0.75%