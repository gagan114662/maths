"""
Tests for the Volatility Assessment Agent.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.agents.volatility_agent import VolatilityAssessmentAgent

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'symbol': 'AAPL',
        'close': np.random.normal(100, 10, 100),
        'high': np.random.normal(105, 10, 100),
        'low': np.random.normal(95, 10, 100),
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    return data

@pytest.fixture
def volatility_agent():
    """Create a VolatilityAssessmentAgent instance."""
    return VolatilityAssessmentAgent(name="TestVolatilityAgent", weight=1.0)

def test_agent_initialization(volatility_agent):
    """Test agent initialization."""
    assert volatility_agent.name == "TestVolatilityAgent"
    assert volatility_agent.weight == 1.0
    assert volatility_agent.state == {}
    assert volatility_agent.history == {}

def test_required_columns(volatility_agent):
    """Test that the agent specifies required columns."""
    required_cols = volatility_agent._get_required_columns()
    assert isinstance(required_cols, list)
    assert all(col in required_cols for col in ['date', 'symbol', 'close', 'high', 'low', 'volume'])

def test_process_valid_data(volatility_agent, sample_data):
    """Test processing of valid market data."""
    results = volatility_agent.process(sample_data)
    
    # Check results structure
    assert isinstance(results, dict)
    assert 'volatility_metrics' in results
    assert 'noise_metrics' in results
    assert 'stationarity' in results
    assert 'autocorrelation' in results
    assert 'regime_analysis' in results

def test_volatility_metrics_calculation(volatility_agent, sample_data):
    """Test calculation of volatility metrics."""
    results = volatility_agent.process(sample_data)
    metrics = results['volatility_metrics']
    
    # Check required metrics
    assert 'short_term_volatility' in metrics
    assert 'medium_term_volatility' in metrics
    assert 'long_term_volatility' in metrics
    assert 'parkinson_volatility' in metrics
    
    # Check metric values
    assert 0 <= metrics['parkinson_volatility'] <= 1
    assert all(0 <= metrics[f'{term}_term_volatility'] <= 1 
              for term in ['short', 'medium', 'long'])

def test_noise_metrics_calculation(volatility_agent, sample_data):
    """Test calculation of noise metrics."""
    results = volatility_agent.process(sample_data)
    metrics = results['noise_metrics']
    
    # Check required metrics
    assert 'noise_ratio' in metrics
    assert 'skewness' in metrics
    assert 'kurtosis' in metrics
    assert 'jarque_bera_stat' in metrics
    
    # Check metric ranges
    assert 0 <= metrics['noise_ratio'] <= 1
    assert isinstance(metrics['jarque_bera_stat'], float)

def test_stationarity_analysis(volatility_agent, sample_data):
    """Test stationarity analysis."""
    results = volatility_agent.process(sample_data)
    stationarity = results['stationarity']
    
    # Check required fields
    assert 'adf_statistic' in stationarity
    assert 'p_value' in stationarity
    assert 'is_stationary' in stationarity
    assert 'critical_values' in stationarity
    
    # Check types
    assert isinstance(stationarity['is_stationary'], bool)
    assert isinstance(stationarity['p_value'], float)

def test_regime_analysis(volatility_agent, sample_data):
    """Test volatility regime analysis."""
    results = volatility_agent.process(sample_data)
    regime = results['regime_analysis']
    
    # Check required fields
    assert 'current_regime' in regime
    assert 'volatility_z_score' in regime
    assert 'regime_confidence' in regime
    
    # Check regime classification
    assert regime['current_regime'] in ['low_volatility', 'normal', 'high_volatility']
    assert 0 <= regime['regime_confidence'] <= 1

def test_confidence_score(volatility_agent, sample_data):
    """Test confidence score calculation."""
    confidence = volatility_agent.get_confidence_score(sample_data)
    
    # Check confidence score range
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1

def test_agent_update(volatility_agent):
    """Test agent update mechanism."""
    feedback = {
        'weight_adjustment': 0.8,
        'lookback_periods': {
            'short': 10,
            'medium': 30,
            'long': 126
        }
    }
    
    volatility_agent.update(feedback)
    
    # Check state updates
    assert volatility_agent.weight == 0.8
    assert volatility_agent.lookback_periods['short'] == 10
    assert 'last_feedback' in volatility_agent.state
    assert 'last_feedback_time' in volatility_agent.state

def test_invalid_data_handling(volatility_agent):
    """Test handling of invalid data."""
    invalid_data = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'close': np.random.normal(100, 10, 10)  # Missing required columns
    })
    
    # Check that validation fails
    assert not volatility_agent.validate_data(invalid_data)
    
    # Check that processing returns empty results
    results = volatility_agent.process(invalid_data)
    assert results == {}

def test_state_persistence(volatility_agent, tmp_path):
    """Test agent state saving and loading."""
    # Set some state
    volatility_agent.state = {'test_key': 'test_value'}
    volatility_agent.history = {'test_action': [{'timestamp': '2024-01-01', 'result': 'test'}]}
    
    # Save state
    save_path = tmp_path / "agent_state"
    save_path.mkdir()
    volatility_agent.save_state(save_path)
    
    # Create new agent and load state
    new_agent = VolatilityAssessmentAgent(name="TestVolatilityAgent")
    new_agent.load_state(save_path)
    
    # Check state was preserved
    assert new_agent.state['test_key'] == 'test_value'
    assert new_agent.history['test_action'][0]['result'] == 'test'