"""
Tests for trading strategy templates.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.templates import (
    StrategyFactory,
    MomentumStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
    VolatilityStrategy,
    MomentumParameters,
    MeanReversionParameters,
    TrendFollowingParameters,
    VolatilityParameters
)

@pytest.fixture
def sample_data():
    """Create sample market data."""
    dates = pd.date_range(start='2025-01-01', end='2025-02-01', freq='D')
    
    # Create price series with known pattern
    prices = np.array([100.0] * len(dates))
    prices[10:20] *= 1.1  # Uptrend
    prices[30:40] *= 0.9  # Downtrend
    prices[50:60] = np.linspace(prices[49], prices[49]*1.2, 10)  # Volatility
    
    return pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': [1000000] * len(dates)
    }, index=dates)

def test_momentum_strategy():
    """Test momentum strategy signals."""
    # Create strategy
    params = MomentumParameters(
        name="Test Momentum",
        description="Test strategy",
        lookback=10,
        threshold=0.05,
        position_size=0.1
    )
    strategy = MomentumStrategy(params)
    
    # Create test data
    dates = pd.date_range(start='2025-01-01', periods=30)
    data = pd.DataFrame({
        'close': [100] * 10 + [110] * 10 + [90] * 10
    }, index=dates)
    
    # Generate signals
    signals = strategy.generate_signals(data)
    positions = strategy.calculate_position_sizes(signals, data)
    
    # Verify signals
    assert signals[15] == 1  # Buy signal after uptrend
    assert signals[25] == -1  # Sell signal after downtrend
    assert positions[15] == params.position_size
    assert positions[25] == -params.position_size

def test_mean_reversion_strategy():
    """Test mean reversion strategy signals."""
    # Create strategy
    params = MeanReversionParameters(
        name="Test Mean Reversion",
        description="Test strategy",
        window=10,
        std_dev=2.0,
        position_size=0.1
    )
    strategy = MeanReversionStrategy(params)
    
    # Create test data with mean reversion pattern
    dates = pd.date_range(start='2025-01-01', periods=30)
    prices = [100] * 30
    prices[10:15] = [120, 115, 110, 105, 100]  # Price spike and reversion
    data = pd.DataFrame({'close': prices}, index=dates)
    
    # Generate signals
    signals = strategy.generate_signals(data)
    positions = strategy.calculate_position_sizes(signals, data)
    
    # Verify signals
    assert signals[10] == -1  # Sell signal at peak
    assert signals[14] == 1  # Buy signal at reversion
    assert positions[10] == -params.position_size
    assert positions[14] == params.position_size

def test_trend_following_strategy():
    """Test trend following strategy signals."""
    # Create strategy
    params = TrendFollowingParameters(
        name="Test Trend Following",
        description="Test strategy",
        fast_ma=5,
        slow_ma=10,
        position_size=0.1,
        trend_strength=0.02
    )
    strategy = TrendFollowingStrategy(params)
    
    # Create test data with trend
    dates = pd.date_range(start='2025-01-01', periods=30)
    prices = np.linspace(100, 150, 30)  # Upward trend
    data = pd.DataFrame({'close': prices}, index=dates)
    
    # Generate signals
    signals = strategy.generate_signals(data)
    positions = strategy.calculate_position_sizes(signals, data)
    
    # Verify signals
    assert signals[15] == 1  # Buy signal in uptrend
    assert positions[15] == params.position_size
    
    # Test downtrend
    prices = np.linspace(150, 100, 30)  # Downward trend
    data = pd.DataFrame({'close': prices}, index=dates)
    signals = strategy.generate_signals(data)
    
    assert signals[15] == -1  # Sell signal in downtrend

def test_volatility_strategy():
    """Test volatility strategy signals."""
    # Create strategy
    params = VolatilityParameters(
        name="Test Volatility",
        description="Test strategy",
        calculation_window=10,
        entry_threshold=0.02,
        position_size=0.1,
        volatility_scaling=True
    )
    strategy = VolatilityStrategy(params)
    
    # Create test data with volatility pattern
    dates = pd.date_range(start='2025-01-01', periods=30)
    prices = [100] * 30
    # Add volatility spike
    prices[15:20] = [100, 105, 95, 110, 90]
    data = pd.DataFrame({'close': prices}, index=dates)
    
    # Generate signals
    signals = strategy.generate_signals(data)
    positions = strategy.calculate_position_sizes(signals, data)
    
    # Verify signals
    assert signals[17] == 1  # Entry signal during high volatility
    assert abs(positions[17]) <= params.position_size  # Scaled position

def test_strategy_factory():
    """Test strategy factory creation."""
    # Test each strategy type
    configs = [
        {
            'type': 'momentum',
            'name': 'Test Momentum',
            'lookback': 10,
            'threshold': 0.05,
            'position_size': 0.1
        },
        {
            'type': 'mean_reversion',
            'name': 'Test Mean Reversion',
            'window': 20,
            'std_dev': 2.0,
            'position_size': 0.1
        },
        {
            'type': 'trend_following',
            'name': 'Test Trend',
            'fast_ma': 5,
            'slow_ma': 20,
            'position_size': 0.1,
            'trend_strength': 0.02
        },
        {
            'type': 'volatility',
            'name': 'Test Volatility',
            'calculation_window': 10,
            'entry_threshold': 0.02,
            'position_size': 0.1
        }
    ]
    
    for config in configs:
        strategy = StrategyFactory.create_strategy(config)
        assert strategy is not None
        assert strategy.parameters.name == config['name']

def test_invalid_strategy_type():
    """Test error handling for invalid strategy type."""
    config = {
        'type': 'invalid_type',
        'name': 'Invalid Strategy'
    }
    
    with pytest.raises(ValueError):
        StrategyFactory.create_strategy(config)

@pytest.mark.parametrize("strategy_type,param_class", [
    ('momentum', MomentumParameters),
    ('mean_reversion', MeanReversionParameters),
    ('trend_following', TrendFollowingParameters),
    ('volatility', VolatilityParameters)
])
def test_strategy_parameters(strategy_type, param_class):
    """Test strategy parameters for each strategy type."""
    config = {
        'type': strategy_type,
        'name': f'Test {strategy_type}',
        'description': 'Test strategy',
        # Add required parameters for each type
        'lookback': 10,
        'threshold': 0.05,
        'window': 20,
        'std_dev': 2.0,
        'fast_ma': 5,
        'slow_ma': 20,
        'trend_strength': 0.02,
        'calculation_window': 10,
        'entry_threshold': 0.02,
        'position_size': 0.1
    }
    
    strategy = StrategyFactory.create_strategy(config)
    assert isinstance(strategy.parameters, param_class)
    assert strategy.parameters.name == config['name']
    assert strategy.parameters.description == 'Test strategy'

def test_strategy_consistency(sample_data):
    """Test consistency of strategy signals and positions."""
    # Test each strategy type
    strategies = [
        MomentumStrategy(MomentumParameters(
            name="Momentum", description="Test",
            lookback=10, threshold=0.05, position_size=0.1
        )),
        MeanReversionStrategy(MeanReversionParameters(
            name="Mean Reversion", description="Test",
            window=20, std_dev=2.0, position_size=0.1
        )),
        TrendFollowingStrategy(TrendFollowingParameters(
            name="Trend", description="Test",
            fast_ma=5, slow_ma=20, position_size=0.1,
            trend_strength=0.02
        )),
        VolatilityStrategy(VolatilityParameters(
            name="Volatility", description="Test",
            calculation_window=10, entry_threshold=0.02,
            position_size=0.1
        ))
    ]
    
    for strategy in strategies:
        signals = strategy.generate_signals(sample_data)
        positions = strategy.calculate_position_sizes(signals, sample_data)
        
        # Check signal properties
        assert len(signals) == len(sample_data)
        assert signals.abs().max() <= 1
        
        # Check position properties
        assert len(positions) == len(sample_data)
        assert positions.abs().max() <= strategy.parameters.position_size
        
        # Check alignment
        assert all(signals.index == sample_data.index)
        assert all(positions.index == sample_data.index)