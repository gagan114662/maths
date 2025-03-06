# Implementing a Custom Trading Strategy

This tutorial walks through the process of implementing a custom trading strategy using the Enhanced Trading Strategy System.

## Overview

We'll create a momentum-based strategy that:
1. Analyzes price momentum
2. Considers volatility
3. Generates trading signals
4. Implements risk management

## Prerequisites

- Basic Python knowledge
- System installed and configured
- Sample data downloaded

## Implementation Steps

### 1. Create Strategy Class

```python
from src.strategies import BaseStrategy
from src.utils.indicators import calculate_momentum
from src.utils.validation import validate_data
from typing import Dict, Any

class MomentumStrategy(BaseStrategy):
    """
    A momentum-based trading strategy with volatility adjustment.
    
    Attributes:
        momentum_window: Lookback period for momentum calculation
        volatility_window: Window for volatility calculation
        entry_threshold: Signal threshold for entry
        position_size_limit: Maximum position size
    """
    
    def __init__(
        self,
        momentum_window: int = 20,
        volatility_window: int = 10,
        entry_threshold: float = 0.5,
        position_size_limit: float = 0.1
    ):
        """Initialize strategy parameters."""
        super().__init__()
        self.momentum_window = momentum_window
        self.volatility_window = volatility_window
        self.entry_threshold = entry_threshold
        self.position_size_limit = position_size_limit
        
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if self.momentum_window <= 0:
            raise ValueError("Momentum window must be positive")
        if self.volatility_window <= 0:
            raise ValueError("Volatility window must be positive")
        if not 0 < self.entry_threshold <= 1:
            raise ValueError("Entry threshold must be between 0 and 1")
        if not 0 < self.position_size_limit <= 1:
            raise ValueError("Position size limit must be between 0 and 1")
        return True
```

### 2. Implement Signal Generation

```python
def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate trading signals based on momentum and volatility.
    
    Args:
        data: DataFrame with market data
        
    Returns:
        Dictionary containing signals and metrics
    """
    # Validate input data
    validate_data(data, required_columns=['close', 'high', 'low', 'volume'])
    
    # Calculate momentum
    momentum = calculate_momentum(
        data['close'],
        window=self.momentum_window
    )
    
    # Calculate volatility
    volatility = data['close'].pct_change().rolling(
        window=self.volatility_window
    ).std()
    
    # Generate raw signals
    raw_signals = (momentum > self.entry_threshold).astype(int)
    
    # Adjust for volatility
    position_sizes = self._calculate_position_sizes(
        raw_signals,
        volatility
    )
    
    return {
        'signals': raw_signals,
        'position_sizes': position_sizes,
        'metrics': {
            'momentum': momentum,
            'volatility': volatility
        }
    }
```

### 3. Implement Position Sizing

```python
def _calculate_position_sizes(
    self,
    signals: pd.Series,
    volatility: pd.Series
) -> pd.Series:
    """
    Calculate position sizes based on volatility.
    
    Args:
        signals: Raw trading signals
        volatility: Volatility series
        
    Returns:
        Adjusted position sizes
    """
    # Base position sizes on signals
    position_sizes = signals.copy()
    
    # Adjust for volatility
    vol_adjustment = (1 / volatility).clip(0, 1)
    position_sizes = position_sizes * vol_adjustment
    
    # Apply position size limit
    position_sizes = position_sizes.clip(
        -self.position_size_limit,
        self.position_size_limit
    )
    
    return position_sizes
```

### 4. Add Risk Management

```python
def apply_risk_management(
    self,
    signals: Dict[str, Any],
    market_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Apply risk management rules to signals.
    
    Args:
        signals: Generated signals and metrics
        market_data: Current market data
        
    Returns:
        Risk-adjusted signals
    """
    # Calculate risk metrics
    risk_metrics = self._calculate_risk_metrics(market_data)
    
    # Adjust signals based on risk
    if risk_metrics['market_risk'] > 0.8:
        signals['position_sizes'] *= 0.5
    
    # Add stop-loss orders
    signals['stop_loss'] = self._calculate_stop_levels(
        market_data,
        signals['position_sizes']
    )
    
    return signals

def _calculate_risk_metrics(
    self,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate various risk metrics."""
    return {
        'market_risk': self._estimate_market_risk(market_data),
        'liquidity_risk': self._estimate_liquidity_risk(market_data),
        'volatility_risk': self._estimate_volatility_risk(market_data)
    }
```

### 5. Implement Performance Tracking

```python
def track_performance(
    self,
    signals: Dict[str, Any],
    market_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Track strategy performance metrics.
    
    Args:
        signals: Generated signals
        market_data: Market data
        
    Returns:
        Performance metrics
    """
    returns = self._calculate_returns(signals, market_data)
    
    return {
        'cumulative_return': returns.cumsum(),
        'sharpe_ratio': self._calculate_sharpe_ratio(returns),
        'max_drawdown': self._calculate_drawdown(returns),
        'win_rate': self._calculate_win_rate(returns)
    }
```

## Usage Example

```python
# Initialize strategy
strategy = MomentumStrategy(
    momentum_window=20,
    volatility_window=10,
    entry_threshold=0.5,
    position_size_limit=0.1
)

# Load market data
data = load_market_data('AAPL', '2024-01-01', '2024-02-01')

# Generate signals
signals = strategy.generate_signals(data)

# Apply risk management
risk_adjusted_signals = strategy.apply_risk_management(
    signals,
    data
)

# Track performance
performance = strategy.track_performance(
    risk_adjusted_signals,
    data
)

# Print results
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
```

## Testing the Strategy

```python
import pytest
from src.utils.testing import generate_test_data

def test_momentum_strategy():
    # Initialize strategy
    strategy = MomentumStrategy()
    
    # Generate test data
    test_data = generate_test_data(days=100)
    
    # Generate signals
    signals = strategy.generate_signals(test_data)
    
    # Verify signal properties
    assert isinstance(signals, dict)
    assert 'signals' in signals
    assert 'position_sizes' in signals
    assert 'metrics' in signals
    
    # Verify signal values
    assert signals['signals'].abs().max() <= 1
    assert signals['position_sizes'].abs().max() <= strategy.position_size_limit
```

## Best Practices

1. Parameter Validation
   - Validate all input parameters
   - Check data requirements
   - Handle edge cases

2. Risk Management
   - Implement position sizing
   - Add stop-loss levels
   - Monitor exposure

3. Performance Tracking
   - Calculate key metrics
   - Monitor drawdowns
   - Track win rate

4. Testing
   - Unit tests
   - Integration tests
   - Backtesting

## Next Steps

1. Explore [Advanced Strategy Topics](../advanced/strategies.md)
2. Learn about [Portfolio Management](../advanced/portfolio.md)
3. Study [Risk Management](../advanced/risk.md)
4. Review [Performance Optimization](../advanced/optimization.md)

## Common Issues

1. Data Quality
   ```python
   # Verify data quality
   from src.utils.validation import verify_data_quality
   verify_data_quality(market_data)
   ```

2. Performance Issues
   ```python
   # Profile strategy
   from src.utils.profiling import profile_strategy
   profile_strategy(strategy, market_data)
   ```

3. Risk Management
   ```python
   # Monitor risk metrics
   from src.utils.risk import monitor_risk_metrics
   monitor_risk_metrics(strategy, market_data)