# Code Examples

## Overview

This directory contains practical examples and code snippets demonstrating the usage of the Enhanced Trading Strategy System.

## Basic Examples

### 1. Setting Up an Agent
```python
from src.agents import VolatilityAgent

# Initialize agent
agent = VolatilityAgent(
    name="volatility_tracker",
    config={
        "window_size": 20,
        "threshold": 0.1
    }
)

# Process market data
results = agent.process(market_data)
print(f"Volatility metrics: {results['volatility_metrics']}")
```

### 2. Data Processing
```python
from src.data_processors import DataPipeline

# Set up pipeline
pipeline = DataPipeline(
    validators=[DataValidator()],
    transformers=[FeatureTransformer()]
)

# Process data
processed_data = pipeline.process(raw_data)
print(f"Processed {len(processed_data)} records")
```

## Intermediate Examples

### 1. Training a Model
```python
from src.training import ModelTrainer
from src.utils.config import load_config

# Load configuration
config = load_config('FinTSB/configs/fintsb_lstm.yaml')

# Initialize trainer
trainer = ModelTrainer(config)

# Train model
model = trainer.train(train_data, val_data)
predictions = model.predict(test_data)
```

### 2. Strategy Evaluation
```python
from src.strategies import StrategyEvaluator
from src.utils.metrics import calculate_metrics

# Initialize evaluator
evaluator = StrategyEvaluator()

# Evaluate strategy
results = evaluator.evaluate_strategy(
    predictions=predictions,
    actuals=actuals,
    returns=returns,
    trades=trades
)

# Print metrics
print(f"Sharpe Ratio: {results['portfolio_metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['portfolio_metrics']['max_drawdown']:.2%}")
```

## Advanced Examples

### 1. Multi-Agent System
```python
from src.agents import AgentSystem
from src.agents.volatility import VolatilityAgent
from src.agents.signal import SignalExtractionAgent

# Create agent system
system = AgentSystem()

# Add agents
system.add_agent(VolatilityAgent(name="vol_agent"))
system.add_agent(SignalExtractionAgent(name="signal_agent"))

# Process data through all agents
results = system.process(market_data)
```

### 2. Custom Strategy Implementation
```python
from src.strategies import BaseStrategy
from src.utils.indicators import calculate_indicators

class MomentumStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20):
        super().__init__()
        self.lookback = lookback
    
    def generate_signals(self, data):
        # Calculate momentum
        momentum = calculate_indicators(
            data,
            indicator_type="momentum",
            params={"window": self.lookback}
        )
        
        # Generate signals
        signals = (momentum > 0).astype(int)
        return signals

# Use strategy
strategy = MomentumStrategy(lookback=20)
signals = strategy.generate_signals(market_data)
```

## Integration Examples

### 1. Using FinTSB with Custom Evaluation
```python
from src.training import EnhancedFinTSBTrainer
from src.strategies import StrategyEvaluator

# Initialize components
trainer = EnhancedFinTSBTrainer(config_path="configs/custom_lstm.yaml")
evaluator = StrategyEvaluator()

# Train and evaluate
model = trainer.train()
predictions = trainer.predict(test_data)
evaluation = evaluator.evaluate_strategy(predictions, actuals)
```

### 2. Real-time Data Processing
```python
from src.data_processors import RealTimeProcessor
from src.agents import VolatilityAgent
import asyncio

async def process_stream():
    # Initialize components
    processor = RealTimeProcessor()
    agent = VolatilityAgent()
    
    async for data in processor.stream():
        # Process real-time data
        result = agent.process(data)
        
        # Take action based on results
        if result['volatility_metrics']['current_regime'] == 'high':
            await reduce_exposure()

# Run the processor
asyncio.run(process_stream())
```

## Testing Examples

### Unit Testing
```python
import pytest
from src.agents import VolatilityAgent

def test_volatility_calculation():
    agent = VolatilityAgent()
    data = generate_test_data()
    
    results = agent.process(data)
    assert 'volatility_metrics' in results
    assert 0 <= results['volatility_metrics']['current_volatility'] <= 1
```

### Integration Testing
```python
def test_end_to_end_pipeline():
    # Set up pipeline
    pipeline = create_test_pipeline()
    
    # Process data
    results = pipeline.run(test_data)
    
    # Verify results
    assert_metrics_valid(results)
    verify_no_data_leakage(results)
```

## Additional Resources

- [API Documentation](../api/index.md)
- [Architecture Guide](../architecture/index.md)
- [Tutorials](../tutorials/index.md)

## Contributing Examples

When adding new examples:
1. Include complete, working code
2. Add comments and explanations
3. Show example output
4. Include error handling
5. Follow code style guidelines