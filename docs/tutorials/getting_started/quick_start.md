# Quick Start Guide

This guide will help you get started with the Enhanced Trading Strategy System in less than 30 minutes.

## Prerequisites

- Python 3.8 or higher
- Git
- 4GB RAM minimum
- Internet connection

## Installation

1. Clone the repository:
```bash
git clone --recursive https://github.com/username/enhanced-trading-system.git
cd enhanced-trading-system
```

2. Run the quick setup script:
```bash
./quick_setup.sh
```

## First Steps

### 1. Set Up API Credentials

```bash
# Run the interactive setup
./download_eastmoney_data.py --interactive
```

### 2. Download Sample Data

```bash
# Download test dataset
make download
```

### 3. Run Your First Strategy

```python
from src.agents import VolatilityAgent
from src.data_processors import DataPipeline
from src.strategies import SimpleMovingAverageStrategy

# Initialize components
agent = VolatilityAgent()
pipeline = DataPipeline()
strategy = SimpleMovingAverageStrategy()

# Load and process data
data = pipeline.process(market_data)

# Get volatility analysis
volatility_results = agent.process(data)

# Generate trading signals
signals = strategy.generate_signals(data)

print(f"Volatility regime: {volatility_results['regime']}")
print(f"Signal strength: {signals.mean():.2f}")
```

## Basic Usage Examples

### 1. Market Analysis

```python
# Analyze market conditions
results = agent.process(market_data)

if results['regime'] == 'high_volatility':
    print("High volatility detected - reducing position sizes")
elif results['regime'] == 'low_volatility':
    print("Low volatility - normal operations")
```

### 2. Strategy Evaluation

```python
from src.strategies import StrategyEvaluator

# Initialize evaluator
evaluator = StrategyEvaluator()

# Evaluate strategy
results = evaluator.evaluate_strategy(
    predictions=signals,
    actuals=market_data,
    returns=strategy_returns
)

print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### 3. Model Training

```python
from src.training import ModelTrainer

# Train a model
trainer = ModelTrainer(config_path='FinTSB/configs/fintsb_lstm.yaml')
model = trainer.train(train_data)
predictions = model.predict(test_data)
```

## Common Operations

### Running Tests

```bash
# Run all tests
./run_tests.py

# Run specific test
./run_tests.py tests/agents/test_volatility_agent.py
```

### Updating Dependencies

```bash
# Update all dependencies
make update-deps
```

### Checking System Status

```bash
# Run validation
./validate_setup.py

# Generate debug report
./generate_debug_report.py
```

## Next Steps

1. Read the [Documentation Standards](../architecture/documentation.md)
2. Explore [Example Code](../examples/index.md)
3. Review [API Documentation](../api/index.md)
4. Check [Tutorials](../tutorials/index.md)

## Common Issues

### 1. Import Errors
```bash
# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 2. Permission Issues
```bash
# Make scripts executable
chmod +x *.py *.sh
```

### 3. Data Download Fails
```bash
# Check API credentials
./download_eastmoney_data.py --interactive

# Verify network connection
ping quantapi.eastmoney.com
```

## Getting Help

1. Generate debug report:
```bash
./generate_debug_report.py
```

2. Check logs:
```bash
tail -f logs/system.log
```

3. Contact support:
- Open GitHub issue
- Join Discord community
- Email support team

## Maintaining Your Installation

1. Regular updates:
```bash
git pull
make update-deps
```

2. Verify system:
```bash
./validate_setup.py
```

3. Run tests:
```bash
make test
```

Remember to check the [full documentation](../index.md) for detailed information about each component and feature.