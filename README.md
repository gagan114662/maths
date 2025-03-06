# AI Co-Scientist Trading System

## Overview

An AI-powered system that develops and manages sophisticated trading strategies using scientific methodology and machine learning.

## Key Features

### Advanced QuantConnect Integration ✅
- Multi-asset class strategy generation for cross-market opportunities
- Market regime detection for adaptive strategy behavior
- Alternative data source integration for enhanced signals
- Neural network-based feature discovery module
- Automated factor analysis module for alpha discovery

### Market Outperformance Focus ✅
- Benchmark-relative performance scoring
- Dynamic asset allocation based on relative strengths
- Sophisticated benchmark-tracking with dynamic beta adjustment
- Stress testing against historical market regimes
- Statistical arbitrage modes for market-neutral performance

### Enhanced Risk Management ✅
- Tail risk analysis with extreme value theory
- Conditional drawdown-at-risk metrics
- Regime-dependent position sizing algorithms
- Sophisticated options-based hedging strategies

### Optimization Enhancements ✅
- Bayesian optimization for parameter tuning
- Genetic algorithm for strategy evolution
- Ensemble learning for strategy combination
- Transfer learning across asset classes
- Reinforcement learning for dynamic adaptation

### Statistical Arbitrage Capabilities
The system includes a sophisticated statistical arbitrage module that provides:
- Automated pair selection using correlation and cointegration analysis
- Dynamic spread calculation and monitoring
- Market-neutral position sizing
- Risk management and performance tracking
- Real-time signal generation

### Market Structure Analysis ✅
- Market regime classification using unsupervised learning (clustering, HMMs)
- Graph Neural Networks for capturing complex market relationships
- Causal discovery for market relationships

## Usage Examples

### Statistical Arbitrage

```python
from src.strategies.statistical_arbitrage import StatisticalArbitrageStrategy

# Initialize the strategy
stat_arb = StatisticalArbitrageStrategy(
    correlation_threshold=0.7,
    zscore_entry=2.0,
    zscore_exit=0.0,
    lookback_period=252
)

# Select pairs from universe
pairs = stat_arb.select_pairs(price_data)

# Calculate spreads and generate signals
spreads = stat_arb.calculate_spreads(price_data)
signals = stat_arb.generate_signals()

# Get position sizes
positions = stat_arb.calculate_position_sizes(portfolio_value, price_data)
```

### Deep Learning Integration

```python
from src.models import TemporalFusionTransformer
from src.features import NeuralFeatureDiscovery

# Initialize neural feature discovery
feature_discoverer = NeuralFeatureDiscovery()
features = feature_discoverer.discover_features(market_data)

# Initialize and train TFT model
model = TemporalFusionTransformer(
    num_features=len(features),
    hidden_size=64
)
model.fit(features, targets)
```

## Installation

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
python -m pytest tests/
```

## Configuration

Strategy parameters can be configured in config files or passed directly to strategy constructors.

## Contributing

Please see CONTRIBUTING.md for guidelines on contributing to this project.

## License

This project is licensed under the terms specified in LICENSE file.