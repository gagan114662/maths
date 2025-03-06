# Market Regime Detection and Adaptive Trading Strategies

This system implements sophisticated market regime detection and automatically adapts trading strategies to the current market conditions. By detecting the prevailing market regime (such as bull market, bear market, high volatility, etc.), the system optimizes strategy parameters to maximize performance in different market environments.

## Components

The system consists of three main components:

1. **Market Regime Detector**: Uses unsupervised learning algorithms (HMM, GMM, K-means, hierarchical clustering) to identify and classify market regimes based on price data and various features.

2. **Regime-Aware Strategy Adapter**: Adapts trading strategies to the detected market regime by adjusting parameters, position sizing, and risk management rules based on the characteristics of the current regime.

3. **QuantConnect Integration**: Converts the adapted strategies to QuantConnect format for backtesting and live trading.

## Market Regime Detection

The market regime detector uses the following features to identify market regimes:

- Returns and volatility
- Volume patterns
- Technical indicators (RSI, MACD, etc.)
- Price action relative to moving averages
- Volatility measures (ATR, Bollinger Bands width)

Multiple unsupervised learning algorithms are available:

- **Hidden Markov Models (HMM)**: Models the market as a system that transitions between unobserved (hidden) states.
- **Gaussian Mixture Models (GMM)**: Identifies clusters of data points that represent different market regimes.
- **K-means Clustering**: Divides market conditions into distinct clusters.
- **Hierarchical Clustering**: Builds a hierarchy of clusters representing different market regimes.

## Strategy Adaptation

Once a market regime is detected, the system adapts the trading strategy in several ways:

1. **Parameter Adjustment**: Adjusts strategy parameters based on the regime (e.g., looser entry criteria in bull markets, tighter stop losses in high volatility regimes).

2. **Position Sizing**: Modifies position sizing rules based on the regime (e.g., larger positions in favorable regimes, smaller positions in high-risk regimes).

3. **Risk Management**: Adapts risk management rules to the regime (e.g., wider stops in high volatility regimes, lower maximum drawdown thresholds in bear markets).

## Usage

### Prerequisites

- Python 3.7+
- Required packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `hmmlearn`, `yfinance`
- Optional: QuantConnect API access for backtesting and live trading

### Installation

```bash
pip install numpy pandas scikit-learn matplotlib seaborn hmmlearn yfinance
```

For QuantConnect integration (optional):
```bash
pip install quantconnect
```

### Basic Usage

1. Detect market regime:

```bash
python adapt_and_run_strategy.py --strategy strategies/momentum_rsi_strategy.json --method hmm --n-regimes 4 --symbol SPY --start 2018-01-01
```

2. Adapt strategy to detected regime and visualize:

```bash
python adapt_and_run_strategy.py --strategy strategies/momentum_rsi_strategy.json --method hmm --n-regimes 4 --symbol SPY --start 2018-01-01 --visualize
```

3. Generate QuantConnect algorithm from adapted strategy:

```bash
python adapt_and_run_strategy.py --strategy strategies/momentum_rsi_strategy.json --method hmm --n-regimes 4 --symbol SPY --start 2018-01-01 --qc-backtest-start 2018-01-01 --qc-backtest-end 2023-12-31
```

### Advanced Usage

For custom regime configurations:

```bash
python adapt_and_run_strategy.py --strategy strategies/momentum_rsi_strategy.json --method hmm --n-regimes 4 --symbol SPY --start 2018-01-01 --regime-config configs/custom_regime_config.json
```

For QuantConnect live trading preparation:

```bash
python adapt_and_run_strategy.py --strategy strategies/momentum_rsi_strategy.json --method hmm --n-regimes 4 --symbol SPY --start 2018-01-01 --qc-live
```

## Configuration

### Market Regime Detector Configuration

The market regime detector can be configured with various parameters:

- **method**: Algorithm to use for regime detection ('hmm', 'gmm', 'kmeans', 'hierarchical')
- **n_regimes**: Number of regimes to detect
- **features**: List of features to use for regime detection
- **lookback**: Number of days to use for training/detection

### Strategy Adaptation Configuration

Strategy adaptation can be customized through regime-specific configurations:

- **parameters**: Strategy-specific parameters for each regime
- **position_sizing**: Position sizing rules for each regime
- **risk_management**: Risk management rules for each regime

## QuantConnect Integration

To use the QuantConnect integration, you need to set up your QuantConnect API credentials:

1. Set environment variables:
```bash
export QC_USER_ID=your_user_id
export QC_API_TOKEN=your_api_token
```

2. Or create a configuration file:
```json
{
  "user_id": "your_user_id",
  "token": "your_api_token"
}
```

Then run with the config:
```bash
python adapt_and_run_strategy.py --strategy strategies/momentum_rsi_strategy.json --method hmm --qc-config path/to/config.json
```

## Visualization

The system provides visualizations to help understand the detected market regimes and their impact on strategy performance:

1. **Market Regime Visualization**: Shows the price chart with different regimes highlighted in different colors.

2. **Performance Metrics by Regime**: Displays key performance metrics (returns, Sharpe ratio, win rate, volatility) for each regime.

3. **Regime Transition Matrix**: Shows the probabilities of transitioning from one regime to another.

## Benefits

Using this regime-aware adaptive system provides several advantages:

1. **Improved Performance**: By adapting to different market conditions, strategies can maintain performance across varying regimes.

2. **Reduced Drawdowns**: Risk management rules are tightened in unfavorable regimes, potentially reducing drawdowns.

3. **Better Risk Management**: Position sizing is adjusted based on the current market regime, potentially improving risk-adjusted returns.

4. **Enhanced Strategy Robustness**: Strategies become more robust by automatically adapting to changing market conditions.

## Extending the System

The system can be extended in several ways:

1. **Additional Regime Detection Algorithms**: Implement other unsupervised learning algorithms for regime detection.

2. **Custom Feature Engineering**: Add domain-specific features for improved regime detection.

3. **Strategy-Specific Adaptations**: Create custom adaptation rules for specific strategy types.

4. **Multi-Asset Regime Detection**: Extend the system to detect regimes across multiple asset classes.

5. **Reinforcement Learning Integration**: Use reinforcement learning to optimize adaptation rules based on historical performance.

## API Documentation

For detailed API documentation, please refer to the docstrings in the source code.