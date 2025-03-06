# AI Co-Scientist System for Trading Strategy Development: Simple Overview

## What This System Does

The AI Co-Scientist system uses artificial intelligence to automatically develop profitable trading strategies by following the scientific method.

## How It Works (Step-by-Step)

1. **Goal Setting**: You tell the system what kind of trading strategy you want (e.g., "Create a strategy with high returns and low risk").

2. **Data Analysis**: The system analyzes historical market data looking for patterns and anomalies.

3. **Hypothesis Creation**: The system creates scientific hypotheses about what might work in the markets (e.g., "Stocks that drop significantly after earnings reports tend to rebound within 3 days").

4. **Strategy Design**: Based on these hypotheses, it designs specific trading strategies with clear rules.

5. **Backtesting**: It tests these strategies on 10+ years of historical data to see if they would have worked.

6. **Risk Assessment**: It evaluates how risky each strategy is, looking at maximum drawdowns, volatility, etc.

7. **Strategy Ranking**: It compares all the strategies it developed and ranks them based on performance.

8. **Strategy Evolution**: It takes the best strategies and tries to improve them further through iteration.

9. **Documentation**: It documents everything about the winning strategies, including why they work.

10. **Google Sheets Integration**: All results are automatically updated in Google Sheets so you can track performance.

## The Team of AI Agents

The system uses multiple specialized AI agents that work together:

- **Generation Agent**: Creates hypotheses and strategy designs
- **Backtesting Agent**: Tests strategies on historical data
- **Risk Assessment Agent**: Evaluates potential risks
- **Ranking Agent**: Compares different strategies
- **Evolution Agent**: Refines strategies
- **Meta-Review Agent**: Reviews the entire process

## Key Features

- Uses real market data with 10+ years of history
- Follows scientific methodology with hypothesis testing
- Produces strategies with clear entry/exit rules
- Considers transaction costs and slippage
- Provides comprehensive performance metrics
- Runs automatically without human intervention
- Continuously learns and improves

## How to Run the System

```bash
./run_deepseek.sh --plan-name "My Strategy Plan" --goal "Develop a strategy with a cagr of above 25%, sharpe ratio >1, max drawdown <20%" --market us_equities --time-horizon daily --use-mathematricks --use-fintsb --interactive
```

## Monitoring Progress

You can watch the development in real-time:
- View output in the terminal
- Check Google Sheets for updated results
- Review log files in the logs/ directory
- Examine generated strategies in generated_strategies/

## The Output

The final output includes:
- Complete trading strategy with entry/exit rules
- Performance metrics (CAGR, Sharpe ratio, drawdown, etc.)
- Risk analysis
- Explanation of why the strategy works
- Transaction history from backtesting
- Charts and visualizations