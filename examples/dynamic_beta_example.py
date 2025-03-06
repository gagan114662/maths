#!/usr/bin/env python
"""
Example script demonstrating the dynamic beta adjustment module
for sophisticated benchmark-tracking with dynamic beta adjustment.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the dynamic beta manager and benchmark tracker
from src.performance_scoring.dynamic_beta import (
    DynamicBetaManager, BenchmarkTracker, BetaMethod, BetaTarget
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(start_date='2020-01-01', end_date='2022-12-31', n_assets=20):
    """
    Generate sample market data for demonstration.
    
    Args:
        start_date: Start date
        end_date: End date
        n_assets: Number of assets
        
    Returns:
        Tuple of (benchmark_data, constituent_data, regime_data)
    """
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Generate market regimes
    regimes = ['bull', 'bear', 'low_volatility', 'high_volatility']
    regime_periods = []
    
    # Split the date range into regime periods
    period_length = len(dates) // len(regimes)
    for i in range(len(regimes)):
        start_idx = i * period_length
        end_idx = (i + 1) * period_length if i < len(regimes) - 1 else len(dates)
        regime_periods.append((dates[start_idx:end_idx], regimes[i]))
    
    # Create regime series
    regime_data = pd.Series(index=dates, data=None)
    for period_dates, regime in regime_periods:
        regime_data.loc[period_dates] = regime
    
    # Generate benchmark data
    np.random.seed(42)  # For reproducibility
    
    # Base parameters
    market_drift = 0.00025  # Daily drift ~6.5% annually
    base_volatility = 0.01  # Base daily volatility
    
    # Generate benchmark prices
    benchmark_prices = [100.0]  # Starting price
    
    for i in range(1, len(dates)):
        date = dates[i]
        current_regime = regime_data[date]
        
        # Adjust parameters based on regime
        if current_regime == 'bull':
            drift = market_drift * 2.0
            vol = base_volatility * 1.2
        elif current_regime == 'bear':
            drift = -market_drift
            vol = base_volatility * 1.5
        elif current_regime == 'low_volatility':
            drift = market_drift * 0.5
            vol = base_volatility * 0.7
        elif current_regime == 'high_volatility':
            drift = market_drift * 0.1
            vol = base_volatility * 2.0
        else:
            drift = market_drift
            vol = base_volatility
        
        # Generate random return
        daily_return = np.random.normal(drift, vol)
        benchmark_prices.append(benchmark_prices[-1] * (1 + daily_return))
    
    # Create benchmark DataFrame
    benchmark_data = pd.DataFrame({
        'close': benchmark_prices
    }, index=dates)
    
    # Calculate benchmark returns
    benchmark_data['returns'] = benchmark_data['close'].pct_change().fillna(0)
    
    # Generate constituent data
    constituent_data = {}
    
    for i in range(n_assets):
        asset_id = i + 1
        constituent_name = f'Asset_{asset_id}'
        
        # Set asset characteristics
        beta = 0.5 + np.random.rand()  # Beta between 0.5 and 1.5
        alpha = np.random.normal(0, 0.0005)  # Daily alpha
        specific_vol = np.random.uniform(0.005, 0.015)  # Specific volatility
        
        # Generate prices
        asset_prices = [100.0]  # Starting price
        
        for j in range(1, len(dates)):
            date = dates[j]
            benchmark_return = benchmark_data.loc[date, 'returns']
            
            # Model return = alpha + beta * benchmark_return + specific_return
            specific_return = np.random.normal(0, specific_vol)
            
            # For certain assets, make them more responsive in specific regimes
            current_regime = regime_data[date]
            
            if current_regime == 'bull' and asset_id % 4 == 0:
                # These assets outperform in bull markets
                daily_return = alpha * 1.5 + beta * benchmark_return * 1.2 + specific_return
            elif current_regime == 'bear' and asset_id % 5 == 0:
                # These assets are more defensive in bear markets
                daily_return = alpha + beta * 0.6 * benchmark_return + specific_return
            elif current_regime == 'low_volatility' and asset_id % 3 == 0:
                # These assets do well in low volatility
                daily_return = alpha * 1.3 + beta * benchmark_return + specific_return * 0.7
            elif current_regime == 'high_volatility' and asset_id % 2 == 0:
                # These assets handle high volatility better
                daily_return = alpha + beta * benchmark_return + specific_return * 0.5
            else:
                daily_return = alpha + beta * benchmark_return + specific_return
            
            asset_prices.append(asset_prices[-1] * (1 + daily_return))
        
        # Create asset DataFrame
        asset_data = pd.DataFrame({
            'close': asset_prices
        }, index=dates)
        
        # Calculate returns
        asset_data['returns'] = asset_data['close'].pct_change().fillna(0)
        
        constituent_data[constituent_name] = asset_data
    
    # Convert regime data to DataFrame
    regime_df = pd.DataFrame({
        'regime': regime_data
    }, index=dates)
    
    return benchmark_data, constituent_data, regime_df

def main():
    """
    Main function demonstrating dynamic beta adjustment.
    """
    # Create output directory for results
    output_dir = os.path.join(os.path.dirname(__file__), 'dynamic_beta_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample data
    logger.info("Generating sample data...")
    benchmark_data, constituent_data, regime_data = generate_sample_data(
        start_date='2020-01-01',
        end_date='2022-12-31',
        n_assets=20
    )
    
    # 1. Demonstrate DynamicBetaManager with different methods
    logger.info("Demonstrating different beta calculation methods...")
    
    # Create beta managers with different methods
    beta_managers = {
        'Standard Beta': DynamicBetaManager(
            calculation_method=BetaMethod.STANDARD,
            beta_target_strategy=BetaTarget.BENCHMARK
        ),
        'Rolling Beta': DynamicBetaManager(
            calculation_method=BetaMethod.ROLLING,
            beta_target_strategy=BetaTarget.BENCHMARK,
            window_size=63
        ),
        'Kalman Filter Beta': DynamicBetaManager(
            calculation_method=BetaMethod.KALMAN,
            beta_target_strategy=BetaTarget.BENCHMARK
        ),
        'Regime-Conditional Beta': DynamicBetaManager(
            calculation_method=BetaMethod.REGIME_CONDITIONAL,
            beta_target_strategy=BetaTarget.BENCHMARK
        )
    }
    
    # Set data for each beta manager
    for name, manager in beta_managers.items():
        # Sample portfolio - equal weight of first 5 constituents
        sample_portfolio_assets = list(constituent_data.keys())[:5]
        sample_weights = {asset: 0.2 for asset in sample_portfolio_assets}
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=benchmark_data.index)
        
        for asset, weight in sample_weights.items():
            asset_returns = constituent_data[asset]['returns']
            portfolio_returns += weight * asset_returns
        
        # Set data for beta manager
        manager.set_data(
            strategy_returns=portfolio_returns,
            benchmark_returns=benchmark_data['returns'],
            regime_data=regime_data['regime'] if name == 'Regime-Conditional Beta' else None
        )
        
        # Calculate historical betas
        historical_betas = manager.calculate_historical_betas()
        
        # Generate and save beta report
        beta_report = manager.generate_beta_report(
            output_path=os.path.join(output_dir, name.replace(' ', '_').lower()),
            include_plots=True
        )
        
        logger.info(f"Generated beta report for {name}")
    
    # 2. Demonstrate BenchmarkTracker with different beta targets
    logger.info("Demonstrating benchmark tracking with different beta targets...")
    
    beta_targets = {
        'Market-Neutral': BetaTarget.MARKET_NEUTRAL,
        'Low Beta': BetaTarget.LOW_BETA,
        'Benchmark Tracking': BetaTarget.BENCHMARK,
        'High Beta': BetaTarget.HIGH_BETA,
        'Dynamic Beta': BetaTarget.DYNAMIC
    }
    
    for name, target in beta_targets.items():
        # Create beta manager with the target
        beta_manager = DynamicBetaManager(
            calculation_method=BetaMethod.ROLLING,
            beta_target_strategy=target,
            window_size=63
        )
        
        # Create benchmark tracker
        tracker = BenchmarkTracker(
            benchmark_symbol="BENCHMARK",
            tracking_portfolio_size=10,
            beta_manager=beta_manager
        )
        
        # Load benchmark data
        tracker.load_benchmark_data(benchmark_data)
        
        # Add constituent data
        for symbol, data in constituent_data.items():
            tracker.add_constituent_data(symbol, data)
        
        # Construct tracking portfolio
        tracking_portfolio = tracker.construct_tracking_portfolio(method='optimization')
        
        # Calculate tracking performance
        tracking_performance = tracker.calculate_tracking_performance()
        
        # Generate and save tracking report
        tracking_report = tracker.generate_tracking_report(
            output_path=os.path.join(output_dir, f"tracking_{name.replace(' ', '_').lower()}"),
            include_plots=True
        )
        
        logger.info(f"Generated tracking report for {name}: Beta={tracking_performance.get('beta', 'N/A'):.4f}, TE={tracking_performance.get('tracking_error', 'N/A'):.4f}")
    
    # 3. Compare dynamic vs static beta tracking
    logger.info("Comparing dynamic vs static beta tracking...")
    
    # Create figure to compare all tracking portfolios
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot benchmark
    benchmark_returns = benchmark_data['returns']
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    normalized_benchmark = benchmark_cumulative / benchmark_cumulative.iloc[0]
    ax.plot(normalized_benchmark.index, normalized_benchmark, 
           label='Benchmark', linewidth=2, color='black')
    
    # Colors for different strategies
    colors = plt.cm.tab10(np.linspace(0, 1, len(beta_targets)))
    
    # Plot regime background
    regimes = regime_data['regime'].unique()
    regime_colors = {
        'bull': 'lightgreen',
        'bear': 'lightcoral',
        'low_volatility': 'lightblue',
        'high_volatility': 'lightyellow'
    }
    
    # Add colored background for regimes
    for regime in regimes:
        regime_dates = regime_data.index[regime_data['regime'] == regime]
        if len(regime_dates) > 0:
            # Find consecutive date ranges
            breaks = np.where(np.diff(regime_dates) > pd.Timedelta(days=1))[0]
            start_idx = 0
            
            for break_idx in list(breaks) + [len(regime_dates) - 1]:
                regime_start = regime_dates[start_idx]
                regime_end = regime_dates[break_idx]
                
                ax.axvspan(regime_start, regime_end, 
                         alpha=0.2, color=regime_colors.get(regime, 'lightgray'))
                
                start_idx = break_idx + 1
    
    # Plot each tracking portfolio's performance
    for i, (name, target) in enumerate(beta_targets.items()):
        # Load tracking report
        report_path = os.path.join(output_dir, f"tracking_{name.replace(' ', '_').lower()}", "tracking_report.json")
        
        if os.path.exists(report_path):
            import json
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            # Get performance data
            perf = report_data.get('performance', {})
            beta = perf.get('beta', "N/A")
            
            # Generate synthetic portfolio values based on beta for visualization
            # In a real scenario, we would use actual portfolio returns
            synthetic_returns = benchmark_returns * beta
            if isinstance(beta, (int, float)):
                portfolio_cumulative = (1 + synthetic_returns).cumprod()
                normalized_portfolio = portfolio_cumulative / portfolio_cumulative.iloc[0]
                
                # Plot normalized portfolio values
                ax.plot(normalized_portfolio.index, normalized_portfolio, 
                       label=f"{name} (Beta={beta:.2f})", 
                       linewidth=2, color=colors[i])
    
    # Add legend and labels
    ax.set_title('Performance Comparison: Dynamic vs Static Beta Tracking')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save comparison figure
    comparison_path = os.path.join(output_dir, "strategy_comparison.png")
    fig.savefig(comparison_path)
    plt.close(fig)
    
    logger.info(f"Saved strategy comparison to {comparison_path}")
    
    # Print summary of results
    print("\nDynamic Beta Adjustment Results:")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Beta':<10} {'Tracking Error':<15} {'Info Ratio':<12} {'Return':<10}")
    print("-" * 80)
    
    for name, target in beta_targets.items():
        report_path = os.path.join(output_dir, f"tracking_{name.replace(' ', '_').lower()}", "tracking_report.json")
        
        if os.path.exists(report_path):
            import json
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            perf = report_data.get('performance', {})
            beta = perf.get('beta', "N/A")
            te = perf.get('tracking_error', "N/A")
            ir = perf.get('information_ratio', "N/A")
            ret = perf.get('tracking_annual_return', "N/A")
            
            if isinstance(beta, (int, float)):
                beta = f"{beta:.2f}"
            if isinstance(te, (int, float)):
                te = f"{te:.4f}"
            if isinstance(ir, (int, float)):
                ir = f"{ir:.2f}"
            if isinstance(ret, (int, float)):
                ret = f"{ret*100:.2f}%"
            
            print(f"{name:<20} {beta:<10} {te:<15} {ir:<12} {ret:<10}")
    
    print("-" * 80)
    print(f"\nAll results and visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()