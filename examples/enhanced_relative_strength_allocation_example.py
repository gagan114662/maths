#!/usr/bin/env python
"""
Example script demonstrating the enhanced relative strength allocation module
for dynamic asset allocation based on relative strengths.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from enum import Enum

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the EnhancedRelativeStrengthAllocator
from src.allocation.dynamic import EnhancedRelativeStrengthAllocator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(start_date='2020-01-01', end_date='2022-12-31', n_assets=10, n_regimes=4):
    """
    Generate sample price and regime data for demonstration.
    
    Args:
        start_date: Start date
        end_date: End date
        n_assets: Number of assets
        n_regimes: Number of market regimes
        
    Returns:
        Tuple of (asset_data, benchmark_data, regime_data)
    """
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Create asset names
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    
    # Generate synthetic price data
    np.random.seed(42)  # For reproducibility
    
    # Create regime data
    regime_changes = [
        dates[0],  # Start with regime 0
        dates[int(len(dates) * 0.25)],  # Switch to regime 1 at 25%
        dates[int(len(dates) * 0.5)],   # Switch to regime 2 at 50%
        dates[int(len(dates) * 0.75)],  # Switch to regime 3 at 75%
    ]
    
    regime_names = ['bull', 'bear', 'low_volatility', 'high_volatility']
    regime_data = pd.Series(index=dates, data=None)
    
    for i in range(len(regime_changes)):
        if i < len(regime_changes) - 1:
            mask = (dates >= regime_changes[i]) & (dates < regime_changes[i+1])
        else:
            mask = dates >= regime_changes[i]
            
        regime_data[mask] = regime_names[i % len(regime_names)]
    
    # Convert to DataFrame
    regime_df = pd.DataFrame({'regime': regime_data})
    
    # Generate asset price data with regime-dependent behavior
    asset_data = {}
    
    for asset_name in asset_names:
        # Base parameters
        drift = 0.0002  # Daily drift (0.02% per day)
        volatility = 0.01  # Base volatility
        
        # Initial price
        price = 100.0
        prices = [price]
        
        # Generate prices for each date
        for i in range(1, len(dates)):
            date = dates[i]
            
            # Get current regime
            current_regime = regime_data[date]
            
            # Adjust parameters based on regime
            if current_regime == 'bull':
                regime_drift = drift * 2
                regime_vol = volatility * 1.0
            elif current_regime == 'bear':
                regime_drift = -drift
                regime_vol = volatility * 1.5
            elif current_regime == 'low_volatility':
                regime_drift = drift * 0.5
                regime_vol = volatility * 0.7
            elif current_regime == 'high_volatility':
                regime_drift = drift * 0.1
                regime_vol = volatility * 2.0
            else:
                regime_drift = drift
                regime_vol = volatility
                
            # Add asset-specific bias
            asset_id = int(asset_name.split('_')[1])
            
            # Some assets perform better in specific regimes
            if current_regime == 'bull' and asset_id % 4 == 0:
                asset_drift = regime_drift * 1.5  # Assets 4, 8 outperform in bull markets
            elif current_regime == 'bear' and asset_id % 5 == 0:
                asset_drift = regime_drift * 0.5  # Assets 5, 10 outperform in bear markets
            elif current_regime == 'low_volatility' and asset_id % 3 == 0:
                asset_drift = regime_drift * 1.3  # Assets 3, 6, 9 outperform in low vol
            elif current_regime == 'high_volatility' and asset_id % 2 == 0:
                asset_drift = regime_drift * 0.8  # Even-numbered assets resist high vol
            else:
                asset_drift = regime_drift
            
            # Random return with regime-dependent drift and volatility
            daily_return = np.random.normal(asset_drift, regime_vol)
            price = price * (1 + daily_return)
            prices.append(price)
        
        # Create DataFrame with prices
        asset_df = pd.DataFrame({
            'close': prices
        }, index=dates)
        
        asset_data[asset_name] = asset_df
    
    # Create benchmark data (average of all assets)
    benchmark_prices = []
    
    for i in range(len(dates)):
        avg_price = np.mean([asset_data[asset]['close'].iloc[i] for asset in asset_names])
        benchmark_prices.append(avg_price)
    
    benchmark_df = pd.DataFrame({
        'close': benchmark_prices
    }, index=dates)
    
    return asset_data, benchmark_df, regime_df

def main():
    """
    Main function demonstrating enhanced relative strength allocation.
    """
    # Create output directory for results
    output_dir = os.path.join(os.path.dirname(__file__), 'allocation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample data
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    logger.info("Generating sample data...")
    asset_data, benchmark_data, regime_data = generate_sample_data(
        start_date=start_date,
        end_date=end_date,
        n_assets=15
    )
    
    # Create allocators with different configurations
    logger.info("Creating allocators with different configurations...")
    
    # 1. Basic relative strength allocator
    basic_allocator = EnhancedRelativeStrengthAllocator(
        benchmark_data=benchmark_data,
        lookback_periods=[21, 63, 126, 252],
        min_rs_score=0.0,
        ranking_method=EnhancedRelativeStrengthAllocator.RankingMethod.SIMPLE,
        weighting_scheme=EnhancedRelativeStrengthAllocator.WeightingScheme.RANK_INVERSE,
        max_allocation_pct=0.3,
        rebalance_frequency=21
    )
    
    # 2. Momentum-adjusted relative strength allocator
    momentum_allocator = EnhancedRelativeStrengthAllocator(
        benchmark_data=benchmark_data,
        lookback_periods=[21, 63, 126, 252],
        min_rs_score=0.0,
        ranking_method=EnhancedRelativeStrengthAllocator.RankingMethod.MOMENTUM_ADJUSTED,
        weighting_scheme=EnhancedRelativeStrengthAllocator.WeightingScheme.SCORE_PROPORTIONAL,
        max_allocation_pct=0.3,
        rebalance_frequency=21
    )
    
    # 3. Volatility-weighted relative strength allocator
    volatility_allocator = EnhancedRelativeStrengthAllocator(
        benchmark_data=benchmark_data,
        lookback_periods=[21, 63, 126, 252],
        min_rs_score=0.0,
        ranking_method=EnhancedRelativeStrengthAllocator.RankingMethod.VOLATILITY_WEIGHTED,
        weighting_scheme=EnhancedRelativeStrengthAllocator.WeightingScheme.VOLATILITY_ADJUSTED,
        max_allocation_pct=0.3,
        rebalance_frequency=21
    )
    
    # 4. Regime-aware allocator
    regime_allocator = EnhancedRelativeStrengthAllocator(
        benchmark_data=benchmark_data,
        lookback_periods=[21, 63, 126, 252],
        min_rs_score=0.0,
        ranking_method=EnhancedRelativeStrengthAllocator.RankingMethod.SIMPLE,
        weighting_scheme=EnhancedRelativeStrengthAllocator.WeightingScheme.RANK_INVERSE,
        regime_data=regime_data,
        max_allocation_pct=0.3,
        rebalance_frequency=21
    )
    
    # 5. Cluster-based allocator
    cluster_allocator = EnhancedRelativeStrengthAllocator(
        benchmark_data=benchmark_data,
        lookback_periods=[21, 63, 126, 252],
        min_rs_score=0.0,
        ranking_method=EnhancedRelativeStrengthAllocator.RankingMethod.SIMPLE,
        weighting_scheme=EnhancedRelativeStrengthAllocator.WeightingScheme.CLUSTER_BASED,
        correlation_threshold=0.6,
        max_cluster_allocation=0.4,
        max_allocation_pct=0.3,
        rebalance_frequency=21
    )
    
    # Load asset data into allocators
    for allocator in [basic_allocator, momentum_allocator, volatility_allocator,
                      regime_allocator, cluster_allocator]:
        for asset, data in asset_data.items():
            allocator.add_asset_data(asset, data)
    
    # Run backtest for each allocator
    allocators = {
        'Basic RS': basic_allocator,
        'Momentum-Adjusted RS': momentum_allocator,
        'Volatility-Weighted RS': volatility_allocator,
        'Regime-Aware RS': regime_allocator,
        'Cluster-Based RS': cluster_allocator
    }
    
    backtest_results = {}
    
    for name, allocator in allocators.items():
        logger.info(f"Running backtest for {name}...")
        if name == 'Regime-Aware RS':
            # For regime allocator, use the backtest_relative_strength method
            results = allocator.backtest_relative_strength(
                start_date=start_date,
                end_date=end_date,
                top_n=10,
                initial_capital=10000.0
            )
        else:
            # For other allocators, use the backtest_relative_strength method
            results = allocator.backtest_relative_strength(
                start_date=start_date,
                end_date=end_date,
                top_n=10,
                initial_capital=10000.0
            )
        
        backtest_results[name] = results
        
        # Plot backtest results
        fig = allocator.plot_backtest_results(results, title=f'{name} Allocation Strategy')
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_backtest_{timestamp}.png")
        fig.savefig(plot_path)
        plt.close(fig)
        
        logger.info(f"Saved backtest plot to {plot_path}")
        
        # For regime allocator, analyze performance attribution
        if name == 'Regime-Aware RS' or name == 'Cluster-Based RS':
            attribution = allocator.analyze_performance_attribution(results, detailed=True)
            
            # Plot attribution by regime or cluster
            if 'regime_attribution' in attribution and attribution['regime_attribution']:
                plt.figure(figsize=(10, 6))
                regimes = list(attribution['regime_attribution'].keys())
                values = list(attribution['regime_attribution'].values())
                plt.pie(values, labels=regimes, autopct='%1.1f%%')
                plt.title(f'Performance Attribution by Regime - {name}')
                
                regime_plot_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_regime_attribution_{timestamp}.png")
                plt.savefig(regime_plot_path)
                plt.close()
                
                logger.info(f"Saved regime attribution plot to {regime_plot_path}")
            
            # For cluster allocator, plot cluster attribution
            if 'cluster_attribution' in attribution and attribution['cluster_attribution']:
                plt.figure(figsize=(10, 6))
                clusters = list(attribution['cluster_attribution'].keys())
                values = list(attribution['cluster_attribution'].values())
                plt.bar(clusters, values)
                plt.title(f'Performance Attribution by Cluster - {name}')
                plt.ylabel('Contribution')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                cluster_plot_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_cluster_attribution_{timestamp}.png")
                plt.savefig(cluster_plot_path)
                plt.close()
                
                logger.info(f"Saved cluster attribution plot to {cluster_plot_path}")
    
    # Compare performance of different allocators
    plt.figure(figsize=(12, 8))
    
    for name, results in backtest_results.items():
        dates = results['dates']
        portfolio_values = results['portfolio_values']
        # Normalize to starting value of 1.0
        normalized_values = [v / portfolio_values[0] for v in portfolio_values]
        plt.plot(dates, normalized_values, label=name, linewidth=2)
    
    # Plot benchmark for comparison
    if 'Basic RS' in backtest_results and backtest_results['Basic RS']['benchmark_values']:
        benchmark_values = backtest_results['Basic RS']['benchmark_values']
        normalized_benchmark = [v / benchmark_values[0] for v in benchmark_values]
        plt.plot(dates, normalized_benchmark, label='Benchmark', linewidth=2, linestyle='--', color='black')
    
    plt.title('Performance Comparison of Different Allocation Strategies')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(output_dir, f"strategy_comparison_{timestamp}.png")
    plt.savefig(comparison_path)
    plt.close()
    
    logger.info(f"Saved strategy comparison plot to {comparison_path}")
    
    # Print performance summary
    print("\nPerformance Summary:")
    print("-" * 80)
    print(f"{'Strategy':<25} {'Total Return':<15} {'Annual Return':<15} {'Sharpe':<10} {'Max DD':<10} {'Info Ratio':<10}")
    print("-" * 80)
    
    for name, results in backtest_results.items():
        total_return = results['total_return'] * 100
        annual_return = results['annualized_return'] * 100
        sharpe = results['sharpe_ratio']
        max_dd = results['max_drawdown'] * 100
        info_ratio = results['information_ratio']
        
        print(f"{name:<25} {total_return:>6.2f}% {annual_return:>10.2f}% {sharpe:>10.2f} {max_dd:>9.2f}% {info_ratio:>10.2f}")
    
    print("-" * 80)

if __name__ == "__main__":
    main()