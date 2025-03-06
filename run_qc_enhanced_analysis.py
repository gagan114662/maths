#!/usr/bin/env python
"""
QuantConnect Enhanced Analysis

This script performs advanced analysis on QuantConnect backtests:
1. Market regime detection and classification
2. Benchmark-relative performance analysis
3. Regime-specific strategy optimization
4. Factor-based return attribution
5. Comprehensive reporting with visualizations

Usage:
    python run_qc_enhanced_analysis.py --results qc_results/ --market-data data/market_data.csv --output qc_analysis_output/
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from pathlib import Path

# Import custom modules
from quantconnect_market_regimes import MarketRegimeClassifier, QuantConnectRegimeIntegration
from benchmark_relative_performance import BenchmarkPerformanceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qc_enhanced_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class EnhancedAnalysisRunner:
    """Class for running enhanced analysis on QuantConnect backtest results."""
    
    def __init__(self, market_data_path=None, benchmark_data_path=None):
        """
        Initialize the enhanced analysis runner.
        
        Args:
            market_data_path (str): Path to market data CSV file
            benchmark_data_path (str): Path to benchmark data CSV file
        """
        self.market_data_path = market_data_path
        self.benchmark_data_path = benchmark_data_path or market_data_path
        self.market_data = None
        self.benchmark_data = None
        self.regime_integration = None
        self.benchmark_analyzer = None
        self.backtest_results = []
        self.output_dir = None
    
    def load_data(self):
        """
        Load market and benchmark data.
        
        Returns:
            bool: True if data loaded successfully
        """
        try:
            # Load market data
            if self.market_data_path:
                self.market_data = pd.read_csv(self.market_data_path, parse_dates=True, index_col=0)
                logger.info(f"Loaded market data from {self.market_data_path} with {len(self.market_data)} rows")
            else:
                logger.error("No market data path specified")
                return False
            
            # Load benchmark data (can be the same as market data)
            if self.benchmark_data_path:
                self.benchmark_data = pd.read_csv(self.benchmark_data_path, parse_dates=True, index_col=0)
                logger.info(f"Loaded benchmark data from {self.benchmark_data_path} with {len(self.benchmark_data)} rows")
            else:
                self.benchmark_data = self.market_data
                logger.info("Using market data as benchmark data")
            
            # Initialize components
            self.regime_integration = QuantConnectRegimeIntegration(self.market_data)
            self.benchmark_analyzer = BenchmarkPerformanceAnalyzer(benchmark_data=self.benchmark_data)
            
            # Check required columns for market data
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in self.market_data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns in market data: {missing_columns}")
                return False
            
            # Calculate benchmark returns for performance analysis
            if 'close' in self.benchmark_data.columns:
                self.benchmark_analyzer.benchmark_returns = self.benchmark_data['close'].pct_change().dropna()
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_backtest_results(self, results_path):
        """
        Load backtest results from file or directory.
        
        Args:
            results_path (str): Path to backtest results JSON file or directory
            
        Returns:
            list: Loaded backtest results
        """
        results = []
        
        try:
            # Handle directory of results
            if os.path.isdir(results_path):
                for file_name in os.listdir(results_path):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(results_path, file_name)
                        try:
                            with open(file_path, 'r') as f:
                                result = json.load(f)
                                results.append(result)
                        except Exception as e:
                            logger.error(f"Error loading result file {file_path}: {e}")
            # Handle single result file
            elif os.path.isfile(results_path) and results_path.endswith('.json'):
                try:
                    with open(results_path, 'r') as f:
                        result = json.load(f)
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error loading result file {results_path}: {e}")
            else:
                logger.error(f"Invalid results path: {results_path}")
                return []
            
            logger.info(f"Loaded {len(results)} backtest results")
            self.backtest_results = results
            return results
        
        except Exception as e:
            logger.error(f"Error loading backtest results: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def extract_equity_curves(self):
        """
        Extract equity curves from backtest results.
        
        Returns:
            DataFrame: Equity curves for all backtest results
        """
        if not self.backtest_results:
            logger.error("No backtest results available")
            return pd.DataFrame()
        
        # Create dictionary to store equity curves
        equity_data = {}
        
        for result in self.backtest_results:
            try:
                # Extract strategy name
                strategy_name = result.get('strategy_name', 'Unknown Strategy')
                
                # Look for equity curve data in different possible locations
                equity_curve = None
                
                if 'statistics' in result and 'equity' in result['statistics']:
                    # Extract raw equity curve data
                    equity_curve_data = result['statistics']['equity']
                    
                    # Convert to pandas Series with date index
                    if isinstance(equity_curve_data, dict):
                        dates = [datetime.fromisoformat(d) for d in equity_curve_data.keys()]
                        values = list(equity_curve_data.values())
                        equity_curve = pd.Series(values, index=dates, name=strategy_name)
                
                # Alternatively, look for returns that can be cumulated
                elif 'returns' in result:
                    returns = result['returns']
                    if isinstance(returns, dict):
                        dates = [datetime.fromisoformat(d) for d in returns.keys()]
                        values = list(returns.values())
                        returns_series = pd.Series(values, index=dates)
                        equity_curve = (1 + returns_series).cumprod()
                        equity_curve.name = strategy_name
                
                # If found, add to the dictionary
                if equity_curve is not None:
                    equity_data[strategy_name] = equity_curve
                    logger.info(f"Extracted equity curve for {strategy_name} with {len(equity_curve)} data points")
                else:
                    logger.warning(f"Could not find equity curve data for {strategy_name}")
            
            except Exception as e:
                logger.error(f"Error extracting equity curve: {e}")
                continue
        
        # If no equity curves were found, return empty DataFrame
        if not equity_data:
            logger.error("No equity curves found in backtest results")
            return pd.DataFrame()
        
        # Combine all equity curves into a single DataFrame
        equity_df = pd.DataFrame(equity_data)
        
        return equity_df
    
    def run_market_regime_analysis(self, num_regimes=4, method="kmeans"):
        """
        Run market regime analysis.
        
        Args:
            num_regimes (int): Number of regimes to identify
            method (str): Classification method ('kmeans' or 'hmm')
            
        Returns:
            dict: Market regime analysis results
        """
        if self.market_data is None:
            logger.error("No market data available for regime analysis")
            return {}
        
        if self.regime_integration is None:
            self.regime_integration = QuantConnectRegimeIntegration(self.market_data)
        
        try:
            # Train regime model
            classifier = self.regime_integration.train_regime_model(
                method=method,
                num_regimes=num_regimes
            )
            
            if classifier is None:
                logger.error("Failed to train regime classifier")
                return {}
            
            # Generate regime report
            regime_report = self.regime_integration.generate_regime_report(self.output_dir)
            
            # Extract equity curves from backtest results
            equity_curves = self.extract_equity_curves()
            
            # If we have backtest results, analyze performance by regime
            if not equity_curves.empty:
                # Extract regime classifications
                regime_series = pd.Series(index=self.market_data.index)
                regimes = classifier.predict(self.market_data)
                regime_series.loc[self.market_data.index[-len(regimes):]] = regimes
                
                # Analyze each strategy's performance across regimes
                regime_performance = {}
                
                for strategy_name, equity_curve in equity_curves.items():
                    # Calculate returns from equity curve
                    returns = equity_curve.pct_change().dropna()
                    
                    # Analyze performance by regime
                    performance = self.regime_integration.analyze_regime_performance(
                        {
                            'equity_curve': equity_curve,
                            'strategy_name': strategy_name
                        }
                    )
                    
                    regime_performance[strategy_name] = performance
                
                regime_report['strategy_regime_performance'] = regime_performance
            
            logger.info("Completed market regime analysis")
            return regime_report
        
        except Exception as e:
            logger.error(f"Error in market regime analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def run_benchmark_relative_analysis(self):
        """
        Run benchmark-relative performance analysis.
        
        Returns:
            dict: Benchmark performance analysis results
        """
        if self.benchmark_analyzer is None or self.benchmark_analyzer.benchmark_returns is None:
            logger.error("Benchmark analyzer not initialized or no benchmark returns available")
            return {}
        
        if not self.backtest_results:
            logger.error("No backtest results available for benchmark analysis")
            return {}
        
        try:
            # Extract equity curves from backtest results
            equity_curves = self.extract_equity_curves()
            
            if equity_curves.empty:
                logger.error("No equity curves available for benchmark analysis")
                return {}
            
            # Analyze each strategy relative to the benchmark
            benchmark_analysis = {}
            
            for strategy_name, equity_curve in equity_curves.items():
                # Calculate returns from equity curve
                returns = equity_curve.pct_change().dropna()
                
                # Set strategy returns for analysis
                self.benchmark_analyzer.set_strategy_returns(returns)
                
                # Generate performance report
                strategy_output_dir = os.path.join(self.output_dir, strategy_name) if self.output_dir else None
                if strategy_output_dir:
                    os.makedirs(strategy_output_dir, exist_ok=True)
                
                report = self.benchmark_analyzer.generate_performance_report(strategy_output_dir)
                benchmark_analysis[strategy_name] = report
            
            # Create summary of benchmark-relative performance
            summary = {}
            for strategy_name, report in benchmark_analysis.items():
                if 'outperformance_summary' in report:
                    summary[strategy_name] = report['outperformance_summary']
            
            benchmark_analysis['summary'] = summary
            
            # Save overall summary if output directory is specified
            if self.output_dir:
                summary_path = os.path.join(self.output_dir, "benchmark_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
            
            logger.info("Completed benchmark-relative performance analysis")
            return benchmark_analysis
        
        except Exception as e:
            logger.error(f"Error in benchmark-relative analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def create_strategy_dashboard(self):
        """
        Create a comprehensive dashboard of strategy performance across regimes and relative to benchmark.
        
        Returns:
            dict: Dashboard data and file paths
        """
        if not self.backtest_results:
            logger.error("No backtest results available for dashboard")
            return {}
        
        try:
            # Create dashboard directory
            dashboard_dir = os.path.join(self.output_dir, "dashboard") if self.output_dir else None
            if dashboard_dir:
                os.makedirs(dashboard_dir, exist_ok=True)
            
            # Extract equity curves
            equity_curves = self.extract_equity_curves()
            
            if equity_curves.empty:
                logger.error("No equity curves available for dashboard")
                return {}
            
            # Create performance comparison plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot each strategy's equity curve
            for strategy_name, equity_curve in equity_curves.items():
                ax.plot(equity_curve.index, equity_curve, label=strategy_name)
            
            # Plot benchmark if available
            if self.benchmark_data is not None and 'close' in self.benchmark_data.columns:
                benchmark_returns = self.benchmark_data['close'].pct_change().dropna()
                benchmark_equity = (1 + benchmark_returns).cumprod()
                ax.plot(benchmark_equity.index, benchmark_equity, label='Benchmark', 
                       linestyle='--', color='black', linewidth=2)
            
            ax.set_title('Strategy Performance Comparison')
            ax.set_ylabel('Growth of $1')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot if dashboard directory is specified
            if dashboard_dir:
                plot_path = os.path.join(dashboard_dir, "strategy_comparison.png")
                fig.savefig(plot_path)
                plt.close(fig)
            
            # Create regime-overlay plot if regime data is available
            if self.regime_integration and self.regime_integration.regime_classifier.current_regime is not None:
                try:
                    # Create figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
                    
                    # Plot equity curves on top subplot
                    for strategy_name, equity_curve in equity_curves.items():
                        ax1.plot(equity_curve.index, equity_curve, label=strategy_name)
                    
                    # Plot benchmark if available
                    if self.benchmark_data is not None and 'close' in self.benchmark_data.columns:
                        benchmark_returns = self.benchmark_data['close'].pct_change().dropna()
                        benchmark_equity = (1 + benchmark_returns).cumprod()
                        ax1.plot(benchmark_equity.index, benchmark_equity, label='Benchmark', 
                               linestyle='--', color='black', linewidth=2)
                    
                    ax1.set_title('Strategy Performance by Market Regime')
                    ax1.set_ylabel('Growth of $1')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Get regime classifications
                    regimes = self.regime_integration.regime_classifier.predict(self.market_data)
                    regime_dates = self.market_data.index[-len(regimes):]
                    
                    # Define colors for regimes
                    colors = ['#90EE90', '#FF9999', '#FFCC99', '#FF99CC', '#99CCFF', '#CC99FF']
                    
                    # Plot regimes as background colors
                    regime_changes = np.where(np.diff(np.append([-1], regimes)))[0]
                    for i in range(len(regime_changes)):
                        start = regime_changes[i]
                        end = regime_changes[i+1] if i < len(regime_changes) - 1 else len(regimes)
                        regime = regimes[start]
                        
                        # Add colored background for the regime period
                        ax1.axvspan(regime_dates[start], regime_dates[end-1], 
                                  alpha=0.2, color=colors[regime % len(colors)])
                        
                        # Add regime label in the middle of the period
                        mid_point = start + (end - start) // 2
                        ax1.text(regime_dates[mid_point], ax1.get_ylim()[1] * 0.95,
                                self.regime_integration.regime_classifier.REGIMES.get(regime, f"Regime {regime}"),
                                ha='center', va='top', backgroundcolor='white', alpha=0.7)
                    
                    # Plot regime index in the second subplot
                    ax2.plot(regime_dates, regimes, color='blue', linewidth=1.5)
                    ax2.set_ylabel('Regime')
                    ax2.set_yticks(list(self.regime_integration.regime_classifier.REGIMES.keys()))
                    ax2.set_yticklabels([self.regime_integration.regime_classifier.REGIMES.get(i, f"Regime {i}") 
                                      for i in self.regime_integration.regime_classifier.REGIMES.keys()])
                    ax2.grid(True, alpha=0.3)
                    
                    # Save plot if dashboard directory is specified
                    if dashboard_dir:
                        plot_path = os.path.join(dashboard_dir, "strategy_by_regime.png")
                        fig.savefig(plot_path)
                        plt.close(fig)
                
                except Exception as e:
                    logger.error(f"Error creating regime-overlay plot: {e}")
            
            # Create outperformance comparison table
            outperformance_data = []
            
            for result in self.backtest_results:
                strategy_name = result.get('strategy_name', 'Unknown Strategy')
                
                # Try to extract market outperformance metrics
                market_outperformance = None
                alpha = None
                
                if 'market_outperformance' in result:
                    market_outperformance = result['market_outperformance']
                elif 'alpha' in result:
                    alpha = result['alpha']
                
                outperformance_data.append({
                    'strategy_name': strategy_name,
                    'market_outperformance': market_outperformance,
                    'alpha': alpha,
                    'annual_return': result.get('annual_return', None),
                    'sharpe_ratio': result.get('sharpe_ratio', None),
                    'max_drawdown': result.get('max_drawdown', None),
                    'win_rate': result.get('win_rate', None)
                })
            
            # Create dashboard data
            dashboard = {
                'strategies': list(equity_curves.columns),
                'outperformance_data': outperformance_data,
                'dashboard_dir': dashboard_dir
            }
            
            # Save dashboard data if dashboard directory is specified
            if dashboard_dir:
                dashboard_path = os.path.join(dashboard_dir, "dashboard_data.json")
                with open(dashboard_path, 'w') as f:
                    json.dump(dashboard, f, indent=2, default=str)
            
            logger.info("Created strategy performance dashboard")
            return dashboard
        
        except Exception as e:
            logger.error(f"Error creating strategy dashboard: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def run_enhanced_analysis(self, results_path, output_dir=None, num_regimes=4, method="kmeans"):
        """
        Run all enhanced analysis components.
        
        Args:
            results_path (str): Path to backtest results JSON file or directory
            output_dir (str, optional): Directory to save analysis results
            num_regimes (int): Number of regimes to identify
            method (str): Classification method ('kmeans' or 'hmm')
            
        Returns:
            dict: Comprehensive analysis results
        """
        # Set output directory
        self.output_dir = output_dir
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load data. Analysis aborted.")
            return {}
        
        # Load backtest results
        results = self.load_backtest_results(results_path)
        if not results:
            logger.error("No backtest results found. Analysis aborted.")
            return {}
        
        # Run market regime analysis
        logger.info("Running market regime analysis...")
        regime_results = self.run_market_regime_analysis(num_regimes, method)
        
        # Run benchmark-relative analysis
        logger.info("Running benchmark-relative performance analysis...")
        benchmark_results = self.run_benchmark_relative_analysis()
        
        # Create strategy dashboard
        logger.info("Creating strategy performance dashboard...")
        dashboard = self.create_strategy_dashboard()
        
        # Compile comprehensive results
        comprehensive_results = {
            'regime_analysis': regime_results,
            'benchmark_analysis': benchmark_results,
            'dashboard': dashboard,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save comprehensive results if output directory is specified
        if self.output_dir:
            results_path = os.path.join(self.output_dir, "comprehensive_analysis.json")
            with open(results_path, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            
            logger.info(f"Saved comprehensive analysis results to {results_path}")
        
        logger.info("Enhanced analysis completed")
        return comprehensive_results

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="QuantConnect Enhanced Analysis")
    parser.add_argument("--results", required=True, help="Path to backtest results JSON file or directory")
    parser.add_argument("--market-data", required=True, help="Path to market data CSV file")
    parser.add_argument("--benchmark", help="Path to benchmark data CSV file (defaults to market data)")
    parser.add_argument("--output", default="qc_analysis_output", help="Output directory for analysis results")
    parser.add_argument("--regimes", type=int, default=4, help="Number of market regimes to identify")
    parser.add_argument("--method", choices=["kmeans", "hmm"], default="kmeans", help="Regime classification method")
    
    args = parser.parse_args()
    
    # Initialize analysis runner
    runner = EnhancedAnalysisRunner(
        market_data_path=args.market_data,
        benchmark_data_path=args.benchmark
    )
    
    # Run enhanced analysis
    results = runner.run_enhanced_analysis(
        results_path=args.results,
        output_dir=args.output,
        num_regimes=args.regimes,
        method=args.method
    )
    
    # Print summary
    if results and 'benchmark_analysis' in results and 'summary' in results['benchmark_analysis']:
        summary = results['benchmark_analysis']['summary']
        print("\nStrategy Outperformance Summary:")
        for strategy_name, outperformance in summary.items():
            outperforms = "outperforms" if outperformance.get('outperforms_benchmark', False) else "underperforms"
            percentage = outperformance.get('outperformance_percentage', "N/A")
            significance = "Significant" if outperformance.get('outperformance_significant', False) else "Not significant"
            
            print(f"{strategy_name}: {outperforms} benchmark by {percentage} ({significance})")
    
    if results and 'regime_analysis' in results and 'current_regime' in results['regime_analysis']:
        current_regime = results['regime_analysis']['current_regime']
        print(f"\nCurrent Market Regime: {current_regime.get('regime_name', 'Unknown')}")
    
    if args.output:
        print(f"\nAnalysis results saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())