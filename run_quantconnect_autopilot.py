#!/usr/bin/env python
"""
QuantConnect Autopilot

This script automates the entire process of generating trading strategies, 
converting them to QuantConnect format, running backtests, and evaluating performance.

It integrates the strategy generation system with QuantConnect backtesting to:
1. Generate optimized strategies
2. Convert strategies to QuantConnect format
3. Run backtests on the QuantConnect platform
4. Analyze and compare strategy performance
5. Refine strategies based on backtest results

Usage:
    python run_quantconnect_autopilot.py --strategies 5 --iterations 3
"""

import os
import sys
import json
import time
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import concurrent.futures
import traceback

# Import custom modules
from quant_connect_adapter import QuantConnectAdapter
from quantconnect_api import QuantConnectAPIClient
from generate_qc_algorithm import QuantConnectGenerator
from quantconnect_metrics import QuantConnectMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qc_autopilot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class QuantConnectAutopilot:
    """Class to automate strategy generation, conversion, and backtesting on QuantConnect."""
    
    def __init__(self, config_path=None, output_dir="qc_strategies", credentials_path=None):
        """
        Initialize the autopilot.
        
        Args:
            config_path (str): Path to QuantConnect credentials
            output_dir (str): Directory to save generated algorithms
            credentials_path (str): Path to Google API credentials
        """
        self.config_path = config_path or os.path.join(os.getcwd(), 'qc_config.json')
        self.output_dir = output_dir
        self.credentials_path = credentials_path or os.path.join(os.getcwd(), 'google_credentials.json')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize components
        self.generator = QuantConnectGenerator(self.output_dir)
        self.api_client = QuantConnectAPIClient(config_path=self.config_path)
        self.metrics_processor = QuantConnectMetrics(credentials_path=self.credentials_path)
        
        # Initialize results storage
        self.strategies = []
        self.backtest_results = []
        
        # Create results directory if it doesn't exist
        self.results_dir = os.path.join(os.getcwd(), 'qc_results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def _load_historical_results(self):
        """Load historical backtest results if available."""
        results_dir = os.path.join(os.getcwd(), 'qc_results')
        if not os.path.exists(results_dir):
            return []
        
        results = []
        for file_name in os.listdir(results_dir):
            if file_name.endswith('.json'):
                try:
                    with open(os.path.join(results_dir, file_name), 'r') as f:
                        results.append(json.load(f))
                except Exception as e:
                    logger.error(f"Error loading results from {file_name}: {e}")
        
        return results
    
    def generate_strategies(self, num_strategies=5, mode="auto"):
        """
        Generate strategies for testing.
        
        Args:
            num_strategies (int): Number of strategies to generate
            mode (str): Generation mode - "auto", "latest", or "template"
            
        Returns:
            list: Paths to generated QuantConnect algorithm files
        """
        logger.info(f"Generating {num_strategies} strategies in {mode} mode")
        
        if mode == "auto":
            # Generate new optimized strategies
            self.strategies = self.generator.generate_auto(num_strategies)
            
        elif mode == "latest":
            # Use latest generated strategies
            self.strategies = self.generator.generate_from_latest(num_strategies)
            
        elif mode == "template":
            # Use template-based strategies (for specific strategy types)
            template_dir = os.path.join(os.getcwd(), 'strategy_templates')
            self.strategies = []
            
            # Find templates and generate strategies
            for template_file in os.listdir(template_dir):
                if template_file.endswith('.json') and len(self.strategies) < num_strategies:
                    strategy_path = os.path.join(template_dir, template_file)
                    qc_file = self.generator.generate_from_json(strategy_path)
                    self.strategies.append(qc_file)
        
        logger.info(f"Generated {len(self.strategies)} strategy files")
        return self.strategies
    
    def run_backtests(self, start_date=None, end_date=None, parallel=True):
        """
        Run backtests for all generated strategies.
        
        Args:
            start_date (str): Backtest start date (YYYY-MM-DD)
            end_date (str): Backtest end date (YYYY-MM-DD)
            parallel (bool): Run backtests in parallel
            
        Returns:
            list: Backtest results
        """
        if not self.strategies:
            logger.error("No strategies available to backtest")
            return []
        
        # Set default dates if not provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Running backtests for {len(self.strategies)} strategies from {start_date} to {end_date}")
        
        self.backtest_results = []
        
        if parallel:
            # Run backtests in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(self.strategies))) as executor:
                future_to_strategy = {
                    executor.submit(self._run_single_backtest, strategy, start_date, end_date): strategy
                    for strategy in self.strategies
                }
                
                for future in concurrent.futures.as_completed(future_to_strategy):
                    strategy = future_to_strategy[future]
                    try:
                        result = future.result()
                        if result:
                            self.backtest_results.append(result)
                    except Exception as e:
                        logger.error(f"Error in backtest for {strategy}: {e}")
                        logger.error(traceback.format_exc())
        else:
            # Run backtests sequentially
            for strategy in self.strategies:
                try:
                    result = self._run_single_backtest(strategy, start_date, end_date)
                    if result:
                        self.backtest_results.append(result)
                except Exception as e:
                    logger.error(f"Error in backtest for {strategy}: {e}")
                    logger.error(traceback.format_exc())
        
        logger.info(f"Completed {len(self.backtest_results)} backtests")
        return self.backtest_results
    
    def _run_single_backtest(self, strategy_path, start_date, end_date):
        """Run a single backtest."""
        strategy_name = os.path.basename(strategy_path).replace('.py', '')
        backtest_name = f"Backtest_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Running backtest for {strategy_name}")
        
        try:
            # Run algorithm on QuantConnect
            metrics = self.api_client.run_algorithm(
                algorithm_path=strategy_path,
                project_name=f"Mathematricks_{strategy_name}",
                backtest_name=backtest_name
            )
            
            # Add strategy information
            metrics['strategy_name'] = strategy_name
            metrics['strategy_path'] = strategy_path
            metrics['backtest_date'] = datetime.now().isoformat()
            
            # Enhance metrics with market comparison
            enhanced_metrics = self.metrics_processor.enhance_metrics(metrics)
            
            # Save results to file
            result_path = os.path.join(self.results_dir, f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(result_path, 'w') as f:
                json.dump(enhanced_metrics, f, indent=2, default=str)
            
            # Log results including market comparison
            if 'market_outperformance' in enhanced_metrics:
                market_out = enhanced_metrics['market_outperformance'] * 100
                beats_market = "outperforms" if market_out > 0 else "underperforms"
                logger.info(f"Backtest completed for {strategy_name}: Sharpe {enhanced_metrics.get('sharpe_ratio', 0):.2f}, " +
                           f"{beats_market} market by {abs(market_out):.2f}%")
            else:
                logger.info(f"Backtest completed for {strategy_name} with Sharpe ratio {enhanced_metrics.get('sharpe_ratio', 0):.2f}")
            
            return enhanced_metrics
        except Exception as e:
            logger.error(f"Error in backtest for {strategy_name}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def analyze_results(self):
        """
        Analyze backtest results and identify top strategies.
        
        Returns:
            dict: Analysis results with best strategies by metric
        """
        if not self.backtest_results:
            logger.error("No backtest results available to analyze")
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.backtest_results)
        
        # Calculate overall score based on multiple metrics
        if 'sharpe_ratio' in df.columns and 'annual_return' in df.columns and 'max_drawdown' in df.columns:
            # Normalize metrics
            df['sharpe_normalized'] = df['sharpe_ratio'] / df['sharpe_ratio'].max() if df['sharpe_ratio'].max() > 0 else 0
            df['return_normalized'] = df['annual_return'] / df['annual_return'].max() if df['annual_return'].max() > 0 else 0
            df['drawdown_normalized'] = 1 - (df['max_drawdown'].abs() / df['max_drawdown'].abs().max() if df['max_drawdown'].abs().max() > 0 else 0)
            
            # Calculate overall score (weighted sum)
            df['overall_score'] = (df['sharpe_normalized'] * 0.4 + 
                                df['return_normalized'] * 0.4 + 
                                df['drawdown_normalized'] * 0.2)
        
        # Get top strategies by different metrics
        analysis = {}
        
        if 'overall_score' in df.columns:
            best_overall = df.loc[df['overall_score'].idxmax()] if not df.empty else None
            analysis['best_overall'] = best_overall.to_dict() if best_overall is not None else None
        
        if 'sharpe_ratio' in df.columns:
            best_sharpe = df.loc[df['sharpe_ratio'].idxmax()] if not df.empty else None
            analysis['best_sharpe'] = best_sharpe.to_dict() if best_sharpe is not None else None
        
        if 'annual_return' in df.columns:
            best_return = df.loc[df['annual_return'].idxmax()] if not df.empty else None
            analysis['best_return'] = best_return.to_dict() if best_return is not None else None
        
        if 'max_drawdown' in df.columns:
            best_drawdown = df.loc[df['max_drawdown'].idxmin()] if not df.empty else None
            analysis['best_drawdown'] = best_drawdown.to_dict() if best_drawdown is not None else None
        
        # Generate statistics
        analysis['statistics'] = {
            'num_strategies': len(df),
            'avg_sharpe': df['sharpe_ratio'].mean() if 'sharpe_ratio' in df.columns else None,
            'avg_return': df['annual_return'].mean() if 'annual_return' in df.columns else None,
            'avg_drawdown': df['max_drawdown'].mean() if 'max_drawdown' in df.columns else None,
            'avg_win_rate': df['win_rate'].mean() if 'win_rate' in df.columns else None
        }
        
        return analysis
    
    def save_analysis(self, analysis, output_path=None):
        """
        Save analysis results to a file.
        
        Args:
            analysis (dict): Analysis results
            output_path (str): Path to save results
            
        Returns:
            str: Path to saved analysis
        """
        # Generate output path if not provided
        if not output_path:
            output_dir = os.path.join(os.getcwd(), 'qc_analysis')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_path = os.path.join(output_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Save analysis to file
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Saved analysis results to {output_path}")
        return output_path
    
    def refine_strategies(self, num_strategies=3):
        """
        Refine top strategies to generate improved versions.
        
        Args:
            num_strategies (int): Number of strategies to refine
            
        Returns:
            list: Paths to refined strategy files
        """
        if not self.backtest_results:
            logger.error("No backtest results available for refinement")
            return []
        
        # Get top strategies based on overall score
        df = pd.DataFrame(self.backtest_results)
        
        if 'overall_score' not in df.columns:
            # Calculate overall score if not already done
            if 'sharpe_ratio' in df.columns and 'annual_return' in df.columns and 'max_drawdown' in df.columns:
                # Normalize metrics
                df['sharpe_normalized'] = df['sharpe_ratio'] / df['sharpe_ratio'].max() if df['sharpe_ratio'].max() > 0 else 0
                df['return_normalized'] = df['annual_return'] / df['annual_return'].max() if df['annual_return'].max() > 0 else 0
                df['drawdown_normalized'] = 1 - (df['max_drawdown'].abs() / df['max_drawdown'].abs().max() if df['max_drawdown'].abs().max() > 0 else 0)
                
                # Calculate overall score (weighted sum)
                df['overall_score'] = (df['sharpe_normalized'] * 0.4 + 
                                    df['return_normalized'] * 0.4 + 
                                    df['drawdown_normalized'] * 0.2)
        
        if 'overall_score' not in df.columns or df.empty:
            logger.error("Cannot refine strategies: missing score data")
            return []
        
        # Get top N strategies
        top_strategies = df.nlargest(num_strategies, 'overall_score')
        
        refined_strategies = []
        
        for _, strategy in top_strategies.iterrows():
            strategy_name = strategy.get('strategy_name', 'unknown')
            strategy_path = strategy.get('strategy_path', None)
            
            if not strategy_path or not os.path.exists(strategy_path):
                logger.error(f"Strategy file not found: {strategy_path}")
                continue
            
            logger.info(f"Refining strategy: {strategy_name}")
            
            try:
                # Create a refined version with parameter improvements
                refined_name = f"Refined_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Read the original algorithm file
                with open(strategy_path, 'r') as f:
                    algorithm_code = f.read()
                
                # Modify parameters based on backtest results
                # (This would ideally be a more sophisticated process)
                if 'self.ema_short_period = ' in algorithm_code:
                    algorithm_code = algorithm_code.replace('self.ema_short_period = 8', 'self.ema_short_period = 10')
                
                if 'self.rsi_period = ' in algorithm_code:
                    algorithm_code = algorithm_code.replace('self.rsi_period = 14', 'self.rsi_period = 12')
                
                # Save refined algorithm
                refined_path = os.path.join(self.output_dir, f"{refined_name}.py")
                with open(refined_path, 'w') as f:
                    f.write(algorithm_code)
                
                refined_strategies.append(refined_path)
                logger.info(f"Created refined strategy: {refined_path}")
                
            except Exception as e:
                logger.error(f"Error refining strategy {strategy_name}: {e}")
                logger.error(traceback.format_exc())
        
        return refined_strategies
    
    def run_iteration(self, num_strategies=5, mode="auto", num_refine=2, update_sheets=True):
        """
        Run a complete iteration of strategy generation, backtesting, and refinement.
        
        Args:
            num_strategies (int): Number of strategies to generate
            mode (str): Generation mode - "auto", "latest", or "template"
            num_refine (int): Number of strategies to refine
            update_sheets (bool): Whether to update Google Sheets with results
            
        Returns:
            dict: Iteration results with analysis and top strategies
        """
        logger.info(f"Starting iteration with {num_strategies} strategies in {mode} mode")
        
        # Generate strategies
        self.generate_strategies(num_strategies, mode)
        
        # Run backtests
        self.run_backtests()
        
        # Analyze results
        analysis = self.analyze_results()
        
        # Save analysis
        analysis_path = self.save_analysis(analysis)
        
        # Create visualizations and update Google Sheets
        if self.backtest_results:
            # Create visualizations
            output_dir = os.path.join(os.getcwd(), 'qc_visualizations')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            visualization_paths = self.metrics_processor.create_visualizations(
                self.backtest_results, output_dir
            )
            
            # Update Google Sheets if requested
            if update_sheets:
                sheet_updated = self.metrics_processor.update_google_sheet(
                    self.backtest_results, 
                    sheet_name=f"QuantConnect Strategies {datetime.now().strftime('%Y-%m-%d')}"
                )
                if sheet_updated:
                    logger.info("Updated Google Sheets with backtest results")
                else:
                    logger.warning("Failed to update Google Sheets")
        
        # Refine top strategies
        refined_strategies = self.refine_strategies(num_refine)
        
        # Add refined strategies to the list
        self.strategies.extend(refined_strategies)
        
        # Return results
        return {
            'analysis': analysis,
            'analysis_path': analysis_path,
            'strategies': self.strategies,
            'refined_strategies': refined_strategies,
            'backtest_results': self.backtest_results,
            'visualizations': visualization_paths if 'visualization_paths' in locals() else {}
        }
    
    def run_multiple_iterations(self, num_iterations=3, num_strategies=5, mode="auto", num_refine=2, update_sheets=True):
        """
        Run multiple iterations of the strategy generation and testing process.
        
        Args:
            num_iterations (int): Number of iterations to run
            num_strategies (int): Number of strategies per iteration
            mode (str): Generation mode - "auto", "latest", or "template"
            num_refine (int): Number of strategies to refine per iteration
            update_sheets (bool): Whether to update Google Sheets with results
            
        Returns:
            dict: Overall results with best strategies
        """
        logger.info(f"Starting {num_iterations} iterations with {num_strategies} strategies per iteration")
        
        all_results = []
        
        for i in range(num_iterations):
            logger.info(f"Starting iteration {i+1}/{num_iterations}")
            
            try:
                # Run iteration with sheets update only on the last iteration to avoid too many updates
                iteration_update_sheets = update_sheets and (i == num_iterations - 1)
                iteration_results = self.run_iteration(num_strategies, mode, num_refine, iteration_update_sheets)
                all_results.append(iteration_results)
                
                # Print summary of this iteration
                if 'analysis' in iteration_results and 'best_overall' in iteration_results['analysis']:
                    best = iteration_results['analysis']['best_overall']
                    if best:
                        logger.info(f"Iteration {i+1} best strategy: {best.get('strategy_name', 'unknown')}")
                        logger.info(f"  Sharpe: {best.get('sharpe_ratio', 0):.2f}, Return: {best.get('annual_return', 0):.2%}, Drawdown: {best.get('max_drawdown', 0):.2%}")
                
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {e}")
                logger.error(traceback.format_exc())
        
        # Collect all results across iterations
        all_backtest_results = []
        for result in all_results:
            if 'backtest_results' in result:
                all_backtest_results.extend(result['backtest_results'])
        
        # Find the best strategy across all iterations
        if all_backtest_results:
            df = pd.DataFrame(all_backtest_results)
            
            if 'sharpe_ratio' in df.columns and 'annual_return' in df.columns and 'max_drawdown' in df.columns:
                # Normalize metrics
                df['sharpe_normalized'] = df['sharpe_ratio'] / df['sharpe_ratio'].max() if df['sharpe_ratio'].max() > 0 else 0
                df['return_normalized'] = df['annual_return'] / df['annual_return'].max() if df['annual_return'].max() > 0 else 0
                df['drawdown_normalized'] = 1 - (df['max_drawdown'].abs() / df['max_drawdown'].abs().max() if df['max_drawdown'].abs().max() > 0 else 0)
                
                # Calculate overall score (weighted sum)
                df['overall_score'] = (df['sharpe_normalized'] * 0.4 + 
                                    df['return_normalized'] * 0.4 + 
                                    df['drawdown_normalized'] * 0.2)
            
                # Get top strategies
                top_strategies = df.nlargest(5, 'overall_score')
                
                logger.info("Top 5 strategies across all iterations:")
                for i, (_, strategy) in enumerate(top_strategies.iterrows()):
                    logger.info(f"{i+1}. {strategy.get('strategy_name', 'unknown')}")
                    logger.info(f"   Sharpe: {strategy.get('sharpe_ratio', 0):.2f}, Return: {strategy.get('annual_return', 0):.2%}, Drawdown: {strategy.get('max_drawdown', 0):.2%}")
        
        # Create final report
        final_report = {
            'num_iterations': num_iterations,
            'total_strategies': sum(len(result.get('strategies', [])) for result in all_results),
            'total_backtests': len(all_backtest_results),
            'top_strategies': top_strategies.to_dict('records') if 'top_strategies' in locals() else [],
            'iterations': all_results
        }
        
        # Save final report
        report_dir = os.path.join(os.getcwd(), 'qc_reports')
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
            
        report_path = os.path.join(report_dir, f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"Saved final report to {report_path}")
        
        # Final visualization and Google Sheets update with all results
        if all_backtest_results and update_sheets:
            # Create summary visualizations with all results
            output_dir = os.path.join(os.getcwd(), 'qc_visualizations', 'summary')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            visualization_paths = self.metrics_processor.create_visualizations(
                all_backtest_results, output_dir
            )
            
            # Final Google Sheets update with all results
            sheet_updated = self.metrics_processor.update_google_sheet(
                all_backtest_results, 
                sheet_name=f"QuantConnect Summary {datetime.now().strftime('%Y-%m-%d')}"
            )
            
            if sheet_updated:
                logger.info("Updated Google Sheets with final summary of all results")
            else:
                logger.warning("Failed to update Google Sheets with final summary")
                
            final_report['summary_visualizations'] = visualization_paths
        
        return final_report

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="QuantConnect Autopilot")
    parser.add_argument("--config", default="qc_config.json", help="Path to QuantConnect config file")
    parser.add_argument("--output", default="qc_strategies", help="Output directory for generated strategies")
    parser.add_argument("--strategies", type=int, default=5, help="Number of strategies per iteration")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations to run")
    parser.add_argument("--mode", choices=["auto", "latest", "template"], default="auto", help="Strategy generation mode")
    parser.add_argument("--refine", type=int, default=2, help="Number of strategies to refine per iteration")
    parser.add_argument("--start", default=None, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--credentials", default="google_credentials.json", help="Path to Google API credentials")
    parser.add_argument("--no-sheets", action="store_true", help="Skip updating Google Sheets")
    
    args = parser.parse_args()
    
    # Initialize autopilot
    autopilot = QuantConnectAutopilot(args.config, args.output, args.credentials)
    
    # Run multiple iterations
    final_report = autopilot.run_multiple_iterations(
        num_iterations=args.iterations,
        num_strategies=args.strategies,
        mode=args.mode,
        num_refine=args.refine,
        update_sheets=not args.no_sheets
    )
    
    # Print summary of best strategies
    if 'top_strategies' in final_report and final_report['top_strategies']:
        print("\nQuantConnect Autopilot Results:")
        print(f"Total strategies tested: {final_report['total_strategies']}")
        print(f"Total backtests run: {final_report['total_backtests']}")
        print("\nTop Strategies:")
        
        for i, strategy in enumerate(final_report['top_strategies'][:3]):
            print(f"{i+1}. {strategy.get('strategy_name', 'unknown')}")
            print(f"   Sharpe Ratio: {strategy.get('sharpe_ratio', 0):.2f}")
            print(f"   Annual Return: {strategy.get('annual_return', 0):.2%}")
            print(f"   Max Drawdown: {strategy.get('max_drawdown', 0):.2%}")
            print(f"   Win Rate: {strategy.get('win_rate', 0):.2%}")
            print()

if __name__ == "__main__":
    main()