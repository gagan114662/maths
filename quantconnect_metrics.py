#!/usr/bin/env python
"""
QuantConnect Metrics Processor

This module processes and enhances the metrics from QuantConnect backtests:
1. Calculates additional performance metrics
2. Compares strategy performance against market benchmarks
3. Provides visualization and reporting functions
4. Integrates with Google Sheets for easy monitoring
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from itertools import cycle
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qc_metrics.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class QuantConnectMetrics:
    """Class to process and enhance metrics from QuantConnect backtests."""
    
    def __init__(self, credentials_path=None):
        """
        Initialize the metrics processor.
        
        Args:
            credentials_path (str): Path to Google API credentials file
        """
        self.credentials_path = credentials_path or 'google_credentials.json'
        self.google_client = None
        
        # Market benchmark stats (S&P 500 long-term averages)
        self.market_benchmarks = {
            'annual_return': 0.10,  # 10% average annual return
            'sharpe_ratio': 1.0,    # Typical market Sharpe ratio
            'volatility': 0.15,     # Typical S&P 500 volatility
            'max_drawdown': -0.20,  # Typical max drawdown
        }
        
        # Initialize the Google Sheets client if credentials exist
        if os.path.exists(self.credentials_path):
            try:
                self._init_google_sheets()
            except Exception as e:
                logger.error(f"Error initializing Google Sheets: {e}")
    
    def _init_google_sheets(self):
        """Initialize the Google Sheets client."""
        # Set up credentials
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                self.credentials_path, scope
            )
            self.google_client = gspread.authorize(creds)
            logger.info("Initialized Google Sheets client")
        except Exception as e:
            logger.error(f"Error authorizing with Google Sheets: {e}")
            self.google_client = None
    
    def enhance_metrics(self, backtest_results):
        """
        Enhance the metrics from a backtest with additional calculations.
        
        Args:
            backtest_results (dict): Original backtest results from QuantConnect
            
        Returns:
            dict: Enhanced metrics
        """
        # Extract original metrics
        metrics = backtest_results.copy()
        
        # Calculate market outperformance
        if 'annual_return' in metrics:
            metrics['market_outperformance'] = metrics['annual_return'] - self.market_benchmarks['annual_return']
            metrics['market_outperformance_pct'] = (metrics['annual_return'] / self.market_benchmarks['annual_return']) - 1
            metrics['beats_market'] = metrics['annual_return'] > self.market_benchmarks['annual_return']
        
        # Calculate risk-adjusted metrics
        if 'alpha' not in metrics and 'annual_return' in metrics and 'beta' in metrics:
            # Estimate alpha if not provided
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            expected_return = risk_free_rate + metrics['beta'] * (self.market_benchmarks['annual_return'] - risk_free_rate)
            metrics['alpha'] = metrics['annual_return'] - expected_return
        
        # Calculate Sortino ratio if not provided (downside risk-adjusted return)
        if 'sortino_ratio' not in metrics and 'annual_return' in metrics:
            # Estimate Sortino using max drawdown as a proxy for downside risk
            risk_free_rate = 0.02
            downside_risk = abs(metrics.get('max_drawdown', 0.20))
            metrics['sortino_ratio'] = (metrics['annual_return'] - risk_free_rate) / downside_risk
        
        # Calculate Calmar ratio (return / max drawdown)
        if 'calmar_ratio' not in metrics and 'annual_return' in metrics and 'max_drawdown' in metrics:
            metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else float('inf')
        
        # Calculate profit factor if not provided
        if 'profit_factor' not in metrics and 'avg_win' in metrics and 'avg_loss' in metrics:
            win_rate = metrics.get('win_rate', 0.5)
            avg_win = metrics.get('avg_win', 0)
            avg_loss = abs(metrics.get('avg_loss', 0))
            
            if avg_loss > 0:
                metrics['profit_factor'] = (win_rate * avg_win) / ((1 - win_rate) * avg_loss)
            else:
                metrics['profit_factor'] = float('inf') if avg_win > 0 else 0
        
        # Add human-readable assessment
        metrics['assessment'] = self._generate_assessment(metrics)
        
        return metrics
    
    def _generate_assessment(self, metrics):
        """Generate a human-readable assessment of strategy performance."""
        assessment = []
        
        # Check if the strategy beats the market
        if 'market_outperformance' in metrics:
            if metrics['market_outperformance'] > 0:
                outperf = metrics['market_outperformance'] * 100
                outperf_pct = metrics['market_outperformance_pct'] * 100
                assessment.append(f"Outperforms market by {outperf:.1f}% ({outperf_pct:.1f}% better)")
            else:
                underperf = -metrics['market_outperformance'] * 100
                assessment.append(f"Underperforms market by {underperf:.1f}%")
        
        # Assess Sharpe ratio
        if 'sharpe_ratio' in metrics:
            if metrics['sharpe_ratio'] > 2.0:
                assessment.append("Excellent risk-adjusted returns (Sharpe > 2)")
            elif metrics['sharpe_ratio'] > 1.0:
                assessment.append("Good risk-adjusted returns (Sharpe > 1)")
            elif metrics['sharpe_ratio'] > 0.5:
                assessment.append("Average risk-adjusted returns (Sharpe > 0.5)")
            else:
                assessment.append("Poor risk-adjusted returns (Sharpe < 0.5)")
        
        # Assess drawdown
        if 'max_drawdown' in metrics:
            drawdown = abs(metrics['max_drawdown']) * 100
            if drawdown < 10:
                assessment.append(f"Low drawdown risk ({drawdown:.1f}%)")
            elif drawdown < 20:
                assessment.append(f"Moderate drawdown risk ({drawdown:.1f}%)")
            else:
                assessment.append(f"High drawdown risk ({drawdown:.1f}%)")
        
        # Assess win rate
        if 'win_rate' in metrics:
            win_rate = metrics['win_rate'] * 100
            if win_rate > 60:
                assessment.append(f"High win rate ({win_rate:.1f}%)")
            elif win_rate > 50:
                assessment.append(f"Above average win rate ({win_rate:.1f}%)")
            else:
                assessment.append(f"Below average win rate ({win_rate:.1f}%)")
        
        # Assess profit factor
        if 'profit_factor' in metrics:
            if metrics['profit_factor'] > 2.0:
                assessment.append(f"Excellent profit factor ({metrics['profit_factor']:.2f})")
            elif metrics['profit_factor'] > 1.5:
                assessment.append(f"Good profit factor ({metrics['profit_factor']:.2f})")
            elif metrics['profit_factor'] > 1.0:
                assessment.append(f"Positive profit factor ({metrics['profit_factor']:.2f})")
            else:
                assessment.append(f"Negative profit factor ({metrics['profit_factor']:.2f})")
        
        return assessment
    
    def create_visualizations(self, results_list, output_dir):
        """
        Create visualizations from multiple backtest results.
        
        Args:
            results_list (list): List of enhanced metrics dictionaries
            output_dir (str): Directory to save visualizations
            
        Returns:
            dict: Paths to generated visualization files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(results_list)
        
        # Generate plots only if we have data
        if df.empty:
            logger.warning("No data available for visualizations")
            return {}
        
        visualization_paths = {}
        
        # Plot key metrics comparison
        try:
            key_metrics = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, metric in enumerate(key_metrics):
                if metric in df.columns:
                    # Sort values for better visualization
                    sorted_df = df.sort_values(metric, ascending=(metric == 'max_drawdown'))
                    
                    # Plot only top 10 strategies for readability
                    plot_df = sorted_df.head(10)
                    
                    # Plot with appropriate formatting
                    if metric == 'max_drawdown':
                        plot_df[metric] = plot_df[metric].abs() * 100
                        axes[i].set_title('Max Drawdown (%)')
                    elif metric == 'annual_return':
                        plot_df[metric] = plot_df[metric] * 100
                        axes[i].set_title('Annual Return (%)')
                    elif metric == 'win_rate':
                        plot_df[metric] = plot_df[metric] * 100
                        axes[i].set_title('Win Rate (%)')
                    else:
                        axes[i].set_title(metric.replace('_', ' ').title())
                    
                    # Create bar chart
                    bars = axes[i].barh(plot_df['strategy_name'], plot_df[metric])
                    
                    # Add values at the end of bars
                    for bar in bars:
                        width = bar.get_width()
                        label_x_pos = width if width >= 0 else 0
                        axes[i].text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                                    f'{width:.1f}', va='center')
                    
                    # Add market benchmark line if applicable
                    if metric in self.market_benchmarks:
                        benchmark = self.market_benchmarks[metric]
                        if metric == 'max_drawdown':
                            benchmark = abs(benchmark) * 100
                        elif metric in ['annual_return', 'win_rate']:
                            benchmark = benchmark * 100
                        
                        axes[i].axvline(x=benchmark, color='r', linestyle='--', 
                                      label=f'Market ({benchmark:.1f})')
                        axes[i].legend()
            
            plt.tight_layout()
            comparison_path = os.path.join(output_dir, 'strategy_comparison.png')
            plt.savefig(comparison_path)
            plt.close()
            
            visualization_paths['comparison'] = comparison_path
            logger.info(f"Generated strategy comparison visualization: {comparison_path}")
        except Exception as e:
            logger.error(f"Error creating comparison visualization: {e}")
        
        # Plot market outperformance
        try:
            if 'market_outperformance' in df.columns:
                plt.figure(figsize=(12, 8))
                
                # Sort by outperformance
                sorted_df = df.sort_values('market_outperformance', ascending=False)
                
                # Plot only top 10 strategies
                plot_df = sorted_df.head(10)
                
                # Convert to percentage
                plot_df['market_outperformance'] = plot_df['market_outperformance'] * 100
                
                # Create bar chart with color coding
                bars = plt.barh(plot_df['strategy_name'], plot_df['market_outperformance'])
                
                # Color bars based on outperformance
                for i, bar in enumerate(bars):
                    if bar.get_width() >= 0:
                        bar.set_color('green')
                    else:
                        bar.set_color('red')
                    
                    # Add value labels
                    width = bar.get_width()
                    label_x_pos = width if width >= 0 else width - 2
                    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                            f'{width:.1f}%', va='center')
                
                plt.axvline(x=0, color='black', linestyle='--')
                plt.title('Market Outperformance (%)')
                plt.tight_layout()
                
                outperformance_path = os.path.join(output_dir, 'market_outperformance.png')
                plt.savefig(outperformance_path)
                plt.close()
                
                visualization_paths['outperformance'] = outperformance_path
                logger.info(f"Generated market outperformance visualization: {outperformance_path}")
        except Exception as e:
            logger.error(f"Error creating outperformance visualization: {e}")
        
        # Plot risk-return scatter plot
        try:
            if 'annual_return' in df.columns and 'volatility' in df.columns:
                plt.figure(figsize=(10, 8))
                
                # Convert to percentage
                x = df['volatility'] * 100
                y = df['annual_return'] * 100
                
                # Plot scatter
                plt.scatter(x, y, s=100, alpha=0.7)
                
                # Add labels for each point
                for i, name in enumerate(df['strategy_name']):
                    plt.annotate(name, (x.iloc[i], y.iloc[i]), 
                                xytext=(5, 5), textcoords='offset points')
                
                # Add market benchmark
                plt.scatter([self.market_benchmarks['volatility'] * 100], 
                          [self.market_benchmarks['annual_return'] * 100], 
                          s=150, color='red', marker='*', label='Market')
                
                # Add efficient frontier (simplified as a curve)
                frontier_x = np.linspace(0, max(x) * 1.2, 100)
                frontier_y = 0.05 * 100 + 0.5 * np.sqrt(frontier_x)  # Simplified model
                plt.plot(frontier_x, frontier_y, 'k--', label='Efficient Frontier')
                
                plt.xlabel('Volatility (%)')
                plt.ylabel('Annual Return (%)')
                plt.title('Risk-Return Profile')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                risk_return_path = os.path.join(output_dir, 'risk_return.png')
                plt.savefig(risk_return_path)
                plt.close()
                
                visualization_paths['risk_return'] = risk_return_path
                logger.info(f"Generated risk-return visualization: {risk_return_path}")
        except Exception as e:
            logger.error(f"Error creating risk-return visualization: {e}")
        
        return visualization_paths
    
    def update_google_sheet(self, results_list, sheet_name=None):
        """
        Update a Google Sheet with backtest results.
        
        Args:
            results_list (list): List of enhanced metrics dictionaries
            sheet_name (str): Name of the Google Sheet to update
            
        Returns:
            bool: True if successful
        """
        if not self.google_client:
            logger.error("Google Sheets client not initialized")
            return False
        
        if not results_list:
            logger.warning("No results to update in Google Sheet")
            return False
        
        try:
            # Use default sheet name if not provided
            sheet_name = sheet_name or "QuantConnect Backtest Results"
            
            # Try to open existing sheet, create if not exists
            try:
                sheet = self.google_client.open(sheet_name)
            except gspread.exceptions.SpreadsheetNotFound:
                sheet = self.google_client.create(sheet_name)
                logger.info(f"Created new Google Sheet: {sheet_name}")
            
            # Get or create the results worksheet
            try:
                worksheet = sheet.worksheet("Strategy Results")
            except gspread.exceptions.WorksheetNotFound:
                worksheet = sheet.add_worksheet("Strategy Results", 1000, 20)
                logger.info("Created 'Strategy Results' worksheet")
            
            # Convert results to DataFrame for easier manipulation
            results_df = pd.DataFrame(results_list)
            
            # Select and order columns for the sheet
            important_columns = [
                'strategy_name', 'backtest_date', 'annual_return', 'sharpe_ratio', 
                'max_drawdown', 'win_rate', 'total_trades', 'market_outperformance',
                'alpha', 'beta', 'calmar_ratio', 'sortino_ratio'
            ]
            
            # Ensure all columns exist, otherwise create with None values
            for col in important_columns:
                if col not in results_df.columns:
                    results_df[col] = None
            
            # Reorder columns and select only the ones we want
            columns_to_use = [col for col in important_columns if col in results_df.columns]
            sheet_df = results_df[columns_to_use].copy()
            
            # Format dates
            if 'backtest_date' in sheet_df.columns:
                sheet_df['backtest_date'] = pd.to_datetime(sheet_df['backtest_date']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Format percentage values
            for col in ['annual_return', 'max_drawdown', 'win_rate', 'market_outperformance']:
                if col in sheet_df.columns:
                    sheet_df[col] = sheet_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "")
            
            # Format numeric values
            for col in ['sharpe_ratio', 'alpha', 'beta', 'calmar_ratio', 'sortino_ratio']:
                if col in sheet_df.columns:
                    sheet_df[col] = sheet_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
            
            # Convert to list of lists for Google Sheets
            header = sheet_df.columns.tolist()
            values = sheet_df.values.tolist()
            
            # Clear existing data and update
            worksheet.clear()
            
            # Add headers
            formatted_header = [col.replace('_', ' ').title() for col in header]
            worksheet.append_row(formatted_header)
            
            # Add data rows
            for row in values:
                worksheet.append_row(row)
            
            # Format the header row
            worksheet.format('A1:Z1', {
                'backgroundColor': {'red': 0.2, 'green': 0.2, 'blue': 0.2},
                'textFormat': {'bold': True, 'foregroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}}
            })
            
            # Auto-resize columns
            for i, col in enumerate(header):
                column_letter = chr(65 + i)  # A, B, C, ...
                worksheet.columns_auto_resize(i, i+1)
            
            # Create or update summary worksheet
            try:
                summary_ws = sheet.worksheet("Summary")
            except gspread.exceptions.WorksheetNotFound:
                summary_ws = sheet.add_worksheet("Summary", 500, 10)
                logger.info("Created 'Summary' worksheet")
            
            # Update summary worksheet
            self._update_summary_worksheet(summary_ws, results_list)
            
            logger.info(f"Updated Google Sheet: {sheet_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating Google Sheet: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _update_summary_worksheet(self, worksheet, results_list):
        """Update the summary worksheet with key statistics and top strategies."""
        try:
            # Clear the worksheet
            worksheet.clear()
            
            # Create DataFrame from results
            df = pd.DataFrame(results_list)
            
            # Calculate summary statistics
            num_strategies = len(df)
            
            # Count strategies that beat the market
            if 'beats_market' in df.columns:
                market_beaters = df['beats_market'].sum()
                market_beaters_pct = market_beaters / num_strategies if num_strategies > 0 else 0
            else:
                market_beaters = 0
                market_beaters_pct = 0
            
            # Get average metrics
            avg_return = df['annual_return'].mean() if 'annual_return' in df.columns else 0
            avg_sharpe = df['sharpe_ratio'].mean() if 'sharpe_ratio' in df.columns else 0
            avg_drawdown = df['max_drawdown'].mean() if 'max_drawdown' in df.columns else 0
            
            # Add summary section
            worksheet.update('A1', 'QuantConnect Backtest Summary')
            worksheet.format('A1', {
                'textFormat': {'bold': True, 'fontSize': 14}
            })
            
            summary_data = [
                ['Total Strategies Tested', num_strategies],
                ['Strategies That Beat Market', f"{market_beaters} ({market_beaters_pct:.1%})"],
                ['Average Annual Return', f"{avg_return:.2%}"],
                ['Average Sharpe Ratio', f"{avg_sharpe:.2f}"],
                ['Average Max Drawdown', f"{avg_drawdown:.2%}"],
                ['Last Updated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]
            
            worksheet.update('A3:B8', summary_data)
            
            # Format the summary section
            worksheet.format('A3:A8', {
                'textFormat': {'bold': True}
            })
            
            # Add top strategies section
            worksheet.update('A10', 'Top Strategies by Performance')
            worksheet.format('A10', {
                'textFormat': {'bold': True, 'fontSize': 12}
            })
            
            # Add headers for top strategies
            top_headers = ['Strategy Name', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Market Outperformance']
            worksheet.update('A12:E12', [top_headers])
            worksheet.format('A12:E12', {
                'backgroundColor': {'red': 0.2, 'green': 0.2, 'blue': 0.2},
                'textFormat': {'bold': True, 'foregroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}}
            })
            
            # Sort and get top 5 strategies by return
            if 'annual_return' in df.columns and len(df) > 0:
                top_by_return = df.sort_values('annual_return', ascending=False).head(5)
                
                # Format data for sheet
                top_data = []
                for _, row in top_by_return.iterrows():
                    strategy_row = [
                        row.get('strategy_name', ''),
                        f"{row.get('annual_return', 0):.2%}",
                        f"{row.get('sharpe_ratio', 0):.2f}",
                        f"{row.get('max_drawdown', 0):.2%}",
                        f"{row.get('market_outperformance', 0):.2%}"
                    ]
                    top_data.append(strategy_row)
                
                # Update sheet with top strategies
                if top_data:
                    worksheet.update('A13:E17', top_data)
            
            # Add sections for other categories (top by Sharpe, etc.)
            row_offset = 19
            
            # Top by Sharpe ratio
            worksheet.update(f'A{row_offset}', 'Top Strategies by Risk-Adjusted Return (Sharpe)')
            worksheet.format(f'A{row_offset}', {
                'textFormat': {'bold': True, 'fontSize': 12}
            })
            
            worksheet.update(f'A{row_offset+2}:E{row_offset+2}', [top_headers])
            worksheet.format(f'A{row_offset+2}:E{row_offset+2}', {
                'backgroundColor': {'red': 0.2, 'green': 0.2, 'blue': 0.2},
                'textFormat': {'bold': True, 'foregroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}}
            })
            
            if 'sharpe_ratio' in df.columns and len(df) > 0:
                top_by_sharpe = df.sort_values('sharpe_ratio', ascending=False).head(5)
                
                top_data = []
                for _, row in top_by_sharpe.iterrows():
                    strategy_row = [
                        row.get('strategy_name', ''),
                        f"{row.get('annual_return', 0):.2%}",
                        f"{row.get('sharpe_ratio', 0):.2f}",
                        f"{row.get('max_drawdown', 0):.2%}",
                        f"{row.get('market_outperformance', 0):.2%}"
                    ]
                    top_data.append(strategy_row)
                
                if top_data:
                    worksheet.update(f'A{row_offset+3}:E{row_offset+7}', top_data)
            
            # Auto-resize columns
            worksheet.columns_auto_resize(0, 5)
            
            logger.info("Updated summary worksheet")
            return True
            
        except Exception as e:
            logger.error(f"Error updating summary worksheet: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def process_results(self, results_path, update_sheets=True, create_visuals=True):
        """
        Process results from a single file or directory of result files.
        
        Args:
            results_path (str): Path to result file or directory
            update_sheets (bool): Whether to update Google Sheets
            create_visuals (bool): Whether to create visualizations
            
        Returns:
            dict: Processed results and paths to visualizations
        """
        results_list = []
        
        # Handle directory of results
        if os.path.isdir(results_path):
            for file_name in os.listdir(results_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(results_path, file_name)
                    try:
                        with open(file_path, 'r') as f:
                            result = json.load(f)
                            results_list.append(result)
                    except Exception as e:
                        logger.error(f"Error loading result file {file_path}: {e}")
        # Handle single result file
        elif os.path.isfile(results_path) and results_path.endswith('.json'):
            try:
                with open(results_path, 'r') as f:
                    result = json.load(f)
                    results_list.append(result)
            except Exception as e:
                logger.error(f"Error loading result file {results_path}: {e}")
        else:
            logger.error(f"Invalid results path: {results_path}")
            return {}
        
        # Enhance metrics for each result
        enhanced_results = [self.enhance_metrics(result) for result in results_list]
        
        # Create visualizations if requested
        visualization_paths = {}
        if create_visuals and enhanced_results:
            output_dir = os.path.join(os.getcwd(), 'qc_visualizations')
            visualization_paths = self.create_visualizations(enhanced_results, output_dir)
        
        # Update Google Sheets if requested
        sheets_updated = False
        if update_sheets and enhanced_results:
            sheets_updated = self.update_google_sheet(enhanced_results)
        
        return {
            'results': enhanced_results,
            'visualizations': visualization_paths,
            'sheets_updated': sheets_updated
        }

def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantConnect Metrics Processor")
    parser.add_argument("--results", required=True, help="Path to results file or directory")
    parser.add_argument("--credentials", default="google_credentials.json", help="Path to Google API credentials")
    parser.add_argument("--no-sheets", action="store_true", help="Skip updating Google Sheets")
    parser.add_argument("--no-visuals", action="store_true", help="Skip creating visualizations")
    
    args = parser.parse_args()
    
    # Process results
    processor = QuantConnectMetrics(args.credentials)
    processed = processor.process_results(
        args.results,
        update_sheets=not args.no_sheets,
        create_visuals=not args.no_visuals
    )
    
    # Print summary
    if processed.get('results'):
        # Print top strategies
        results_df = pd.DataFrame(processed['results'])
        
        if 'market_outperformance' in results_df.columns:
            top_strategies = results_df.sort_values('market_outperformance', ascending=False).head(3)
            
            print("\nTop Strategies by Market Outperformance:")
            for i, (_, strategy) in enumerate(top_strategies.iterrows()):
                name = strategy.get('strategy_name', f"Strategy {i+1}")
                outperf = strategy.get('market_outperformance', 0) * 100
                annual = strategy.get('annual_return', 0) * 100
                sharpe = strategy.get('sharpe_ratio', 0)
                
                print(f"{i+1}. {name}")
                print(f"   Return: {annual:.2f}% | Outperforms Market: {outperf:.2f}% | Sharpe: {sharpe:.2f}")
        
        # Print visualization paths
        if processed.get('visualizations'):
            print("\nVisualizations created:")
            for name, path in processed['visualizations'].items():
                print(f"- {name}: {path}")
        
        # Print Google Sheets status
        if args.no_sheets:
            print("\nGoogle Sheets update skipped")
        elif processed.get('sheets_updated'):
            print("\nGoogle Sheets successfully updated")
        else:
            print("\nFailed to update Google Sheets")
    else:
        print("\nNo results processed")

if __name__ == "__main__":
    main()