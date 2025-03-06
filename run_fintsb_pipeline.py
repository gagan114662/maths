#!/usr/bin/env python3
"""
Complete pipeline script for FinTSB integration with enhanced evaluation.
This script runs the comprehensive FinTSB integration pipeline,
implementing all components from data preprocessing to portfolio construction.
"""
import os
import sys
import argparse
from pathlib import Path
import logging
import yaml
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set up top-level logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger('FinTSB_Pipeline')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Import our FinTSB integration
from src.training.fintsb_integration import FinTSBPipeline, FINTSB_AVAILABLE

def check_environment():
    """Check if all required components are available."""
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        raise RuntimeError("Python 3.8 or higher is required")
    
    # Check FinTSB availability
    if not FINTSB_AVAILABLE:
        logger.warning("FinTSB modules not fully available. Some features may be limited.")
    
    # Check required directories
    required_dirs = [
        'FinTSB',
        'mathematricks',
        'src',
        'data'
    ]
    
    missing_dirs = [d for d in required_dirs if not Path(d).exists()]
    if missing_dirs:
        raise RuntimeError(f"Missing required directories: {missing_dirs}")
    
    # Check configuration files
    config_path = Path('FinTSB/configs/config_comprehensive.yaml')
    if not config_path.exists():
        raise RuntimeError(f"Missing configuration file: {config_path}")
    
    logger.info("Environment check completed successfully")

def run_pipeline(config_path: str, output_dir: Path = None):
    """
    Run the FinTSB pipeline with the specified configuration.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results (overrides config)
        
    Returns:
        Results of the pipeline run
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override output directory if specified
    if output_dir:
        config['output_dir'] = str(output_dir)
    
    # Create output directory if it doesn't exist
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Create and run pipeline
    pipeline = FinTSBPipeline(config_path)
    results = pipeline.run()
    
    # Generate summary visuals
    generate_summary_visuals(results, Path(config['output_dir']))
    
    return results

def generate_summary_visuals(results: dict, output_dir: Path):
    """
    Generate summary visualizations of pipeline results.
    
    Args:
        results: Pipeline results dictionary
        output_dir: Output directory for visualizations
    """
    logger.info("Generating summary visualizations")
    
    # Create visualizations directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Portfolio Allocation
    try:
        weights = results['portfolio']['weights']
        plt.figure(figsize=(10, 6))
        plt.bar(weights.keys(), weights.values())
        plt.title("Portfolio Allocation")
        plt.ylabel("Weight")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(viz_dir / "portfolio_allocation.png")
        plt.close()
    except Exception as e:
        logger.error(f"Error generating portfolio allocation visualization: {str(e)}")
    
    # 2. Forecast Directions
    try:
        symbols = []
        directions = []
        magnitudes = []
        
        for symbol, forecast in results['forecasts'].items():
            symbols.append(symbol)
            directions.append(1 if forecast['direction'] == 'up' else -1)
            magnitudes.append(abs(forecast['magnitude']))
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if d > 0 else 'red' for d in directions]
        plt.bar(symbols, [d * m for d, m in zip(directions, magnitudes)], color=colors)
        plt.title("Forecast Directions and Magnitudes")
        plt.ylabel("Forecast Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(viz_dir / "forecast_directions.png")
        plt.close()
    except Exception as e:
        logger.error(f"Error generating forecast directions visualization: {str(e)}")
    
    # 3. Risk Metrics Comparison
    try:
        risk_data = []
        for symbol, metrics in results['risk_metrics'].items():
            risk_data.append({
                'Symbol': symbol,
                'Volatility': metrics['volatility'],
                'Max Drawdown': abs(metrics['max_drawdown']),
                'Sharpe Ratio': metrics['sharpe_ratio']
            })
        
        risk_df = pd.DataFrame(risk_data)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        sns.barplot(data=risk_df, x='Symbol', y='Volatility')
        plt.title("Volatility by Symbol")
        plt.xticks(rotation=45)
        
        plt.subplot(3, 1, 2)
        sns.barplot(data=risk_df, x='Symbol', y='Max Drawdown')
        plt.title("Max Drawdown by Symbol")
        plt.xticks(rotation=45)
        
        plt.subplot(3, 1, 3)
        sns.barplot(data=risk_df, x='Symbol', y='Sharpe Ratio')
        plt.title("Sharpe Ratio by Symbol")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "risk_metrics.png")
        plt.close()
    except Exception as e:
        logger.error(f"Error generating risk metrics visualization: {str(e)}")
    
    # 4. Generate summary report
    try:
        report = []
        report.append("# FinTSB Pipeline Summary Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Forecast summary
        report.append("## Forecast Summary")
        for symbol, forecast in results['forecasts'].items():
            report.append(f"- {symbol}: {forecast['direction'].upper()} with magnitude {forecast['magnitude']:.4f}")
        
        # Portfolio summary
        report.append("\n## Portfolio Summary")
        report.append(f"- Allocation Method: {results['portfolio']['allocation_method']}")
        report.append(f"- Expected Return: {results['portfolio']['expected_return']:.4f}")
        report.append(f"- Expected Volatility: {results['portfolio']['expected_volatility']:.4f}")
        report.append(f"- Expected Sharpe: {results['portfolio']['expected_sharpe']:.4f}")
        
        report.append("\n### Portfolio Weights")
        for symbol, weight in results['portfolio']['weights'].items():
            report.append(f"- {symbol}: {weight:.4f}")
        
        report.append("\n### Trading Signals")
        for symbol, signal in results['portfolio']['signals'].items():
            report.append(f"- {symbol}: {signal}")
        
        # Risk metrics summary
        report.append("\n## Risk Metrics Summary")
        for symbol, metrics in results['risk_metrics'].items():
            report.append(f"\n### {symbol}")
            report.append(f"- Volatility: {metrics['volatility']:.4f}")
            report.append(f"- Max Drawdown: {abs(metrics['max_drawdown']):.4f}")
            report.append(f"- Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        
        # Sample explanations
        if results['explanations']:
            report.append("\n## Sample Forecast Explanations")
            
            # Take up to 3 explanations
            samples = list(results['explanations'].items())[:3]
            
            for symbol, explanation in samples:
                report.append(f"\n### {symbol}")
                report.append(explanation['text'])
        
        # Write report
        report_path = output_dir / "summary_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Summary report generated at {report_path}")
    except Exception as e:
        logger.error(f"Error generating summary report: {str(e)}")

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run FinTSB integration pipeline')
    parser.add_argument('--config', default='FinTSB/configs/config_comprehensive.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', default=None,
                        help='Output directory for results')
    parser.add_argument('--symbols', nargs='+',
                        help='Symbols to analyze (overrides config)')
    parser.add_argument('--start-date', help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for analysis (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        # Check environment
        logger.info("Checking environment")
        check_environment()
        
        # Determine output directory
        output_dir = None
        if args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"{args.output}_{timestamp}")
        
        # Modify configuration if needed
        if args.symbols or args.start_date or args.end_date:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            if args.symbols:
                config['data']['symbols'] = args.symbols
            
            if args.start_date:
                config['data']['start_date'] = args.start_date
            
            if args.end_date:
                config['data']['end_date'] = args.end_date
            
            # Write to temporary config
            temp_config = f"temp_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            with open(temp_config, 'w') as f:
                yaml.dump(config, f)
            
            config_path = temp_config
        else:
            config_path = args.config
        
        # Run the pipeline
        logger.info(f"Running FinTSB pipeline with config: {config_path}")
        results = run_pipeline(config_path, output_dir)
        
        # Clean up temporary config if created
        if config_path != args.config and os.path.exists(config_path):
            os.remove(config_path)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()