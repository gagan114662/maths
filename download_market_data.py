#!/usr/bin/env python3
"""
Market Data Downloader Script.
Downloads, processes, and caches market data from multiple sources.
Ensures 10-year history and proper data quality.
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('market_data.log')
    ]
)
logger = logging.getLogger('MarketDataDownloader')

# Import our market data processor
from src.data_processors.market_data_processor import MarketDataProcessor

def parse_symbols_file(file_path: str) -> list:
    """Parse a file containing symbols, one per line."""
    try:
        with open(file_path, 'r') as f:
            # Read lines and strip whitespace
            symbols = [line.strip() for line in f.readlines()]
            # Remove empty lines and comments
            symbols = [s for s in symbols if s and not s.startswith('#')]
            return symbols
    except Exception as e:
        logger.error(f"Error reading symbols file {file_path}: {str(e)}")
        return []

def download_data(args):
    """Download market data based on provided arguments."""
    # Initialize the market data processor
    processor = MarketDataProcessor(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir
    )
    
    # Get list of symbols
    symbols = []
    if args.symbols:
        symbols = args.symbols
    elif args.symbols_file:
        symbols = parse_symbols_file(args.symbols_file)
    else:
        # Default to some common symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    logger.info(f"Processing {len(symbols)} symbols: {', '.join(symbols[:5])}{' and more' if len(symbols) > 5 else ''}")
    
    # Calculate start date if not provided (10 years ago)
    if not args.start_date:
        start_date = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
    else:
        start_date = args.start_date
        
    # Calculate end date if not provided (today)
    if not args.end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
    
    # Download and process the data
    data = processor.get_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        source=args.source,
        timeframe=args.timeframe,
        clean=not args.no_clean,
        augment=args.augment,
        use_cache=not args.no_cache,
        force_download=args.force_download
    )
    
    logger.info(f"Successfully processed data for {len(data)} symbols")
    
    # Get data quality report
    quality_report = processor.get_data_quality_report()
    
    # Generate quality report
    quality_summary = {"symbol": [], "completeness": [], "missing_percentage": [], "overall_score": []}
    for symbol, metrics in quality_report.items():
        quality_summary["symbol"].append(symbol)
        quality_summary["completeness"].append(metrics.get("completeness", 0))
        quality_summary["missing_percentage"].append(metrics.get("missing_percentage", 0))
        quality_summary["overall_score"].append(metrics.get("overall_score", 0))
    
    quality_df = pd.DataFrame(quality_summary)
    
    # Export data if requested
    if args.export:
        export_dir = args.export_dir if args.export_dir else "exported_data"
        export_paths = processor.export_data(data, export_dir, args.export_format)
        logger.info(f"Exported data to {export_dir}")
        
        # Export quality report
        quality_path = Path(export_dir) / "data_quality_report.csv"
        quality_df.to_csv(quality_path)
        logger.info(f"Exported data quality report to {quality_path}")
    
    # Generate visualizations if requested
    if args.visualize:
        generate_visualizations(data, quality_df, args.visualize_dir)
    
    return data, quality_df

def generate_visualizations(data: dict, quality_df: pd.DataFrame, output_dir: str = "visualizations"):
    """Generate visualizations of the data and quality metrics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating visualizations in {output_path}")
    
    # 1. Data Quality Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    sns.barplot(data=quality_df, x='symbol', y='completeness')
    plt.title('Data Completeness by Symbol')
    plt.xticks(rotation=45)
    
    plt.subplot(3, 1, 2)
    sns.barplot(data=quality_df, x='symbol', y='missing_percentage')
    plt.title('Missing Data Percentage by Symbol')
    plt.xticks(rotation=45)
    
    plt.subplot(3, 1, 3)
    sns.barplot(data=quality_df, x='symbol', y='overall_score')
    plt.title('Overall Data Quality Score by Symbol')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path / "data_quality.png")
    plt.close()
    
    # 2. Price History Visualization (up to 5 symbols)
    plt.figure(figsize=(12, 8))
    
    for i, (symbol, df) in enumerate(list(data.items())[:5]):  # Limit to 5 symbols
        if 'close' in df.columns:
            plt.plot(df.index, df['close'], label=symbol)
    
    plt.title('Price History')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / "price_history.png")
    plt.close()
    
    # 3. Volatility Comparison
    plt.figure(figsize=(12, 6))
    
    volatilities = []
    symbols = []
    
    for symbol, df in data.items():
        if 'returns' in df.columns:
            volatility = df['returns'].std() * 100  # Convert to percentage
            volatilities.append(volatility)
            symbols.append(symbol)
    
    # Create volatility DataFrame
    vol_df = pd.DataFrame({'symbol': symbols, 'volatility': volatilities})
    vol_df = vol_df.sort_values('volatility', ascending=False)
    
    sns.barplot(data=vol_df, x='symbol', y='volatility')
    plt.title('Volatility by Symbol (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / "volatility_comparison.png")
    plt.close()
    
    logger.info(f"Generated visualizations in {output_path}")

def main():
    """Main function to parse arguments and execute the download."""
    parser = argparse.ArgumentParser(description='Download and process market data')
    
    # Symbol selection
    symbol_group = parser.add_mutually_exclusive_group()
    symbol_group.add_argument('--symbols', nargs='+', help='List of symbols to download')
    symbol_group.add_argument('--symbols-file', help='Path to file containing symbols (one per line)')
    
    # Date range
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    # Data source and format
    parser.add_argument('--source', default='yahoo', 
                        choices=['yahoo', 'ibkr', 'alphavantage', 'kraken', 'eastmoney'],
                        help='Data source')
    parser.add_argument('--timeframe', default='daily',
                        choices=['daily', 'hourly', 'minute', 'weekly', 'monthly'],
                        help='Data timeframe')
    
    # Processing options
    parser.add_argument('--no-clean', action='store_true', help='Skip data cleaning')
    parser.add_argument('--augment', action='store_true', help='Augment data with synthetic samples')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--force-download', action='store_true', help='Force download even if cached data exists')
    
    # Output options
    parser.add_argument('--export', action='store_true', help='Export data to files')
    parser.add_argument('--export-dir', help='Directory for exported data')
    parser.add_argument('--export-format', default='csv',
                        choices=['csv', 'json', 'excel', 'parquet'],
                        help='Export format')
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--visualize-dir', default='visualizations', help='Directory for visualizations')
    
    # Directory options
    parser.add_argument('--data-dir', help='Directory for storing raw data')
    parser.add_argument('--cache-dir', help='Directory for caching processed data')
    
    args = parser.parse_args()
    
    try:
        # Download and process the data
        data, quality_df = download_data(args)
        
        # Print summary
        print("\nData Processing Summary")
        print("=" * 50)
        print(f"Total symbols processed: {len(data)}")
        print(f"Date range: {args.start_date or '10 years ago'} to {args.end_date or 'today'}")
        print(f"Data source: {args.source}")
        print(f"Timeframe: {args.timeframe}")
        
        print("\nData Quality Summary")
        print("=" * 50)
        print(quality_df.to_string(index=False))
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing market data: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()