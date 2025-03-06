#!/usr/bin/env python3
"""
Command-line interface for downloading data from EastMoney.
"""
import argparse
import sys
from pathlib import Path
import logging
from typing import Optional
import yaml

from src.utils.credentials import setup_credentials
from FinTSB.data.eastmoney_downloader import EastMoneyDataDownloader

def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("EastMoneyDownloader")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    
    return logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download data from EastMoney API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets with interactive credential setup
  %(prog)s --interactive
  
  # Download specific categories
  %(prog)s --categories extreme,fall
  
  # Download with custom config
  %(prog)s --config path/to/config.yaml
  
  # Download with specific output directory
  %(prog)s --output path/to/output
"""
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactively set up API credentials'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('FinTSB/data/eastmoney_config.yaml'),
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('FinTSB/data'),
        help='Output directory for downloaded data'
    )
    
    parser.add_argument(
        '--categories',
        type=str,
        help='Comma-separated list of categories to download'
    )
    
    parser.add_argument(
        '--log-file',
        type=Path,
        help='Path to log file'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip data validation'
    )
    
    return parser.parse_args()

def validate_paths(args) -> None:
    """Validate input and output paths."""
    if not args.config.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
    args.output.mkdir(parents=True, exist_ok=True)

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    """Main execution function."""
    args = parse_args()
    logger = setup_logging(args.log_file)
    
    try:
        # Validate paths
        validate_paths(args)
        
        # Set up credentials
        credentials = setup_credentials(interactive=args.interactive)
        api_key = credentials.get_credentials()['EASTMONEY_API_KEY']
        
        # Load configuration
        config = load_config(args.config)
        
        # Initialize downloader
        downloader = EastMoneyDataDownloader(args.config, api_key)
        
        # Filter categories if specified
        if args.categories:
            categories = args.categories.split(',')
            config['categories'] = [
                cat for cat in config['categories']
                if cat['name'] in categories
            ]
            
        if not config['categories']:
            raise ValueError("No valid categories specified")
        
        # Update configuration
        if args.skip_validation:
            config['validation']['check_data_quality'] = False
        
        # Download data
        logger.info("Starting data download...")
        success_counts = downloader.download_all(args.output)
        
        # Print summary
        total_success = sum(success_counts.values())
        total_datasets = sum(cat['datasets'] for cat in config['categories'])
        
        logger.info("\nDownload Summary:")
        for category, count in success_counts.items():
            logger.info(f"{category}: {count} datasets downloaded")
        logger.info(f"\nTotal: {total_success}/{total_datasets} datasets downloaded successfully")
        
        # Exit with appropriate status
        sys.exit(0 if total_success == total_datasets else 1)
        
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()