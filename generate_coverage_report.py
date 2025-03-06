#!/usr/bin/env python3
"""
Command-line tool to generate coverage reports.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import yaml

from src.coverage.reporter import CoverageReporter
from src.coverage.utils import parse_coverage_xml, calculate_coverage_stats
from src.coverage import DEFAULT_CONFIG_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate coverage reports in various formats"
    )
    
    parser.add_argument(
        '--coverage-file',
        default='coverage.xml',
        help='Path to coverage XML file'
    )
    
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG_PATH,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        default='reports/coverage',
        help='Output directory for reports'
    )
    
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['html', 'markdown', 'json', 'all'],
        default=['html'],
        help='Report formats to generate'
    )
    
    parser.add_argument(
        '--notify',
        action='store_true',
        help='Send notifications about coverage'
    )
    
    parser.add_argument(
        '--badge',
        action='store_true',
        help='Generate coverage badge'
    )
    
    parser.add_argument(
        '--compare',
        help='Path to previous coverage data for comparison'
    )
    
    parser.add_argument(
        '--archive',
        action='store_true',
        help='Archive the generated reports'
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        return {}

def validate_paths(args) -> bool:
    """Validate required paths exist."""
    if not Path(args.coverage_file).exists():
        logger.error(f"Coverage file not found: {args.coverage_file}")
        return False
        
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return False
        
    if args.compare and not Path(args.compare).exists():
        logger.error(f"Comparison file not found: {args.compare}")
        return False
        
    return True

def get_report_formats(formats: List[str]) -> List[str]:
    """Get list of report formats to generate."""
    if 'all' in formats:
        return ['html', 'markdown', 'json']
    return formats

def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate paths
    if not validate_paths(args):
        return 1
        
    # Load configuration
    config = load_config(args.config)
    if not config:
        return 1
        
    try:
        # Parse coverage data
        coverage_data = parse_coverage_xml(args.coverage_file)
        if not coverage_data:
            logger.error("Failed to parse coverage data")
            return 1
            
        # Calculate statistics
        stats = calculate_coverage_stats(coverage_data)
        
        # Load comparison data if specified
        comparison_data = None
        if args.compare:
            comparison_data = parse_coverage_xml(args.compare)
            
        # Initialize reporter
        reporter = CoverageReporter(config)
        
        # Generate reports
        formats = get_report_formats(args.formats)
        reports = reporter.generate_reports(
            coverage_data,
            args.output_dir,
            formats
        )
        
        if not reports:
            logger.error("Failed to generate reports")
            return 1
            
        logger.info("Generated reports:")
        for fmt, path in reports.items():
            logger.info(f"  {fmt}: {path}")
            
        # Generate badge if requested
        if args.badge:
            reporter._generate_badge(
                stats['total_coverage'],
                Path(args.output_dir) / 'coverage-badge.svg'
            )
            logger.info("Generated coverage badge")
            
        # Send notifications if requested
        if args.notify:
            reporter.notify(coverage_data, comparison_data)
            logger.info("Sent coverage notifications")
            
        # Archive reports if requested
        if args.archive:
            archive_dir = Path(args.output_dir) / 'archive'
            for path in reports.values():
                reporter.archive_report(path, str(archive_dir))
            logger.info(f"Archived reports to {archive_dir}")
            
        return 0
        
    except Exception as e:
        logger.error(f"Error generating reports: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())