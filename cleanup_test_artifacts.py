#!/usr/bin/env python3
"""
Utility script for cleaning up test artifacts, logs, and reports.
"""
import os
import sys
import shutil
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class TestArtifactCleaner:
    """Manages cleanup of test artifacts."""
    
    def __init__(self, config_path: str = 'config/test_config.yaml'):
        """Initialize cleaner with configuration."""
        self.config = self._load_config(config_path)
        self.base_dir = Path.cwd()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}
            
    def cleanup_reports(self, days: int = 30, dry_run: bool = False) -> None:
        """Clean up old test reports."""
        reports_dir = self.base_dir / 'tests' / 'reports'
        if not reports_dir.exists():
            logger.warning(f"Reports directory not found: {reports_dir}")
            return
            
        cutoff = datetime.now() - timedelta(days=days)
        
        for item in reports_dir.rglob('*'):
            if item.is_file() and not item.name.startswith('.'):
                try:
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    if mtime < cutoff:
                        if dry_run:
                            logger.info(f"Would delete: {item}")
                        else:
                            item.unlink()
                            logger.info(f"Deleted: {item}")
                except Exception as e:
                    logger.error(f"Error processing {item}: {str(e)}")
                    
    def cleanup_logs(self, days: int = 7, dry_run: bool = False) -> None:
        """Clean up old log files."""
        log_dirs = [
            '/var/log/trading-system',
            self.base_dir / 'logs'
        ]
        
        cutoff = datetime.now() - timedelta(days=days)
        
        for log_dir in log_dirs:
            log_dir = Path(log_dir)
            if not log_dir.exists():
                continue
                
            for item in log_dir.rglob('*.log*'):
                try:
                    if item.is_file():
                        mtime = datetime.fromtimestamp(item.stat().st_mtime)
                        if mtime < cutoff:
                            if dry_run:
                                logger.info(f"Would delete: {item}")
                            else:
                                item.unlink()
                                logger.info(f"Deleted: {item}")
                except Exception as e:
                    logger.error(f"Error processing {item}: {str(e)}")
                    
    def cleanup_cache(self, dry_run: bool = False) -> None:
        """Clean up cache directories."""
        cache_dirs = [
            self.base_dir / '.pytest_cache',
            self.base_dir / '__pycache__',
            self.base_dir / 'tests' / '__pycache__'
        ]
        
        for cache_dir in cache_dirs:
            try:
                if cache_dir.exists():
                    if dry_run:
                        logger.info(f"Would remove: {cache_dir}")
                    else:
                        shutil.rmtree(cache_dir)
                        logger.info(f"Removed: {cache_dir}")
            except Exception as e:
                logger.error(f"Error removing {cache_dir}: {str(e)}")
                
    def cleanup_coverage(self, dry_run: bool = False) -> None:
        """Clean up coverage reports."""
        coverage_files = [
            '.coverage',
            'coverage.xml',
            'htmlcov'
        ]
        
        for item in coverage_files:
            path = self.base_dir / item
            try:
                if path.exists():
                    if dry_run:
                        logger.info(f"Would remove: {path}")
                    else:
                        if path.is_file():
                            path.unlink()
                        else:
                            shutil.rmtree(path)
                        logger.info(f"Removed: {path}")
            except Exception as e:
                logger.error(f"Error removing {path}: {str(e)}")
                
    def cleanup_temp(self, dry_run: bool = False) -> None:
        """Clean up temporary test files."""
        temp_patterns = ['*.pyc', '*.pyo', '*.pyd', '*.so']
        
        for pattern in temp_patterns:
            for item in self.base_dir.rglob(pattern):
                try:
                    if dry_run:
                        logger.info(f"Would delete: {item}")
                    else:
                        item.unlink()
                        logger.info(f"Deleted: {item}")
                except Exception as e:
                    logger.error(f"Error deleting {item}: {str(e)}")
                    
    def get_disk_usage(self) -> dict:
        """Get disk usage statistics."""
        stats = {}
        
        # Check test reports
        reports_dir = self.base_dir / 'tests' / 'reports'
        if reports_dir.exists():
            stats['reports'] = sum(f.stat().st_size for f in reports_dir.rglob('*') if f.is_file())
            
        # Check logs
        log_dirs = ['/var/log/trading-system', self.base_dir / 'logs']
        stats['logs'] = 0
        for log_dir in log_dirs:
            log_dir = Path(log_dir)
            if log_dir.exists():
                stats['logs'] += sum(f.stat().st_size for f in log_dir.rglob('*') if f.is_file())
                
        # Format sizes
        for key in stats:
            stats[key] = self._format_size(stats[key])
            
        return stats
        
    def _format_size(self, size: int) -> str:
        """Format size in bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Clean up test artifacts")
    
    parser.add_argument(
        '--reports-days',
        type=int,
        default=30,
        help='Days to keep reports'
    )
    
    parser.add_argument(
        '--logs-days',
        type=int,
        default=7,
        help='Days to keep logs'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Clean everything'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show disk usage statistics'
    )
    
    args = parser.parse_args()
    
    cleaner = TestArtifactCleaner()
    
    if args.stats:
        stats = cleaner.get_disk_usage()
        print("\nDisk Usage Statistics:")
        for key, value in stats.items():
            print(f"{key.title()}: {value}")
        return 0
        
    if args.all or args.reports_days:
        cleaner.cleanup_reports(days=args.reports_days, dry_run=args.dry_run)
        
    if args.all or args.logs_days:
        cleaner.cleanup_logs(days=args.logs_days, dry_run=args.dry_run)
        
    if args.all:
        cleaner.cleanup_cache(dry_run=args.dry_run)
        cleaner.cleanup_coverage(dry_run=args.dry_run)
        cleaner.cleanup_temp(dry_run=args.dry_run)
        
    return 0

if __name__ == '__main__':
    sys.exit(main())