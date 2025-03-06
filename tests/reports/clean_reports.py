#!/usr/bin/env python3
"""
Script to clean old test reports and manage report retention.
"""
import os
import shutil
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class ReportCleaner:
    """Manages test report cleanup and retention."""
    
    def __init__(self, reports_dir: Path):
        """Initialize report cleaner."""
        self.reports_dir = Path(reports_dir)
        self.retention_periods = {
            'runs': timedelta(days=30),
            'coverage': timedelta(days=90),
            'logs': timedelta(days=7)
        }
        
    def clean_old_reports(self, dry_run: bool = False) -> None:
        """Clean reports older than retention period."""
        for report_type, retention in self.retention_periods.items():
            report_dir = self.reports_dir / report_type
            if not report_dir.exists():
                continue
                
            logger.info(f"Cleaning {report_type} reports...")
            cutoff_date = datetime.now() - retention
            
            # Process all files in directory
            for item in report_dir.glob('*'):
                # Skip .gitkeep files
                if item.name == '.gitkeep':
                    continue
                    
                # Get file modification time
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                
                # Check if file is older than retention period
                if mtime < cutoff_date:
                    if dry_run:
                        logger.info(f"Would delete: {item}")
                    else:
                        try:
                            if item.is_file():
                                item.unlink()
                            else:
                                shutil.rmtree(item)
                            logger.info(f"Deleted: {item}")
                        except Exception as e:
                            logger.error(f"Error deleting {item}: {str(e)}")
                            
    def archive_reports(self, archive_dir: Path) -> None:
        """Archive reports to specified directory."""
        archive_dir = Path(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Create archive timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for report_type in self.retention_periods.keys():
            report_dir = self.reports_dir / report_type
            if not report_dir.exists():
                continue
                
            # Create archive for this report type
            archive_path = archive_dir / f"{report_type}_{timestamp}.tar.gz"
            
            try:
                # Create tar archive
                shutil.make_archive(
                    str(archive_path.with_suffix('')),
                    'gztar',
                    report_dir
                )
                logger.info(f"Archived {report_type} reports to {archive_path}")
            except Exception as e:
                logger.error(f"Error archiving {report_type} reports: {str(e)}")
                
    def get_report_stats(self) -> dict:
        """Get statistics about reports."""
        stats = {}
        
        for report_type in self.retention_periods.keys():
            report_dir = self.reports_dir / report_type
            if not report_dir.exists():
                stats[report_type] = {'count': 0, 'size': 0}
                continue
                
            # Calculate statistics
            count = sum(1 for _ in report_dir.glob('*') if _.name != '.gitkeep')
            size = sum(f.stat().st_size for f in report_dir.glob('**/*') if f.is_file())
            
            stats[report_type] = {
                'count': count,
                'size': size,
                'oldest': self._get_oldest_file_date(report_dir),
                'newest': self._get_newest_file_date(report_dir)
            }
            
        return stats
        
    def _get_oldest_file_date(self, directory: Path) -> datetime:
        """Get date of oldest file in directory."""
        files = [f for f in directory.glob('*') if f.name != '.gitkeep']
        if not files:
            return None
        return datetime.fromtimestamp(
            min(f.stat().st_mtime for f in files)
        )
        
    def _get_newest_file_date(self, directory: Path) -> datetime:
        """Get date of newest file in directory."""
        files = [f for f in directory.glob('*') if f.name != '.gitkeep']
        if not files:
            return None
        return datetime.fromtimestamp(
            max(f.stat().st_mtime for f in files)
        )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Clean test reports')
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without deleting'
    )
    
    parser.add_argument(
        '--archive',
        type=str,
        help='Archive reports to specified directory'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show report statistics'
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize cleaner
    reports_dir = Path(__file__).parent
    cleaner = ReportCleaner(reports_dir)
    
    # Show statistics if requested
    if args.stats:
        stats = cleaner.get_report_stats()
        print("\nReport Statistics:")
        for report_type, data in stats.items():
            print(f"\n{report_type.upper()}:")
            print(f"  Count: {data['count']} files")
            print(f"  Size: {data['size'] / 1024 / 1024:.2f} MB")
            if data['oldest']:
                print(f"  Oldest: {data['oldest']}")
            if data['newest']:
                print(f"  Newest: {data['newest']}")
                
    # Archive if requested
    if args.archive:
        cleaner.archive_reports(args.archive)
        
    # Clean old reports
    cleaner.clean_old_reports(dry_run=args.dry_run)
    
    return 0

if __name__ == '__main__':
    exit(main())