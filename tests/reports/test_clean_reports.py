"""
Tests for report cleaning functionality.
"""
import pytest
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import os

from tests.reports.clean_reports import ReportCleaner

@pytest.fixture
def test_reports_dir(tmp_path):
    """Create temporary test reports directory."""
    reports_dir = tmp_path / "reports"
    for subdir in ['runs', 'coverage', 'logs']:
        (reports_dir / subdir).mkdir(parents=True)
    return reports_dir

@pytest.fixture
def report_cleaner(test_reports_dir):
    """Create ReportCleaner instance."""
    return ReportCleaner(test_reports_dir)

def create_test_file(path: Path, days_old: int) -> None:
    """Create test file with specified age."""
    path.write_text("test content")
    mtime = (datetime.now() - timedelta(days=days_old)).timestamp()
    os.utime(path, (mtime, mtime))

def test_clean_old_reports(report_cleaner, test_reports_dir):
    """Test cleaning old reports."""
    # Create test files with different ages
    create_test_file(test_reports_dir / "runs" / "old_run.txt", 40)  # Older than retention
    create_test_file(test_reports_dir / "runs" / "new_run.txt", 5)   # Within retention
    create_test_file(test_reports_dir / "coverage" / "old_coverage.html", 100)
    create_test_file(test_reports_dir / "coverage" / "new_coverage.html", 10)
    create_test_file(test_reports_dir / "logs" / "old_log.txt", 10)
    create_test_file(test_reports_dir / "logs" / "new_log.txt", 1)
    
    # Run cleaner
    report_cleaner.clean_old_reports()
    
    # Verify results
    assert not (test_reports_dir / "runs" / "old_run.txt").exists()
    assert (test_reports_dir / "runs" / "new_run.txt").exists()
    assert not (test_reports_dir / "coverage" / "old_coverage.html").exists()
    assert (test_reports_dir / "coverage" / "new_coverage.html").exists()
    assert not (test_reports_dir / "logs" / "old_log.txt").exists()
    assert (test_reports_dir / "logs" / "new_log.txt").exists()

def test_archive_reports(report_cleaner, test_reports_dir, tmp_path):
    """Test archiving reports."""
    # Create test files
    create_test_file(test_reports_dir / "runs" / "test_run.txt", 1)
    create_test_file(test_reports_dir / "coverage" / "test_coverage.html", 1)
    create_test_file(test_reports_dir / "logs" / "test_log.txt", 1)
    
    # Create archive directory
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    
    # Archive reports
    report_cleaner.archive_reports(archive_dir)
    
    # Verify archives
    archives = list(archive_dir.glob("*.tar.gz"))
    assert len(archives) == 3  # One for each report type
    
    # Verify archive contents
    for archive in archives:
        assert archive.exists()
        assert archive.stat().st_size > 0

def test_get_report_stats(report_cleaner, test_reports_dir):
    """Test getting report statistics."""
    # Create test files
    create_test_file(test_reports_dir / "runs" / "test1.txt", 1)
    create_test_file(test_reports_dir / "runs" / "test2.txt", 2)
    create_test_file(test_reports_dir / "coverage" / "test.html", 1)
    
    # Get stats
    stats = report_cleaner.get_report_stats()
    
    # Verify stats
    assert stats['runs']['count'] == 2
    assert stats['runs']['size'] > 0
    assert stats['coverage']['count'] == 1
    assert stats['logs']['count'] == 0
    
    # Verify timestamps
    assert isinstance(stats['runs']['oldest'], datetime)
    assert isinstance(stats['runs']['newest'], datetime)
    assert stats['runs']['newest'] > stats['runs']['oldest']

def test_dry_run(report_cleaner, test_reports_dir):
    """Test dry run mode doesn't delete files."""
    # Create old test file
    old_file = test_reports_dir / "runs" / "old_run.txt"
    create_test_file(old_file, 40)
    
    # Run cleaner in dry run mode
    report_cleaner.clean_old_reports(dry_run=True)
    
    # Verify file wasn't deleted
    assert old_file.exists()

def test_gitkeep_preservation(report_cleaner, test_reports_dir):
    """Test .gitkeep files are preserved."""
    # Create .gitkeep files
    for subdir in ['runs', 'coverage', 'logs']:
        gitkeep = test_reports_dir / subdir / ".gitkeep"
        gitkeep.touch()
        
    # Create old test files
    create_test_file(test_reports_dir / "runs" / "old_run.txt", 40)
    
    # Run cleaner
    report_cleaner.clean_old_reports()
    
    # Verify .gitkeep files still exist
    for subdir in ['runs', 'coverage', 'logs']:
        assert (test_reports_dir / subdir / ".gitkeep").exists()

def test_missing_directory_handling(report_cleaner, test_reports_dir):
    """Test handling of missing directories."""
    # Remove a directory
    shutil.rmtree(test_reports_dir / "runs")
    
    # Run cleaner - should not raise exception
    try:
        report_cleaner.clean_old_reports()
    except Exception as e:
        pytest.fail(f"clean_old_reports raised {e}")