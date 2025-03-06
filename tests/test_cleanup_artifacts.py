"""
Tests for test artifact cleanup utility.
"""
import pytest
from pathlib import Path
from datetime import datetime, timedelta
import os
import shutil
from unittest.mock import Mock, patch

from cleanup_test_artifacts import TestArtifactCleaner, main

@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary test directory."""
    return tmp_path

@pytest.fixture
def sample_config(temp_test_dir):
    """Create sample configuration file."""
    config_path = temp_test_dir / "test_config.yaml"
    config_content = """
    cleanup:
        reports_retention_days: 30
        logs_retention_days: 7
        temp_patterns: ["*.pyc", "*.pyo", "*.pyd", "*.so"]
    """
    config_path.write_text(config_content)
    return str(config_path)

@pytest.fixture
def test_env(temp_test_dir):
    """Create test environment with sample files."""
    # Create directory structure
    (temp_test_dir / "tests" / "reports").mkdir(parents=True)
    (temp_test_dir / "logs").mkdir()
    (temp_test_dir / ".pytest_cache").mkdir()
    (temp_test_dir / "__pycache__").mkdir()
    
    # Function to create files with specific dates
    def create_dated_file(path: Path, days_old: int, content: str = "test content"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        mtime = (datetime.now() - timedelta(days=days_old)).timestamp()
        os.utime(path, (mtime, mtime))
        return path
    
    # Create test reports
    create_dated_file(temp_test_dir / "tests" / "reports" / "old_report.txt", 40)
    create_dated_file(temp_test_dir / "tests" / "reports" / "new_report.txt", 5)
    
    # Create log files
    create_dated_file(temp_test_dir / "logs" / "old.log", 10)
    create_dated_file(temp_test_dir / "logs" / "current.log", 1)
    create_dated_file(temp_test_dir / "logs" / "error.log", 2)
    
    # Create coverage files
    create_dated_file(temp_test_dir / ".coverage", 1)
    create_dated_file(temp_test_dir / "coverage.xml", 1)
    (temp_test_dir / "htmlcov").mkdir()
    create_dated_file(temp_test_dir / "htmlcov" / "index.html", 1)
    
    # Create cache files
    create_dated_file(temp_test_dir / "__pycache__" / "test.pyc", 1)
    create_dated_file(temp_test_dir / ".pytest_cache" / "v" / "cache.py", 1)
    
    return temp_test_dir

@pytest.fixture
def cleaner(sample_config, test_env):
    """Create TestArtifactCleaner instance."""
    with patch('pathlib.Path.cwd', return_value=test_env):
        return TestArtifactCleaner(sample_config)

def test_cleanup_reports(cleaner, test_env):
    """Test cleaning up old reports."""
    reports_dir = test_env / "tests" / "reports"
    
    # Run cleanup
    cleaner.cleanup_reports(days=30, dry_run=False)
    
    # Verify
    assert not (reports_dir / "old_report.txt").exists()
    assert (reports_dir / "new_report.txt").exists()

def test_cleanup_logs(cleaner, test_env):
    """Test cleaning up old logs."""
    logs_dir = test_env / "logs"
    
    # Run cleanup
    cleaner.cleanup_logs(days=7, dry_run=False)
    
    # Verify
    assert not (logs_dir / "old.log").exists()
    assert (logs_dir / "current.log").exists()
    assert (logs_dir / "error.log").exists()

def test_cleanup_cache(cleaner, test_env):
    """Test cleaning up cache directories."""
    # Run cleanup
    cleaner.cleanup_cache(dry_run=False)
    
    # Verify
    assert not (test_env / "__pycache__").exists()
    assert not (test_env / ".pytest_cache").exists()

def test_cleanup_coverage(cleaner, test_env):
    """Test cleaning up coverage files."""
    # Run cleanup
    cleaner.cleanup_coverage(dry_run=False)
    
    # Verify
    assert not (test_env / ".coverage").exists()
    assert not (test_env / "coverage.xml").exists()
    assert not (test_env / "htmlcov").exists()

def test_dry_run_mode(cleaner, test_env):
    """Test dry run mode doesn't delete files."""
    reports_dir = test_env / "tests" / "reports"
    old_report = reports_dir / "old_report.txt"
    
    # Run cleanup in dry run mode
    cleaner.cleanup_reports(days=30, dry_run=True)
    cleaner.cleanup_logs(days=7, dry_run=True)
    cleaner.cleanup_cache(dry_run=True)
    cleaner.cleanup_coverage(dry_run=True)
    
    # Verify nothing was deleted
    assert old_report.exists()
    assert (test_env / "logs" / "old.log").exists()
    assert (test_env / "__pycache__").exists()
    assert (test_env / ".coverage").exists()

def test_get_disk_usage(cleaner, test_env):
    """Test getting disk usage statistics."""
    stats = cleaner.get_disk_usage()
    
    # Verify stats structure
    assert 'reports' in stats
    assert 'logs' in stats
    assert isinstance(stats['reports'], str)
    assert isinstance(stats['logs'], str)
    
    # Verify size formatting
    assert 'B' in stats['reports'] or 'KB' in stats['reports']
    assert 'B' in stats['logs'] or 'KB' in stats['logs']

@pytest.mark.parametrize("size,expected", [
    (100, "100.0B"),
    (1500, "1.5KB"),
    (1500000, "1.4MB"),
    (1500000000, "1.4GB"),
    (1500000000000, "1.4TB")
])
def test_size_formatting(cleaner, size, expected):
    """Test size formatting."""
    assert cleaner._format_size(size) == expected

def test_error_handling(cleaner, test_env):
    """Test error handling for various scenarios."""
    # Test with non-existent directory
    shutil.rmtree(test_env / "tests" / "reports")
    cleaner.cleanup_reports()  # Should not raise exception
    
    # Test with permission error
    with patch('pathlib.Path.unlink', side_effect=PermissionError):
        cleaner.cleanup_logs()  # Should not raise exception
        
    # Test with invalid config
    with patch('yaml.safe_load', side_effect=Exception):
        TestArtifactCleaner("invalid_config.yaml")  # Should not raise exception

def test_main_function():
    """Test main function with different arguments."""
    with patch('sys.argv', ['cleanup_test_artifacts.py', '--stats']):
        assert main() == 0
        
    with patch('sys.argv', ['cleanup_test_artifacts.py', '--all', '--dry-run']):
        assert main() == 0
        
    with patch('sys.argv', ['cleanup_test_artifacts.py', '--reports-days', '7']):
        assert main() == 0

if __name__ == '__main__':
    pytest.main(['-v', __file__])