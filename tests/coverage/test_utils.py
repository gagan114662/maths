"""
Tests for coverage utilities.
"""
import pytest
from pathlib import Path
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import json
from unittest.mock import patch, mock_open, Mock

from src.coverage.utils import (
    parse_coverage_xml,
    calculate_coverage_stats,
    find_uncovered_lines,
    merge_coverage_data,
    generate_coverage_badge,
    cleanup_old_reports,
    archive_coverage_data,
    compare_coverage
)

@pytest.fixture
def sample_coverage_xml(tmp_path):
    """Create sample coverage XML file."""
    xml_content = """<?xml version="1.0" ?>
    <coverage version="5.5" timestamp="1614556800" lines-valid="100" lines-covered="75" branches-valid="20" branches-covered="15">
        <packages>
            <package name="src">
                <classes>
                    <class name="module1.py">
                        <lines>
                            <line number="1" hits="1"/>
                            <line number="2" hits="1"/>
                            <line number="3" hits="0"/>
                            <line number="4" hits="1"/>
                        </lines>
                    </class>
                    <class name="module2.py">
                        <lines>
                            <line number="1" hits="1"/>
                            <line number="2" hits="0"/>
                            <line number="3" hits="1"/>
                        </lines>
                    </class>
                </classes>
            </package>
        </packages>
    </coverage>"""
    
    xml_file = tmp_path / "coverage.xml"
    xml_file.write_text(xml_content)
    return xml_file

@pytest.fixture
def sample_coverage_data():
    """Create sample coverage data."""
    return {
        'lines': {'total': 100, 'covered': 75, 'missed': 25},
        'branches': {'total': 20, 'covered': 15, 'missed': 5},
        'files': [
            {'name': 'module1.py', 'lines': {'total': 4, 'covered': 3, 'missed': 1}},
            {'name': 'module2.py', 'lines': {'total': 3, 'covered': 2, 'missed': 1}}
        ]
    }

class TestCoverageUtils:
    """Tests for coverage utilities."""
    
    def test_parse_coverage_xml(self, sample_coverage_xml):
        """Test parsing coverage XML."""
        data = parse_coverage_xml(str(sample_coverage_xml))
        
        assert data['lines']['total'] == 100
        assert data['lines']['covered'] == 75
        assert data['branches']['total'] == 20
        assert data['branches']['covered'] == 15
        assert len(data['files']) == 2
    
    def test_calculate_coverage_stats(self, sample_coverage_data):
        """Test coverage statistics calculation."""
        stats = calculate_coverage_stats(sample_coverage_data)
        
        assert stats['line_coverage'] == 75.0
        assert stats['branch_coverage'] == 75.0
        assert stats['total_coverage'] == 75.0
    
    def test_find_uncovered_lines(self, sample_coverage_xml):
        """Test finding uncovered lines."""
        data = {'xml_file': str(sample_coverage_xml)}
        uncovered = find_uncovered_lines(data)
        
        assert 'module1.py' in uncovered
        assert 'module2.py' in uncovered
        assert 3 in uncovered['module1.py']
        assert 2 in uncovered['module2.py']
    
    def test_merge_coverage_data(self, sample_coverage_xml):
        """Test merging coverage data."""
        data_files = [str(sample_coverage_xml), str(sample_coverage_xml)]
        merged = merge_coverage_data(data_files)
        
        assert merged['lines']['total'] == 200
        assert merged['lines']['covered'] == 150
        assert merged['branches']['total'] == 40
        assert merged['branches']['covered'] == 30
    
    def test_generate_coverage_badge(self, tmp_path):
        """Test coverage badge generation."""
        thresholds = {
            'excellent': 90.0,
            'good': 80.0,
            'acceptable': 70.0,
            'poor': 60.0,
            'critical': 50.0
        }
        
        output_path = tmp_path / "coverage.svg"
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.content = b"badge content"
            
            generate_coverage_badge(75.0, thresholds, str(output_path))
            
            assert output_path.exists()
            assert output_path.read_bytes() == b"badge content"
    
    def test_cleanup_old_reports(self, tmp_path):
        """Test cleaning up old reports."""
        # Create old and new files
        old_file = tmp_path / "old_report.xml"
        new_file = tmp_path / "new_report.xml"
        
        old_file.touch()
        new_file.touch()
        
        # Set old file's modification time
        old_time = datetime.now() - timedelta(days=40)
        os.utime(str(old_file), (old_time.timestamp(), old_time.timestamp()))
        
        cleanup_old_reports(str(tmp_path), max_age_days=30)
        
        assert not old_file.exists()
        assert new_file.exists()
    
    def test_archive_coverage_data(self, tmp_path, sample_coverage_data):
        """Test archiving coverage data."""
        archive_dir = tmp_path / "archive"
        archive_coverage_data(sample_coverage_data, str(archive_dir))
        
        archive_files = list(archive_dir.glob("*.json"))
        assert len(archive_files) == 1
        
        with open(archive_files[0]) as f:
            archived_data = json.load(f)
            assert archived_data == sample_coverage_data
    
    def test_compare_coverage(self, sample_coverage_data):
        """Test comparing coverage data."""
        current = sample_coverage_data
        previous = {
            'lines': {'total': 90, 'covered': 60, 'missed': 30},
            'branches': {'total': 18, 'covered': 12, 'missed': 6},
            'files': [
                {'name': 'module1.py', 'lines': {'total': 4, 'covered': 3, 'missed': 1}},
                {'name': 'old_module.py', 'lines': {'total': 3, 'covered': 2, 'missed': 1}}
            ]
        }
        
        comparison = compare_coverage(current, previous)
        
        assert comparison['line_coverage_change'] == pytest.approx(8.33, 0.01)
        assert comparison['branch_coverage_change'] == pytest.approx(8.33, 0.01)
        assert 'module2.py' in comparison['new_files']
        assert 'old_module.py' in comparison['removed_files']
    
    def test_error_handling_invalid_xml(self, tmp_path):
        """Test error handling with invalid XML."""
        invalid_xml = tmp_path / "invalid.xml"
        invalid_xml.write_text("invalid xml content")
        
        data = parse_coverage_xml(str(invalid_xml))
        assert data == {}
    
    def test_error_handling_missing_file(self):
        """Test error handling with missing file."""
        data = parse_coverage_xml("nonexistent.xml")
        assert data == {}
    
    def test_zero_division_handling(self):
        """Test handling of zero division cases."""
        data = {
            'lines': {'total': 0, 'covered': 0, 'missed': 0},
            'branches': {'total': 0, 'covered': 0, 'missed': 0},
            'files': []
        }
        
        stats = calculate_coverage_stats(data)
        assert stats['line_coverage'] == 0.0
        assert stats['branch_coverage'] == 0.0
        assert stats['total_coverage'] == 0.0