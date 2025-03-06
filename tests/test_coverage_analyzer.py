"""
Tests for coverage trend analyzer.
"""
import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import pandas as pd
from unittest.mock import patch, mock_open

from analyze_coverage_trends import CoverageAnalyzer

@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary test directory."""
    return tmp_path

@pytest.fixture
def sample_coverage_xml(temp_test_dir):
    """Create sample coverage XML file."""
    xml_content = """<?xml version="1.0" ?>
    <coverage version="5.5" timestamp="1614556800" lines-valid="100" lines-covered="75">
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
    
    xml_file = temp_test_dir / "coverage.xml"
    xml_file.write_text(xml_content)
    return xml_file

@pytest.fixture
def sample_trends():
    """Create sample coverage trends data."""
    return [
        {
            "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
            "total_lines": 100,
            "covered_lines": 70,
            "coverage_percent": 70.0,
            "files_analyzed": 2
        },
        {
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "total_lines": 110,
            "covered_lines": 80,
            "coverage_percent": 72.73,
            "files_analyzed": 2
        }
    ]

@pytest.fixture
def analyzer(temp_test_dir):
    """Create CoverageAnalyzer instance."""
    return CoverageAnalyzer(str(temp_test_dir))

class TestCoverageAnalyzer:
    """Tests for CoverageAnalyzer class."""
    
    def test_analyze_current_coverage(self, analyzer, sample_coverage_xml):
        """Test analyzing current coverage from XML."""
        coverage = analyzer.analyze_current_coverage(str(sample_coverage_xml))
        
        assert coverage is not None
        assert coverage['total_lines'] == 7
        assert coverage['covered_lines'] == 5
        assert coverage['coverage_percent'] == pytest.approx(71.43, 0.01)
        assert coverage['files_analyzed'] == 2
        assert 'timestamp' in coverage
    
    def test_load_empty_trends(self, analyzer):
        """Test loading trends when no file exists."""
        trends = analyzer.load_trends()
        assert trends == []
    
    def test_save_and_load_trends(self, analyzer, sample_trends):
        """Test saving and loading trends."""
        analyzer.save_trends(sample_trends)
        loaded_trends = analyzer.load_trends()
        
        assert len(loaded_trends) == len(sample_trends)
        assert loaded_trends[0]['coverage_percent'] == 70.0
        assert loaded_trends[1]['coverage_percent'] == 72.73
    
    def test_update_trends(self, analyzer, sample_coverage_xml):
        """Test updating trends with new coverage data."""
        # First analyze coverage
        analyzer.analyze_current_coverage(str(sample_coverage_xml))
        
        # Then update trends
        analyzer.update_trends()
        
        # Verify trends were updated
        trends = analyzer.load_trends()
        assert len(trends) == 1
        assert trends[0]['coverage_percent'] == pytest.approx(71.43, 0.01)
    
    def test_generate_report(self, analyzer, sample_trends, temp_test_dir):
        """Test report generation."""
        # Save sample trends
        analyzer.save_trends(sample_trends)
        
        # Generate report
        output_dir = temp_test_dir / 'reports'
        analyzer.generate_report(str(output_dir))
        
        # Verify report files
        assert (output_dir / 'coverage_trend.png').exists()
        assert (output_dir / 'lines_trend.png').exists()
        assert (output_dir / 'coverage_summary.json').exists()
        assert (output_dir / 'coverage_report.md').exists()
    
    def test_report_contents(self, analyzer, sample_trends, temp_test_dir):
        """Test report file contents."""
        analyzer.save_trends(sample_trends)
        output_dir = temp_test_dir / 'reports'
        analyzer.generate_report(str(output_dir))
        
        # Check summary JSON
        with open(output_dir / 'coverage_summary.json') as f:
            summary = json.load(f)
            assert 'current_coverage' in summary
            assert 'coverage_change' in summary
            assert 'total_lines' in summary
    
    def test_error_handling(self, analyzer, temp_test_dir):
        """Test error handling scenarios."""
        # Invalid XML file
        invalid_xml = temp_test_dir / "invalid.xml"
        invalid_xml.write_text("invalid content")
        result = analyzer.analyze_current_coverage(str(invalid_xml))
        assert result is None
        
        # Invalid trends file
        trends_file = Path(analyzer.coverage_dir) / 'trends.json'
        trends_file.parent.mkdir(parents=True, exist_ok=True)
        trends_file.write_text("invalid json")
        assert analyzer.load_trends() == []
    
    def test_empty_coverage_file(self, analyzer, temp_test_dir):
        """Test handling of empty coverage file."""
        empty_xml = temp_test_dir / "empty.xml"
        empty_xml.write_text("<?xml version='1.0'?><coverage></coverage>")
        
        coverage = analyzer.analyze_current_coverage(str(empty_xml))
        assert coverage['total_lines'] == 0
        assert coverage['covered_lines'] == 0
        assert coverage['coverage_percent'] == 0
    
    @pytest.mark.parametrize("total,covered,expected", [
        (100, 75, 75.0),
        (100, 0, 0.0),
        (0, 0, 0.0),
        (100, 100, 100.0)
    ])
    def test_coverage_calculation(self, analyzer, total, covered, expected):
        """Test coverage percentage calculation with different values."""
        analyzer.current_coverage = {
            'timestamp': datetime.now().isoformat(),
            'total_lines': total,
            'covered_lines': covered,
            'coverage_percent': (covered / total * 100) if total > 0 else 0,
            'files_analyzed': 1
        }
        
        assert analyzer.current_coverage['coverage_percent'] == expected

def test_main_function(temp_test_dir, sample_coverage_xml):
    """Test main function execution."""
    from analyze_coverage_trends import main
    
    with patch('sys.argv', [
        'analyze_coverage_trends.py',
        '--coverage-file', str(sample_coverage_xml),
        '--output-dir', str(temp_test_dir / 'reports'),
        '--update'
    ]):
        assert main() == 0
        assert (temp_test_dir / 'reports' / 'coverage_report.md').exists()