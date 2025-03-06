"""
Tests for coverage report generator script.
"""
import pytest
from pathlib import Path
import yaml
from unittest.mock import patch, Mock, mock_open
import json

from generate_coverage_report import (
    parse_args,
    load_config,
    validate_paths,
    get_report_formats,
    main
)

@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        'coverage': {
            'requirements': {
                'minimum_coverage': 80.0,
                'fail_under': 75.0,
                'branch_coverage': True
            },
            'paths': {
                'coverage_dir': 'tests/reports/coverage',
                'trends_file': 'tests/reports/coverage/trends.json',
                'archive_dir': 'tests/reports/coverage/archive'
            },
            'notifications': {
                'email': {
                    'enabled': False,
                    'recipients': ['test@example.com']
                },
                'slack': {
                    'enabled': False,
                    'webhook_url': 'https://hooks.slack.com/test'
                }
            },
            'badges': {
                'generate': True,
                'style': 'flat'
            }
        }
    }

@pytest.fixture
def sample_coverage_data():
    """Create sample coverage data."""
    return {
        'lines': {'total': 100, 'covered': 75, 'missed': 25},
        'branches': {'total': 20, 'covered': 15, 'missed': 5},
        'timestamp': '2025-03-03T09:00:00',
        'files': [
            {
                'name': 'test.py',
                'lines': {'total': 10, 'covered': 8, 'missed': 2}
            }
        ]
    }

@pytest.fixture
def temp_project(tmp_path):
    """Create temporary project structure."""
    (tmp_path / "tests" / "reports" / "coverage").mkdir(parents=True)
    return tmp_path

class TestCoverageReportGenerator:
    """Test coverage report generator functionality."""

    def test_parse_args_defaults(self):
        """Test argument parsing with defaults."""
        with patch('sys.argv', ['script.py']):
            args = parse_args()
            assert args.coverage_file == 'coverage.xml'
            assert args.formats == ['html']
            assert not args.notify
            assert not args.badge
            assert not args.archive

    def test_parse_args_custom(self):
        """Test argument parsing with custom values."""
        test_args = [
            'script.py',
            '--coverage-file', 'test.xml',
            '--formats', 'html', 'markdown',
            '--notify',
            '--badge',
            '--archive'
        ]
        with patch('sys.argv', test_args):
            args = parse_args()
            assert args.coverage_file == 'test.xml'
            assert set(args.formats) == {'html', 'markdown'}
            assert args.notify
            assert args.badge
            assert args.archive

    def test_load_config_success(self, temp_project, sample_config):
        """Test successful config loading."""
        config_path = temp_project / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)

        config = load_config(str(config_path))
        assert config['coverage']['requirements']['minimum_coverage'] == 80.0
        assert config['coverage']['paths']['coverage_dir'] == 'tests/reports/coverage'

    def test_load_config_invalid_yaml(self, temp_project):
        """Test loading invalid YAML config."""
        config_path = temp_project / "invalid.yaml"
        config_path.write_text("invalid: {")

        config = load_config(str(config_path))
        assert config == {}

    def test_validate_paths_all_exist(self, temp_project):
        """Test path validation when all files exist."""
        coverage_file = temp_project / "coverage.xml"
        config_file = temp_project / "config.yaml"
        coverage_file.touch()
        config_file.touch()

        class Args:
            coverage_file = str(coverage_file)
            config = str(config_file)
            compare = None

        assert validate_paths(Args()) is True

    def test_validate_paths_missing_files(self):
        """Test path validation with missing files."""
        class Args:
            coverage_file = "nonexistent.xml"
            config = "nonexistent.yaml"
            compare = None

        assert validate_paths(Args()) is False

    @pytest.mark.parametrize("formats,expected", [
        (['html'], ['html']),
        (['markdown', 'json'], ['markdown', 'json']),
        (['all'], ['html', 'markdown', 'json']),
        ([], ['html']),  # Default format
    ])
    def test_get_report_formats(self, formats, expected):
        """Test report format determination."""
        assert set(get_report_formats(formats)) == set(expected)

    def test_main_success(self, temp_project, sample_config, sample_coverage_data):
        """Test successful execution of main function."""
        coverage_file = temp_project / "coverage.xml"
        config_file = temp_project / "config.yaml"
        output_dir = temp_project / "reports"
        coverage_file.touch()

        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)

        with patch('generate_coverage_report.parse_coverage_xml') as mock_parse:
            mock_parse.return_value = sample_coverage_data

            mock_reporter = Mock()
            mock_reporter.generate_reports.return_value = {
                'html': str(output_dir / 'coverage.html')
            }

            with patch('generate_coverage_report.CoverageReporter', return_value=mock_reporter):
                with patch('sys.argv', [
                    'script.py',
                    '--coverage-file', str(coverage_file),
                    '--config', str(config_file),
                    '--output-dir', str(output_dir)
                ]):
                    assert main() == 0
                    mock_reporter.generate_reports.assert_called_once()

    def test_main_with_notifications(self, temp_project, sample_config, sample_coverage_data):
        """Test main function with notifications enabled."""
        config_file = temp_project / "config.yaml"
        coverage_file = temp_project / "coverage.xml"
        coverage_file.touch()

        sample_config['coverage']['notifications']['email']['enabled'] = True
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)

        with patch('generate_coverage_report.parse_coverage_xml') as mock_parse:
            mock_parse.return_value = sample_coverage_data

            mock_reporter = Mock()
            with patch('generate_coverage_report.CoverageReporter', return_value=mock_reporter):
                with patch('sys.argv', [
                    'script.py',
                    '--coverage-file', str(coverage_file),
                    '--config', str(config_file),
                    '--notify'
                ]):
                    assert main() == 0
                    mock_reporter.notify.assert_called_once()

    def test_error_handling(self, temp_project):
        """Test error handling in main function."""
        with patch('sys.argv', ['script.py']):
            assert main() == 1  # Should fail due to missing files

    @pytest.mark.integration
    def test_full_report_generation(self, temp_project, sample_config, sample_coverage_data):
        """Integration test for full report generation."""
        # Setup test environment
        coverage_file = temp_project / "coverage.xml"
        config_file = temp_project / "config.yaml"
        output_dir = temp_project / "reports"
        
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)
            
        with open(coverage_file, 'w') as f:
            f.write("""<?xml version="1.0" ?>
            <coverage version="5.5">
                <packages><package name="test"/></packages>
            </coverage>""")

        # Run with all features enabled
        with patch('generate_coverage_report.parse_coverage_xml') as mock_parse:
            mock_parse.return_value = sample_coverage_data
            
            with patch('sys.argv', [
                'script.py',
                '--coverage-file', str(coverage_file),
                '--config', str(config_file),
                '--output-dir', str(output_dir),
                '--formats', 'all',
                '--notify',
                '--badge',
                '--archive'
            ]):
                assert main() == 0