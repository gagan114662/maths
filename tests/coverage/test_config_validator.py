"""
Tests for coverage configuration validator.
"""
import pytest
import yaml
import json
from pathlib import Path
import jsonschema
from unittest.mock import patch, mock_open

from src.coverage.config_validator import CoverageConfigValidator, CoverageRequirements, CoveragePaths

@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        "coverage": {
            "requirements": {
                "minimum_coverage": 80.0,
                "fail_under": 75.0,
                "branch_coverage": True
            },
            "paths": {
                "coverage_dir": "tests/reports/coverage",
                "trends_file": "tests/reports/coverage/trends.json",
                "archive_dir": "tests/reports/coverage/archive"
            },
            "patterns": {
                "include": ["src/**/*.py", "tests/**/*.py"],
                "exclude": ["tests/data/*", "**/__init__.py"]
            },
            "thresholds": {
                "excellent": 90.0,
                "good": 80.0,
                "acceptable": 70.0,
                "poor": 60.0,
                "critical": 50.0
            },
            "badges": {
                "colors": {
                    "excellent": "green",
                    "good": "yellowgreen",
                    "acceptable": "yellow",
                    "poor": "orange",
                    "critical": "red"
                }
            },
            "notifications": {
                "enabled": True,
                "email": {
                    "enabled": True,
                    "recipients": ["test@example.com"]
                }
            },
            "trends": {
                "alert_threshold": 5.0
            }
        }
    }

@pytest.fixture
def config_file(tmp_path, sample_config):
    """Create temporary config file."""
    config_path = tmp_path / "coverage_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path

@pytest.fixture
def validator(config_file):
    """Create validator instance."""
    return CoverageConfigValidator(str(config_file))

class TestCoverageConfigValidator:
    """Tests for CoverageConfigValidator."""

    def test_load_valid_config(self, validator):
        """Test loading valid configuration."""
        assert validator.config is not None
        assert "coverage" in validator.config

    def test_invalid_config_schema(self, tmp_path):
        """Test handling invalid configuration schema."""
        invalid_config = {"coverage": {"invalid": "config"}}
        config_path = tmp_path / "invalid_config.yaml"
        
        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)
            
        with pytest.raises(jsonschema.exceptions.ValidationError):
            CoverageConfigValidator(str(config_path))

    def test_get_requirements(self, validator):
        """Test getting coverage requirements."""
        requirements = validator.get_requirements()
        
        assert isinstance(requirements, CoverageRequirements)
        assert requirements.minimum_coverage == 80.0
        assert requirements.fail_under == 75.0
        assert requirements.branch_coverage is True

    def test_get_paths(self, validator):
        """Test getting coverage paths."""
        paths = validator.get_paths()
        
        assert isinstance(paths, CoveragePaths)
        assert paths.coverage_dir.name == "coverage"
        assert paths.trends_file.name == "trends.json"
        assert paths.archive_dir.name == "archive"

    def test_get_patterns(self, validator):
        """Test getting include/exclude patterns."""
        patterns = validator.get_patterns()
        
        assert "include" in patterns
        assert "exclude" in patterns
        assert "src/**/*.py" in patterns["include"]
        assert "**/__init__.py" in patterns["exclude"]

    @pytest.mark.parametrize("coverage,expected", [
        (85.0, True),
        (75.0, False),
        (60.0, False)
    ])
    def test_validate_coverage_result(self, validator, coverage, expected):
        """Test coverage validation."""
        assert validator.validate_coverage_result(coverage) == expected

    @pytest.mark.parametrize("coverage,expected", [
        (80.0, False),
        (74.9, True),
        (50.0, True)
    ])
    def test_should_fail_build(self, validator, coverage, expected):
        """Test build failure determination."""
        assert validator.should_fail_build(coverage) == expected

    def test_create_directories(self, validator, tmp_path):
        """Test directory creation."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            validator.create_directories()
            assert mock_mkdir.call_count == 2

    def test_validate_paths(self, validator):
        """Test path validation."""
        with patch('os.access', return_value=True):
            errors = validator.validate_paths()
            assert len(errors) > 0  # Should have errors since paths don't exist

    def test_get_report_settings(self, validator):
        """Test getting report settings."""
        settings = validator.get_report_settings()
        assert isinstance(settings, dict)

    def test_get_notification_settings(self, validator):
        """Test getting notification settings."""
        settings = validator.get_notification_settings()
        assert settings["enabled"] is True
        assert "email" in settings

    @pytest.mark.parametrize("coverage,expected_color", [
        (95.0, "green"),
        (85.0, "yellowgreen"),
        (75.0, "yellow"),
        (65.0, "orange"),
        (45.0, "red")
    ])
    def test_get_threshold_color(self, validator, coverage, expected_color):
        """Test threshold color determination."""
        color = validator.get_threshold_color(coverage)
        assert color == expected_color

    @pytest.mark.parametrize("current,previous,expected", [
        (70.0, 80.0, True),   # Drop of 10%
        (75.0, 78.0, False),  # Small drop
        (70.0, None, True),   # Below minimum
        (85.0, None, False)   # Above minimum
    ])
    def test_should_notify(self, validator, current, previous, expected):
        """Test notification determination."""
        assert validator.should_notify(current, previous) == expected

    def test_file_not_found(self, tmp_path):
        """Test handling of missing config file."""
        nonexistent_file = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            CoverageConfigValidator(str(nonexistent_file))

    def test_invalid_yaml(self, tmp_path):
        """Test handling of invalid YAML."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("{invalid: yaml: content}")
        
        with pytest.raises(yaml.YAMLError):
            CoverageConfigValidator(str(invalid_yaml))