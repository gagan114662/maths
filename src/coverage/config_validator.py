"""
Configuration validator for test coverage settings.
"""
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import jsonschema
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CoverageRequirements:
    """Coverage requirement settings."""
    minimum_coverage: float
    fail_under: float
    branch_coverage: bool

@dataclass
class CoveragePaths:
    """Coverage path settings."""
    coverage_dir: Path
    trends_file: Path
    archive_dir: Path

class CoverageConfigValidator:
    """Validates coverage configuration settings."""
    
    # Schema for coverage configuration validation
    CONFIG_SCHEMA = {
        "type": "object",
        "required": ["coverage"],
        "properties": {
            "coverage": {
                "type": "object",
                "required": ["requirements", "paths", "patterns"],
                "properties": {
                    "requirements": {
                        "type": "object",
                        "required": ["minimum_coverage", "fail_under"],
                        "properties": {
                            "minimum_coverage": {"type": "number", "minimum": 0, "maximum": 100},
                            "fail_under": {"type": "number", "minimum": 0, "maximum": 100},
                            "branch_coverage": {"type": "boolean"}
                        }
                    },
                    "paths": {
                        "type": "object",
                        "required": ["coverage_dir", "trends_file", "archive_dir"],
                        "properties": {
                            "coverage_dir": {"type": "string"},
                            "trends_file": {"type": "string"},
                            "archive_dir": {"type": "string"}
                        }
                    },
                    "patterns": {
                        "type": "object",
                        "required": ["include", "exclude"],
                        "properties": {
                            "include": {"type": "array", "items": {"type": "string"}},
                            "exclude": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            }
        }
    }
    
    def __init__(self, config_path: str):
        """Initialize validator with config path."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                self._validate_schema(config)
                return config
        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {str(e)}")
            raise
            
    def _validate_schema(self, config: Dict[str, Any]) -> None:
        """Validate configuration against schema."""
        try:
            jsonschema.validate(instance=config, schema=self.CONFIG_SCHEMA)
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"Config validation error: {str(e)}")
            raise
            
    def get_requirements(self) -> CoverageRequirements:
        """Get coverage requirements."""
        req = self.config['coverage']['requirements']
        return CoverageRequirements(
            minimum_coverage=req['minimum_coverage'],
            fail_under=req['fail_under'],
            branch_coverage=req.get('branch_coverage', False)
        )
        
    def get_paths(self) -> CoveragePaths:
        """Get coverage paths."""
        paths = self.config['coverage']['paths']
        return CoveragePaths(
            coverage_dir=Path(paths['coverage_dir']),
            trends_file=Path(paths['trends_file']),
            archive_dir=Path(paths['archive_dir'])
        )
        
    def get_patterns(self) -> Dict[str, List[str]]:
        """Get include/exclude patterns."""
        return self.config['coverage']['patterns']
        
    def validate_coverage_result(self, coverage_percent: float) -> bool:
        """Validate if coverage meets requirements."""
        requirements = self.get_requirements()
        return coverage_percent >= requirements.minimum_coverage
        
    def should_fail_build(self, coverage_percent: float) -> bool:
        """Determine if build should fail based on coverage."""
        requirements = self.get_requirements()
        return coverage_percent < requirements.fail_under
        
    def create_directories(self) -> None:
        """Create necessary directories from paths configuration."""
        paths = self.get_paths()
        
        for path in [paths.coverage_dir, paths.archive_dir]:
            path.mkdir(parents=True, exist_ok=True)
            
    def validate_paths(self) -> List[str]:
        """Validate path configurations."""
        errors = []
        paths = self.get_paths()
        
        # Check parent directories exist
        for path in [paths.coverage_dir, paths.trends_file, paths.archive_dir]:
            if not path.parent.exists():
                errors.append(f"Parent directory does not exist: {path.parent}")
                
        # Check write permissions
        for path in [paths.coverage_dir, paths.archive_dir]:
            if path.exists() and not os.access(path, os.W_OK):
                errors.append(f"No write permission for: {path}")
                
        return errors
        
    def get_report_settings(self) -> Dict[str, Any]:
        """Get report generation settings."""
        return self.config['coverage'].get('report', {})
        
    def get_notification_settings(self) -> Dict[str, Any]:
        """Get notification settings."""
        return self.config['coverage'].get('notifications', {})
        
    def get_badge_settings(self) -> Dict[str, Any]:
        """Get badge generation settings."""
        return self.config['coverage'].get('badges', {})
        
    def get_threshold_color(self, coverage_percent: float) -> str:
        """Get badge color based on coverage thresholds."""
        thresholds = self.config['coverage']['thresholds']
        badges = self.config['coverage']['badges']
        
        if coverage_percent >= thresholds['excellent']:
            return badges['colors']['excellent']
        elif coverage_percent >= thresholds['good']:
            return badges['colors']['good']
        elif coverage_percent >= thresholds['acceptable']:
            return badges['colors']['acceptable']
        elif coverage_percent >= thresholds['poor']:
            return badges['colors']['poor']
        else:
            return badges['colors']['critical']
            
    def should_notify(self, coverage_percent: float, previous_percent: Optional[float] = None) -> bool:
        """Determine if notification should be sent."""
        notifications = self.get_notification_settings()
        if not notifications.get('enabled', False):
            return False
            
        # Check if coverage dropped significantly
        if previous_percent is not None:
            threshold = self.config['coverage']['trends']['alert_threshold']
            if (previous_percent - coverage_percent) >= threshold:
                return True
                
        # Check if coverage is below minimum
        requirements = self.get_requirements()
        return coverage_percent < requirements.minimum_coverage