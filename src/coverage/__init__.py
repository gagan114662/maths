"""
Coverage analysis and management package.
"""
from .config_validator import CoverageConfigValidator, CoverageRequirements, CoveragePaths

__all__ = ['CoverageConfigValidator', 'CoverageRequirements', 'CoveragePaths']

# Package version
__version__ = '1.0.0'

# Default configuration paths
DEFAULT_CONFIG_PATH = 'config/coverage_config.yaml'
DEFAULT_COVERAGE_DIR = 'tests/reports/coverage'
DEFAULT_TRENDS_FILE = 'tests/reports/coverage/trends.json'
DEFAULT_ARCHIVE_DIR = 'tests/reports/coverage/archive'

# Coverage thresholds
COVERAGE_THRESHOLDS = {
    'excellent': 90.0,
    'good': 80.0,
    'acceptable': 70.0,
    'poor': 60.0,
    'critical': 50.0
}

# Badge colors
BADGE_COLORS = {
    'excellent': 'green',
    'good': 'yellowgreen',
    'acceptable': 'yellow',
    'poor': 'orange',
    'critical': 'red'
}

# Report formats
REPORT_FORMATS = ['xml', 'html', 'json', 'term', 'term-missing']

# Default patterns
DEFAULT_INCLUDE_PATTERNS = ['src/**/*.py', 'tests/**/*.py']
DEFAULT_EXCLUDE_PATTERNS = [
    'tests/data/*',
    '**/__init__.py',
    'setup.py',
    'conf.py',
    'tests/*/data/*'
]

# Default notification settings
DEFAULT_NOTIFICATION_SETTINGS = {
    'enabled': True,
    'email': {
        'enabled': True,
        'on_failure': True,
        'on_success': False
    },
    'slack': {
        'enabled': False,
        'on_failure': True,
        'on_success': False
    }
}

# Coverage collection settings
DEFAULT_COLLECTION_SETTINGS = {
    'branch': True,
    'concurrency': ['thread', 'multiprocessing'],
    'data_file': '.coverage',
    'context': None,
    'debug': {
        'trace': False,
        'config': False,
        'sys': False
    }
}
