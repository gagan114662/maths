"""
Enhanced trading strategy system integrating FinTSB and mathematricks frameworks.
"""
from pathlib import Path
import sys

# Version control
__version__ = '0.1.0'

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Module imports
from . import agents
from . import data_processors
from . import strategies
from . import training
from . import utils

__all__ = [
    'agents',
    'data_processors',
    'strategies',
    'training',
    'utils'
]

# System configuration
SYSTEM_CONFIG = {
    'frameworks': {
        'fintsb_path': PROJECT_ROOT / 'FinTSB',
        'mathematricks_path': PROJECT_ROOT / 'mathematricks'
    },
    'data': {
        'local_data_path': PROJECT_ROOT / 'data',
        'output_path': PROJECT_ROOT / 'output'
    },
    'requirements': {
        'python_version': '>=3.8.0',
        'frameworks': [
            'torch>=1.9.0',
            'xgboost>=1.5.0',
            'qlib>=0.8.0',
            'tensorflow>=2.7.0'
        ]
    }
}

# Initialize logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create necessary directories
for path in SYSTEM_CONFIG['data'].values():
    path.mkdir(exist_ok=True)
    
logger = logging.getLogger(__name__)
logger.info(f"Initialized enhanced trading system v{__version__}")