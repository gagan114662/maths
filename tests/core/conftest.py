"""
Shared test configuration and fixtures for core components.
"""
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.llm_interface import LLMInterface
from src.core.mcp import ModelContextProtocol
from src.core.safety_checker import SafetyChecker
from src.core.memory_manager import MemoryManager, Base

@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        'model': 'gpt-4',
        'max_tokens': 1000,
        'temperature': 0.7,
        'max_context_length': 4096,
        'max_history': 100,
        'cache_size': 1000,
        'separate_storage_threshold': 1000,
        'position_limits': {
            'max_position': 0.1,
            'max_concentration': 0.2
        },
        'risk_limits': {
            'max_drawdown': 0.2,
            'var_limit': 0.05,
            'volatility_cap': 0.15
        },
        'trading_hours': {
            'start': '09:30',
            'end': '16:00'
        }
    }

@pytest.fixture
def temp_config_file(test_config, tmp_path):
    """Create temporary configuration file."""
    config_file = tmp_path / 'config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(test_config, f)
    return str(config_file)

@pytest.fixture
def test_db():
    """Create test database."""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    return engine

@pytest.fixture
def test_session(test_db):
    """Create test database session."""
    Session = sessionmaker(bind=test_db)
    return Session()

@pytest.fixture
def mock_openai():
    """Mock OpenAI API."""
    with patch('openai.ChatCompletion') as mock:
        mock.acreate = Mock()
        mock.acreate.return_value = {
            'choices': [{'message': {'content': 'Test response', 'role': 'assistant'}}],
            'usage': {'total_tokens': 10},
            'model': 'gpt-4'
        }
        yield mock

@pytest.fixture
def temp_memory_dir():
    """Create temporary directory for memory storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def llm_interface(test_config, mock_openai):
    """Create LLM interface instance."""
    with patch('src.core.llm_interface.load_config', return_value=test_config):
        return LLMInterface()

@pytest.fixture
def mcp(test_config):
    """Create Model Context Protocol instance."""
    with patch('src.core.mcp.load_config', return_value=test_config):
        return ModelContextProtocol()

@pytest.fixture
def safety_checker(test_config):
    """Create Safety Checker instance."""
    with patch('src.core.safety_checker.load_config', return_value=test_config):
        return SafetyChecker()

@pytest.fixture
def memory_manager(test_db, temp_memory_dir, test_config):
    """Create Memory Manager instance."""
    test_config.update({
        'db_path': 'sqlite:///:memory:',
        'memory_dir': str(temp_memory_dir)
    })
    
    with patch('src.core.memory_manager.load_config', return_value=test_config):
        manager = MemoryManager()
        manager.engine = test_db
        manager.Session = sessionmaker(bind=test_db)
        return manager

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return {
        'symbol': 'AAPL',
        'price': 150.0,
        'volume': 1000000,
        'timestamp': '2025-02-03T14:30:00Z',
        'indicators': {
            'sma_20': 148.5,
            'rsi': 65,
            'volatility': 0.15
        }
    }

@pytest.fixture
def sample_strategy_data():
    """Create sample strategy data for testing."""
    return {
        'id': 'momentum_01',
        'type': 'momentum',
        'parameters': {
            'lookback': 20,
            'threshold': 0.5,
            'position_size': 0.1
        },
        'performance': {
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.15,
            'win_rate': 0.6
        }
    }

@pytest.fixture
def sample_context():
    """Create sample context data for testing."""
    return {
        'system': {
            'role': 'trading_system',
            'capabilities': ['strategy_generation', 'risk_assessment'],
            'constraints': ['ethical_trading', 'risk_limits']
        },
        'memory': {
            'short_term': [],
            'long_term': {},
            'working': {}
        },
        'market': {
            'current_state': {},
            'indicators': {},
            'sentiment': {}
        },
        'tools': {
            'available': ['data_analysis', 'backtesting'],
            'permissions': {},
            'constraints': {}
        }
    }

def pytest_configure(config):
    """Configure pytest for the test suite."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow to run"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection for markers."""
    # Skip slow tests unless explicitly requested
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )