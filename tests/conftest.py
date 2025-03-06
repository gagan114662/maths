"""
Common test fixtures and utilities.
"""
import os
import pytest
import logging
from pathlib import Path
from typing import Dict, Any
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from src.utils.config import load_config
from src.agents import AgentFactory
from src.web.app import app as web_app
from src.monitoring.dashboard import MonitoringDashboard
from src.core.memory_manager import MemoryManager

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Load test configuration."""
    config_path = os.getenv('CONFIG_PATH', 'tests/config/test_config.yaml')
    return load_config(config_path)

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def memory_manager(test_config):
    """Create memory manager instance."""
    manager = MemoryManager(test_config)
    return manager

@pytest.fixture(scope="session")
def agent_factory(test_config, memory_manager):
    """Create agent factory instance."""
    factory = AgentFactory(config=test_config)
    factory.memory = memory_manager
    return factory

@pytest.fixture
def web_client():
    """Create test client for web interface."""
    with TestClient(web_app) as client:
        yield client

@pytest.fixture
def dashboard(test_config):
    """Create monitoring dashboard instance."""
    dashboard = MonitoringDashboard(test_config)
    yield dashboard
    dashboard.stop()

@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
        freq='1min'
    )
    
    # Create price series with known patterns
    n = len(dates)
    base_price = 100
    prices = np.random.normal(0, 0.01, n).cumsum() + base_price
    
    # Add some trends and patterns
    trends = np.sin(np.linspace(0, 8*np.pi, n)) * 5
    prices += trends
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    return data

@pytest.fixture
def sample_strategy_config():
    """Create sample strategy configuration."""
    return {
        'type': 'momentum',
        'name': 'Test Momentum Strategy',
        'parameters': {
            'lookback': 20,
            'threshold': 0.02,
            'stop_loss': 0.05,
            'take_profit': 0.1
        },
        'position_sizing': {
            'type': 'fixed',
            'size': 0.1
        },
        'risk_management': {
            'max_position_size': 0.2,
            'max_correlation': 0.7,
            'max_drawdown': 0.15
        }
    }

@pytest.fixture
def mock_websocket():
    """Create mock WebSocket connection."""
    class MockWebSocket:
        def __init__(self):
            self.sent_messages = []
            self.closed = False
            
        async def accept(self):
            pass
            
        async def send_json(self, data: Dict[str, Any]):
            self.sent_messages.append(data)
            
        async def receive_json(self):
            return {'type': 'test'}
            
        async def close(self):
            self.closed = True
            
    return MockWebSocket()

@pytest.fixture
def sample_user():
    """Create sample user data."""
    return {
        'username': 'testuser',
        'hashed_password': 'hashed_password',
        'email': 'test@example.com',
        'is_active': True
    }

@pytest.fixture
def auth_headers(sample_user):
    """Create authentication headers."""
    import jwt
    from src.web.app import SECRET_KEY, ALGORITHM
    
    token = jwt.encode(
        {'sub': sample_user['username']},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return {'Authorization': f'Bearer {token}'}

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "web: marks tests related to web interface")
    config.addinivalue_line("markers", "monitoring: marks tests related to monitoring")
    config.addinivalue_line("markers", "strategy: marks tests related to strategies")

@pytest.fixture(autouse=True)
def setup_test_env(tmp_path):
    """Setup test environment."""
    # Create necessary directories
    (tmp_path / "logs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "cache").mkdir()
    
    # Set environment variables
    os.environ['TEST_MODE'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    yield
    
    # Cleanup
    os.environ.pop('TEST_MODE', None)
    os.environ.pop('LOG_LEVEL', None)