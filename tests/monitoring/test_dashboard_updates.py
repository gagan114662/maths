"""
Tests for dashboard updates functionality.
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import psutil

from src.monitoring.dashboard_updates import DashboardUpdater
from src.web.app import WebInterface

@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        'monitoring': {
            'update_interval': 1,
            'max_history_size': 100
        }
    }

@pytest.fixture
def dashboard_updater(sample_config):
    """Create dashboard updater instance."""
    return DashboardUpdater(sample_config)

@pytest.fixture
def mock_interface():
    """Mock web interface."""
    interface = Mock(spec=WebInterface)
    interface.agent_factory = Mock()
    interface.broadcast_update = AsyncMock()
    return interface

@pytest.fixture
def mock_psutil():
    """Mock psutil functions."""
    with patch('psutil.cpu_percent', return_value=50.0), \
         patch('psutil.virtual_memory', return_value=Mock(percent=60.0)), \
         patch('psutil.disk_usage', return_value=Mock(percent=70.0)):
        yield

@pytest.mark.asyncio
async def test_system_metrics_collection(dashboard_updater, mock_psutil):
    """Test system metrics collection."""
    metrics = dashboard_updater._collect_system_metrics()
    
    assert 'timestamp' in metrics
    assert metrics['cpu'] == 50.0
    assert metrics['memory'] == 60.0
    assert metrics['disk'] == 70.0
    assert 'active_strategies' in metrics
