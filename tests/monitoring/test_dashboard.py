"""
Tests for monitoring dashboard.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import threading
import queue
import time

from src.monitoring.dashboard import MonitoringDashboard

@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        'max_history_size': 100,
        'update_intervals': {
            'system': 5,
            'strategy': 10,
            'execution': 5,
            'risk': 15
        }
    }

@pytest.fixture
def dashboard(sample_config):
    """Create dashboard instance."""
    with patch('src.monitoring.dashboard.dash.Dash') as mock_dash:
        dashboard = MonitoringDashboard(sample_config)
        yield dashboard

def test_dashboard_initialization(dashboard):
    """Test dashboard initialization."""
    assert dashboard.config is not None
    assert isinstance(dashboard.data_queue, queue.Queue)
    assert all(category in dashboard.metrics_history 
              for category in ['system', 'strategies', 'execution', 'performance'])
    assert not dashboard.collection_active

@pytest.mark.asyncio
async def test_metrics_collection(dashboard):
    """Test metrics collection process."""
    with patch('psutil.cpu_percent', return_value=50.0), \
         patch('psutil.virtual_memory', return_value=Mock(
             percent=60.0,
             used=8000000000,
             total=16000000000
         )):
        
        # Start collection
        dashboard.collection_active = True
        collection_thread = threading.Thread(
            target=dashboard._collect_metrics,
            daemon=True
        )
        collection_thread.start()
        
        # Wait for some metrics to be collected
        time.sleep(2)
        
        # Stop collection
        dashboard.collection_active = False
        collection_thread.join()
        
        # Verify metrics
        assert len(dashboard.metrics_history['system']) > 0
        latest_metrics = dashboard.metrics_history['system'][-1]
        assert 'cpu_percent' in latest_metrics
        assert 'memory_percent' in latest_metrics
        assert latest_metrics['cpu_percent'] == 50.0
        assert latest_metrics['memory_percent'] == 60.0

def test_metrics_update(dashboard):
    """Test updating metrics through queue."""
    test_metrics = {
        'strategy': 'test_strategy',
        'returns': 0.1,
        'drawdown': -0.05
    }
    
    dashboard.update_metrics('strategies', test_metrics)
    
    # Verify queue
    data = dashboard.data_queue.get()
    assert data['category'] == 'strategies'
    assert data['strategy'] == 'test_strategy'
    assert data['returns'] == 0.1
    assert data['drawdown'] == -0.05
    assert 'timestamp' in data

def test_history_size_limit(dashboard):
    """Test history size limitation."""
    max_size = dashboard.config['max_history_size']
    
    # Add more metrics than max_size
    for i in range(max_size + 10):
        dashboard.metrics_history['system'].append({
            'timestamp': datetime.now(),
            'cpu_percent': 50.0,
            'memory_percent': 60.0
        })
        
    # Process metrics
    dashboard._collect_metrics()
    
    # Verify size limit
    assert len(dashboard.metrics_history['system']) <= max_size

def test_system_metrics_figure(dashboard):
    """Test system metrics figure creation."""
    # Add sample metrics
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(minutes=5),
        end=datetime.now(),
        freq='1min'
    )
    
    for ts in timestamps:
        dashboard.metrics_history['system'].append({
            'timestamp': ts,
            'cpu_percent': np.random.uniform(0, 100),
            'memory_percent': np.random.uniform(0, 100)
        })
    
    figure = dashboard._create_system_metrics_figure()
    
    assert figure is not None
    assert len(figure.data) == 2  # CPU and Memory traces
    assert figure.data[0].name == 'CPU Usage %'
    assert figure.data[1].name == 'Memory Usage %'

def test_strategy_figure(dashboard):
    """Test strategy performance figure creation."""
    # Add sample strategy metrics
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(minutes=5),
        end=datetime.now(),
        freq='1min'
    )
    
    for ts in timestamps:
        dashboard.metrics_history['strategies'].append({
            'timestamp': ts,
            'strategy': 'test_strategy',
            'returns': np.random.uniform(-0.1, 0.1),
            'drawdown': np.random.uniform(-0.2, 0)
        })
    
    figure = dashboard._create_strategy_figure('test_strategy')
    
    assert figure is not None
    assert len(figure.data) == 2  # Returns and Drawdown traces
    assert figure.data[0].name == 'Returns'
    assert figure.data[1].name == 'Drawdown'

def test_execution_figure(dashboard):
    """Test execution metrics figure creation."""
    # Add sample execution metrics
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(minutes=5),
        end=datetime.now(),
        freq='1min'
    )
    
    for ts in timestamps:
        dashboard.metrics_history['execution'].append({
            'timestamp': ts,
            'orders': np.random.randint(0, 10),
            'fill_rate': np.random.uniform(0.8, 1.0)
        })
    
    figure = dashboard._create_execution_figure()
    
    assert figure is not None
    assert len(figure.data) == 2  # Orders and Fill Rate traces
    assert figure.data[0].name == 'Orders'
    assert figure.data[1].name == 'Fill Rate'

def test_risk_figure(dashboard):
    """Test risk metrics figure creation."""
    # Add sample risk metrics
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(minutes=5),
        end=datetime.now(),
        freq='1min'
    )
    
    for ts in timestamps:
        dashboard.metrics_history['performance'].append({
            'timestamp': ts,
            'var': np.random.uniform(-0.1, -0.01),
            'sharpe': np.random.uniform(0, 2)
        })
    
    figure = dashboard._create_risk_figure()
    
    assert figure is not None
    assert len(figure.data) == 2  # VaR and Sharpe traces
    assert figure.data[0].name == 'VaR'
    assert figure.data[1].name == 'Sharpe Ratio'

def test_dashboard_start_stop(dashboard):
    """Test dashboard start and stop functionality."""
    with patch('src.monitoring.dashboard.dash.Dash.run_server') as mock_run:
        # Start dashboard
        dashboard_thread = threading.Thread(
            target=dashboard.start,
            kwargs={'host': 'localhost', 'port': 8050},
            daemon=True
        )
        dashboard_thread.start()
        
        # Wait for startup
        time.sleep(1)
        
        # Verify collection is active
        assert dashboard.collection_active
        assert dashboard.collection_thread is not None
        assert dashboard.collection_thread.is_alive()
        
        # Stop dashboard
        dashboard.stop()
        
        # Verify shutdown
        assert not dashboard.collection_active
        dashboard_thread.join(timeout=1)
        assert not dashboard_thread.is_alive()
        
        # Verify server was started
        mock_run.assert_called_once_with(host='localhost', port=8050)

def test_error_handling(dashboard):
    """Test error handling in metrics collection."""
    with patch('psutil.cpu_percent', side_effect=Exception("Test error")):
        # Start collection
        dashboard.collection_active = True
        collection_thread = threading.Thread(
            target=dashboard._collect_metrics,
            daemon=True
        )
        collection_thread.start()
        
        # Wait for some cycles
        time.sleep(2)
        
        # Stop collection
        dashboard.collection_active = False
        collection_thread.join()
        
        # Verify system continues running despite errors
        assert not collection_thread.is_alive()