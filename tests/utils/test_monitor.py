"""
Tests for system monitoring utilities.
"""
import pytest
import psutil
import threading
import time
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
from src.utils.monitor import SystemMonitor, start_monitoring, get_system_status

@pytest.fixture
def mock_psutil():
    """Mock psutil functions."""
    with patch('psutil.virtual_memory') as mock_vm, \
         patch('psutil.cpu_percent') as mock_cpu, \
         patch('psutil.disk_partitions') as mock_dp, \
         patch('psutil.disk_usage') as mock_du, \
         patch('psutil.net_connections') as mock_nc, \
         patch('psutil.net_io_counters') as mock_nio:
        
        # Setup mock memory
        mock_vm.return_value = Mock(
            total=16000000000,
            available=8000000000,
            used=8000000000,
            free=8000000000,
            percent=50.0
        )
        
        # Setup mock CPU
        mock_cpu.return_value = 50.0
        
        # Setup mock disk
        mock_dp.return_value = [
            Mock(mountpoint='/'),
            Mock(mountpoint='/home')
        ]
        mock_du.return_value = Mock(
            total=100000000000,
            used=50000000000,
            free=50000000000,
            percent=50.0
        )
        
        # Setup mock network
        mock_nc.return_value = []
        mock_nio.return_value = Mock(
            bytes_sent=1000000,
            bytes_recv=1000000,
            packets_sent=1000,
            packets_recv=1000,
            _asdict=lambda: {
                'bytes_sent': 1000000,
                'bytes_recv': 1000000
            }
        )
        
        yield {
            'virtual_memory': mock_vm,
            'cpu_percent': mock_cpu,
            'disk_partitions': mock_dp,
            'disk_usage': mock_du,
            'net_connections': mock_nc,
            'net_io_counters': mock_nio
        }

@pytest.fixture
def monitor():
    """Create SystemMonitor instance."""
    return SystemMonitor()

def test_monitor_initialization(monitor):
    """Test monitor initialization."""
    assert monitor is not None
    assert isinstance(monitor.logger, logging.Logger)
    assert isinstance(monitor.metrics, dict)
    assert not monitor._running

def test_start_stop_monitoring(monitor):
    """Test starting and stopping monitoring."""
    monitor.start(interval=1)
    assert monitor._running
    assert monitor._monitor_thread.is_alive()
    
    monitor.stop()
    assert not monitor._running
    assert not monitor._monitor_thread.is_alive()

def test_metric_collection(monitor, mock_psutil):
    """Test metric collection."""
    monitor._collect_metrics()
    metrics = monitor.metrics
    
    assert 'timestamp' in metrics
    assert isinstance(metrics['timestamp'], datetime)
    
    assert 'system' in metrics
    assert 'memory' in metrics
    assert 'cpu' in metrics
    assert 'disk' in metrics
    assert 'network' in metrics
    assert 'trading' in metrics

def test_memory_metrics(monitor, mock_psutil):
    """Test memory metric collection."""
    metrics = monitor._get_memory_metrics()
    
    assert metrics['total'] == 16000000000
    assert metrics['available'] == 8000000000
    assert metrics['used'] == 8000000000
    assert metrics['free'] == 8000000000
    assert metrics['percent'] == 50.0

def test_cpu_metrics(monitor, mock_psutil):
    """Test CPU metric collection."""
    metrics = monitor._get_cpu_metrics()
    
    assert metrics['percent'] == 50.0
    assert 'count' in metrics
    assert 'load_avg' in metrics

def test_disk_metrics(monitor, mock_psutil):
    """Test disk metric collection."""
    metrics = monitor._get_disk_metrics()
    
    assert '/' in metrics
    assert '/home' in metrics
    assert metrics['/']['percent'] == 50.0

def test_network_metrics(monitor, mock_psutil):
    """Test network metric collection."""
    metrics = monitor._get_network_metrics()
    
    assert metrics['connections'] == 0
    assert 'bytes_sent' in metrics['io_counters']
    assert 'bytes_recv' in metrics['io_counters']

def test_trading_metrics(monitor):
    """Test trading metric collection."""
    metrics = monitor._get_trading_metrics()
    
    assert 'active_strategies' in metrics
    assert 'pending_orders' in metrics
    assert 'data_pipeline' in metrics
    assert 'model_status' in metrics

def test_metric_analysis(monitor, mock_psutil):
    """Test metric analysis."""
    # Set up high resource usage
    mock_psutil['virtual_memory'].return_value.percent = 95.0
    mock_psutil['cpu_percent'].return_value = 90.0
    
    monitor._collect_metrics()
    monitor._analyze_metrics()
    
    # Check log for warnings
    log_file = Path("logs/system_monitor.log")
    assert log_file.exists()
    log_content = log_file.read_text()
    assert "High memory usage" in log_content
    assert "High CPU usage" in log_content

def test_metric_saving(monitor, tmp_path):
    """Test metric saving to file."""
    metrics_file = tmp_path / "metrics.csv"
    monitor.metrics_file = metrics_file
    
    monitor._collect_metrics()
    monitor._save_metrics()
    
    assert metrics_file.exists()
    content = metrics_file.read_text()
    assert "timestamp" in content
    assert "memory" in content
    assert "cpu" in content

def test_status_check(monitor):
    """Test system status check."""
    status = monitor.get_status()
    
    assert 'status' in status
    assert 'metrics' in status
    assert 'warnings' in status

def test_warning_detection(monitor, mock_psutil):
    """Test system warning detection."""
    # Set up warning conditions
    mock_psutil['virtual_memory'].return_value.percent = 95.0
    mock_psutil['cpu_percent'].return_value = 90.0
    
    monitor._collect_metrics()
    warnings = monitor._check_warnings()
    
    assert len(warnings) > 0
    assert "Critical memory usage" in warnings
    assert "High CPU load" in warnings

@pytest.mark.integration
def test_monitoring_integration():
    """Test monitoring integration."""
    monitor = start_monitoring()
    
    # Wait for some metrics to be collected
    time.sleep(2)
    
    status = get_system_status()
    assert status['status'] in ['healthy', 'warning']
    assert 'metrics' in status
    
    monitor.stop()

def test_error_handling(monitor):
    """Test error handling in monitoring."""
    with patch('psutil.virtual_memory', side_effect=Exception("Test error")):
        monitor._collect_metrics()
        
        # Check that monitoring continues despite errors
        assert monitor.metrics != {}
        assert 'system' in monitor.metrics

def test_threading_safety(monitor):
    """Test thread safety of monitoring."""
    def update_metrics():
        for _ in range(100):
            monitor._collect_metrics()
            
    threads = [
        threading.Thread(target=update_metrics)
        for _ in range(5)
    ]
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
        
    # Verify no exceptions occurred
    assert monitor.metrics is not None