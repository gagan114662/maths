"""
Tests for the monitoring management system.
"""
import pytest
import threading
import time
from unittest.mock import Mock, patch
from monitor_system import MonitoringManager

@pytest.fixture
def manager():
    """Create MonitoringManager instance."""
    return MonitoringManager()

@pytest.fixture
def mock_monitor():
    """Mock SystemMonitor."""
    with patch('monitor_system.SystemMonitor') as mock:
        monitor_instance = Mock()
        mock.return_value = monitor_instance
        yield monitor_instance

@pytest.fixture
def mock_web_monitor():
    """Mock web monitor."""
    with patch('monitor_system.start_web_monitor') as mock:
        yield mock

def test_manager_initialization(manager):
    """Test manager initialization."""
    assert manager.monitor is None
    assert manager.web_thread is None
    assert not manager.running

def test_start_monitoring_basic(manager, mock_monitor):
    """Test basic monitoring start."""
    manager.start_monitoring(web=False)
    
    assert manager.running
    assert manager.monitor is not None
    assert manager.monitor.start.called
    assert manager.web_thread is None

def test_start_monitoring_with_web(manager, mock_monitor, mock_web_monitor):
    """Test monitoring start with web interface."""
    manager.start_monitoring(web=True, port=5000)
    
    assert manager.running
    assert manager.monitor is not None
    assert manager.monitor.start.called
    assert manager.web_thread is not None
    assert manager.web_thread.is_alive()

def test_stop_monitoring(manager, mock_monitor):
    """Test monitoring stop."""
    manager.start_monitoring(web=False)
    manager.stop_monitoring()
    
    assert not manager.running
    assert manager.monitor.stop.called

def test_web_interface_thread(manager, mock_monitor, mock_web_monitor):
    """Test web interface thread management."""
    manager.start_monitoring(web=True, port=5000)
    
    # Wait for thread to start
    time.sleep(0.1)
    
    assert manager.web_thread.is_alive()
    assert mock_web_monitor.called
    
    # Cleanup
    manager.stop_monitoring()

def test_error_handling(manager):
    """Test error handling during startup."""
    with patch('monitor_system.SystemMonitor', side_effect=Exception("Test error")):
        with pytest.raises(SystemExit):
            manager.start_monitoring()
            
    assert not manager.running

def test_signal_handling(manager, mock_monitor):
    """Test signal handling."""
    import signal
    
    # Store original handler
    original_handler = signal.getsignal(signal.SIGINT)
    
    try:
        manager.start_monitoring(web=False)
        
        # Verify signal handlers are set
        current_handler = signal.getsignal(signal.SIGINT)
        assert current_handler != original_handler
        
        # Simulate signal
        with pytest.raises(SystemExit):
            current_handler(signal.SIGINT, None)
            
        assert not manager.running
        assert manager.monitor.stop.called
        
    finally:
        # Restore original handler
        signal.signal(signal.SIGINT, original_handler)

def test_multiple_starts(manager, mock_monitor):
    """Test starting monitoring multiple times."""
    manager.start_monitoring(web=False)
    first_monitor = manager.monitor
    
    manager.start_monitoring(web=False)
    second_monitor = manager.monitor
    
    assert first_monitor != second_monitor
    assert first_monitor.stop.called

def test_web_interface_options(manager, mock_monitor, mock_web_monitor):
    """Test web interface configuration options."""
    custom_host = '127.0.0.1'
    custom_port = 8080
    
    manager.start_monitoring(
        web=True,
        host=custom_host,
        port=custom_port
    )
    
    mock_web_monitor.assert_called_with(
        host=custom_host,
        port=custom_port,
        debug=False
    )

@pytest.mark.integration
def test_full_monitoring_cycle():
    """Test complete monitoring cycle."""
    manager = MonitoringManager()
    
    # Start monitoring
    manager.start_monitoring(web=True, port=5000)
    assert manager.running
    
    # Let it run briefly
    time.sleep(2)
    
    # Stop monitoring
    manager.stop_monitoring()
    assert not manager.running
    
    # Verify cleanup
    assert not manager.web_thread.is_alive()

def test_browser_launch(manager, mock_monitor):
    """Test browser launch option."""
    with patch('webbrowser.open') as mock_browser:
        manager.start_monitoring(
            web=True,
            port=5000,
            open_browser=True
        )
        
        assert mock_browser.called
        mock_browser.assert_called_with('http://localhost:5000')

def test_logs_directory_creation(manager, mock_monitor, tmp_path):
    """Test logs directory creation."""
    with patch('pathlib.Path.mkdir') as mock_mkdir:
        manager.start_monitoring(web=False)
        mock_mkdir.assert_called_with(exist_ok=True)

def test_thread_cleanup(manager, mock_monitor):
    """Test proper thread cleanup on stop."""
    manager.start_monitoring(web=True)
    web_thread = manager.web_thread
    
    manager.stop_monitoring()
    
    # Give thread time to stop
    time.sleep(0.1)
    assert not web_thread.is_alive()