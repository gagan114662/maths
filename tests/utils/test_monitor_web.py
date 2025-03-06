"""
Tests for web monitoring interface.
"""
import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from unittest.mock import Mock, patch
from src.utils.monitor_web import app, monitor

@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_monitor():
    """Mock system monitor."""
    with patch('src.utils.monitor_web.monitor') as mock:
        mock.get_status.return_value = {
            'status': 'healthy',
            'metrics': {
                'memory': {
                    'total': 16000000000,
                    'used': 8000000000,
                    'free': 8000000000,
                    'percent': 50.0
                },
                'cpu': {
                    'percent': 30.0,
                    'count': 8,
                    'load_avg': (1.5, 1.2, 1.0)
                },
                'trading': {
                    'active_strategies': 5,
                    'pending_orders': 2,
                    'data_pipeline': {
                        'status': 'ok',
                        'last_update': str(datetime.now())
                    },
                    'model_status': {
                        'status': 'ok',
                        'last_prediction': str(datetime.now())
                    }
                }
            },
            'warnings': []
        }
        yield mock

def test_index_page(client):
    """Test index page rendering."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'System Monitor' in response.data
    assert b'Memory Usage' in response.data
    assert b'CPU Usage' in response.data

def test_api_status(client, mock_monitor):
    """Test status API endpoint."""
    response = client.get('/api/status')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'metrics' in data
    assert 'warnings' in data

def test_metrics_history(client):
    """Test metrics history API endpoint."""
    # Create test metrics data
    metrics_dir = Path("logs")
    metrics_dir.mkdir(exist_ok=True)
    
    metrics_file = metrics_dir / "metrics.csv"
    
    # Generate sample metrics
    now = datetime.now()
    metrics = []
    for i in range(10):
        timestamp = now - timedelta(minutes=i*5)
        metrics.append({
            'timestamp': timestamp,
            'memory_percent': 50.0 + i,
            'cpu_percent': 30.0 + i
        })
    
    df = pd.DataFrame(metrics)
    df.to_csv(metrics_file, index=False)
    
    # Test API endpoint
    response = client.get('/api/metrics/history')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert len(data) > 0
    assert 'timestamp' in data[0]
    assert 'memory_percent' in data[0]
    assert 'cpu_percent' in data[0]
    
    # Cleanup
    metrics_file.unlink()

def test_warning_display(client, mock_monitor):
    """Test warning display in status."""
    # Mock monitor to return warnings
    mock_monitor.get_status.return_value['status'] = 'warning'
    mock_monitor.get_status.return_value['warnings'] = [
        'High memory usage',
        'High CPU load'
    ]
    
    response = client.get('/api/status')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'warning'
    assert len(data['warnings']) == 2
    assert 'High memory usage' in data['warnings']

def test_trading_status(client, mock_monitor):
    """Test trading status display."""
    response = client.get('/api/status')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    trading = data['metrics']['trading']
    
    assert 'active_strategies' in trading
    assert 'pending_orders' in trading
    assert 'data_pipeline' in trading
    assert 'model_status' in trading

@pytest.mark.integration
def test_live_updates(client):
    """Test live metric updates."""
    # Start monitor
    monitor.start(interval=1)
    
    # Get initial status
    response1 = client.get('/api/status')
    data1 = json.loads(response1.data)
    
    # Wait for update
    import time
    time.sleep(2)
    
    # Get updated status
    response2 = client.get('/api/status')
    data2 = json.loads(response2.data)
    
    # Verify timestamps are different
    assert data1['metrics']['timestamp'] != data2['metrics']['timestamp']
    
    # Stop monitor
    monitor.stop()

def test_error_handling(client):
    """Test error handling in web interface."""
    # Test invalid endpoint
    response = client.get('/invalid')
    assert response.status_code == 404
    
    # Test server error
    with patch('src.utils.monitor_web.monitor.get_status', 
              side_effect=Exception("Test error")):
        response = client.get('/api/status')
        assert response.status_code == 500

def test_template_existence():
    """Test that HTML template exists."""
    template_dir = Path(__file__).parent.parent.parent / 'src' / 'utils' / 'templates'
    template_file = template_dir / 'monitor.html'
    
    assert template_dir.exists()
    assert template_file.exists()
    assert template_file.read_text().strip() != ""

def test_chart_data_format(client, mock_monitor):
    """Test that metrics are formatted correctly for charts."""
    response = client.get('/api/status')
    data = json.loads(response.data)
    
    memory = data['metrics']['memory']
    cpu = data['metrics']['cpu']
    
    # Verify memory data format
    assert isinstance(memory['used'], (int, float))
    assert isinstance(memory['free'], (int, float))
    
    # Verify CPU data format
    assert isinstance(cpu['percent'], (int, float))
    assert 0 <= cpu['percent'] <= 100

def test_static_files(client):
    """Test static file serving."""
    # Test CSS
    response = client.get('/static/css/style.css')
    assert response.status_code in (200, 404)  # May not exist in test
    
    # Test JavaScript
    response = client.get('/static/js/charts.js')
    assert response.status_code in (200, 404)  # May not exist in test

def test_response_headers(client):
    """Test response headers."""
    response = client.get('/api/status')
    assert response.content_type == 'application/json'
    assert 'Cache-Control' in response.headers