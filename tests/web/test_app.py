"""
Tests for web interface.
"""
import pytest
from fastapi.testclient import TestClient
import jwt
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.web.app import WebInterface, app

@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        'web': {
            'host': 'localhost',
            'port': 8000,
            'jwt_secret': 'test_secret',
            'jwt_algorithm': 'HS256'
        }
    }

@pytest.fixture
def web_interface(sample_config):
    """Create web interface instance."""
    return WebInterface(sample_config)

@pytest.fixture
def client(web_interface):
    """Create test client."""
    return TestClient(app)

@pytest.fixture
def test_user():
    """Create test user data."""
    return {
        'username': 'testuser',
        'hashed_password': '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewYpwRU.KSinFR2.',  # 'password'
        'email': 'test@example.com',
        'is_active': True
    }

@pytest.fixture
def auth_headers(test_user):
    """Create authentication headers."""
    token = jwt.encode(
        {'sub': test_user['username']},
        'test_secret',
        algorithm='HS256'
    )
    return {'Authorization': f'Bearer {token}'}

@pytest.mark.asyncio
async def test_authenticate_user(web_interface, test_user):
    """Test user authentication."""
    with patch.object(web_interface, '_get_user', return_value=test_user):
        # Test valid credentials
        user = await web_interface.authenticate_user('testuser', 'password')
        assert user is not None
        assert user['username'] == 'testuser'
        
        # Test invalid password
        user = await web_interface.authenticate_user('testuser', 'wrongpass')
        assert user is None
        
        # Test invalid username
        user = await web_interface.authenticate_user('nonexistent', 'password')
        assert user is None

def test_login_endpoint(client, test_user):
    """Test login endpoint."""
    with patch('src.web.app.WebInterface.authenticate_user', 
              AsyncMock(return_value=test_user)):
        response = client.post(
            '/token',
            data={'username': 'testuser', 'password': 'password'}
        )
        
        assert response.status_code == 200
        assert 'access_token' in response.json()
        assert response.json()['token_type'] == 'bearer'

def test_invalid_login(client):
    """Test login with invalid credentials."""
    with patch('src.web.app.WebInterface.authenticate_user', 
              AsyncMock(return_value=None)):
        response = client.post(
            '/token',
            data={'username': 'testuser', 'password': 'wrongpass'}
        )
        
        assert response.status_code == 401

def test_get_strategies(client, auth_headers):
    """Test getting strategies."""
    test_strategies = [
        {'id': 'strat1', 'name': 'Strategy 1'},
        {'id': 'strat2', 'name': 'Strategy 2'}
    ]
    
    with patch('src.web.app.AgentFactory.get_agents_by_type', return_value=[
        Mock(get_generated_strategies=AsyncMock(return_value=test_strategies))
    ]):
        response = client.get('/api/strategies', headers=auth_headers)
        
        assert response.status_code == 200
        assert len(response.json()) == 2
        assert response.json()[0]['id'] == 'strat1'

def test_create_strategy(client, auth_headers):
    """Test strategy creation."""
    strategy_config = {
        'type': 'momentum',
        'parameters': {'lookback': 20}
    }
    
    with patch('src.web.app.StrategyFactory.create_strategy') as mock_create, \
         patch('src.web.app.AgentFactory.create_agent') as mock_agent:
        
        mock_strategy = Mock(id='new_strategy')
        mock_create.return_value = mock_strategy
        mock_agent.return_value.generate_strategy = AsyncMock()
        
        response = client.post(
            '/api/strategies',
            json=strategy_config,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert response.json()['strategy_id'] == 'new_strategy'

def test_get_strategy_details(client, auth_headers):
    """Test getting strategy details."""
    test_strategy = {
        'id': 'strat1',
        'name': 'Strategy 1',
        'type': 'momentum',
        'parameters': {'lookback': 20}
    }
    
    with patch('src.web.app.AgentFactory.get_agents_by_type', return_value=[
        Mock(get_strategy=AsyncMock(return_value=test_strategy))
    ]):
        response = client.get('/api/strategies/strat1', headers=auth_headers)
        
        assert response.status_code == 200
        assert response.json()['id'] == 'strat1'
        assert response.json()['type'] == 'momentum'

def test_get_performance_metrics(client, auth_headers):
    """Test getting performance metrics."""
    test_metrics = [
        {'strategy_id': 'strat1', 'returns': 0.1, 'sharpe': 1.5},
        {'strategy_id': 'strat2', 'returns': 0.2, 'sharpe': 1.8}
    ]
    
    with patch('src.web.app.AgentFactory.get_agents_by_type', return_value=[
        Mock(get_all_performance=AsyncMock(return_value=test_metrics))
    ]):
        response = client.get('/api/performance', headers=auth_headers)
        
        assert response.status_code == 200
        assert len(response.json()) == 2
        assert response.json()[0]['strategy_id'] == 'strat1'

def test_get_risk_metrics(client, auth_headers):
    """Test getting risk metrics."""
    test_metrics = [
        {'strategy_id': 'strat1', 'var': -0.05, 'max_drawdown': -0.15},
        {'strategy_id': 'strat2', 'var': -0.03, 'max_drawdown': -0.10}
    ]
    
    with patch('src.web.app.AgentFactory.get_agents_by_type', return_value=[
        Mock(get_all_risk_metrics=AsyncMock(return_value=test_metrics))
    ]):
        response = client.get('/api/risk', headers=auth_headers)
        
        assert response.status_code == 200
        assert len(response.json()) == 2
        assert response.json()[0]['strategy_id'] == 'strat1'

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket connection."""
    async with TestClient(app).websocket_connect('/ws/updates') as websocket:
        # Test subscription
        await websocket.send_json({'type': 'subscribe'})
        response = await websocket.receive_json()
        assert response['status'] == 'subscribed'
        
        # Test unsubscribe
        await websocket.send_json({'type': 'unsubscribe'})
        response = await websocket.receive_json()
        assert response['status'] == 'unsubscribed'

def test_unauthorized_access(client):
    """Test unauthorized access to protected endpoints."""
    endpoints = [
        '/api/strategies',
        '/api/performance',
        '/api/risk'
    ]
    
    for endpoint in endpoints:
        response = client.get(endpoint)
        assert response.status_code == 401

def test_invalid_token(client):
    """Test access with invalid token."""
    headers = {'Authorization': 'Bearer invalid_token'}
    response = client.get('/api/strategies', headers=headers)
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_broadcast_update(web_interface):
    """Test broadcasting updates to WebSocket clients."""
    # Create mock WebSocket clients
    mock_websockets = [Mock(send_json=AsyncMock()) for _ in range(3)]
    web_interface.active_websockets = mock_websockets
    
    # Broadcast update
    test_update = {'type': 'strategy_update', 'data': {'id': 'strat1'}}
    await web_interface.broadcast_update(test_update)
    
    # Verify all clients received the update
    for ws in mock_websockets:
        ws.send_json.assert_called_once_with(test_update)

@pytest.mark.asyncio
async def test_websocket_error_handling():
    """Test WebSocket error handling."""
    with patch('src.web.app.WebInterface._process_websocket_message',
              side_effect=Exception("Test error")):
        async with TestClient(app).websocket_connect('/ws/updates') as websocket:
            await websocket.send_json({'type': 'invalid'})
            response = await websocket.receive_json()
            assert response['status'] == 'error'

@pytest.mark.asyncio
async def test_start_stop_interface(web_interface):
    """Test starting and stopping web interface."""
    with patch('uvicorn.run') as mock_run:
        # Start interface
        web_interface.start()
        mock_run.assert_called_once()
        
        # Stop interface
        with patch('asyncio.create_task') as mock_task:
            await web_interface.stop()  # Make stop method async
            assert len(web_interface.active_websockets) == 0