"""
Tests for trade execution system.
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal

from src.execution.executor import (
    Order,
    Position,
    ExecutionManager,
    BinanceHandler,
    InteractiveBrokersHandler
)

@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        'handlers': {
            'binance': {
                'api_key': 'test_key',
                'api_secret': 'test_secret'
            },
            'ib': {
                'host': 'localhost',
                'port': 7497,
                'client_id': 1
            }
        }
    }

@pytest.fixture
def mock_ccxt():
    """Mock CCXT module."""
    with patch('ccxt.binance') as mock:
        client = Mock()
        client.load_markets = AsyncMock()
        client.create_order = AsyncMock(return_value={'id': 'test_order_id'})
        client.cancel_order = AsyncMock(return_value={'id': 'test_order_id'})
        client.fetch_order = AsyncMock(return_value={
            'id': 'test_order_id',
            'status': 'closed',
            'filled': 1.0,
            'remaining': 0.0,
            'average': 100.0
        })
        client.fetch_balance = AsyncMock(return_value={
            'total': {'BTC': 1.0, 'ETH': 10.0}
        })
        mock.return_value = client
        yield mock

@pytest.fixture
def execution_manager(sample_config):
    """Create execution manager instance."""
    return ExecutionManager(sample_config)

@pytest.fixture
def sample_order():
    """Create sample order."""
    return Order(
        id='test_order',
        symbol='BTC/USDT',
        order_type='market',
        side='buy',
        quantity=1.0,
        price=50000.0
    )

@pytest.mark.asyncio
async def test_binance_handler(mock_ccxt):
    """Test Binance execution handler."""
    handler = BinanceHandler({
        'api_key': 'test_key',
        'api_secret': 'test_secret'
    })
    
    # Test connection
    await handler.connect()
    assert handler.client is not None
    
    # Test order submission
    order = Order(
        id='test_order',
        symbol='BTC/USDT',
        order_type='limit',
        side='buy',
        quantity=1.0,
        price=50000.0
    )
    
    order_id = await handler.submit_order(order)
    assert order_id == 'test_order'
    assert order.broker_order_id == 'test_order_id'
    
    # Test order cancellation
    success = await handler.cancel_order('test_order')
    assert success
    
    # Test order status
    status = await handler.get_order_status('test_order')
    assert status['status'] == 'closed'
    assert status['filled'] == 1.0
    
    # Test position retrieval
    positions = await handler.get_positions()
    assert 'BTC' in positions
    assert positions['BTC'].quantity == 1.0
    
    # Test disconnection
    await handler.disconnect()
    assert handler.client is None

@pytest.mark.asyncio
async def test_execution_manager(execution_manager, sample_order, mock_ccxt):
    """Test execution manager functionality."""
    # Test connection
    await execution_manager.connect_all()
    
    # Test order submission
    order_id = await execution_manager.submit_order('binance', sample_order)
    assert order_id == sample_order.id
    
    # Test position retrieval
    positions = await execution_manager.get_positions('binance')
    assert len(positions) > 0
    
    # Test disconnection
    await execution_manager.disconnect_all()

def test_order_creation():
    """Test order object creation."""
    order = Order(
        id='test_order',
        symbol='BTC/USDT',
        order_type='limit',
        side='buy',
        quantity=1.0,
        price=50000.0
    )
    
    assert order.id == 'test_order'
    assert order.symbol == 'BTC/USDT'
    assert order.order_type == 'limit'
    assert order.side == 'buy'
    assert order.quantity == 1.0
    assert order.price == 50000.0
    assert order.status == 'created'
    assert order.client_order_id is not None
    assert order.timestamp is not None

def test_position_creation():
    """Test position object creation."""
    position = Position(
        symbol='BTC/USDT',
        quantity=1.0,
        average_price=50000.0
    )
    
    assert position.symbol == 'BTC/USDT'
    assert position.quantity == 1.0
    assert position.average_price == 50000.0
    assert position.timestamp is not None

def test_execution_manager_initialization(sample_config):
    """Test execution manager initialization."""
    manager = ExecutionManager(sample_config)
    
    assert 'binance' in manager.handlers
    assert 'ib' in manager.handlers
    assert isinstance(manager.handlers['binance'], BinanceHandler)
    assert isinstance(manager.handlers['ib'], InteractiveBrokersHandler)

@pytest.mark.asyncio
async def test_error_handling(execution_manager, sample_order):
    """Test error handling in execution system."""
    # Test invalid broker
    with pytest.raises(ValueError):
        await execution_manager.submit_order('invalid_broker', sample_order)
    
    # Test order submission failure
    with patch.dict(
        execution_manager.handlers,
        {'binance': Mock(submit_order=AsyncMock(side_effect=Exception("API Error")))}
    ):
        with pytest.raises(Exception):
            await execution_manager.submit_order('binance', sample_order)

@pytest.mark.asyncio
async def test_multiple_orders(execution_manager, mock_ccxt):
    """Test handling multiple orders."""
    orders = [
        Order(
            id=f'test_order_{i}',
            symbol='BTC/USDT',
            order_type='market',
            side='buy',
            quantity=1.0
        )
        for i in range(3)
    ]
    
    # Submit orders
    order_ids = []
    for order in orders:
        order_id = await execution_manager.submit_order('binance', order)
        order_ids.append(order_id)
    
    assert len(order_ids) == 3
    assert all(isinstance(id, str) for id in order_ids)

@pytest.mark.asyncio
async def test_order_lifecycle(execution_manager, sample_order, mock_ccxt):
    """Test complete order lifecycle."""
    # Submit order
    order_id = await execution_manager.submit_order('binance', sample_order)
    
    # Get status
    handler = execution_manager.handlers['binance']
    status = await handler.get_order_status(order_id)
    assert status['status'] == 'closed'
    
    # Cancel order
    success = await execution_manager.cancel_order('binance', order_id)
    assert success
    
    # Check positions
    positions = await execution_manager.get_positions('binance')
    assert len(positions) > 0

def test_available_brokers(execution_manager):
    """Test getting available brokers."""
    brokers = execution_manager.get_available_brokers()
    assert 'binance' in brokers
    assert 'ib' in brokers
    assert len(brokers) == 2

@pytest.mark.asyncio
async def test_connection_error_handling(execution_manager):
    """Test handling of connection errors."""
    # Make connection fail
    with patch.dict(
        execution_manager.handlers,
        {'binance': Mock(connect=AsyncMock(side_effect=Exception("Connection Error")))}
    ):
        # Should not raise exception
        await execution_manager.connect_all()

@pytest.mark.asyncio
async def test_position_updates(execution_manager, mock_ccxt):
    """Test position updates after trades."""
    # Submit buy order
    buy_order = Order(
        id='buy_order',
        symbol='BTC/USDT',
        order_type='market',
        side='buy',
        quantity=1.0
    )
    await execution_manager.submit_order('binance', buy_order)
    
    # Check positions
    positions = await execution_manager.get_positions('binance')
    assert positions['BTC'].quantity == 1.0
    
    # Submit sell order
    sell_order = Order(
        id='sell_order',
        symbol='BTC/USDT',
        order_type='market',
        side='sell',
        quantity=1.0
    )
    await execution_manager.submit_order('binance', sell_order)