"""
Tests for Model Context Protocol.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.core.mcp import ModelContextProtocol
from src.core.safety_checker import SafetyChecker

@pytest.fixture
def mock_safety_checker():
    """Mock safety checker."""
    with patch('src.core.mcp.SafetyChecker') as mock:
        checker = Mock()
        checker.verify_context.return_value = True
        mock.return_value = checker
        yield checker

@pytest.fixture
def mcp(mock_safety_checker):
    """Create MCP instance."""
    return ModelContextProtocol()

def test_prepare_context_basic(mcp):
    """Test basic context preparation."""
    context = mcp.prepare_context()
    
    # Check required sections
    assert 'system' in context
    assert 'memory' in context
    assert 'market' in context
    assert 'tools' in context
    
    # Check system section
    assert 'role' in context['system']
    assert 'capabilities' in context['system']
    assert 'constraints' in context['system']
    assert 'timestamp' in context['system']

def test_prepare_context_with_additional(mcp):
    """Test context preparation with additional context."""
    additional = {
        'custom_data': {'key': 'value'},
        'system': {'additional': 'info'}
    }
    
    context = mcp.prepare_context(additional)
    
    assert context['custom_data'] == {'key': 'value'}
    assert 'additional' in context['system']
    assert context['system']['additional'] == 'info'

def test_context_safety_check(mcp, mock_safety_checker):
    """Test context safety verification."""
    mock_safety_checker.verify_context.return_value = False
    
    with pytest.raises(ValueError, match="Context failed safety verification"):
        mcp.prepare_context()

def test_get_capabilities(mcp):
    """Test capabilities retrieval."""
    capabilities = mcp._get_capabilities()
    
    assert isinstance(capabilities, list)
    assert 'strategy_generation' in capabilities
    assert 'risk_assessment' in capabilities
    assert 'backtesting' in capabilities

def test_get_constraints(mcp):
    """Test constraints retrieval."""
    constraints = mcp._get_constraints()
    
    assert isinstance(constraints, list)
    assert 'ethical_trading' in constraints
    assert 'position_limits' in constraints
    assert 'risk_limits' in constraints

def test_get_memory_context(mcp):
    """Test memory context retrieval."""
    memory = mcp._get_memory_context()
    
    assert 'short_term' in memory
    assert 'long_term' in memory
    assert 'working' in memory

def test_get_market_context(mcp):
    """Test market context retrieval."""
    market = mcp._get_market_context()
    
    assert 'current_state' in market
    assert 'indicators' in market
    assert 'sentiment' in market

def test_get_tool_context(mcp):
    """Test tool context retrieval."""
    tools = mcp._get_tool_context()
    
    assert 'available' in tools
    assert 'permissions' in tools
    assert 'constraints' in tools

def test_context_history_tracking(mcp):
    """Test context history tracking."""
    # Initial history should be empty
    assert len(mcp.context_history) == 0
    
    # Prepare context and check history
    mcp.prepare_context()
    assert len(mcp.context_history) == 1
    
    # Check history entry
    entry = mcp.context_history[0]
    assert 'timestamp' in entry
    assert 'context' in entry
    assert 'tokens' in entry

def test_context_optimization(mcp):
    """Test context optimization."""
    # Create large context
    large_context = {
        'large_data': 'x' * 5000  # Large string
    }
    
    optimized = mcp._optimize_context(large_context)
    
    # Verify optimization
    assert len(str(optimized)) < len(str(large_context))

def test_merge_contexts(mcp):
    """Test context merging."""
    base = {
        'system': {'key1': 'value1'},
        'memory': {'mem1': 'value1'}
    }
    
    new = {
        'system': {'key2': 'value2'},
        'custom': 'value'
    }
    
    merged = mcp._merge_contexts(base, new)
    
    assert merged['system']['key1'] == 'value1'
    assert merged['system']['key2'] == 'value2'
    assert merged['memory']['mem1'] == 'value1'
    assert merged['custom'] == 'value'

def test_context_size_estimation(mcp):
    """Test context size estimation."""
    context = {
        'text': 'test ' * 100  # 500 characters
    }
    
    tokens = mcp._estimate_tokens(context)
    assert tokens > 0
    assert isinstance(tokens, int)

def test_recent_history(mcp):
    """Test recent history retrieval."""
    # Add some history entries
    for i in range(15):
        mcp.context_history.append({
            'timestamp': datetime.now(),
            'context': f'test{i}',
            'tokens': 10
        })
    
    recent = mcp._get_recent_history()
    assert len(recent) == 10  # Should only get last 10 entries

def test_current_context(mcp):
    """Test current context retrieval."""
    current = mcp._get_current_context()
    
    assert 'active_strategies' in current
    assert 'pending_decisions' in current
    assert 'recent_actions' in current

def test_context_validation(mcp, mock_safety_checker):
    """Test context validation with different scenarios."""
    # Missing required sections
    incomplete = {'system': {}}
    with pytest.raises(ValueError):
        mcp.prepare_context(incomplete)
    
    # Invalid content type
    invalid = {'system': 'not_a_dict'}
    with pytest.raises(ValueError):
        mcp.prepare_context(invalid)
    
    # Valid complex context
    valid = {
        'system': {'custom': 'value'},
        'memory': {'data': 'test'},
        'market': {'price': 100},
        'tools': {'tool1': True}
    }
    
    result = mcp.prepare_context(valid)
    assert result['system']['custom'] == 'value'
    assert result['memory']['data'] == 'test'

def test_context_stats(mcp):
    """Test context statistics retrieval."""
    # Add some context history
    for i in range(5):
        mcp.context_history.append({
            'timestamp': datetime.now(),
            'context': f'test{i}',
            'tokens': 10
        })
    
    stats = mcp.get_context_stats()
    assert stats['total_contexts'] == 5
    assert stats['average_tokens'] == 10
    assert stats['last_context_tokens'] == 10

def test_clear_history(mcp):
    """Test history clearing."""
    # Add some history
    mcp.context_history.append({
        'timestamp': datetime.now(),
        'context': 'test',
        'tokens': 10
    })
    
    assert len(mcp.context_history) > 0
    
    # Clear history
    mcp.clear_history()
    assert len(mcp.context_history) == 0