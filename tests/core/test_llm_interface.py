"""
Tests for LLM interface.
"""
import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.core.llm_interface import LLMInterface
from src.core.mcp import ModelContextProtocol
from src.core.safety_checker import SafetyChecker

@pytest.fixture
def mock_openai():
    """Mock OpenAI API."""
    with patch('src.core.llm_interface.openai') as mock:
        mock.ChatCompletion.acreate = AsyncMock(return_value={
            'choices': [{'message': {'content': 'Test response', 'role': 'assistant'}}],
            'usage': {'total_tokens': 10},
            'model': 'gpt-4'
        })
        yield mock

@pytest.fixture
def mock_safety_checker():
    """Mock safety checker."""
    with patch('src.core.llm_interface.SafetyChecker') as mock:
        checker = Mock()
        checker.verify_prompt.return_value = True
        checker.verify_response.return_value = True
        checker.verify_context.return_value = True
        mock.return_value = checker
        yield checker

@pytest.fixture
def mock_mcp():
    """Mock Model Context Protocol."""
    with patch('src.core.llm_interface.ModelContextProtocol') as mock:
        protocol = Mock()
        protocol.prepare_context.return_value = {
            'system': 'test context',
            'memory': {},
            'market': {},
            'tools': {}
        }
        mock.return_value = protocol
        yield protocol

@pytest.fixture
def llm_interface(mock_openai, mock_safety_checker, mock_mcp):
    """Create LLM interface instance."""
    return LLMInterface()

@pytest.mark.asyncio
async def test_generate_response_basic(llm_interface, mock_openai):
    """Test basic response generation."""
    response = await llm_interface.generate_response("Test prompt")
    
    assert response['content'] == 'Test response'
    assert response['usage']['total_tokens'] == 10
    assert response['model'] == 'gpt-4'
    assert 'timestamp' in response

@pytest.mark.asyncio
async def test_generate_response_with_context(llm_interface, mock_openai, mock_mcp):
    """Test response generation with context."""
    context = {'test': 'context'}
    response = await llm_interface.generate_response("Test prompt", context=context)
    
    mock_mcp.prepare_context.assert_called_once_with(context)
    assert response['content'] == 'Test response'

@pytest.mark.asyncio
async def test_generate_response_safety_check(llm_interface, mock_safety_checker):
    """Test safety checks during response generation."""
    mock_safety_checker.verify_prompt.return_value = False
    
    with pytest.raises(ValueError, match="Safety check failed"):
        await llm_interface.generate_response("Test prompt")

@pytest.mark.asyncio
async def test_generate_response_streaming(llm_interface, mock_openai):
    """Test streaming response generation."""
    mock_openai.ChatCompletion.acreate.return_value = [
        {'choices': [{'delta': {'content': 'Test'}}]},
        {'choices': [{'delta': {'content': ' response'}}]}
    ]
    
    response = await llm_interface.generate_response("Test prompt", stream=True)
    chunks = [chunk async for chunk in response]
    
    assert len(chunks) == 2
    assert chunks[0]['chunk'] == 'Test'
    assert chunks[1]['chunk'] == ' response'

def test_token_usage_tracking(llm_interface):
    """Test token usage tracking."""
    initial_usage = llm_interface.get_token_usage()
    assert initial_usage['total_tokens'] == 0
    
    # Simulate some token usage
    llm_interface.total_tokens = 100
    
    updated_usage = llm_interface.get_token_usage()
    assert updated_usage['total_tokens'] == 100

def test_conversation_history(llm_interface):
    """Test conversation history management."""
    # Initial history should be empty
    assert len(llm_interface.get_history()) == 0
    
    # Add some history
    llm_interface.context_history.append({
        'timestamp': datetime.now().isoformat(),
        'prompt': 'test',
        'content': 'response',
        'tokens': 10
    })
    
    history = llm_interface.get_history()
    assert len(history) == 1
    assert history[0]['prompt'] == 'test'
    assert history[0]['content'] == 'response'

def test_history_persistence(llm_interface, tmp_path):
    """Test history saving and loading."""
    # Add test history
    llm_interface.context_history.append({
        'timestamp': datetime.now().isoformat(),
        'prompt': 'test',
        'content': 'response',
        'tokens': 10
    })
    
    # Save history
    filepath = tmp_path / "history.json"
    llm_interface.save_history(filepath)
    
    # Clear history
    llm_interface.clear_history()
    assert len(llm_interface.get_history()) == 0
    
    # Load history
    llm_interface.load_history(filepath)
    loaded_history = llm_interface.get_history()
    assert len(loaded_history) == 1
    assert loaded_history[0]['prompt'] == 'test'

@pytest.mark.asyncio
async def test_error_handling(llm_interface, mock_openai):
    """Test error handling in response generation."""
    mock_openai.ChatCompletion.acreate.side_effect = Exception("API Error")
    
    with pytest.raises(Exception, match="API Error"):
        await llm_interface.generate_response("Test prompt")

@pytest.mark.asyncio
async def test_response_validation(llm_interface, mock_openai, mock_safety_checker):
    """Test response content validation."""
    mock_safety_checker.verify_response.return_value = False
    mock_openai.ChatCompletion.acreate.return_value = {
        'choices': [{'message': {'content': 'Invalid response', 'role': 'assistant'}}],
        'usage': {'total_tokens': 10},
        'model': 'gpt-4'
    }
    
    with pytest.raises(ValueError, match="Response failed safety check"):
        await llm_interface.generate_response("Test prompt")

def test_context_management(llm_interface):
    """Test context management."""
    # Get initial context size
    initial_tokens = llm_interface.total_tokens
    
    # Add some context history
    llm_interface.context_history.extend([
        {
            'timestamp': datetime.now().isoformat(),
            'prompt': f'test{i}',
            'content': f'response{i}',
            'tokens': 10
        }
        for i in range(5)
    ])
    
    # Check token accumulation
    assert llm_interface.total_tokens == initial_tokens + 50

def test_configuration(llm_interface):
    """Test configuration management."""
    assert hasattr(llm_interface, 'config')
    assert isinstance(llm_interface.config, dict)

@pytest.mark.asyncio
async def test_retry_mechanism(llm_interface, mock_openai):
    """Test retry mechanism for failed requests."""
    # Make the first two calls fail, third succeeds
    mock_openai.ChatCompletion.acreate.side_effect = [
        Exception("First error"),
        Exception("Second error"),
        {
            'choices': [{'message': {'content': 'Success', 'role': 'assistant'}}],
            'usage': {'total_tokens': 10},
            'model': 'gpt-4'
        }
    ]
    
    response = await llm_interface.generate_response("Test prompt")
    assert response['content'] == 'Success'
    assert mock_openai.ChatCompletion.acreate.call_count == 3