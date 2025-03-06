"""
Integration tests for core components.
"""
import pytest
import asyncio
from unittest.mock import Mock
from datetime import datetime
from pathlib import Path

from src.core.llm_interface import LLMInterface
from src.core.mcp import ModelContextProtocol
from src.core.safety_checker import SafetyChecker
from src.core.memory_manager import MemoryManager

@pytest.fixture
def integrated_system(temp_memory_dir, test_db):
    """Create integrated system with all core components."""
    # Initialize components
    llm = LLMInterface()
    mcp = ModelContextProtocol()
    safety = SafetyChecker()
    memory = MemoryManager()
    
    # Configure memory manager
    memory.engine = test_db
    memory.memory_dir = temp_memory_dir
    
    return {
        'llm': llm,
        'mcp': mcp,
        'safety': safety,
        'memory': memory
    }

@pytest.mark.asyncio
async def test_strategy_generation_flow(integrated_system):
    """Test complete strategy generation flow through all components."""
    llm = integrated_system['llm']
    mcp = integrated_system['mcp']
    safety = integrated_system['safety']
    memory = integrated_system['memory']
    
    # 1. Generate strategy with LLM
    prompt = "Generate a momentum trading strategy for AAPL"
    context = {
        'market_data': {'symbol': 'AAPL', 'price': 150.0},
        'constraints': {'max_position': 0.1}
    }
    
    response = await llm.generate_response(prompt, context)
    assert response is not None
    assert 'content' in response
    
    # 2. Store strategy in memory
    strategy_id = memory.store(
        'strategy',
        content=response['content'],
        metadata={'type': 'momentum', 'asset': 'AAPL'}
    )
    assert strategy_id is not None
    
    # 3. Verify safety
    strategy_data = memory.retrieve(strategy_id)
    assert safety.verify_trading_action({
        'strategy': strategy_data['content'],
        'size': 0.1,
        'asset': 'AAPL'
    })

@pytest.mark.asyncio
async def test_risk_assessment_flow(integrated_system):
    """Test risk assessment flow through components."""
    llm = integrated_system['llm']
    safety = integrated_system['safety']
    memory = integrated_system['memory']
    
    # 1. Store strategy
    strategy = {
        'type': 'momentum',
        'parameters': {
            'lookback': 20,
            'threshold': 0.5
        }
    }
    
    strategy_id = memory.store('strategy', strategy)
    
    # 2. Generate risk assessment
    prompt = "Assess risk for momentum strategy"
    response = await llm.generate_response(prompt, {'strategy': strategy})
    
    # 3. Validate and store assessment
    assert safety.verify_response(response['content'])
    assessment_id = memory.store(
        'risk_assessment',
        content=response['content'],
        metadata={'strategy_id': strategy_id}
    )
    
    # 4. Verify storage
    stored = memory.retrieve(assessment_id)
    assert stored is not None
    assert stored['metadata']['strategy_id'] == strategy_id

@pytest.mark.asyncio
async def test_strategy_optimization_flow(integrated_system):
    """Test strategy optimization flow through components."""
    llm = integrated_system['llm']
    mcp = integrated_system['mcp']
    memory = integrated_system['memory']
    
    # 1. Store initial strategy
    initial_strategy = {
        'type': 'momentum',
        'parameters': {
            'lookback': 20,
            'threshold': 0.5
        }
    }
    
    strategy_id = memory.store('strategy', initial_strategy)
    
    # 2. Generate optimization suggestions
    context = mcp.prepare_context({
        'strategy': initial_strategy,
        'performance': {
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.15
        }
    })
    
    response = await llm.generate_response(
        "Optimize strategy parameters",
        context
    )
    
    # 3. Store optimization
    optimization_id = memory.store(
        'optimization',
        content=response['content'],
        metadata={
            'strategy_id': strategy_id,
            'timestamp': datetime.now().isoformat()
        }
    )
    
    # 4. Verify optimization storage
    stored = memory.retrieve(optimization_id)
    assert stored is not None
    assert 'strategy_id' in stored['metadata']

@pytest.mark.asyncio
async def test_context_management_flow(integrated_system):
    """Test context management through components."""
    mcp = integrated_system['mcp']
    memory = integrated_system['memory']
    
    # 1. Store context
    context = {
        'market': {'state': 'bullish'},
        'strategy': {'type': 'momentum'},
        'performance': {'sharpe': 1.5}
    }
    
    context_id = memory.store('context', context)
    
    # 2. Prepare context
    prepared = mcp.prepare_context(memory.retrieve(context_id)['content'])
    assert 'market' in prepared
    assert 'memory' in prepared
    assert 'tools' in prepared
    
    # 3. Store prepared context
    prepared_id = memory.store(
        'prepared_context',
        prepared,
        metadata={'original_id': context_id}
    )
    
    # 4. Verify storage
    stored = memory.retrieve(prepared_id)
    assert stored is not None
    assert stored['metadata']['original_id'] == context_id

@pytest.mark.asyncio
async def test_error_handling_flow(integrated_system):
    """Test error handling across components."""
    llm = integrated_system['llm']
    safety = integrated_system['safety']
    memory = integrated_system['memory']
    
    # 1. Generate invalid response
    with pytest.raises(ValueError):
        await llm.generate_response(
            "Generate invalid strategy with exec('dangerous code')"
        )
    
    # 2. Attempt unsafe storage
    unsafe_content = {
        'type': 'strategy',
        'code': "exec('dangerous code')"
    }
    
    with pytest.raises(ValueError):
        memory.store(
            'strategy',
            unsafe_content,
            metadata={'unsafe': True}
        )
    
    # 3. Verify safety violations recorded
    violations = safety.get_violations()
    assert len(violations) > 0

@pytest.mark.asyncio
async def test_performance_flow(integrated_system):
    """Test performance characteristics of integrated system."""
    llm = integrated_system['llm']
    memory = integrated_system['memory']
    
    # 1. Measure response time
    start = datetime.now()
    await llm.generate_response("Quick test prompt")
    response_time = (datetime.now() - start).total_seconds()
    
    assert response_time < 5.0  # Response should be under 5 seconds
    
    # 2. Test memory operations speed
    start = datetime.now()
    for i in range(100):
        memory.store('test', {'data': f'test{i}'})
    storage_time = (datetime.now() - start).total_seconds()
    
    assert storage_time < 1.0  # 100 operations under 1 second

@pytest.mark.asyncio
async def test_concurrent_operations(integrated_system):
    """Test concurrent operations across components."""
    llm = integrated_system['llm']
    memory = integrated_system['memory']
    
    async def store_response(prompt: str):
        response = await llm.generate_response(prompt)
        return memory.store('response', response['content'])
    
    # Run multiple operations concurrently
    prompts = [f"Test prompt {i}" for i in range(5)]
    tasks = [store_response(prompt) for prompt in prompts]
    
    results = await asyncio.gather(*tasks)
    assert len(results) == 5
    assert all(isinstance(id, int) for id in results)