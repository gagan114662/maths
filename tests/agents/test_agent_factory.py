"""
Tests for agent factory and pipeline coordination.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.agents import (
    AgentFactory,
    BaseAgent,
    GenerationAgent,
    BacktestingAgent,
    RiskAssessmentAgent,
    RankingAgent,
    EvolutionAgent,
    MetaReviewAgent
)

@pytest.fixture
def agent_factory():
    """Create agent factory instance."""
    return AgentFactory()

@pytest.fixture
def mock_agents():
    """Mock agent classes."""
    with patch('src.agents.agent_factory.GenerationAgent') as gen_mock, \
         patch('src.agents.agent_factory.BacktestingAgent') as back_mock, \
         patch('src.agents.agent_factory.RiskAssessmentAgent') as risk_mock, \
         patch('src.agents.agent_factory.RankingAgent') as rank_mock, \
         patch('src.agents.agent_factory.EvolutionAgent') as evol_mock, \
         patch('src.agents.agent_factory.MetaReviewAgent') as meta_mock:
        
        # Configure mocks
        agents = {
            'generation': gen_mock,
            'backtesting': back_mock,
            'risk': risk_mock,
            'ranking': rank_mock,
            'evolution': evol_mock,
            'meta_review': meta_mock
        }
        
        for name, mock in agents.items():
            mock_instance = Mock()
            mock_instance.name = f"{name}_test"
            mock_instance.state = {
                'status': 'initialized',
                'metrics': {},
                'errors': []
            }
            mock.return_value = mock_instance
            
        yield agents

def test_create_agent(agent_factory, mock_agents):
    """Test agent creation."""
    # Create each type of agent
    for agent_type in mock_agents.keys():
        agent = agent_factory.create_agent(agent_type)
        
        assert agent is not None
        assert agent.name.startswith(agent_type)
        assert agent_type in agent_factory.agents
        assert agent_factory.agent_states[agent.name]['type'] == agent_type

def test_get_agent(agent_factory, mock_agents):
    """Test agent retrieval."""
    agent = agent_factory.create_agent('generation')
    retrieved = agent_factory.get_agent(agent.name)
    
    assert retrieved is agent
    assert agent_factory.get_agent('nonexistent') is None

def test_get_agents_by_type(agent_factory, mock_agents):
    """Test retrieving agents by type."""
    # Create multiple agents
    agent1 = agent_factory.create_agent('generation')
    agent2 = agent_factory.create_agent('generation')
    agent3 = agent_factory.create_agent('backtesting')
    
    generation_agents = agent_factory.get_agents_by_type('generation')
    assert len(generation_agents) == 2
    assert all(isinstance(a, mock_agents['generation'].return_value.__class__) for a in generation_agents)

@pytest.mark.asyncio
async def test_start_agent(agent_factory, mock_agents):
    """Test starting agent."""
    agent = agent_factory.create_agent('generation')
    success = await agent_factory.start_agent(agent.name)
    
    assert success
    assert agent_factory.agent_states[agent.name]['status'] == 'running'
    assert agent.start.called

def test_stop_agent(agent_factory, mock_agents):
    """Test stopping agent."""
    agent = agent_factory.create_agent('generation')
    success = agent_factory.stop_agent(agent.name)
    
    assert success
    assert agent_factory.agent_states[agent.name]['status'] == 'stopped'
    assert agent.stop.called

def test_remove_agent(agent_factory, mock_agents):
    """Test agent removal."""
    agent = agent_factory.create_agent('generation')
    success = agent_factory.remove_agent(agent.name)
    
    assert success
    assert agent.name not in agent_factory.agents
    assert agent.name not in agent_factory.agent_states

def test_get_agent_status(agent_factory, mock_agents):
    """Test getting agent status."""
    agent = agent_factory.create_agent('generation')
    status = agent_factory.get_agent_status(agent.name)
    
    assert status is not None
    assert status['type'] == 'generation'
    assert 'metrics' in status
    assert 'error_count' in status

def test_get_all_agent_status(agent_factory, mock_agents):
    """Test getting all agent statuses."""
    agent1 = agent_factory.create_agent('generation')
    agent2 = agent_factory.create_agent('backtesting')
    
    statuses = agent_factory.get_all_agent_status()
    
    assert len(statuses) == 2
    assert agent1.name in statuses
    assert agent2.name in statuses

def test_create_agent_pipeline(agent_factory, mock_agents):
    """Test creating agent pipeline."""
    pipeline_config = {
        'agents': [
            {'type': 'generation'},
            {'type': 'backtesting'},
            {'type': 'risk'}
        ]
    }
    
    pipeline_agents = agent_factory.create_agent_pipeline(pipeline_config)
    
    assert len(pipeline_agents) == 3
    assert any(a.name.startswith('generation') for a in pipeline_agents.values())
    assert any(a.name.startswith('backtesting') for a in pipeline_agents.values())
    assert any(a.name.startswith('risk') for a in pipeline_agents.values())

@pytest.mark.asyncio
async def test_process_pipeline(agent_factory, mock_agents):
    """Test processing data through pipeline."""
    # Create pipeline
    pipeline_config = {
        'agents': [
            {'type': 'generation'},
            {'type': 'backtesting'}
        ]
    }
    pipeline_agents = agent_factory.create_agent_pipeline(pipeline_config)
    
    # Configure mock responses
    for agent in pipeline_agents.values():
        agent.process = Mock(return_value={'status': 'success', 'data': {}})
    
    # Process data
    input_data = {'test': 'data'}
    results = await agent_factory.process_pipeline(pipeline_agents, input_data)
    
    assert results['pipeline_status'] == 'success'
    assert len(results['agent_results']) == 2
    assert not results['errors']

@pytest.mark.asyncio
async def test_pipeline_error_handling(agent_factory, mock_agents):
    """Test pipeline error handling."""
    # Create pipeline
    pipeline_config = {
        'agents': [
            {'type': 'generation'},
            {'type': 'backtesting'}
        ]
    }
    pipeline_agents = agent_factory.create_agent_pipeline(pipeline_config)
    
    # Make second agent fail
    agents = list(pipeline_agents.values())
    agents[0].process = Mock(return_value={'status': 'success', 'data': {}})
    agents[1].process = Mock(side_effect=Exception("Test error"))
    
    # Process data
    input_data = {'test': 'data'}
    results = await agent_factory.process_pipeline(pipeline_agents, input_data)
    
    assert results['pipeline_status'] == 'error'
    assert len(results['errors']) == 1
    assert 'Test error' in results['errors'][0]['error']

def test_cleanup(agent_factory, mock_agents):
    """Test agent factory cleanup."""
    # Create some agents
    agent_factory.create_agent('generation')
    agent_factory.create_agent('backtesting')
    agent_factory.create_agent('risk')
    
    # Clean up
    agent_factory.cleanup()
    
    assert len(agent_factory.agents) == 0
    assert len(agent_factory.agent_states) == 0

def test_invalid_agent_type(agent_factory):
    """Test creating invalid agent type."""
    with pytest.raises(ValueError):
        agent_factory.create_agent('invalid_type')

@pytest.mark.asyncio
async def test_pipeline_data_flow(agent_factory, mock_agents):
    """Test data flow through pipeline."""
    # Create pipeline
    pipeline_config = {
        'agents': [
            {'type': 'generation'},
            {'type': 'backtesting'},
            {'type': 'risk'}
        ]
    }
    pipeline_agents = agent_factory.create_agent_pipeline(pipeline_config)
    
    # Configure mock responses with data flow
    agents = list(pipeline_agents.values())
    agents[0].process = Mock(return_value={'status': 'success', 'data': {'strategy': 'test1'}})
    agents[1].process = Mock(return_value={'status': 'success', 'data': {'backtest': 'test2'}})
    agents[2].process = Mock(return_value={'status': 'success', 'data': {'risk': 'test3'}})
    
    # Process data
    input_data = {'initial': 'data'}
    results = await agent_factory.process_pipeline(pipeline_agents, input_data)
    
    # Verify data flow
    assert results['pipeline_status'] == 'success'
    assert agents[1].process.call_args[0][0]['strategy'] == 'test1'
    assert agents[2].process.call_args[0][0]['backtest'] == 'test2'