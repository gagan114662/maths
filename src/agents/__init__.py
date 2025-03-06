"""
Trading system agent module.
"""
from .base_agent import BaseAgent
from .generation_agent import GenerationAgent
from .backtesting_agent import BacktestingAgent
from .risk_agent import RiskAssessmentAgent
from .ranking_agent import RankingAgent
from .evolution_agent import EvolutionAgent
from .meta_review_agent import MetaReviewAgent
from .agent_factory import AgentFactory

__all__ = [
    'BaseAgent',
    'GenerationAgent',
    'BacktestingAgent',
    'RiskAssessmentAgent',
    'RankingAgent',
    'EvolutionAgent',
    'MetaReviewAgent',
    'AgentFactory'
]

# Version
__version__ = '1.0.0'

# Agent Types
AGENT_TYPES = {
    'generation': GenerationAgent,
    'backtesting': BacktestingAgent,
    'risk': RiskAssessmentAgent,
    'ranking': RankingAgent,
    'evolution': EvolutionAgent,
    'meta_review': MetaReviewAgent
}

# Default Pipeline Configuration
DEFAULT_PIPELINE = {
    'agents': [
        {
            'type': 'generation',
            'name': 'strategy_generator',
            'config': {
                'max_strategies': 10,
                'generation_interval': 3600  # 1 hour
            }
        },
        {
            'type': 'backtesting',
            'name': 'strategy_tester',
            'config': {
                'initial_capital': 100000,
                'trading_costs': True
            }
        },
        {
            'type': 'risk',
            'name': 'risk_assessor',
            'config': {
                'risk_limits': {
                    'max_drawdown': 0.2,
                    'position_size': 0.1
                }
            }
        },
        {
            'type': 'ranking',
            'name': 'strategy_ranker',
            'config': {
                'tournament_size': 10,
                'evaluation_period': 30  # days
            }
        },
        {
            'type': 'evolution',
            'name': 'strategy_evolver',
            'config': {
                'population_size': 50,
                'generations': 10
            }
        },
        {
            'type': 'meta_review',
            'name': 'system_analyzer',
            'config': {
                'analysis_interval': 86400  # 24 hours
            }
        }
    ],
    'connections': {
        'strategy_generator': ['strategy_tester'],
        'strategy_tester': ['risk_assessor'],
        'risk_assessor': ['strategy_ranker'],
        'strategy_ranker': ['strategy_evolver'],
        'strategy_evolver': ['system_analyzer']
    }
}

# Create default factory instance
factory = AgentFactory()

def create_agent(agent_type: str, **kwargs):
    """
    Create agent using default factory.
    
    Args:
        agent_type: Type of agent to create
        **kwargs: Agent configuration parameters
        
    Returns:
        Created agent instance
    """
    return factory.create_agent(agent_type, **kwargs)

def create_pipeline(config: dict = None):
    """
    Create agent pipeline using default factory.
    
    Args:
        config: Pipeline configuration (uses DEFAULT_PIPELINE if None)
        
    Returns:
        Dictionary of created agents
    """
    return factory.create_agent_pipeline(config or DEFAULT_PIPELINE)

async def process_pipeline(pipeline_agents: dict, input_data: dict):
    """
    Process data through agent pipeline using default factory.
    
    Args:
        pipeline_agents: Dictionary of pipeline agents
        input_data: Input data for processing
        
    Returns:
        Pipeline processing results
    """
    return await factory.process_pipeline(pipeline_agents, input_data)