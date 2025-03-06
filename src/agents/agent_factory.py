"""
Factory for creating and managing specialized agents.
"""
import logging
from typing import Dict, List, Any, Optional, Type
from datetime import datetime

from .base_agent import BaseAgent
from .generation_agent import GenerationAgent
from .backtesting_agent import BacktestingAgent
from .risk_agent import RiskAssessmentAgent
from .ranking_agent import RankingAgent
from .evolution_agent import EvolutionAgent
from .meta_review_agent import MetaReviewAgent
from ..utils.config import load_config

logger = logging.getLogger(__name__)

class AgentFactory:
    """
    Factory for creating and managing trading system agents.
    
    Attributes:
        config: Configuration dictionary
        agents: Dictionary of active agents
        agent_types: Mapping of agent types to classes
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize agent factory."""
        self.config = load_config(config_path) if config_path else {}
        
        # Initialize agent registry
        self.agents: Dict[str, BaseAgent] = {}
        
        # Define agent types
        self.agent_types: Dict[str, Type[BaseAgent]] = {
            'generation': GenerationAgent,
            'backtesting': BacktestingAgent,
            'risk': RiskAssessmentAgent,
            'ranking': RankingAgent,
            'evolution': EvolutionAgent,
            'meta_review': MetaReviewAgent
        }
        
        # Track agent states
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        
    def create_agent(
        self,
        agent_type: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseAgent:
        """
        Create new agent instance.
        
        Args:
            agent_type: Type of agent to create
            name: Optional agent name
            config: Optional agent configuration
            **kwargs: Additional agent parameters
            
        Returns:
            Created agent instance
            
        Raises:
            ValueError: If agent type is invalid
        """
        if agent_type not in self.agent_types:
            raise ValueError(f"Invalid agent type: {agent_type}")
            
        # Generate name if not provided
        if name is None:
            name = f"{agent_type}_{datetime.now().timestamp()}"
            
        # Create agent instance
        agent_class = self.agent_types[agent_type]
        agent = agent_class(name=name, config=config, **kwargs)
        
        # Register agent
        self.agents[name] = agent
        self.agent_states[name] = {
            'type': agent_type,
            'status': 'initialized',
            'created_at': datetime.now().isoformat(),
            'last_active': datetime.now().isoformat()
        }
        
        logger.info(f"Created {agent_type} agent: {name}")
        return agent
        
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """
        Get agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(name)
        
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """
        Get all agents of specified type.
        
        Args:
            agent_type: Type of agents to retrieve
            
        Returns:
            List of matching agents
        """
        return [
            agent for agent in self.agents.values()
            if isinstance(agent, self.agent_types[agent_type])
        ]
        
    async def start_agent(self, name: str) -> bool:
        """
        Start agent processing.
        
        Args:
            name: Agent name
            
        Returns:
            bool: Success status
        """
        agent = self.get_agent(name)
        if agent:
            await agent.start()
            self.agent_states[name]['status'] = 'running'
            self.agent_states[name]['last_active'] = datetime.now().isoformat()
            return True
        return False
        
    def stop_agent(self, name: str) -> bool:
        """
        Stop agent processing.
        
        Args:
            name: Agent name
            
        Returns:
            bool: Success status
        """
        agent = self.get_agent(name)
        if agent:
            agent.stop()
            self.agent_states[name]['status'] = 'stopped'
            self.agent_states[name]['last_active'] = datetime.now().isoformat()
            return True
        return False
        
    def remove_agent(self, name: str) -> bool:
        """
        Remove agent from factory.
        
        Args:
            name: Agent name
            
        Returns:
            bool: Success status
        """
        if name in self.agents:
            # Stop agent if running
            self.stop_agent(name)
            
            # Remove agent
            del self.agents[name]
            del self.agent_states[name]
            
            logger.info(f"Removed agent: {name}")
            return True
        return False
        
    def get_agent_status(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get agent status.
        
        Args:
            name: Agent name
            
        Returns:
            Agent status dictionary or None if not found
        """
        if name in self.agents and name in self.agent_states:
            agent = self.agents[name]
            state = self.agent_states[name]
            
            return {
                **state,
                'metrics': agent.state.get('metrics', {}),
                'error_count': len(agent.state.get('errors', [])),
                'last_error': agent.state.get('errors', [])[-1] if agent.state.get('errors') else None
            }
        return None
        
    def get_all_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all agents.
        
        Returns:
            Dictionary of agent status information
        """
        return {
            name: self.get_agent_status(name)
            for name in self.agents.keys()
        }
        
    def create_agent_pipeline(
        self,
        pipeline_config: Dict[str, Any]
    ) -> Dict[str, BaseAgent]:
        """
        Create a pipeline of connected agents.
        
        Args:
            pipeline_config: Pipeline configuration
            
        Returns:
            Dictionary of created agents
        """
        pipeline_agents = {}
        
        try:
            # Create agents
            for agent_config in pipeline_config['agents']:
                agent = self.create_agent(
                    agent_type=agent_config['type'],
                    name=agent_config.get('name'),
                    config=agent_config.get('config')
                )
                pipeline_agents[agent.name] = agent
                
            logger.info(f"Created agent pipeline with {len(pipeline_agents)} agents")
            return pipeline_agents
            
        except Exception as e:
            logger.error(f"Error creating agent pipeline: {str(e)}")
            # Cleanup on failure
            for agent in pipeline_agents.values():
                self.remove_agent(agent.name)
            raise
            
    async def process_pipeline(
        self,
        pipeline_agents: Dict[str, BaseAgent],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process data through agent pipeline.
        
        Args:
            pipeline_agents: Dictionary of pipeline agents
            input_data: Input data for processing
            
        Returns:
            Pipeline processing results
        """
        results = {
            'pipeline_status': 'success',
            'agent_results': {},
            'errors': []
        }
        
        current_data = input_data
        
        try:
            # Process through each agent
            for name, agent in pipeline_agents.items():
                try:
                    # Process data
                    agent_result = await agent.process(current_data)
                    
                    # Store results
                    results['agent_results'][name] = agent_result
                    
                    # Update data for next agent
                    current_data = agent_result
                    
                except Exception as e:
                    logger.error(f"Error in agent {name}: {str(e)}")
                    results['errors'].append({
                        'agent': name,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    results['pipeline_status'] = 'error'
                    break
                    
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            results['pipeline_status'] = 'error'
            results['errors'].append({
                'agent': 'pipeline',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
        return results
        
    def cleanup(self):
        """Clean up all agents."""
        for name in list(self.agents.keys()):
            self.remove_agent(name)
            
        self.agents.clear()
        self.agent_states.clear()
        logger.info("Cleaned up all agents")