"""
Base agent class for specialized trading agents.
"""
import logging
import asyncio
import enum
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
import traceback

from ..core.llm import LLMInterface, Message, MessageRole
from ..core.mcp import ModelContextProtocol, Context, ContextType
from ..core.safety import SafetyChecker
from ..core.memory import MemoryManager, MemoryType, MemoryImportance
from ..config.system_config import get_full_config

logger = logging.getLogger(__name__)


class AgentState(str, enum.Enum):
    """Agent state enumeration."""
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class AgentType(str, enum.Enum):
    """Agent type enumeration."""
    SUPERVISOR = "supervisor"
    GENERATION = "generation"
    BACKTESTING = "backtesting"
    RISK = "risk"
    RANKING = "ranking"
    EVOLUTION = "evolution"
    META_REVIEW = "meta_review"


class BaseAgent(ABC):
    """
    Base class for all trading agents.
    
    Attributes:
        name: Agent identifier
        agent_type: Agent type
        config: Configuration dictionary
        llm: LLM interface
        mcp: Model Context Protocol
        safety: Safety checker
        memory: Memory manager
        state: Current agent state
    """
    
    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        config: Optional[Dict[str, Any]] = None,
        llm: Optional[LLMInterface] = None,
        mcp: Optional[ModelContextProtocol] = None,
        safety: Optional[SafetyChecker] = None,
        memory: Optional[MemoryManager] = None
    ):
        """
        Initialize agent.
        
        Args:
            name: Agent identifier
            agent_type: Agent type
            config: Optional configuration dictionary
            llm: Optional LLM interface
            mcp: Optional Model Context Protocol
            safety: Optional Safety checker
            memory: Optional Memory manager
        """
        self.name = name
        self.agent_type = agent_type
        
        # Load configuration
        system_config = get_full_config()
        agent_config = system_config.get("agents", {}).get(agent_type.value, {})
        self.config = {**agent_config, **(config or {})}
        
        # Initialize components
        self.llm = llm or LLMInterface(config=system_config.get("llm", {}))
        self.mcp = mcp or ModelContextProtocol()
        self.safety = safety or SafetyChecker(config=system_config.get("safety", {}))
        self.memory = memory or MemoryManager(config=system_config.get("memory", {}))
        
        # Initialize state
        self.state = AgentState.INITIALIZED
        self.last_action_time = datetime.now()
        self.metrics = {}
        self.errors = []
        self.status_message = "Agent initialized"
        
        # Initialize communication queues
        self.message_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        
        # Set up prompt templates
        self.system_prompt = self._get_default_system_prompt()
        
        # Set up communication channels with other agents
        self.connected_agents = {}
        
        logger.info(f"Agent {self.name} ({self.agent_type.value}) initialized")
        
        # Log to activity logger if available
        try:
            # Attempt to import activity_logger (added in run_deepseek.sh)
            import sys
            import importlib.util
            spec = importlib.util.find_spec('activity_logger')
            if spec:
                activity_logger = importlib.import_module('activity_logger')
                activity_logger.log_activity(
                    action="Agent Initialization",
                    component=f"{self.agent_type.value.capitalize()} Agent",
                    status="Ready",
                    details=f"Agent {self.name} initialized with {self.llm.model_name if hasattr(self, 'llm') and self.llm else 'no'} LLM",
                    next_steps="Waiting for tasks",
                    notes=f"Type: {self.agent_type.value}"
                )
        except Exception:
            # Activity logger is optional
            pass
        
    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for this agent type.
        
        Returns:
            Default system prompt
        """
        base_prompt = f"""You are a {self.agent_type.value} agent in an autonomous trading strategy system.
Your goal is to {self._get_agent_goal()}.

IMPORTANT GUIDELINES:
1. Always base your decisions on data and evidence, not speculation.
2. Consider risk management in all decisions.
3. Maintain ethical standards in all strategy recommendations.
4. Provide clear explanations for your reasoning.
5. Follow all regulatory and compliance requirements.
"""
        return base_prompt
    
    def _get_agent_goal(self) -> str:
        """
        Get the specific goal description for this agent type.
        
        Returns:
            Goal description
        """
        goals = {
            AgentType.SUPERVISOR: "coordinate the activities of specialized agents to develop optimal trading strategies",
            AgentType.GENERATION: "generate novel trading strategy ideas based on market data and research",
            AgentType.BACKTESTING: "evaluate trading strategies using historical data to measure performance",
            AgentType.RISK: "assess the risk profile of trading strategies and ensure they meet safety guidelines",
            AgentType.RANKING: "compare and rank trading strategies based on their performance and risk metrics",
            AgentType.EVOLUTION: "refine trading strategies through iteration and optimization",
            AgentType.META_REVIEW: "analyze overall system performance and identify patterns across strategies"
        }
        return goals.get(self.agent_type, "assist in trading strategy development")
    
    async def initialize(self) -> bool:
        """
        Perform additional initialization steps.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Set up memory for this agent
            self.memory.set_working_memory(f"agent_{self.name}_state", {
                "type": self.agent_type.value,
                "config": self.config,
                "status": self.state.value
            })
            
            # Store agent information in memory
            self.memory.store(
                memory_type=MemoryType.SYSTEM_EVENT,
                content={
                    "event": "agent_initialized",
                    "agent_name": self.name,
                    "agent_type": self.agent_type.value,
                },
                importance=MemoryImportance.MEDIUM,
                tags=["agent", "initialization", self.agent_type.value]
            )
            
            # Set up agent state context
            agent_state_context = Context(
                type=ContextType.AGENT_STATE,
                data={
                    "name": self.name,
                    "type": self.agent_type.value,
                    "state": self.state.value,
                    "status_message": self.status_message
                }
            )
            self.mcp.update_context(agent_state_context)
            
            # Additional initialization steps can be implemented in subclasses
            self.state = AgentState.STARTING
            self.status_message = "Agent starting"
            logger.info(f"Agent {self.name} initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing agent {self.name}: {str(e)}")
            self._log_error(e)
            self.state = AgentState.ERROR
            self.status_message = f"Initialization error: {str(e)}"
            return False
    
    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and generate results.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processing results
        """
        raise NotImplementedError
        
    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming message.
        
        Args:
            message: Input message
            
        Returns:
            Response message
        """
        try:
            # Validate message
            if not self._validate_message(message):
                raise ValueError("Invalid message format")
                
            # Update metrics
            self._update_metrics({
                "messages_received": self.metrics.get("messages_received", 0) + 1
            })
            
            # Log message receipt
            logger.debug(f"Agent {self.name} received message: {message.get('type')}")
            
            # Process message
            processing_start = datetime.now()
            response = await self.process(message.get('data', {}))
            processing_time = (datetime.now() - processing_start).total_seconds()
            
            # Update metrics
            self._update_metrics({
                "messages_processed": self.metrics.get("messages_processed", 0) + 1,
                "avg_processing_time": (
                    (self.metrics.get("avg_processing_time", 0) * 
                     self.metrics.get("messages_processed", 0) + processing_time) / 
                    (self.metrics.get("messages_processed", 0) + 1)
                )
            })
            
            # Store interaction
            self._store_interaction(message, response)
            
            # Create response
            result = {
                'status': 'success',
                'data': response,
                'agent': {
                    'name': self.name,
                    'type': self.agent_type.value,
                    'state': self.state.value
                },
                'metrics': {
                    'processing_time': processing_time
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling message in agent {self.name}: {str(e)}")
            logger.debug(traceback.format_exc())
            self._log_error(e)
            
            return {
                'status': 'error',
                'error': str(e),
                'agent': {
                    'name': self.name,
                    'type': self.agent_type.value,
                    'state': self.state.value
                },
                'timestamp': datetime.now().isoformat()
            }
            
    async def start(self) -> bool:
        """
        Start agent processing loop.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            # Initialize agent if not already done
            if self.state == AgentState.INITIALIZED:
                if not await self.initialize():
                    return False
            
            # Set state to running        
            self.state = AgentState.RUNNING
            self.status_message = "Agent running"
            logger.info(f"Agent {self.name} started")
            
            # Start message processing loop
            asyncio.create_task(self._process_messages())
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting agent {self.name}: {str(e)}")
            self._log_error(e)
            self.state = AgentState.ERROR
            self.status_message = f"Start error: {str(e)}"
            return False
    
    async def _process_messages(self) -> None:
        """Process messages from the queue."""
        try:
            while self.state == AgentState.RUNNING:
                # Get next message
                message = await self.message_queue.get()
                
                # Process message
                response = await self.handle_message(message)
                
                # Send response
                await self.result_queue.put(response)
                
                # Mark task as done
                self.message_queue.task_done()
                
            logger.info(f"Agent {self.name} message processing loop ended")
            
        except asyncio.CancelledError:
            logger.info(f"Agent {self.name} message processing loop cancelled")
            
        except Exception as e:
            logger.error(f"Error in agent {self.name} message loop: {str(e)}")
            logger.debug(traceback.format_exc())
            self._log_error(e)
            self.state = AgentState.ERROR
            self.status_message = f"Processing error: {str(e)}"
            
    def pause(self) -> bool:
        """
        Pause agent processing.
        
        Returns:
            True if paused successfully, False otherwise
        """
        if self.state == AgentState.RUNNING:
            self.state = AgentState.PAUSED
            self.status_message = "Agent paused"
            logger.info(f"Agent {self.name} paused")
            return True
        else:
            logger.warning(f"Cannot pause agent {self.name} in state {self.state}")
            return False
            
    def resume(self) -> bool:
        """
        Resume agent processing.
        
        Returns:
            True if resumed successfully, False otherwise
        """
        if self.state == AgentState.PAUSED:
            self.state = AgentState.RUNNING
            self.status_message = "Agent running"
            logger.info(f"Agent {self.name} resumed")
            return True
        else:
            logger.warning(f"Cannot resume agent {self.name} in state {self.state}")
            return False
            
    def stop(self) -> bool:
        """
        Stop agent processing.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if self.state in [AgentState.RUNNING, AgentState.PAUSED]:
            self.state = AgentState.STOPPED
            self.status_message = "Agent stopped"
            logger.info(f"Agent {self.name} stopped")
            return True
        else:
            logger.warning(f"Cannot stop agent {self.name} in state {self.state}")
            return False
        
    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send message to agent and wait for response.
        
        Args:
            message: Message to send
            
        Returns:
            Response message
        """
        # Ensure message has required fields
        if 'type' not in message:
            message['type'] = 'request'
            
        if 'timestamp' not in message:
            message['timestamp'] = datetime.now().isoformat()
            
        if 'data' not in message:
            message['data'] = {}
            
        # Add message to queue
        await self.message_queue.put(message)
        
        # Wait for response
        return await self.result_queue.get()
        
    async def chat_with_llm(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        contexts: Optional[List[Context]] = None,
        history: Optional[List[Message]] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Chat with the LLM.
        
        Args:
            user_message: User message content
            system_message: Optional system message (defaults to agent's system prompt)
            contexts: Optional list of contexts to include
            history: Optional conversation history
            temperature: Optional temperature parameter for LLM
            
        Returns:
            LLM response content
        """
        # Create messages
        messages = []
        
        # Add system message
        system_content = system_message if system_message is not None else self.system_prompt
        messages.append(self.llm.create_system_message(system_content))
        
        # Add conversation history if provided
        if history:
            messages.extend(history)
            
        # Add user message
        messages.append(self.llm.create_user_message(user_message))
        
        # Get response from LLM
        response = await self.llm.chat(
            messages=messages,
            contexts=contexts,
            temperature=temperature
        )
        
        # Return response content
        return response.message.content or ""
        
    async def connect_agent(self, agent_name: str, agent: 'BaseAgent') -> bool:
        """
        Connect to another agent for communication.
        
        Args:
            agent_name: Name to refer to the agent
            agent: Agent instance
            
        Returns:
            True if connected successfully, False otherwise
        """
        if agent_name in self.connected_agents:
            logger.warning(f"Agent {agent_name} already connected to {self.name}")
            return False
            
        self.connected_agents[agent_name] = agent
        logger.info(f"Agent {self.name} connected to {agent_name}")
        return True
        
    async def send_to_agent(
        self,
        agent_name: str,
        message_type: str,
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Send message to a connected agent.
        
        Args:
            agent_name: Name of the agent to send to
            message_type: Type of message
            data: Message data
            
        Returns:
            Response from the agent, or None if not connected
        """
        if agent_name not in self.connected_agents:
            logger.warning(f"Agent {agent_name} not connected to {self.name}")
            return None
            
        message = {
            'type': message_type,
            'data': data,
            'source': self.name,
            'timestamp': datetime.now().isoformat()
        }
        
        return await self.connected_agents[agent_name].send_message(message)
        
    def get_state_info(self) -> Dict[str, Any]:
        """Get current agent state and info."""
        return {
            'name': self.name,
            'type': self.agent_type.value,
            'state': self.state.value,
            'status_message': self.status_message,
            'last_action_time': self.last_action_time.isoformat(),
            'metrics': self.metrics.copy(),
            'error_count': len(self.errors),
            'connected_agents': list(self.connected_agents.keys())
        }
        
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """
        Validate message format.
        
        Args:
            message: Message to validate
            
        Returns:
            bool: Whether message is valid
        """
        required_fields = ['type', 'timestamp']
        return all(field in message for field in required_fields)
        
    def _store_interaction(
        self,
        message: Dict[str, Any],
        response: Dict[str, Any]
    ) -> None:
        """
        Store interaction in memory.
        
        Args:
            message: Input message
            response: Response message
        """
        # Store the interaction
        self.memory.store(
            memory_type=MemoryType.AGENT_INTERACTION,
            content={
                'input': message,
                'output': response,
                'agent': self.name,
                'agent_type': self.agent_type.value
            },
            metadata={
                'processing_time': response.get('metrics', {}).get('processing_time', 0),
                'status': response.get('status')
            },
            tags=[self.agent_type.value, 'interaction']
        )
        
    def _log_error(self, error: Exception) -> None:
        """
        Log error in agent state.
        
        Args:
            error: Error to log
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'type': error.__class__.__name__,
            'traceback': traceback.format_exc()
        }
        
        self.errors.append(error_entry)
        
        # Store in memory
        self.memory.store(
            memory_type=MemoryType.SYSTEM_EVENT,
            content={
                'event': 'agent_error',
                'agent': self.name,
                'agent_type': self.agent_type.value,
                'error': str(error),
                'error_type': error.__class__.__name__
            },
            importance=MemoryImportance.HIGH,
            tags=['error', self.agent_type.value]
        )
        
    def _update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update agent metrics.
        
        Args:
            metrics: New metrics to update
        """
        self.metrics.update(metrics)
        self.last_action_time = datetime.now()
        
        # Update agent state context
        agent_state_context = Context(
            type=ContextType.AGENT_STATE,
            data={
                "name": self.name,
                "type": self.agent_type.value,
                "state": self.state.value,
                "status_message": self.status_message,
                "metrics": self.metrics
            }
        )
        self.mcp.update_context(agent_state_context)
        
    def __repr__(self) -> str:
        """Get string representation of the agent."""
        return f"{self.__class__.__name__}(name={self.name}, type={self.agent_type.value}, state={self.state.value})"