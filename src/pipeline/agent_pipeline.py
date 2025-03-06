"""
Agent pipeline for orchestrating the trading strategy development process.

This pipeline supports using Ollama with DeepSeek R1 models and a simplified memory system.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..core.llm import LLMInterface, create_llm_interface
from ..core.mcp import ModelContextProtocol, Context, ContextType
from ..core.safety import SafetyChecker
# Support both memory systems
from ..core.memory import MemoryManager, MemoryType
from ..core.simple_memory import SimpleMemoryManager
from ..agents.supervisor import SupervisorAgent
from ..agents.generation import GenerationAgent
from ..agents.base_agent import AgentType
from ..config.system_config import get_full_config

logger = logging.getLogger(__name__)


class AgentPipeline:
    """
    Pipeline for orchestrating trading strategy development using specialized agents.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize agent pipeline.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_full_config()
        
        # Create shared components
        self.mcp = ModelContextProtocol()
        
        # Initialize LLM interface based on provider config
        llm_config = self.config.get("llm", {})
        if config and "llm_provider" in config:
            llm_config["provider"] = config["llm_provider"]
        if config and "llm_model" in config:
            llm_config["model"] = config["llm_model"]
            
        # Create LLM interface using factory method
        self.llm_interface = create_llm_interface(llm_config)
        
        # Initialize memory system (simple or standard)
        use_simple_memory = config.get("use_simple_memory", True) if config else False
        if use_simple_memory:
            self.memory_manager = SimpleMemoryManager(memory_dir=self.config.get("memory_dir", "memory"))
            logger.info("Using simplified JSON file-based memory system")
        else:
            self.memory_manager = MemoryManager(config=self.config.get("memory", {}))
            logger.info("Using standard database-backed memory system")
            
        self.safety_checker = SafetyChecker(config=self.config.get("safety", {}))
        
        # Initialize agents dictionary
        self.agents = {}
        
        # Create supervisor agent
        self.supervisor = SupervisorAgent(
            name="supervisor",
            config=self.config.get("agents", {}).get("supervisor", {}),
            llm=self.llm_interface,
            mcp=self.mcp,
            safety=self.safety_checker,
            memory=self.memory_manager
        )
        
        # Store in agents dictionary
        self.agents["supervisor"] = self.supervisor
        
        logger.info("Agent pipeline initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the pipeline components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize supervisor agent
            logger.info("Initializing supervisor agent")
            if not await self.supervisor.initialize():
                logger.error("Failed to initialize supervisor agent")
                return False
            
            # Create and initialize specialized agents
            logger.info("Creating specialized agents")
            try:
                await self._create_specialized_agents()
            except Exception as e:
                logger.error(f"Error creating specialized agents: {str(e)}", exc_info=True)
                return False
            
            logger.info("Agent pipeline initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing agent pipeline: {str(e)}")
            return False
    
    async def _create_specialized_agents(self) -> None:
        """Create and initialize specialized agents."""
        # Check which agents are enabled in config
        agent_configs = self.config.get("agents", {})
        logger.info(f"Agent configs: {agent_configs}")
        
        try:
            # Import specialized agents
            logger.info("Importing specialized agent modules")
            from ..agents.backtesting_agent import BacktestingAgent
            from ..agents.risk_agent import RiskAssessmentAgent
            from ..agents.ranking_agent import RankingAgent
            from ..agents.evolution_agent import EvolutionAgent
            from ..agents.meta_review_agent import MetaReviewAgent
            logger.info("Successfully imported agent modules")
        except Exception as e:
            logger.error(f"Error importing agent modules: {str(e)}", exc_info=True)
            raise
        
        # Create and initialize Generation Agent if enabled
        if agent_configs.get("generation", {}).get("enabled", True):
            logger.info("Initializing Generation Agent")
            try:
                generation_agent = GenerationAgent(
                    name="generation",
                    config=agent_configs.get("generation", {}),
                    llm=self.llm_interface,
                    mcp=self.mcp,
                    safety=self.safety_checker,
                    memory=self.memory_manager
                )
                logger.info("Generation agent instance created")
                
                # Try to initialize it
                init_success = await generation_agent.initialize()
                logger.info(f"Generation agent initialization result: {init_success}")
                
                if init_success:
                    self.agents["generation"] = generation_agent
                    await self.supervisor.register_agent(generation_agent)
                    logger.info("Generation agent initialized and registered")
                else:
                    logger.error("Failed to initialize generation agent")
            except Exception as e:
                logger.error(f"Error creating or initializing generation agent: {str(e)}", exc_info=True)
                raise
        
        # Create and initialize Backtesting Agent if enabled
        if agent_configs.get("backtesting", {}).get("enabled", True):
            backtesting_agent = BacktestingAgent(
                name="backtesting",
                config=agent_configs.get("backtesting", {}),
                llm=self.llm_interface,
                mcp=self.mcp,
                safety=self.safety_checker,
                memory=self.memory_manager
            )
            
            if await backtesting_agent.initialize():
                self.agents["backtesting"] = backtesting_agent
                await self.supervisor.register_agent(backtesting_agent)
                logger.info("Backtesting agent initialized and registered")
            else:
                logger.error("Failed to initialize backtesting agent")
        
        # Create and initialize Risk Assessment Agent if enabled
        if agent_configs.get("risk_assessment", {}).get("enabled", True):
            risk_agent = RiskAssessmentAgent(
                name="risk_assessment",
                config=agent_configs.get("risk_assessment", {}),
                llm=self.llm_interface,
                mcp=self.mcp,
                safety=self.safety_checker,
                memory=self.memory_manager
            )
            
            if await risk_agent.initialize():
                self.agents["risk_assessment"] = risk_agent
                await self.supervisor.register_agent(risk_agent)
                logger.info("Risk assessment agent initialized and registered")
            else:
                logger.error("Failed to initialize risk assessment agent")
        
        # Create and initialize Ranking Agent if enabled
        if agent_configs.get("ranking", {}).get("enabled", True):
            ranking_agent = RankingAgent(
                name="ranking",
                config=agent_configs.get("ranking", {}),
                llm=self.llm_interface,
                mcp=self.mcp,
                safety=self.safety_checker,
                memory=self.memory_manager
            )
            
            if await ranking_agent.initialize():
                self.agents["ranking"] = ranking_agent
                await self.supervisor.register_agent(ranking_agent)
                logger.info("Ranking agent initialized and registered")
            else:
                logger.error("Failed to initialize ranking agent")
        
        # Create and initialize Evolution Agent if enabled
        if agent_configs.get("evolution", {}).get("enabled", True):
            evolution_agent = EvolutionAgent(
                name="evolution",
                config=agent_configs.get("evolution", {}),
                llm=self.llm_interface,
                mcp=self.mcp,
                safety=self.safety_checker,
                memory=self.memory_manager
            )
            
            if await evolution_agent.initialize():
                self.agents["evolution"] = evolution_agent
                await self.supervisor.register_agent(evolution_agent)
                logger.info("Evolution agent initialized and registered")
            else:
                logger.error("Failed to initialize evolution agent")
        
        # Create and initialize Meta-Review Agent if enabled
        if agent_configs.get("meta_review", {}).get("enabled", True):
            meta_review_agent = MetaReviewAgent(
                name="meta_review",
                config=agent_configs.get("meta_review", {}),
                llm=self.llm_interface,
                mcp=self.mcp,
                safety=self.safety_checker,
                memory=self.memory_manager
            )
            
            if await meta_review_agent.initialize():
                self.agents["meta_review"] = meta_review_agent
                await self.supervisor.register_agent(meta_review_agent)
                logger.info("Meta-review agent initialized and registered")
            else:
                logger.error("Failed to initialize meta-review agent")
    
    async def start(self) -> bool:
        """
        Start the agent pipeline.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            # Start supervisor agent
            if not await self.supervisor.start():
                logger.error("Failed to start supervisor agent")
                return False
            
            # Start all other agents
            for agent_name, agent in self.agents.items():
                if agent_name != "supervisor":
                    if not await agent.start():
                        logger.error(f"Failed to start {agent_name} agent")
                        return False
            
            logger.info("Agent pipeline started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting agent pipeline: {str(e)}")
            return False
    
    async def stop(self) -> None:
        """Stop the agent pipeline."""
        # Stop all agents
        for agent_name, agent in self.agents.items():
            agent.stop()
            logger.info(f"Stopped {agent_name} agent")
        
        logger.info("Agent pipeline stopped")
    
    async def create_research_plan(
        self,
        plan_name: str,
        goal: str,
        constraints: Dict[str, Any] = None,
        deadline: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Create a new research plan for developing trading strategies.
        
        Args:
            plan_name: Name of the research plan
            goal: Goal of the research plan
            constraints: Optional constraints for the research plan
            deadline: Optional deadline for the research plan
            
        Returns:
            Result of plan creation
        """
        # Check if supervisor is running
        from ..agents.base_agent import AgentState
        from ..core.mcp.context import ContextType, Context
        
        if self.supervisor.state != AgentState.RUNNING:
            logger.error(f"Supervisor agent is not running, current state: {self.supervisor.state}")
            return {
                "status": "error",
                "error": f"Supervisor agent is not running, current state: {self.supervisor.state}"
            }
            
        # Create and store USER_GOAL context in MCP
        if hasattr(self, 'mcp') and self.mcp:
            user_goal_context = Context(
                type=ContextType.USER_GOAL,
                data={
                    "goal": goal,
                    "plan_name": plan_name,
                    "constraints": constraints or {}
                },
                metadata={
                    "deadline": deadline.isoformat() if deadline else None,
                    "created_at": datetime.now().isoformat()
                }
            )
            self.mcp.update_context(user_goal_context)
            logger.info(f"Added USER_GOAL context to MCP: {goal}")
        
        # Send command to supervisor
        try:
            message = {
                "type": "command",
                "content": {
                    "command": "create_research_plan",
                    "params": {
                        "plan_name": plan_name,
                        "goal": goal,
                        "constraints": constraints or {},
                        "deadline": deadline.isoformat() if deadline else None
                    }
                }
            }
            
            logger.debug(f"Sending create_research_plan message to supervisor: {message}")
            response = await self.supervisor.send_message(message)
            logger.debug(f"Received response from supervisor: {response}")
            
            return response
        except Exception as e:
            logger.error(f"Error sending message to supervisor: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": f"Error sending message to supervisor: {str(e)}"
            }
    
    async def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """
        Get the status of a research plan.
        
        Args:
            plan_id: ID of the research plan
            
        Returns:
            Plan status
        """
        # Send command to supervisor
        response = await self.supervisor.send_message({
            "type": "command",
            "content": {
                "command": "get_plan_status",
                "params": {
                    "plan_id": plan_id
                }
            }
        })
        
        return response
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of system performance.
        
        Returns:
            Performance summary
        """
        # Send query to supervisor
        response = await self.supervisor.send_message({
            "type": "query",
            "content": {
                "query_type": "get_performance_summary"
            }
        })
        
        return response
    
    async def run_forever(self) -> None:
        """Run the pipeline indefinitely until interrupted."""
        try:
            # Start the pipeline
            if not await self.start():
                return
            
            logger.info("Pipeline running in continuous auto-pilot mode. Press Ctrl+C to stop.")
            
            # Run indefinitely with periodic status reports
            iteration = 0
            while True:
                iteration += 1
                
                # Every 60 iterations (minutes), log a status update
                if iteration % 60 == 0:
                    # Get performance summary
                    performance = await self.get_performance_summary()
                    if performance.get("status") == "success":
                        summary = performance.get("data", {})
                        total_strategies = summary.get("total_strategies", 0)
                        meeting_targets = summary.get("strategies_meeting_targets", 0)
                        
                        logger.info(f"Auto-pilot running for {iteration} minutes | "
                                   f"Total strategies: {total_strategies} | "
                                   f"Strategies meeting targets: {meeting_targets}")
                
                # Sleep for a minute between checks
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            # Try to restart on error
            logger.info("Attempting to restart pipeline after error...")
            await asyncio.sleep(5)
            try:
                await self.run_forever()
            except Exception:
                logger.error("Failed to restart pipeline after error")
            
        finally:
            # Stop the pipeline
            await self.stop()


async def run_pipeline() -> None:
    """Run the agent pipeline."""
    # Create pipeline
    pipeline = AgentPipeline()
    
    # Log the provider and model info
    logger.info(f"Using LLM provider: {pipeline.llm_interface.config.get('provider', 'ollama')}")
    logger.info(f"Using LLM model: {pipeline.llm_interface.config.get('model', 'deepseek-r1')}")
    
    # Initialize pipeline
    if not await pipeline.initialize():
        logger.error("Failed to initialize pipeline")
        return
    
    # Create a research plan
    response = await pipeline.create_research_plan(
        plan_name="High Sharpe Strategy",
        goal="Develop trading strategies with high Sharpe ratio and limited drawdown",
        constraints={
            "max_lookback_period": 252,  # 1 year of data
            "universe": "sp500",
            "min_liquidity": "high"
        }
    )
    
    plan_id = response.get("data", {}).get("plan_id")
    if not plan_id:
        logger.error("Failed to create research plan")
        await pipeline.stop()
        return
    
    logger.info(f"Created research plan with ID: {plan_id}")
    
    # Run the pipeline indefinitely
    await pipeline.run_forever()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the pipeline
    asyncio.run(run_pipeline())