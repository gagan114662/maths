"""
Supervisor agent implementation for coordinating other specialized agents.
"""
import logging
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import json

from ...core.llm import LLMInterface, Message, MessageRole
from ...core.mcp import ModelContextProtocol, Context, ContextType
from ...core.safety import SafetyChecker
from ...core.memory import MemoryManager, MemoryType, MemoryImportance
from ..base_agent import BaseAgent, AgentState, AgentType

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent for coordinating all other specialized agents.
    
    The supervisor agent is responsible for:
    1. Managing the research plan for developing trading strategies
    2. Allocating tasks to specialized agents
    3. Monitoring progress and performance
    4. Adjusting priorities based on ongoing results
    5. Making final decisions on strategy selection
    """
    
    def __init__(
        self,
        name: str = "supervisor",
        config: Optional[Dict[str, Any]] = None,
        llm: Optional[LLMInterface] = None,
        mcp: Optional[ModelContextProtocol] = None,
        safety: Optional[SafetyChecker] = None,
        memory: Optional[MemoryManager] = None
    ):
        """
        Initialize supervisor agent.
        
        Args:
            name: Agent identifier
            config: Optional configuration dictionary
            llm: Optional LLM interface
            mcp: Optional Model Context Protocol
            safety: Optional Safety checker
            memory: Optional Memory manager
        """
        super().__init__(
            name=name,
            agent_type=AgentType.SUPERVISOR,
            config=config,
            llm=llm,
            mcp=mcp,
            safety=safety,
            memory=memory
        )
        
        # Initialize supervisor-specific attributes
        self.agents: Dict[str, BaseAgent] = {}
        self.research_plans: Dict[str, Dict[str, Any]] = {}
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.results_cache: Dict[str, Dict[str, Any]] = {}
        self.status_update_interval = self.config.get("status_update_interval", 60)  # seconds
        self.coordination_interval = self.config.get("coordination_interval", 300)  # seconds
        
        # Set up performance targets
        system_config = self.config.get("system_config", {})
        self.performance_targets = system_config.get("performance_targets", {
            "cagr": 0.25,           # 25% Compound Annual Growth Rate
            "sharpe_ratio": 1.0,    # Sharpe ratio with 5% risk-free rate
            "max_drawdown": 0.20,   # Maximum drawdown of 20%
            "avg_profit": 0.0075,   # Average profit of 0.75%
        })
        
        # Status update and coordination tasks
        self._status_update_task = None
        self._coordination_task = None
        
        logger.info(f"SupervisorAgent {self.name} initialized with performance targets: {self.performance_targets}")
    
    async def initialize(self) -> bool:
        """
        Perform additional initialization steps.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not await super().initialize():
            return False
            
        try:
            # Set up coordination context
            coordination_context = Context(
                type=ContextType.SYSTEM,
                data={
                    "supervisor": {
                        "name": self.name,
                        "performance_targets": self.performance_targets,
                        "connected_agents": []
                    }
                }
            )
            self.mcp.update_context(coordination_context)
            
            logger.info(f"SupervisorAgent {self.name} initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing SupervisorAgent {self.name}: {str(e)}")
            self._log_error(e)
            self.state = AgentState.ERROR
            self.status_message = f"Initialization error: {str(e)}"
            return False
    
    async def start(self) -> bool:
        """
        Start supervisor agent processing.
        
        Returns:
            True if started successfully, False otherwise
        """
        if not await super().start():
            return False
            
        try:
            # Start status update and coordination tasks
            self._status_update_task = asyncio.create_task(self._periodic_status_update())
            self._coordination_task = asyncio.create_task(self._periodic_coordination())
            
            logger.info(f"SupervisorAgent {self.name} started with periodic tasks")
            return True
            
        except Exception as e:
            logger.error(f"Error starting SupervisorAgent {self.name}: {str(e)}")
            self._log_error(e)
            self.state = AgentState.ERROR
            self.status_message = f"Start error: {str(e)}"
            return False
    
    def stop(self) -> bool:
        """
        Stop supervisor agent processing.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        # Stop periodic tasks
        if self._status_update_task:
            self._status_update_task.cancel()
            self._status_update_task = None
            
        if self._coordination_task:
            self._coordination_task.cancel()
            self._coordination_task = None
            
        # Stop base agent
        return super().stop()
    
    async def register_agent(self, agent: BaseAgent) -> bool:
        """
        Register a specialized agent with the supervisor.
        
        Args:
            agent: The agent to register
            
        Returns:
            True if registered successfully, False otherwise
        """
        agent_name = agent.name
        agent_type = agent.agent_type
        
        # Check if agent already registered
        if agent_name in self.agents:
            logger.warning(f"Agent {agent_name} already registered with supervisor")
            return False
            
        # Register agent
        self.agents[agent_name] = agent
        
        # Connect agent for communication
        await self.connect_agent(agent_name, agent)
        await agent.connect_agent(self.name, self)
        
        # Update coordination context
        coordination_context = self.mcp.get_context(ContextType.SYSTEM)
        if coordination_context:
            supervisor_data = coordination_context.data.get("supervisor", {})
            connected_agents = supervisor_data.get("connected_agents", [])
            connected_agents.append({
                "name": agent_name,
                "type": agent_type.value,
                "state": agent.state.value
            })
            supervisor_data["connected_agents"] = connected_agents
            coordination_context.update({"supervisor": supervisor_data})
            self.mcp.update_context(coordination_context)
        
        logger.info(f"Agent {agent_name} ({agent_type.value}) registered with supervisor")
        return True
    
    async def unregister_agent(self, agent_name: str) -> bool:
        """
        Unregister a specialized agent from the supervisor.
        
        Args:
            agent_name: The name of the agent to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        # Check if agent is registered
        if agent_name not in self.agents:
            logger.warning(f"Agent {agent_name} not registered with supervisor")
            return False
            
        # Remove agent from connected agents
        if agent_name in self.connected_agents:
            del self.connected_agents[agent_name]
            
        # Remove agent from agents dict
        agent = self.agents.pop(agent_name)
        
        # Update coordination context
        coordination_context = self.mcp.get_context(ContextType.SYSTEM)
        if coordination_context:
            supervisor_data = coordination_context.data.get("supervisor", {})
            connected_agents = supervisor_data.get("connected_agents", [])
            connected_agents = [a for a in connected_agents if a["name"] != agent_name]
            supervisor_data["connected_agents"] = connected_agents
            coordination_context.update({"supervisor": supervisor_data})
            self.mcp.update_context(coordination_context)
        
        logger.info(f"Agent {agent_name} unregistered from supervisor")
        return True
    
    async def create_research_plan(
        self,
        plan_name: str,
        goal: str,
        constraints: Dict[str, Any],
        deadline: Optional[datetime] = None
    ) -> str:
        """
        Create a new research plan for developing trading strategies.
        
        Args:
            plan_name: Name of the research plan
            goal: Goal of the research plan
            constraints: Constraints for the research plan
            deadline: Optional deadline for the research plan
            
        Returns:
            Plan ID
        """
        # Generate plan ID
        plan_id = str(uuid.uuid4())
        
        # Create plan structure
        plan = {
            "id": plan_id,
            "name": plan_name,
            "goal": goal,
            "constraints": constraints,
            "performance_targets": self.performance_targets.copy(),
            "deadline": deadline.isoformat() if deadline else None,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "tasks": [],
            "results": [],
            "metrics": {}
        }
        
        # Store plan
        self.research_plans[plan_id] = plan
        
        # Store in memory
        self.memory.store(
            memory_type=MemoryType.RESEARCH,
            content={
                "type": "research_plan",
                "plan_id": plan_id,
                "plan_name": plan_name,
                "goal": goal,
                "constraints": constraints,
                "performance_targets": self.performance_targets.copy(),
                "deadline": deadline.isoformat() if deadline else None
            },
            importance=MemoryImportance.HIGH,
            tags=["research_plan", plan_name]
        )
        
        # Create plan context
        plan_context = Context(
            type=ContextType.STRATEGY,
            data={
                "research_plan": plan
            }
        )
        self.mcp.update_context(plan_context)
        
        logger.info(f"Created research plan {plan_name} with ID {plan_id}")
        
        # Start the plan by allocating initial tasks
        await self._allocate_tasks_for_plan(plan_id)
        
        return plan_id
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and generate results.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processing results
        """
        # Get message type and content
        message_type = data.get("type", "command")
        content = data.get("content", {})
        
        # Process based on message type
        if message_type == "command":
            return await self._process_command(content)
        elif message_type == "task_result":
            return await self._process_task_result(content)
        elif message_type == "status_update":
            return await self._process_status_update(content)
        elif message_type == "query":
            return await self._process_query(content)
        else:
            logger.warning(f"Unknown message type: {message_type}")
            return {
                "status": "error",
                "error": f"Unknown message type: {message_type}"
            }
    
    async def _process_command(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a command message.
        
        Args:
            content: Command content
            
        Returns:
            Command result
        """
        command = content.get("command")
        params = content.get("params", {})
        
        if command == "create_research_plan":
            try:
                plan_id = await self.create_research_plan(
                    plan_name=params.get("plan_name", "Default Plan"),
                    goal=params.get("goal", "Develop profitable trading strategies"),
                    constraints=params.get("constraints", {}),
                    deadline=datetime.fromisoformat(params["deadline"]) if "deadline" in params else None
                )
                return {
                    "status": "success",
                    "data": {
                        "plan_id": plan_id
                    }
                }
            except Exception as e:
                logger.error(f"Error creating research plan: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "error": f"Error creating research plan: {str(e)}"
                }
            
        elif command == "get_plan_status":
            plan_id = params.get("plan_id")
            if not plan_id or plan_id not in self.research_plans:
                return {
                    "status": "error",
                    "error": f"Plan not found with ID: {plan_id}"
                }
                
            plan = self.research_plans[plan_id]
            return {
                "status": "success",
                "plan_status": plan["status"],
                "plan_metrics": plan["metrics"],
                "active_tasks": [task for task in plan["tasks"] if task["status"] == "in_progress"],
                "completed_tasks": [task for task in plan["tasks"] if task["status"] == "completed"],
                "results": plan["results"]
            }
            
        elif command == "pause_plan":
            plan_id = params.get("plan_id")
            if not plan_id or plan_id not in self.research_plans:
                return {
                    "status": "error",
                    "error": f"Plan not found with ID: {plan_id}"
                }
                
            plan = self.research_plans[plan_id]
            plan["status"] = "paused"
            
            # Pause all active tasks
            for task in plan["tasks"]:
                if task["status"] == "in_progress":
                    task["status"] = "paused"
                    task_id = task["id"]
                    if task_id in self.active_tasks:
                        self.active_tasks[task_id]["status"] = "paused"
            
            return {
                "status": "success",
                "message": f"Plan {plan_id} paused"
            }
            
        elif command == "resume_plan":
            plan_id = params.get("plan_id")
            if not plan_id or plan_id not in self.research_plans:
                return {
                    "status": "error",
                    "error": f"Plan not found with ID: {plan_id}"
                }
                
            plan = self.research_plans[plan_id]
            plan["status"] = "active"
            
            # Resume paused tasks
            for task in plan["tasks"]:
                if task["status"] == "paused":
                    task["status"] = "in_progress"
                    task_id = task["id"]
                    if task_id in self.active_tasks:
                        self.active_tasks[task_id]["status"] = "in_progress"
            
            return {
                "status": "success",
                "message": f"Plan {plan_id} resumed"
            }
            
        elif command == "get_agent_status":
            agent_name = params.get("agent_name")
            if not agent_name or agent_name not in self.agents:
                return {
                    "status": "error",
                    "error": f"Agent not found with name: {agent_name}"
                }
                
            agent = self.agents[agent_name]
            return {
                "status": "success",
                "agent_status": agent.get_state_info()
            }
            
        else:
            return {
                "status": "error",
                "error": f"Unknown command: {command}"
            }
    
    async def _process_task_result(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task result message.
        
        Args:
            content: Task result content
            
        Returns:
            Processing result
        """
        task_id = content.get("task_id")
        agent_name = content.get("agent_name")
        result = content.get("result", {})
        
        if not task_id or task_id not in self.active_tasks:
            return {
                "status": "error",
                "error": f"Task not found with ID: {task_id}"
            }
            
        # Get task info
        task = self.active_tasks[task_id]
        plan_id = task.get("plan_id")
        
        if not plan_id or plan_id not in self.research_plans:
            return {
                "status": "error",
                "error": f"Plan not found with ID: {plan_id}"
            }
            
        # Update task status
        task["status"] = "completed"
        task["completed_at"] = datetime.now().isoformat()
        task["result"] = result
        
        # Update task in plan
        plan = self.research_plans[plan_id]
        for t in plan["tasks"]:
            if t["id"] == task_id:
                t["status"] = "completed"
                t["completed_at"] = datetime.now().isoformat()
                break
                
        # Add result to plan
        result_entry = {
            "task_id": task_id,
            "agent_name": agent_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        plan["results"].append(result_entry)
        
        # Store result in results cache
        self.results_cache[task_id] = {
            "plan_id": plan_id,
            "agent_name": agent_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in memory
        self.memory.store(
            memory_type=MemoryType.RESEARCH,
            content={
                "type": "task_result",
                "task_id": task_id,
                "plan_id": plan_id,
                "agent_name": agent_name,
                "result": result
            },
            importance=MemoryImportance.MEDIUM,
            tags=["task_result", agent_name, plan_id]
        )
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        
        # Update metrics
        self._update_metrics({
            "completed_tasks": self.metrics.get("completed_tasks", 0) + 1
        })
        
        logger.info(f"Task {task_id} completed by agent {agent_name}")
        
        # Allocate new tasks based on result
        await self._allocate_next_tasks(plan_id, task_id, result)
        
        return {
            "status": "success",
            "message": f"Task result processed for task {task_id}"
        }
    
    async def _process_status_update(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a status update message.
        
        Args:
            content: Status update content
            
        Returns:
            Processing result
        """
        agent_name = content.get("agent_name")
        agent_status = content.get("status", {})
        
        if not agent_name or agent_name not in self.agents:
            return {
                "status": "error",
                "error": f"Agent not found with name: {agent_name}"
            }
            
        # Update agent status in coordination context
        coordination_context = self.mcp.get_context(ContextType.SYSTEM)
        if coordination_context:
            supervisor_data = coordination_context.data.get("supervisor", {})
            connected_agents = supervisor_data.get("connected_agents", [])
            
            for agent_info in connected_agents:
                if agent_info["name"] == agent_name:
                    agent_info.update(agent_status)
                    break
                    
            coordination_context.update({"supervisor": supervisor_data})
            self.mcp.update_context(coordination_context)
        
        logger.debug(f"Status update from agent {agent_name}: {agent_status}")
        
        return {
            "status": "success",
            "message": f"Status update processed for agent {agent_name}"
        }
    
    async def _process_query(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query message.
        
        Args:
            content: Query content
            
        Returns:
            Query result
        """
        query_type = content.get("query_type")
        params = content.get("params", {})
        
        if query_type == "get_all_plans":
            return {
                "status": "success",
                "plans": [
                    {
                        "id": plan_id,
                        "name": plan["name"],
                        "status": plan["status"],
                        "created_at": plan["created_at"],
                        "task_count": len(plan["tasks"]),
                        "result_count": len(plan["results"])
                    }
                    for plan_id, plan in self.research_plans.items()
                ]
            }
            
        elif query_type == "get_all_agents":
            return {
                "status": "success",
                "agents": [
                    {
                        "name": agent_name,
                        "type": agent.agent_type.value,
                        "state": agent.state.value
                    }
                    for agent_name, agent in self.agents.items()
                ]
            }
            
        elif query_type == "get_performance_summary":
            # Collect performance metrics from all plans
            all_metrics = []
            for plan in self.research_plans.values():
                if "metrics" in plan and plan["metrics"]:
                    all_metrics.append(plan["metrics"])
            
            # Calculate average metrics
            avg_metrics = {}
            if all_metrics:
                for metric_name in self.performance_targets:
                    values = [m.get(metric_name, 0) for m in all_metrics if metric_name in m]
                    if values:
                        avg_metrics[metric_name] = sum(values) / len(values)
            
            return {
                "status": "success",
                "performance_targets": self.performance_targets,
                "average_metrics": avg_metrics,
                "best_metrics": self._get_best_metrics(),
                "plan_count": len(self.research_plans),
                "completed_tasks": self.metrics.get("completed_tasks", 0)
            }
            
        else:
            return {
                "status": "error",
                "error": f"Unknown query type: {query_type}"
            }
    
    async def _periodic_status_update(self) -> None:
        """Periodically update supervisor status and check agent statuses."""
        try:
            while self.state == AgentState.RUNNING:
                # Update supervisor status
                self._update_metrics({
                    "agent_count": len(self.agents),
                    "plan_count": len(self.research_plans),
                    "active_task_count": len(self.active_tasks)
                })
                
                # Check agent statuses
                for agent_name, agent in self.agents.items():
                    if agent.state == AgentState.ERROR:
                        logger.warning(f"Agent {agent_name} is in error state: {agent.status_message}")
                        
                        # Try to restart the agent
                        if agent.state == AgentState.ERROR:
                            logger.info(f"Attempting to restart agent {agent_name}")
                            await agent.start()
                
                # Sleep for status update interval
                await asyncio.sleep(self.status_update_interval)
                
        except asyncio.CancelledError:
            logger.info("Periodic status update task cancelled")
            
        except Exception as e:
            logger.error(f"Error in periodic status update: {str(e)}")
            self._log_error(e)
    
    async def _periodic_coordination(self) -> None:
        """Periodically coordinate agent activities and update research plans."""
        try:
            while self.state == AgentState.RUNNING:
                # Coordinate active research plans
                for plan_id, plan in self.research_plans.items():
                    if plan["status"] == "active":
                        # Check if all tasks are completed
                        all_completed = all(task["status"] == "completed" for task in plan["tasks"])
                        
                        if all_completed and plan["tasks"]:
                            # All tasks completed, evaluate results
                            logger.info(f"All tasks completed for plan {plan_id}, evaluating results")
                            await self._evaluate_plan_results(plan_id)
                            
                        elif not self.active_tasks:
                            # No active tasks, allocate new tasks
                            logger.info(f"No active tasks for plan {plan_id}, allocating new tasks")
                            await self._allocate_tasks_for_plan(plan_id)
                
                # Sleep for coordination interval
                await asyncio.sleep(self.coordination_interval)
                
        except asyncio.CancelledError:
            logger.info("Periodic coordination task cancelled")
            
        except Exception as e:
            logger.error(f"Error in periodic coordination: {str(e)}")
            self._log_error(e)
    
    async def _allocate_tasks_for_plan(self, plan_id: str) -> None:
        """
        Allocate initial tasks for a research plan.
        
        Args:
            plan_id: ID of the research plan
        """
        if plan_id not in self.research_plans:
            logger.warning(f"Plan not found with ID: {plan_id}")
            return
            
        plan = self.research_plans[plan_id]
        
        # Set plan status to active
        plan["status"] = "active"
        
        # Check if agents are available
        if not self.agents:
            logger.warning("No agents available for task allocation")
            return
            
        # Create tasks for generation agent
        generation_agents = [a for a in self.agents.values() if a.agent_type == AgentType.GENERATION]
        if generation_agents:
            # Create task for generation agent
            gen_agent = generation_agents[0]
            task_id = str(uuid.uuid4())
            
            task = {
                "id": task_id,
                "plan_id": plan_id,
                "agent_name": gen_agent.name,
                "agent_type": gen_agent.agent_type.value,
                "type": "generate_strategies",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "parameters": {
                    "plan_name": plan["name"],
                    "goal": plan["goal"],
                    "constraints": plan["constraints"],
                    "performance_targets": plan["performance_targets"],
                    "strategy_count": self.config.get("strategies_per_generation", 5)
                }
            }
            
            # Add task to plan
            plan["tasks"].append(task)
            
            # Add task to active tasks
            self.active_tasks[task_id] = task
            
            # Send task to agent
            response = await self.send_to_agent(
                agent_name=gen_agent.name,
                message_type="task",
                data={
                    "task_id": task_id,
                    "task_type": "generate_strategies",
                    "parameters": task["parameters"]
                }
            )
            
            if response and response.get("status") == "success":
                # Update task status
                task["status"] = "in_progress"
                logger.info(f"Task {task_id} assigned to agent {gen_agent.name}")
            else:
                # Task assignment failed
                task["status"] = "failed"
                error_msg = response.get("error", "Unknown error") if response else "Failed to send task to agent"
                logger.error(f"Failed to assign task {task_id} to agent {gen_agent.name}: {error_msg}")
                
                # Remove task from active tasks
                del self.active_tasks[task_id]
        else:
            logger.warning("No generation agent available for task allocation")
    
    async def _allocate_next_tasks(
        self,
        plan_id: str,
        completed_task_id: str,
        result: Dict[str, Any]
    ) -> None:
        """
        Allocate next tasks based on completed task result.
        
        Args:
            plan_id: ID of the research plan
            completed_task_id: ID of the completed task
            result: Result of the completed task
        """
        if plan_id not in self.research_plans:
            logger.warning(f"Plan not found with ID: {plan_id}")
            return
            
        plan = self.research_plans[plan_id]
        
        # Find completed task in plan
        completed_task = None
        for task in plan["tasks"]:
            if task["id"] == completed_task_id:
                completed_task = task
                break
                
        if not completed_task:
            logger.warning(f"Completed task not found in plan: {completed_task_id}")
            return
            
        # Determine next tasks based on completed task type
        task_type = completed_task.get("type")
        
        if task_type == "generate_strategies":
            # Strategies were generated, allocate backtesting tasks
            await self._allocate_backtesting_tasks(plan_id, completed_task_id, result)
            
        elif task_type == "backtest_strategy":
            # Backtesting completed, allocate risk assessment tasks
            await self._allocate_risk_assessment_tasks(plan_id, completed_task_id, result)
            
        elif task_type == "assess_risk":
            # Risk assessment completed, allocate ranking tasks
            await self._allocate_ranking_tasks(plan_id, completed_task_id, result)
            
        elif task_type == "rank_strategies":
            # Ranking completed, allocate evolution tasks
            await self._allocate_evolution_tasks(plan_id, completed_task_id, result)
            
        elif task_type == "evolve_strategies":
            # Evolution completed, allocate meta review tasks
            await self._allocate_meta_review_tasks(plan_id, completed_task_id, result)
            
        elif task_type == "meta_review":
            # Meta review completed, evaluate final results
            await self._evaluate_plan_results(plan_id)
            
        else:
            logger.warning(f"Unknown task type for next task allocation: {task_type}")
    
    async def _allocate_backtesting_tasks(
        self,
        plan_id: str,
        generation_task_id: str,
        result: Dict[str, Any]
    ) -> None:
        """
        Allocate backtesting tasks for generated strategies.
        
        Args:
            plan_id: ID of the research plan
            generation_task_id: ID of the strategy generation task
            result: Result of the strategy generation task
        """
        if "strategies" not in result:
            logger.warning("No strategies in generation result")
            return
            
        strategies = result.get("strategies", [])
        if not strategies:
            logger.warning("Empty strategies list in generation result")
            return
            
        # Find backtesting agent
        backtesting_agents = [a for a in self.agents.values() if a.agent_type == AgentType.BACKTESTING]
        if not backtesting_agents:
            logger.warning("No backtesting agent available")
            return
            
        backtesting_agent = backtesting_agents[0]
        
        # Create backtesting task for each strategy
        for strategy in strategies:
            strategy_id = strategy.get("id", str(uuid.uuid4()))
            task_id = str(uuid.uuid4())
            
            task = {
                "id": task_id,
                "plan_id": plan_id,
                "agent_name": backtesting_agent.name,
                "agent_type": backtesting_agent.agent_type.value,
                "type": "backtest_strategy",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "parameters": {
                    "strategy_id": strategy_id,
                    "strategy": strategy,
                    "performance_targets": self.performance_targets,
                    "backtest_period": self.config.get("backtest_period", "1y"),
                    "transaction_costs": self.config.get("transaction_costs", True)
                }
            }
            
            # Add task to plan
            plan = self.research_plans[plan_id]
            plan["tasks"].append(task)
            
            # Add task to active tasks
            self.active_tasks[task_id] = task
            
            # Send task to agent
            response = await self.send_to_agent(
                agent_name=backtesting_agent.name,
                message_type="task",
                data={
                    "task_id": task_id,
                    "task_type": "backtest_strategy",
                    "parameters": task["parameters"]
                }
            )
            
            if response and response.get("status") == "success":
                # Update task status
                task["status"] = "in_progress"
                logger.info(f"Backtesting task {task_id} assigned to agent {backtesting_agent.name}")
            else:
                # Task assignment failed
                task["status"] = "failed"
                error_msg = response.get("error", "Unknown error") if response else "Failed to send task to agent"
                logger.error(f"Failed to assign backtesting task {task_id}: {error_msg}")
                
                # Remove task from active tasks
                del self.active_tasks[task_id]
    
    async def _allocate_risk_assessment_tasks(
        self,
        plan_id: str,
        backtesting_task_id: str,
        result: Dict[str, Any]
    ) -> None:
        """
        Allocate risk assessment tasks for backtested strategies.
        
        Args:
            plan_id: ID of the research plan
            backtesting_task_id: ID of the backtesting task
            result: Result of the backtesting task
        """
        # Check for valid backtest result
        if "strategy" not in result or "backtest_results" not in result:
            logger.warning("Invalid backtest result format")
            return
            
        strategy = result.get("strategy", {})
        backtest_results = result.get("backtest_results", {})
        
        # Find risk assessment agent
        risk_agents = [a for a in self.agents.values() if a.agent_type == AgentType.RISK]
        if not risk_agents:
            logger.warning("No risk assessment agent available")
            return
            
        risk_agent = risk_agents[0]
        
        # Create risk assessment task
        task_id = str(uuid.uuid4())
        
        task = {
            "id": task_id,
            "plan_id": plan_id,
            "agent_name": risk_agent.name,
            "agent_type": risk_agent.agent_type.value,
            "type": "assess_risk",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "parameters": {
                "strategy_id": strategy.get("id"),
                "strategy": strategy,
                "backtest_results": backtest_results,
                "performance_targets": self.performance_targets,
                "risk_limits": self.config.get("risk_limits", {
                    "max_drawdown": self.performance_targets.get("max_drawdown", 0.2),
                    "max_position_size": 0.1
                })
            }
        }
        
        # Add task to plan
        plan = self.research_plans[plan_id]
        plan["tasks"].append(task)
        
        # Add task to active tasks
        self.active_tasks[task_id] = task
        
        # Send task to agent
        response = await self.send_to_agent(
            agent_name=risk_agent.name,
            message_type="task",
            data={
                "task_id": task_id,
                "task_type": "assess_risk",
                "parameters": task["parameters"]
            }
        )
        
        if response and response.get("status") == "success":
            # Update task status
            task["status"] = "in_progress"
            logger.info(f"Risk assessment task {task_id} assigned to agent {risk_agent.name}")
        else:
            # Task assignment failed
            task["status"] = "failed"
            error_msg = response.get("error", "Unknown error") if response else "Failed to send task to agent"
            logger.error(f"Failed to assign risk assessment task {task_id}: {error_msg}")
            
            # Remove task from active tasks
            del self.active_tasks[task_id]
    
    async def _allocate_ranking_tasks(
        self,
        plan_id: str,
        risk_task_id: str,
        result: Dict[str, Any]
    ) -> None:
        """
        Allocate ranking tasks for risk-assessed strategies.
        
        Args:
            plan_id: ID of the research plan
            risk_task_id: ID of the risk assessment task
            result: Result of the risk assessment task
        """
        # Check for valid risk assessment result
        if "strategy" not in result or "risk_assessment" not in result:
            logger.warning("Invalid risk assessment result format")
            return
            
        strategy = result.get("strategy", {})
        risk_assessment = result.get("risk_assessment", {})
        is_acceptable = result.get("is_acceptable", False)
        
        # Skip ranking if strategy doesn't meet risk criteria
        if not is_acceptable:
            logger.info(f"Strategy {strategy.get('id')} does not meet risk criteria, skipping ranking")
            return
            
        # Find ranking agent
        ranking_agents = [a for a in self.agents.values() if a.agent_type == AgentType.RANKING]
        if not ranking_agents:
            logger.warning("No ranking agent available")
            return
            
        ranking_agent = ranking_agents[0]
        
        # Get all risk-assessed strategies for this plan
        plan = self.research_plans[plan_id]
        risk_assessed_strategies = []
        
        # Collect results from all completed risk assessment tasks
        for task in plan["tasks"]:
            if task["type"] == "assess_risk" and task["status"] == "completed":
                task_id = task["id"]
                if task_id in self.results_cache:
                    result_entry = self.results_cache[task_id]
                    result_data = result_entry.get("result", {})
                    
                    if result_data.get("is_acceptable", False):
                        risk_assessed_strategies.append({
                            "strategy": result_data.get("strategy", {}),
                            "risk_assessment": result_data.get("risk_assessment", {}),
                            "backtest_results": result_data.get("backtest_results", {})
                        })
        
        # Add current strategy to the list
        risk_assessed_strategies.append({
            "strategy": strategy,
            "risk_assessment": risk_assessment,
            "backtest_results": result.get("backtest_results", {})
        })
        
        # Create ranking task
        task_id = str(uuid.uuid4())
        
        task = {
            "id": task_id,
            "plan_id": plan_id,
            "agent_name": ranking_agent.name,
            "agent_type": ranking_agent.agent_type.value,
            "type": "rank_strategies",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "parameters": {
                "strategies": risk_assessed_strategies,
                "performance_targets": self.performance_targets,
                "ranking_criteria": self.config.get("ranking_criteria", {
                    "sharpe_weight": 0.4,
                    "returns_weight": 0.3,
                    "drawdown_weight": 0.2,
                    "consistency_weight": 0.1
                })
            }
        }
        
        # Add task to plan
        plan["tasks"].append(task)
        
        # Add task to active tasks
        self.active_tasks[task_id] = task
        
        # Send task to agent
        response = await self.send_to_agent(
            agent_name=ranking_agent.name,
            message_type="task",
            data={
                "task_id": task_id,
                "task_type": "rank_strategies",
                "parameters": task["parameters"]
            }
        )
        
        if response and response.get("status") == "success":
            # Update task status
            task["status"] = "in_progress"
            logger.info(f"Ranking task {task_id} assigned to agent {ranking_agent.name}")
        else:
            # Task assignment failed
            task["status"] = "failed"
            error_msg = response.get("error", "Unknown error") if response else "Failed to send task to agent"
            logger.error(f"Failed to assign ranking task {task_id}: {error_msg}")
            
            # Remove task from active tasks
            del self.active_tasks[task_id]
    
    async def _allocate_evolution_tasks(
        self,
        plan_id: str,
        ranking_task_id: str,
        result: Dict[str, Any]
    ) -> None:
        """
        Allocate evolution tasks for ranked strategies.
        
        Args:
            plan_id: ID of the research plan
            ranking_task_id: ID of the ranking task
            result: Result of the ranking task
        """
        # Check for valid ranking result
        if "ranked_strategies" not in result:
            logger.warning("Invalid ranking result format")
            return
            
        ranked_strategies = result.get("ranked_strategies", [])
        if not ranked_strategies:
            logger.warning("No ranked strategies in ranking result")
            return
            
        # Find evolution agent
        evolution_agents = [a for a in self.agents.values() if a.agent_type == AgentType.EVOLUTION]
        if not evolution_agents:
            logger.warning("No evolution agent available")
            return
            
        evolution_agent = evolution_agents[0]
        
        # Create evolution task
        task_id = str(uuid.uuid4())
        
        # Take the top N strategies for evolution
        top_n = self.config.get("evolution_top_n", 3)
        top_strategies = ranked_strategies[:min(top_n, len(ranked_strategies))]
        
        task = {
            "id": task_id,
            "plan_id": plan_id,
            "agent_name": evolution_agent.name,
            "agent_type": evolution_agent.agent_type.value,
            "type": "evolve_strategies",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "parameters": {
                "top_strategies": top_strategies,
                "performance_targets": self.performance_targets,
                "evolution_params": self.config.get("evolution_params", {
                    "mutation_rate": 0.2,
                    "crossover_rate": 0.7,
                    "generation_size": 5,
                    "generations": 3
                })
            }
        }
        
        # Add task to plan
        plan = self.research_plans[plan_id]
        plan["tasks"].append(task)
        
        # Add task to active tasks
        self.active_tasks[task_id] = task
        
        # Send task to agent
        response = await self.send_to_agent(
            agent_name=evolution_agent.name,
            message_type="task",
            data={
                "task_id": task_id,
                "task_type": "evolve_strategies",
                "parameters": task["parameters"]
            }
        )
        
        if response and response.get("status") == "success":
            # Update task status
            task["status"] = "in_progress"
            logger.info(f"Evolution task {task_id} assigned to agent {evolution_agent.name}")
        else:
            # Task assignment failed
            task["status"] = "failed"
            error_msg = response.get("error", "Unknown error") if response else "Failed to send task to agent"
            logger.error(f"Failed to assign evolution task {task_id}: {error_msg}")
            
            # Remove task from active tasks
            del self.active_tasks[task_id]
    
    async def _allocate_meta_review_tasks(
        self,
        plan_id: str,
        evolution_task_id: str,
        result: Dict[str, Any]
    ) -> None:
        """
        Allocate meta-review tasks for evolved strategies.
        
        Args:
            plan_id: ID of the research plan
            evolution_task_id: ID of the evolution task
            result: Result of the evolution task
        """
        # Check for valid evolution result
        if "evolved_strategies" not in result:
            logger.warning("Invalid evolution result format")
            return
            
        evolved_strategies = result.get("evolved_strategies", [])
        if not evolved_strategies:
            logger.warning("No evolved strategies in evolution result")
            return
            
        # Find meta-review agent
        meta_review_agents = [a for a in self.agents.values() if a.agent_type == AgentType.META_REVIEW]
        if not meta_review_agents:
            logger.warning("No meta-review agent available")
            return
            
        meta_review_agent = meta_review_agents[0]
        
        # Create meta-review task
        task_id = str(uuid.uuid4())
        
        task = {
            "id": task_id,
            "plan_id": plan_id,
            "agent_name": meta_review_agent.name,
            "agent_type": meta_review_agent.agent_type.value,
            "type": "meta_review",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "parameters": {
                "evolved_strategies": evolved_strategies,
                "performance_targets": self.performance_targets,
                "review_criteria": self.config.get("review_criteria", {
                    "performance_alignment": True,
                    "risk_profile": True,
                    "market_conditions": True,
                    "implementation_feasibility": True
                })
            }
        }
        
        # Add task to plan
        plan = self.research_plans[plan_id]
        plan["tasks"].append(task)
        
        # Add task to active tasks
        self.active_tasks[task_id] = task
        
        # Send task to agent
        response = await self.send_to_agent(
            agent_name=meta_review_agent.name,
            message_type="task",
            data={
                "task_id": task_id,
                "task_type": "meta_review",
                "parameters": task["parameters"]
            }
        )
        
        if response and response.get("status") == "success":
            # Update task status
            task["status"] = "in_progress"
            logger.info(f"Meta-review task {task_id} assigned to agent {meta_review_agent.name}")
        else:
            # Task assignment failed
            task["status"] = "failed"
            error_msg = response.get("error", "Unknown error") if response else "Failed to send task to agent"
            logger.error(f"Failed to assign meta-review task {task_id}: {error_msg}")
            
            # Remove task from active tasks
            del self.active_tasks[task_id]
    
    async def _evaluate_plan_results(self, plan_id: str) -> None:
        """
        Evaluate the results of a research plan.
        
        Args:
            plan_id: ID of the research plan
        """
        if plan_id not in self.research_plans:
            logger.warning(f"Plan not found with ID: {plan_id}")
            return
            
        plan = self.research_plans[plan_id]
        
        # Check for meta-review results
        meta_review_tasks = [t for t in plan["tasks"] if t["type"] == "meta_review" and t["status"] == "completed"]
        if not meta_review_tasks:
            logger.warning(f"No completed meta-review tasks for plan {plan_id}")
            return
            
        # Get latest meta-review task
        meta_review_task = sorted(meta_review_tasks, key=lambda t: t.get("completed_at", ""))[0]
        meta_review_task_id = meta_review_task["id"]
        
        # Get meta-review result
        if meta_review_task_id not in self.results_cache:
            logger.warning(f"Meta-review result not found for task {meta_review_task_id}")
            return
            
        result = self.results_cache[meta_review_task_id].get("result", {})
        
        # Check if we have a final strategy recommendation
        if "recommended_strategy" not in result:
            logger.warning("No recommended strategy in meta-review result")
            return
            
        recommended_strategy = result.get("recommended_strategy", {})
        strategy_metrics = result.get("strategy_metrics", {})
        
        # Update plan status and metrics
        plan["status"] = "completed"
        plan["completed_at"] = datetime.now().isoformat()
        plan["metrics"] = strategy_metrics
        plan["final_recommendation"] = {
            "strategy": recommended_strategy,
            "metrics": strategy_metrics,
            "rationale": result.get("recommendation_rationale", "")
        }
        
        # Check if performance targets were met
        targets_met = True
        for metric, target in self.performance_targets.items():
            if metric in strategy_metrics:
                actual = strategy_metrics[metric]
                # Assume metrics where lower is better (like max_drawdown)
                if metric == "max_drawdown":
                    if actual > target:
                        targets_met = False
                        break
                # For metrics where higher is better
                elif actual < target:
                    targets_met = False
                    break
        
        plan["targets_met"] = targets_met
        
        # Store in memory
        self.memory.store(
            memory_type=MemoryType.RESEARCH,
            content={
                "type": "plan_completion",
                "plan_id": plan_id,
                "plan_name": plan["name"],
                "recommended_strategy": recommended_strategy,
                "strategy_metrics": strategy_metrics,
                "targets_met": targets_met
            },
            importance=MemoryImportance.HIGH,
            tags=["plan_completion", plan["name"]],
            metadata={
                "targets_met": targets_met,
                "metrics": strategy_metrics
            }
        )
        
        logger.info(f"Plan {plan_id} completed with {'success' if targets_met else 'partial success'}")
        
        # If targets were not met, consider starting a new iteration
        if not targets_met and self.config.get("auto_iterate_plans", True):
            logger.info(f"Performance targets not met for plan {plan_id}, starting new iteration")
            await self._create_refined_plan(plan_id)
    
    async def _create_refined_plan(self, original_plan_id: str) -> None:
        """
        Create a refined plan based on the results of a previous plan.
        
        Args:
            original_plan_id: ID of the original plan
        """
        if original_plan_id not in self.research_plans:
            logger.warning(f"Original plan not found with ID: {original_plan_id}")
            return
            
        original_plan = self.research_plans[original_plan_id]
        
        # Create new plan based on original
        new_plan_id = str(uuid.uuid4())
        new_plan_name = f"{original_plan['name']} - Iteration {len([p for p in self.research_plans.values() if p['name'].startswith(original_plan['name'])]) + 1}"
        
        # Get the final recommendation from the original plan
        final_recommendation = original_plan.get("final_recommendation", {})
        recommended_strategy = final_recommendation.get("strategy", {})
        strategy_metrics = final_recommendation.get("metrics", {})
        
        # Create new plan
        await self.create_research_plan(
            plan_name=new_plan_name,
            goal=f"Refine strategy based on previous iteration to meet performance targets. Original goal: {original_plan['goal']}",
            constraints={
                **original_plan.get("constraints", {}),
                "base_strategy": recommended_strategy,
                "current_metrics": strategy_metrics,
                "improvement_areas": self._identify_improvement_areas(strategy_metrics, self.performance_targets)
            },
            deadline=None
        )
        
        logger.info(f"Created refined plan {new_plan_name} based on original plan {original_plan['name']}")
    
    def _identify_improvement_areas(
        self,
        current_metrics: Dict[str, float],
        targets: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Identify areas for improvement based on current metrics vs targets.
        
        Args:
            current_metrics: Current performance metrics
            targets: Target performance metrics
            
        Returns:
            Dictionary of improvement areas
        """
        improvement_areas = {}
        
        for metric, target in targets.items():
            if metric in current_metrics:
                actual = current_metrics[metric]
                
                # Metrics where lower is better (like max_drawdown)
                if metric == "max_drawdown":
                    if actual > target:
                        improvement_areas[metric] = {
                            "current": actual,
                            "target": target,
                            "gap": actual - target,
                            "priority": "high" if (actual - target) / target > 0.25 else "medium"
                        }
                # Metrics where higher is better
                elif actual < target:
                    improvement_areas[metric] = {
                        "current": actual,
                        "target": target,
                        "gap": target - actual,
                        "priority": "high" if (target - actual) / target > 0.25 else "medium"
                    }
        
        return improvement_areas
    
    def _get_best_metrics(self) -> Dict[str, float]:
        """
        Get the best performance metrics across all plans.
        
        Returns:
            Dictionary of best performance metrics
        """
        best_metrics = {}
        
        for plan in self.research_plans.values():
            metrics = plan.get("metrics", {})
            
            for metric, value in metrics.items():
                if metric not in best_metrics:
                    best_metrics[metric] = value
                else:
                    # For metrics where higher is better
                    if metric != "max_drawdown":
                        best_metrics[metric] = max(best_metrics[metric], value)
                    # For metrics where lower is better
                    else:
                        best_metrics[metric] = min(best_metrics[metric], value)
        
        return best_metrics