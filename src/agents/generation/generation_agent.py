"""
Generation agent for creating trading strategies.
"""
import logging
import asyncio
import uuid
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import random

from ...core.llm import LLMInterface, Message, MessageRole
from ...core.mcp import ModelContextProtocol, Context, ContextType
from ...core.safety import SafetyChecker
from ...core.memory import MemoryManager, MemoryType, MemoryImportance
from ..base_agent import BaseAgent, AgentState, AgentType

logger = logging.getLogger(__name__)


class GenerationAgent(BaseAgent):
    """
    Generation agent for creating trading strategies.
    
    This agent is responsible for:
    1. Exploring financial literature for trading strategy patterns
    2. Generating novel trading strategy ideas based on market data
    3. Formulating testable trading hypotheses
    4. Designing implementation logic for strategies
    """
    
    def __init__(
        self,
        name: str = "generation",
        config: Optional[Dict[str, Any]] = None,
        llm: Optional[LLMInterface] = None,
        mcp: Optional[ModelContextProtocol] = None,
        safety: Optional[SafetyChecker] = None,
        memory: Optional[MemoryManager] = None
    ):
        """
        Initialize generation agent.
        
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
            agent_type=AgentType.GENERATION,
            config=config,
            llm=llm,
            mcp=mcp,
            safety=safety,
            memory=memory
        )
        
        # Initialize generation-specific attributes
        self.strategy_templates = self._get_strategy_templates()
        self.exploration_weight = self.config.get("exploration_weight", 0.7)
        self.generated_strategies = {}
    
    def _get_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get strategy templates.
        
        Returns:
            Dictionary of strategy templates
        """
        return {
            "momentum": {
                "name": "Momentum Strategy",
                "description": "A strategy that buys assets that have performed well in the past and sells assets that have performed poorly.",
                "parameters": {
                    "lookback_period": [5, 10, 20, 60, 120, 252],  # days
                    "holding_period": [5, 10, 20, 60],  # days
                    "threshold": [0.05, 0.1, 0.15],  # percent
                    "ranking_method": ["return", "return/volatility"]
                },
                "implementation": "Calculate returns over lookback period, rank assets, buy top performers and sell bottom performers based on threshold."
            },
            "mean_reversion": {
                "name": "Mean Reversion Strategy",
                "description": "A strategy that assumes asset prices will revert to their historical mean over time.",
                "parameters": {
                    "lookback_period": [10, 20, 60, 120],  # days
                    "z_score_threshold": [1.0, 1.5, 2.0, 2.5],  # standard deviations
                    "holding_period": [1, 3, 5, 10],  # days
                    "exit_z_score": [0.5, 0.0, -0.5]  # standard deviations
                },
                "implementation": "Calculate z-score based on historical mean and standard deviation, enter when z-score exceeds threshold, exit when z-score crosses exit level."
            },
            "trend_following": {
                "name": "Trend Following Strategy",
                "description": "A strategy that follows established trends in asset prices.",
                "parameters": {
                    "fast_ma": [5, 10, 20],  # days
                    "slow_ma": [20, 50, 100, 200],  # days
                    "entry_threshold": [0.0, 0.01, 0.02],  # percent
                    "exit_threshold": [0.0, -0.01, -0.02]  # percent
                },
                "implementation": "Calculate fast and slow moving averages, enter when fast crosses above slow by threshold, exit when fast crosses below slow by threshold."
            },
            "breakout": {
                "name": "Breakout Strategy",
                "description": "A strategy that trades breakouts from price ranges or technical patterns.",
                "parameters": {
                    "lookback_period": [10, 20, 30, 50],  # days
                    "threshold": [0.02, 0.03, 0.05],  # percent
                    "stop_loss": [0.01, 0.02, 0.03],  # percent
                    "take_profit": [0.02, 0.03, 0.05, 0.1]  # percent
                },
                "implementation": "Calculate high and low over lookback period, enter when price breaks above high or below low by threshold, set stop loss and take profit levels."
            },
            "volatility": {
                "name": "Volatility Strategy",
                "description": "A strategy that trades based on changes in market volatility.",
                "parameters": {
                    "vol_lookback": [10, 20, 30, 60],  # days
                    "vol_calculation": ["standard_deviation", "atr", "parkinson"],
                    "vol_threshold": [1.5, 2.0, 2.5],  # multiplier
                    "holding_period": [1, 3, 5, 10]  # days
                },
                "implementation": "Calculate historical volatility, enter positions when volatility exceeds or falls below threshold, exit after holding period."
            },
            "pairs_trading": {
                "name": "Pairs Trading Strategy",
                "description": "A strategy that trades pairs of correlated assets when their price relationship deviates from historical norm.",
                "parameters": {
                    "correlation_lookback": [60, 120, 252],  # days
                    "correlation_threshold": [0.7, 0.8, 0.9],  # correlation coefficient
                    "z_score_threshold": [1.5, 2.0, 2.5],  # standard deviations
                    "exit_z_score": [0.5, 0.0, -0.5]  # standard deviations
                },
                "implementation": "Find correlated pairs, calculate spread z-score, enter when z-score exceeds threshold, exit when z-score crosses exit level."
            },
            "calendar_effects": {
                "name": "Calendar Effects Strategy",
                "description": "A strategy that exploits known calendar anomalies in market returns.",
                "parameters": {
                    "effects": ["month_of_year", "day_of_week", "turn_of_month", "holidays"],
                    "lookback_years": [3, 5, 10],  # years
                    "threshold": [0.5, 0.6, 0.7],  # win rate
                    "holding_period": [1, 3, 5]  # days
                },
                "implementation": "Analyze historical returns for calendar effects, enter positions during periods with high probability of positive returns based on threshold."
            },
            "fundamental": {
                "name": "Fundamental Strategy",
                "description": "A strategy that trades based on fundamental metrics like PE ratio, dividend yield, etc.",
                "parameters": {
                    "metrics": ["pe_ratio", "pb_ratio", "dividend_yield", "roe"],
                    "selection_criteria": ["top_percentile", "bottom_percentile", "threshold"],
                    "percentile": [10, 20, 30],  # percent
                    "rebalance_period": [20, 60, 120]  # days
                },
                "implementation": "Rank assets based on fundamental metrics, select top or bottom performers based on criteria, rebalance portfolio periodically."
            }
        }
    
    async def initialize(self) -> bool:
        """
        Perform additional initialization steps.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not await super().initialize():
            return False
            
        try:
            # Load previously generated strategies from memory
            strategy_memories = self.memory.retrieve(
                memory_type=MemoryType.STRATEGY,
                tags=["generation", "strategy"],
                importance_minimum=MemoryImportance.MEDIUM
            )
            
            for memory_id, memory in strategy_memories:
                strategy_id = memory.content.get("strategy_id")
                if strategy_id and strategy_id not in self.generated_strategies:
                    self.generated_strategies[strategy_id] = memory.content.get("strategy", {})
            
            # Initialize generation-specific contexts
            strategy_templates_context = Context(
                type=ContextType.STRATEGY,
                data={
                    "strategy_templates": self.strategy_templates
                }
            )
            self.mcp.update_context(strategy_templates_context)
            
            logger.info(f"GenerationAgent {self.name} initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing GenerationAgent {self.name}: {str(e)}")
            self._log_error(e)
            self.state = AgentState.ERROR
            self.status_message = f"Initialization error: {str(e)}"
            return False
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and generate results.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processing results
        """
        # Determine message type
        message_type = data.get("type", "")
        
        if message_type == "task":
            return await self._process_task(data)
        elif message_type == "query":
            return await self._process_query(data)
        else:
            return {
                "status": "error",
                "error": f"Unknown message type: {message_type}"
            }
    
    async def _process_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task message.
        
        Args:
            data: Task data
            
        Returns:
            Task result
        """
        task_id = data.get("task_id")
        task_type = data.get("task_type")
        parameters = data.get("parameters", {})
        
        if task_type == "generate_strategies":
            return await self._generate_strategies(task_id, parameters)
        else:
            return {
                "status": "error",
                "error": f"Unknown task type: {task_type}"
            }
    
    async def _process_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query message.
        
        Args:
            data: Query data
            
        Returns:
            Query result
        """
        query_type = data.get("query_type")
        parameters = data.get("parameters", {})
        
        if query_type == "get_strategy_templates":
            return {
                "status": "success",
                "templates": self.strategy_templates
            }
        elif query_type == "get_generated_strategies":
            return {
                "status": "success",
                "strategies": list(self.generated_strategies.values())
            }
        else:
            return {
                "status": "error",
                "error": f"Unknown query type: {query_type}"
            }
    
    async def _generate_strategies(
        self,
        task_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate trading strategies.
        
        Args:
            task_id: Task ID
            parameters: Task parameters
            
        Returns:
            Generated strategies
        """
        # Extract parameters
        plan_name = parameters.get("plan_name", "Default Plan")
        goal = parameters.get("goal", "Generate profitable trading strategies")
        constraints = parameters.get("constraints", {})
        performance_targets = parameters.get("performance_targets", {})
        strategy_count = parameters.get("strategy_count", 5)
        
        logger.info(f"Generating {strategy_count} strategies for plan '{plan_name}'")
        
        # Prepare strategies
        strategies = []
        
        # Find market data context if available
        market_data_context = self.mcp.get_context(ContextType.MARKET_DATA)
        
        # Define exploration vs exploitation ratio (higher values favor innovation)
        exploration_ratio = self.exploration_weight
        
        # Generate strategies with a mix of template-based and novel approaches
        for i in range(strategy_count):
            # Decide whether to use a template or generate a novel strategy
            use_template = random.random() > exploration_ratio
            
            if use_template:
                # Template-based generation (exploitation)
                strategy = await self._generate_template_strategy(
                    plan_name=plan_name,
                    performance_targets=performance_targets,
                    constraints=constraints
                )
            else:
                # Novel strategy generation (exploration)
                strategy = await self._generate_novel_strategy(
                    plan_name=plan_name,
                    goal=goal,
                    performance_targets=performance_targets,
                    constraints=constraints,
                    market_data_context=market_data_context
                )
            
            # Generate strategy ID
            strategy_id = str(uuid.uuid4())
            
            # Add metadata
            strategy["id"] = strategy_id
            strategy["created_at"] = datetime.now().isoformat()
            strategy["task_id"] = task_id
            strategy["plan_name"] = plan_name
            
            # Store in memory
            self.memory.store(
                memory_type=MemoryType.STRATEGY,
                content={
                    "type": "generated_strategy",
                    "strategy_id": strategy_id,
                    "strategy": strategy,
                    "task_id": task_id,
                    "plan_name": plan_name
                },
                importance=MemoryImportance.MEDIUM,
                tags=["generation", "strategy", plan_name]
            )
            
            # Store in local cache
            self.generated_strategies[strategy_id] = strategy
            
            # Add to result
            strategies.append(strategy)
            
            logger.info(f"Generated strategy {i+1}/{strategy_count}: {strategy['name']} (ID: {strategy_id})")
        
        # Update metrics
        self._update_metrics({
            "strategies_generated": self.metrics.get("strategies_generated", 0) + strategy_count,
            "total_strategies": len(self.generated_strategies)
        })
        
        return {
            "status": "success",
            "task_id": task_id,
            "strategies": strategies,
            "strategy_count": len(strategies)
        }
    
    async def _generate_template_strategy(
        self,
        plan_name: str,
        performance_targets: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a strategy based on a template.
        
        Args:
            plan_name: Plan name
            performance_targets: Performance targets
            constraints: Strategy constraints
            
        Returns:
            Generated strategy
        """
        # Select a random template
        template_key = random.choice(list(self.strategy_templates.keys()))
        template = self.strategy_templates[template_key]
        
        # Create base strategy
        strategy = {
            "name": f"{template['name']} - {plan_name}",
            "description": template["description"],
            "type": template_key,
            "parameters": {},
            "implementation": template["implementation"],
            "entry_logic": "",
            "exit_logic": "",
            "position_sizing": "",
            "risk_management": "",
            "backtest_settings": {},
            "performance_expectations": {}
        }
        
        # Randomize parameters from template
        for param_name, param_values in template["parameters"].items():
            strategy["parameters"][param_name] = random.choice(param_values)
        
        # Generate detailed logic using LLM
        prompt = f"""
        Generate detailed entry and exit logic, position sizing rules, and risk management rules for a {template['name']}.
        
        Strategy parameters:
        {json.dumps(strategy['parameters'], indent=2)}
        
        Performance targets:
        {json.dumps(performance_targets, indent=2)}
        
        Constraints:
        {json.dumps(constraints, indent=2)}
        
        The implementation should be detailed enough to be directly implemented in code.
        Format the response as JSON with the following keys:
        "entry_logic", "exit_logic", "position_sizing", "risk_management"
        """
        
        response = await self.chat_with_llm(prompt)
        
        try:
            # Extract JSON from response
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
                
            logic_details = json.loads(json_str)
            
            # Update strategy with generated logic
            for key in ["entry_logic", "exit_logic", "position_sizing", "risk_management"]:
                if key in logic_details:
                    strategy[key] = logic_details[key]
                    
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Error parsing LLM response: {str(e)}")
            logger.debug(f"LLM response: {response}")
            
            # Fallback to simple logic
            strategy["entry_logic"] = f"Enter when conditions based on {template_key} parameters are met."
            strategy["exit_logic"] = f"Exit when opposite conditions occur or stop loss is hit."
            strategy["position_sizing"] = "Equal allocation across positions with risk-based position sizing."
            strategy["risk_management"] = "Use stop loss of 2%, max portfolio exposure of 20% per position."
        
        # Set backtest settings
        strategy["backtest_settings"] = {
            "start_date": "2018-01-01",
            "end_date": "2023-01-01",
            "universe": "sp500",
            "rebalance_frequency": "daily",
            "commission": 0.001  # 10 bps
        }
        
        # Set performance expectations
        strategy["performance_expectations"] = {
            "cagr": performance_targets.get("cagr", 0.25),
            "sharpe_ratio": performance_targets.get("sharpe_ratio", 1.0),
            "max_drawdown": performance_targets.get("max_drawdown", 0.2),
            "avg_profit": performance_targets.get("avg_profit", 0.0075)
        }
        
        # Generate code implementation using LLM
        code_prompt = f"""
        Generate Python code that implements the following trading strategy using Pandas and NumPy.
        
        Strategy name: {strategy['name']}
        Strategy description: {strategy['description']}
        
        Parameters:
        {json.dumps(strategy['parameters'], indent=2)}
        
        Entry logic: {strategy['entry_logic']}
        Exit logic: {strategy['exit_logic']}
        Position sizing: {strategy['position_sizing']}
        Risk management: {strategy['risk_management']}
        
        The code should include:
        1. A function to generate trading signals based on historical data
        2. A function to apply position sizing and risk management
        3. A function to simulate execution of the strategy
        
        Return just the Python code without any explanations or commentary.
        """
        
        code_response = await self.chat_with_llm(code_prompt)
        
        # Extract code from response
        code = code_response
        if "```python" in code_response:
            code = code_response.split("```python")[1].split("```")[0]
        elif "```" in code_response:
            code = code_response.split("```")[1].split("```")[0]
            
        strategy["implementation_code"] = code
        
        return strategy
    
    async def _generate_novel_strategy(
        self,
        plan_name: str,
        goal: str,
        performance_targets: Dict[str, Any],
        constraints: Dict[str, Any],
        market_data_context: Optional[Context] = None
    ) -> Dict[str, Any]:
        """
        Generate a novel trading strategy.
        
        Args:
            plan_name: Plan name
            goal: Strategy goal
            performance_targets: Performance targets
            constraints: Strategy constraints
            market_data_context: Optional market data context
            
        Returns:
            Generated strategy
        """
        # Create prompt for novel strategy generation
        market_data_info = ""
        if market_data_context:
            market_data_info = f"""
            Market data context:
            {json.dumps(market_data_context.data, indent=2)}
            """
        
        prompt = f"""
        Generate a completely new and innovative trading strategy that meets the following criteria.
        
        Goal: {goal}
        
        Performance targets:
        {json.dumps(performance_targets, indent=2)}
        
        Constraints:
        {json.dumps(constraints, indent=2)}
        
        {market_data_info}
        
        The strategy should:
        1. Have a unique approach that is different from traditional strategies
        2. Be implementable using standard market data
        3. Have clear entry and exit rules
        4. Include position sizing and risk management
        5. Be designed to meet the performance targets
        
        Format the response as JSON with the following structure:
        {{
            "name": "Strategy name",
            "description": "Detailed description",
            "type": "Category of strategy (e.g., trend, mean-reversion, volatility, etc.)",
            "parameters": {{
                "parameter1": "value1",
                "parameter2": "value2",
                ...
            }},
            "entry_logic": "Detailed entry logic",
            "exit_logic": "Detailed exit logic",
            "position_sizing": "Position sizing rules",
            "risk_management": "Risk management rules",
            "uniqueness_factors": ["factor1", "factor2", ...],
            "expected_market_conditions": "Description of ideal market conditions"
        }}
        """
        
        response = await self.chat_with_llm(prompt)
        
        try:
            # Extract JSON from response
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
                
            strategy = json.loads(json_str)
            
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Error parsing LLM response for novel strategy: {str(e)}")
            logger.debug(f"LLM response: {response}")
            
            # Fallback to simple strategy
            strategy = {
                "name": f"Adaptive Strategy - {plan_name}",
                "description": "A multi-factor adaptive strategy that adjusts to market conditions",
                "type": "adaptive",
                "parameters": {
                    "lookback_period": 60,
                    "regime_threshold": 0.8,
                    "factor_count": 3
                },
                "entry_logic": "Enter based on combined factor signals when market regime is favorable",
                "exit_logic": "Exit when combined signal turns negative or market regime changes",
                "position_sizing": "Risk-based position sizing with adaptive allocation based on market regime",
                "risk_management": "Dynamic stop-loss based on volatility, max drawdown control",
                "uniqueness_factors": ["Regime detection", "Multi-factor combination", "Adaptive allocation"]
            }
        
        # Set backtest settings
        strategy["backtest_settings"] = {
            "start_date": "2018-01-01",
            "end_date": "2023-01-01",
            "universe": "sp500",
            "rebalance_frequency": "daily",
            "commission": 0.001  # 10 bps
        }
        
        # Set performance expectations
        strategy["performance_expectations"] = {
            "cagr": performance_targets.get("cagr", 0.25),
            "sharpe_ratio": performance_targets.get("sharpe_ratio", 1.0),
            "max_drawdown": performance_targets.get("max_drawdown", 0.2),
            "avg_profit": performance_targets.get("avg_profit", 0.0075)
        }
        
        # Generate code implementation using LLM
        code_prompt = f"""
        Generate Python code that implements the following innovative trading strategy using Pandas and NumPy.
        
        Strategy name: {strategy['name']}
        Strategy description: {strategy['description']}
        Strategy type: {strategy['type']}
        
        Parameters:
        {json.dumps(strategy.get('parameters', {}), indent=2)}
        
        Entry logic: {strategy.get('entry_logic', '')}
        Exit logic: {strategy.get('exit_logic', '')}
        Position sizing: {strategy.get('position_sizing', '')}
        Risk management: {strategy.get('risk_management', '')}
        
        The code should include:
        1. A function to generate trading signals based on historical data
        2. A function to apply position sizing and risk management
        3. A function to simulate execution of the strategy
        
        Return just the Python code without any explanations or commentary.
        """
        
        code_response = await self.chat_with_llm(code_prompt)
        
        # Extract code from response
        code = code_response
        if "```python" in code_response:
            code = code_response.split("```python")[1].split("```")[0]
        elif "```" in code_response:
            code = code_response.split("```")[1].split("```")[0]
            
        strategy["implementation_code"] = code
        
        return strategy