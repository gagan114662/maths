"""
Backtesting agent for evaluating trading strategies.
"""
import logging
import asyncio
import uuid
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np
import pandas as pd

from ...core.llm import LLMInterface, Message, MessageRole
from ...core.mcp import ModelContextProtocol, Context, ContextType
from ...core.safety import SafetyChecker
from ...core.memory import MemoryManager, MemoryType, MemoryImportance
from ..base_agent import BaseAgent, AgentState, AgentType

logger = logging.getLogger(__name__)


class BacktestingAgent(BaseAgent):
    """
    Backtesting agent for evaluating trading strategies.
    
    This agent is responsible for:
    1. Converting strategy definitions into executable code
    2. Running backtests to evaluate strategy performance
    3. Calculating performance metrics
    4. Identifying strengths and weaknesses of strategies
    """
    
    def __init__(
        self,
        name: str = "backtesting",
        config: Optional[Dict[str, Any]] = None,
        llm: Optional[LLMInterface] = None,
        mcp: Optional[ModelContextProtocol] = None,
        safety: Optional[SafetyChecker] = None,
        memory: Optional[MemoryManager] = None
    ):
        """
        Initialize backtesting agent.
        
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
            agent_type=AgentType.BACKTESTING,
            config=config,
            llm=llm,
            mcp=mcp,
            safety=safety,
            memory=memory
        )
        
        # Initialize backtesting-specific attributes
        self.transaction_cost = self.config.get("transaction_cost", 0.0015)  # 15 basis points
        self.slippage = self.config.get("slippage", 0.001)  # 10 basis points
        self.data_dir = self.config.get("data_dir", "../data/ibkr/1d")
        self.backtested_strategies = {}
        
        # Default test periods
        self.default_train_period = ("2018-01-01", "2021-12-31")
        self.default_test_period = ("2022-01-01", "2023-12-31")
    
    async def initialize(self) -> bool:
        """
        Perform additional initialization steps.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not await super().initialize():
            return False
            
        try:
            # Load previously backtested strategies from memory
            backtest_memories = self.memory.retrieve(
                memory_type=MemoryType.BACKTEST_RESULTS,
                tags=["backtesting", "strategy_performance"],
                importance_minimum=MemoryImportance.MEDIUM
            )
            
            for memory_id, memory in backtest_memories:
                strategy_id = memory.content.get("strategy_id")
                if strategy_id and strategy_id not in self.backtested_strategies:
                    self.backtested_strategies[strategy_id] = memory.content.get("performance_metrics", {})
            
            # Initialize backtesting-specific contexts
            backtest_context = Context(
                type=ContextType.BACKTEST_RESULTS,
                data={
                    "backtested_strategies": list(self.backtested_strategies.keys()),
                    "default_train_period": self.default_train_period,
                    "default_test_period": self.default_test_period
                }
            )
            self.mcp.update_context(backtest_context)
            
            logger.info(f"BacktestingAgent {self.name} initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing BacktestingAgent {self.name}: {str(e)}")
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
        
        if task_type == "backtest_strategy":
            return await self._backtest_strategy(task_id, parameters)
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
        
        if query_type == "get_backtest_results":
            strategy_id = parameters.get("strategy_id")
            if not strategy_id or strategy_id not in self.backtested_strategies:
                return {
                    "status": "error",
                    "error": f"Strategy not found with ID: {strategy_id}"
                }
                
            return {
                "status": "success",
                "strategy_id": strategy_id,
                "backtest_results": self.backtested_strategies[strategy_id]
            }
        elif query_type == "get_all_backtested_strategies":
            return {
                "status": "success",
                "strategies": [
                    {
                        "strategy_id": strategy_id,
                        "metrics": metrics
                    }
                    for strategy_id, metrics in self.backtested_strategies.items()
                ]
            }
        else:
            return {
                "status": "error",
                "error": f"Unknown query type: {query_type}"
            }
    
    async def _backtest_strategy(
        self,
        task_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Backtest a trading strategy.
        
        Args:
            task_id: Task ID
            parameters: Task parameters
            
        Returns:
            Backtest results
        """
        # Extract parameters
        strategy_id = parameters.get("strategy_id")
        strategy = parameters.get("strategy", {})
        performance_targets = parameters.get("performance_targets", {})
        backtest_period = parameters.get("backtest_period", "1y")
        transaction_costs = parameters.get("transaction_costs", True)
        
        if not strategy_id or not strategy:
            return {
                "status": "error",
                "error": "Missing strategy_id or strategy parameters"
            }
            
        logger.info(f"Backtesting strategy {strategy_id}: {strategy.get('name', 'Unknown')}")
        
        # Check if we have implementation code
        implementation_code = strategy.get("implementation_code")
        if not implementation_code:
            # Generate implementation code using LLM
            implementation_code = await self._generate_implementation_code(strategy)
            
        # Create a module for the strategy
        module_path = await self._create_strategy_module(strategy_id, implementation_code)
        
        # Load market data
        universe = strategy.get("backtest_settings", {}).get("universe", "sp500")
        start_date = strategy.get("backtest_settings", {}).get("start_date", "2018-01-01")
        end_date = strategy.get("backtest_settings", {}).get("end_date", "2023-01-01")
        
        market_data = await self._load_market_data(universe, start_date, end_date)
        
        # Run backtest
        try:
            backtest_results = await self._run_backtest(
                strategy_id=strategy_id,
                strategy=strategy,
                module_path=module_path,
                market_data=market_data,
                transaction_costs=transaction_costs
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(backtest_results)
            
            # Store backtest results in memory
            memory_id = self.memory.store(
                memory_type=MemoryType.BACKTEST_RESULTS,
                content={
                    "strategy_id": strategy_id,
                    "strategy_name": strategy.get("name", "Unknown"),
                    "backtest_results": backtest_results,
                    "performance_metrics": performance_metrics,
                    "transaction_costs": transaction_costs,
                    "universe": universe,
                    "period": {
                        "start_date": start_date,
                        "end_date": end_date
                    }
                },
                importance=MemoryImportance.HIGH,
                tags=["backtesting", "strategy_performance", strategy_id]
            )
            
            # Update local cache
            self.backtested_strategies[strategy_id] = performance_metrics
            
            # Update metrics
            self._update_metrics({
                "strategies_backtested": self.metrics.get("strategies_backtested", 0) + 1,
                "total_backtest_runs": self.metrics.get("total_backtest_runs", 0) + 1
            })
            
            # Evaluate against performance targets
            targets_assessment = self._evaluate_against_targets(performance_metrics, performance_targets)
            
            # Return results
            return {
                "status": "success",
                "task_id": task_id,
                "strategy_id": strategy_id,
                "strategy": strategy,
                "backtest_results": backtest_results,
                "performance_metrics": performance_metrics,
                "targets_assessment": targets_assessment,
                "meets_targets": targets_assessment.get("meets_all_targets", False)
            }
            
        except Exception as e:
            logger.error(f"Error backtesting strategy {strategy_id}: {str(e)}")
            self._log_error(e)
            
            return {
                "status": "error",
                "task_id": task_id,
                "strategy_id": strategy_id,
                "error": str(e)
            }
    
    async def _generate_implementation_code(self, strategy: Dict[str, Any]) -> str:
        """
        Generate implementation code for a strategy using LLM.
        
        Args:
            strategy: Strategy definition
            
        Returns:
            Implementation code
        """
        prompt = f"""
        Generate Python code that implements the following trading strategy using Pandas and NumPy.
        
        Strategy name: {strategy.get('name', 'Unknown')}
        Strategy description: {strategy.get('description', '')}
        Strategy type: {strategy.get('type', 'Unknown')}
        
        Parameters:
        {json.dumps(strategy.get('parameters', {}), indent=2)}
        
        Entry logic: {strategy.get('entry_logic', '')}
        Exit logic: {strategy.get('exit_logic', '')}
        Position sizing: {strategy.get('position_sizing', '')}
        Risk management: {strategy.get('risk_management', '')}
        
        The code should include:
        1. A function called 'generate_signals' that takes a DataFrame of market data and returns buy/sell signals
        2. A function called 'apply_position_sizing' that implements the position sizing rules
        3. A function called 'apply_risk_management' that implements the risk management rules
        4. A function called 'backtest' that ties everything together and returns a DataFrame with performance metrics
        
        The market data DataFrame will have the following columns for each asset:
        - Open: opening price
        - High: high price
        - Low: low price
        - Close: closing price
        - Volume: trading volume
        
        Return just the Python code without any explanations or commentary.
        """
        
        response = await self.chat_with_llm(prompt)
        
        # Extract code from response
        code = response
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
            
        return code
    
    async def _create_strategy_module(self, strategy_id: str, implementation_code: str) -> str:
        """
        Create a Python module for the strategy.
        
        Args:
            strategy_id: Strategy ID
            implementation_code: Implementation code
            
        Returns:
            Path to the created module
        """
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create module files
        init_path = os.path.join(temp_dir, "__init__.py")
        with open(init_path, 'w') as f:
            f.write("")
            
        module_path = os.path.join(temp_dir, f"{strategy_id}.py")
        with open(module_path, 'w') as f:
            f.write("""
import pandas as pd
import numpy as np

# Standard imports for strategy implementation
from typing import Dict, List, Tuple, Any, Optional

""")
            f.write(implementation_code)
            
        return module_path
    
    async def _load_market_data(
        self,
        universe: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Load market data for backtesting.
        
        Args:
            universe: Universe of assets to load
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary of market data by symbol
        """
        market_data = {}
        
        # Get list of data files
        data_files = os.listdir(self.data_dir)
        
        # Filter by universe if needed
        if universe == "sp500":
            # Simple filter for demonstration
            data_files = [f for f in data_files if f.endswith('.csv')][:100]  # Just use first 100 files
        elif universe == "nasdaq":
            data_files = [f for f in data_files if f.endswith('.csv')][:50]  # Just use first 50 files
        elif universe == "custom":
            data_files = [f for f in data_files if f.endswith('.csv')][:20]  # Just use first 20 files
        else:
            data_files = [f for f in data_files if f.endswith('.csv')][:10]  # Just use first 10 files
        
        # Load data files
        for file in data_files:
            try:
                symbol = file.split('.')[0]
                file_path = os.path.join(self.data_dir, file)
                
                # Read data
                df = pd.read_csv(file_path)
                
                # Convert date column
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                elif 'date' in df.columns:
                    df['Date'] = pd.to_datetime(df['date'])
                    df = df.drop('date', axis=1)
                    
                # Filter by date range
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                
                # Set index
                df = df.set_index('Date')
                
                # Add to market data
                if not df.empty:
                    market_data[symbol] = df
                    
            except Exception as e:
                logger.warning(f"Error loading data for {file}: {str(e)}")
                
        logger.info(f"Loaded market data for {len(market_data)} symbols")
        return market_data
    
    async def _run_backtest(
        self,
        strategy_id: str,
        strategy: Dict[str, Any],
        module_path: str,
        market_data: Dict[str, pd.DataFrame],
        transaction_costs: bool = True
    ) -> Dict[str, Any]:
        """
        Run a backtest for the strategy.
        
        Args:
            strategy_id: Strategy ID
            strategy: Strategy definition
            module_path: Path to strategy module
            market_data: Market data dictionary
            transaction_costs: Whether to include transaction costs
            
        Returns:
            Backtest results
        """
        # Import strategy module
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"strategy_{strategy_id}", module_path)
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)
        
        # Run backtest
        try:
            # Check if module has necessary functions
            required_functions = ['generate_signals', 'apply_position_sizing', 'apply_risk_management', 'backtest']
            missing_functions = [f for f in required_functions if not hasattr(strategy_module, f)]
            
            if missing_functions:
                raise ValueError(f"Strategy module missing required functions: {missing_functions}")
                
            # Fix common code issues
            self._fix_common_issues(strategy_module)
                
            # Run backtest
            backtest_results = strategy_module.backtest(
                market_data=market_data,
                parameters=strategy.get('parameters', {}),
                transaction_cost=self.transaction_cost if transaction_costs else 0.0,
                slippage=self.slippage
            )
            
            # Convert results to a standardized format
            standardized_results = self._standardize_results(backtest_results)
            
            return standardized_results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise ValueError(f"Backtest failed: {str(e)}")
    
    def _fix_common_issues(self, module: Any) -> None:
        """
        Fix common issues in generated code.
        
        Args:
            module: Strategy module
        """
        # Monkey patch functions if needed
        if hasattr(module, 'backtest') and callable(module.backtest):
            original_backtest = module.backtest
            
            def safe_backtest(*args, **kwargs):
                try:
                    return original_backtest(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in backtest function: {str(e)}")
                    # Return minimal result structure
                    return {
                        'portfolio_value': pd.Series([1.0]),
                        'positions': {},
                        'trades': [],
                        'metrics': {
                            'total_return': 0.0,
                            'cagr': 0.0,
                            'sharpe_ratio': 0.0,
                            'max_drawdown': 0.0
                        }
                    }
                    
            module.backtest = safe_backtest
    
    def _standardize_results(self, backtest_results: Any) -> Dict[str, Any]:
        """
        Standardize backtest results into a common format.
        
        Args:
            backtest_results: Backtest results from strategy module
            
        Returns:
            Standardized results
        """
        standardized = {}
        
        # Handle different result types
        if isinstance(backtest_results, dict):
            # Already a dictionary
            standardized = backtest_results
        elif isinstance(backtest_results, pd.DataFrame):
            # Convert DataFrame to dictionary
            standardized = {
                'portfolio_value': backtest_results['portfolio_value'] if 'portfolio_value' in backtest_results.columns else pd.Series([1.0]),
                'positions': {},
                'trades': [],
                'metrics': {}
            }
        else:
            # Unknown format
            standardized = {
                'portfolio_value': pd.Series([1.0]),
                'positions': {},
                'trades': [],
                'metrics': {}
            }
            
        # Ensure all required fields exist
        if 'portfolio_value' not in standardized:
            standardized['portfolio_value'] = pd.Series([1.0])
            
        if 'positions' not in standardized:
            standardized['positions'] = {}
            
        if 'trades' not in standardized:
            standardized['trades'] = []
            
        if 'metrics' not in standardized:
            standardized['metrics'] = {}
            
        return standardized
    
    def _calculate_performance_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            backtest_results: Standardized backtest results
            
        Returns:
            Performance metrics
        """
        metrics = {}
        
        # Extract portfolio value series
        portfolio_value = backtest_results.get('portfolio_value')
        
        if isinstance(portfolio_value, pd.Series) and len(portfolio_value) > 1:
            # Calculate returns
            returns = portfolio_value.pct_change().dropna()
            
            # Calculate metrics
            total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
            
            # Calculate annualized return (CAGR)
            years = len(returns) / 252  # Assuming 252 trading days per year
            cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0
            
            # Calculate Sharpe ratio (assuming 5% risk-free rate)
            risk_free_rate = 0.05
            excess_returns = returns - (risk_free_rate / 252)
            sharpe_ratio = (excess_returns.mean() * 252) / (returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
            
            # Calculate Sortino ratio
            downside_returns = returns[returns < 0]
            sortino_ratio = (excess_returns.mean() * 252) / (downside_returns.std() * (252 ** 0.5)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max) - 1
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            # Calculate win rate
            trades = backtest_results.get('trades', [])
            if trades:
                profitable_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
                win_rate = profitable_trades / len(trades) if len(trades) > 0 else 0
            else:
                win_rate = 0
                
            # Calculate average profit
            if trades:
                total_profit = sum(trade.get('profit', 0) for trade in trades)
                avg_profit = total_profit / len(trades) if len(trades) > 0 else 0
            else:
                avg_profit = 0
                
            # Store metrics
            metrics = {
                'total_return': total_return,
                'cagr': cagr,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'trades_count': len(trades)
            }
            
        else:
            # Default metrics for invalid portfolio value
            metrics = {
                'total_return': 0.0,
                'cagr': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'trades_count': 0
            }
            
        return metrics
    
    def _evaluate_against_targets(
        self,
        metrics: Dict[str, float],
        targets: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluate performance metrics against targets.
        
        Args:
            metrics: Performance metrics
            targets: Performance targets
            
        Returns:
            Evaluation results
        """
        evaluation = {
            'metrics_vs_targets': {},
            'meets_all_targets': True
        }
        
        # Check each target
        for target_name, target_value in targets.items():
            if target_name in metrics:
                actual_value = metrics[target_name]
                
                # Different comparison for different metrics
                if target_name == 'max_drawdown':
                    # For drawdown, lower is better
                    meets_target = actual_value <= target_value
                    relative_performance = target_value / max(actual_value, 0.001)  # Avoid division by zero
                else:
                    # For other metrics, higher is better
                    meets_target = actual_value >= target_value
                    relative_performance = actual_value / max(target_value, 0.001)  # Avoid division by zero
                    
                evaluation['metrics_vs_targets'][target_name] = {
                    'target': target_value,
                    'actual': actual_value,
                    'meets_target': meets_target,
                    'relative_performance': relative_performance
                }
                
                # Update overall assessment
                evaluation['meets_all_targets'] = evaluation['meets_all_targets'] and meets_target
                
        return evaluation