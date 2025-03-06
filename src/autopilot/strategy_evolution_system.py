"""
Strategy Evolution System.

This module implements a complete automated strategy development pipeline
with agent coordination, feedback loops, and evolutionary improvement
of trading strategies. It provides continuous operation in autopilot mode,
automatically detecting and rejecting overfitted strategies while creating
checkpoints for the evolution history.
"""

import os
import sys
import logging
import time
import json
import random
import uuid
import shutil
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import required modules
# The exact imports would depend on your project structure
try:
    from src.agents.generation_agent import GenerationAgent
    from src.agents.backtesting_agent import BacktestingAgent
    from src.agents.risk_assessment_agent import RiskAssessmentAgent
    from src.agents.ranking_agent import RankingAgent
    from src.agents.evolution_agent import EvolutionAgent
    from src.agents.meta_review_agent import MetaReviewAgent
except ImportError as e:
    logger.warning(f"Could not import agent modules: {e}")
    logger.warning("Using placeholder agent classes for demonstration")
    
    # Placeholder classes for demonstration
    class BaseAgent:
        def __init__(self, config=None):
            self.config = config or {}
            
        def execute(self, *args, **kwargs):
            return {"status": "success", "message": f"{self.__class__.__name__} execution"}
            
    class GenerationAgent(BaseAgent): pass
    class BacktestingAgent(BaseAgent): pass
    class RiskAssessmentAgent(BaseAgent): pass
    class RankingAgent(BaseAgent): pass
    class EvolutionAgent(BaseAgent): pass
    class MetaReviewAgent(BaseAgent): pass

class StrategyMetrics:
    """
    Helper class to calculate and compare strategy metrics.
    """
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0
    
    @staticmethod
    def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        if not returns or len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_dev = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        
        return np.mean(excess_returns) / downside_dev if downside_dev > 0 else 0.0
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
            
        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        return np.max(drawdown)
    
    @staticmethod
    def calculate_calmar_ratio(returns: List[float], equity_curve: List[float]) -> float:
        """Calculate Calmar ratio."""
        if not returns or len(returns) < 2 or not equity_curve or len(equity_curve) < 2:
            return 0.0
            
        annual_return = np.mean(returns) * 252  # Annualized return (assuming daily returns)
        max_dd = StrategyMetrics.calculate_max_drawdown(equity_curve)
        
        return annual_return / max_dd if max_dd > 0 else 0.0
    
    @staticmethod
    def detect_overfitting(in_sample_metrics: Dict[str, float], 
                         out_of_sample_metrics: Dict[str, float],
                         threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Detect overfitting by comparing in-sample and out-of-sample performance.
        
        Args:
            in_sample_metrics: Dictionary of in-sample performance metrics
            out_of_sample_metrics: Dictionary of out-of-sample performance metrics
            threshold: Threshold for overfitting detection (0.0 to 1.0)
            
        Returns:
            Tuple of (is_overfitted, overfitting_score)
        """
        # Keys to compare
        metrics_to_compare = [
            'sharpe_ratio', 
            'sortino_ratio', 
            'annual_return',
            'win_rate',
            'profit_factor'
        ]
        
        # Calculate performance degradation for each metric
        degradation_scores = []
        
        for metric in metrics_to_compare:
            if metric in in_sample_metrics and metric in out_of_sample_metrics:
                in_sample = in_sample_metrics[metric]
                out_of_sample = out_of_sample_metrics[metric]
                
                # Skip if in-sample metric is zero or negative
                if in_sample <= 0:
                    continue
                    
                # Calculate degradation ratio (1.0 means no degradation, 0.0 means complete failure)
                degradation = max(0.0, min(1.0, out_of_sample / in_sample))
                degradation_scores.append(1.0 - degradation)  # Convert to degradation score
        
        # If no valid comparisons, assume not overfitted
        if not degradation_scores:
            return (False, 0.0)
            
        # Average degradation score
        avg_degradation = sum(degradation_scores) / len(degradation_scores)
        
        # Determine if overfitted based on threshold
        is_overfitted = avg_degradation > threshold
        
        return (is_overfitted, avg_degradation)

class EvolutionaryAlgorithm:
    """
    Implementation of genetic algorithms for strategy evolution.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the evolutionary algorithm.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.population_size = self.config.get('population_size', 10)
        self.mutation_rate = self.config.get('mutation_rate', 0.2)
        self.crossover_rate = self.config.get('crossover_rate', 0.7)
        self.elitism_count = self.config.get('elitism_count', 2)
        
    def initialize_population(self, template_strategy: Dict[str, Any], 
                            parameter_ranges: Dict[str, Tuple]) -> List[Dict[str, Any]]:
        """
        Create an initial population of strategies by randomizing parameters.
        
        Args:
            template_strategy: Base strategy template
            parameter_ranges: Dictionary of parameter ranges (min, max) for each parameter
            
        Returns:
            List of strategy dictionaries
        """
        population = []
        
        for i in range(self.population_size):
            # Clone the template strategy
            strategy = self._deep_copy(template_strategy)
            
            # Randomize parameters
            for param, (min_val, max_val) in parameter_ranges.items():
                # Navigate to the parameter using dot notation
                parts = param.split('.')
                target = strategy
                
                # Navigate to the parent object
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                
                # Set the parameter value
                param_name = parts[-1]
                
                # Handle different types of parameters
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Integer parameter
                    target[param_name] = random.randint(min_val, max_val)
                elif isinstance(min_val, float) and isinstance(max_val, float):
                    # Float parameter
                    target[param_name] = min_val + random.random() * (max_val - min_val)
                elif isinstance(min_val, bool) and isinstance(max_val, bool):
                    # Boolean parameter
                    target[param_name] = random.choice([True, False])
                elif isinstance(min_val, str) and isinstance(max_val, list):
                    # Categorical parameter (string from list)
                    target[param_name] = random.choice(max_val)
                    
            # Add a unique ID
            strategy['id'] = str(uuid.uuid4())
            strategy['generation'] = 0
            
            population.append(strategy)
            
        return population
    
    def evaluate_fitness(self, population: List[Dict[str, Any]], 
                        fitness_function: callable) -> List[float]:
        """
        Evaluate the fitness of each strategy in the population.
        
        Args:
            population: List of strategy dictionaries
            fitness_function: Function that takes a strategy and returns a fitness score
            
        Returns:
            List of fitness scores
        """
        fitness_scores = []
        
        for strategy in population:
            fitness = fitness_function(strategy)
            fitness_scores.append(fitness)
            
        return fitness_scores
    
    def select_parents(self, population: List[Dict[str, Any]], 
                      fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """
        Select parents for reproduction using tournament selection.
        
        Args:
            population: List of strategy dictionaries
            fitness_scores: List of fitness scores
            
        Returns:
            List of selected parent strategies
        """
        # Number of parents to select (same as population size)
        num_parents = self.population_size
        
        # Store selected parents
        parents = []
        
        # Tournament selection
        tournament_size = 3
        
        for _ in range(num_parents):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
            
            # Find the best individual in the tournament
            best_idx = tournament_indices[0]
            best_fitness = fitness_scores[best_idx]
            
            for idx in tournament_indices[1:]:
                if fitness_scores[idx] > best_fitness:
                    best_idx = idx
                    best_fitness = fitness_scores[idx]
            
            # Add the winner to parents
            parents.append(self._deep_copy(population[best_idx]))
            
        return parents
    
    def crossover(self, parents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform crossover between pairs of parents to create offspring.
        
        Args:
            parents: List of parent strategies
            
        Returns:
            List of offspring strategies
        """
        offspring = []
        
        # Keep track of elite individuals
        if self.elitism_count > 0:
            offspring.extend(parents[:self.elitism_count])
        
        # Create remaining offspring through crossover
        for i in range(self.elitism_count, self.population_size, 2):
            # Select two parents
            parent1_idx = random.randint(0, len(parents) - 1)
            parent2_idx = random.randint(0, len(parents) - 1)
            
            # Ensure parents are different
            while parent2_idx == parent1_idx:
                parent2_idx = random.randint(0, len(parents) - 1)
                
            parent1 = parents[parent1_idx]
            parent2 = parents[parent2_idx]
            
            # Perform crossover with probability
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover_strategies(parent1, parent2)
            else:
                # No crossover, just copy parents
                child1 = self._deep_copy(parent1)
                child2 = self._deep_copy(parent2)
                
            # Assign new IDs and update generation
            child1['id'] = str(uuid.uuid4())
            child2['id'] = str(uuid.uuid4())
            
            child1['generation'] = max(parent1.get('generation', 0), parent2.get('generation', 0)) + 1
            child2['generation'] = max(parent1.get('generation', 0), parent2.get('generation', 0)) + 1
            
            # Add to offspring
            offspring.append(child1)
            if len(offspring) < self.population_size:
                offspring.append(child2)
                
        return offspring
    
    def mutate(self, population: List[Dict[str, Any]], 
              parameter_ranges: Dict[str, Tuple]) -> List[Dict[str, Any]]:
        """
        Apply mutation to the population.
        
        Args:
            population: List of strategy dictionaries
            parameter_ranges: Dictionary of parameter ranges for each parameter
            
        Returns:
            List of mutated strategies
        """
        mutated_population = []
        
        # Skip mutation for elite individuals
        for i, strategy in enumerate(population):
            mutated_strategy = self._deep_copy(strategy)
            
            # Skip mutation for elite individuals
            if i < self.elitism_count:
                mutated_population.append(mutated_strategy)
                continue
                
            # Apply mutation to each parameter with probability
            for param, (min_val, max_val) in parameter_ranges.items():
                if random.random() < self.mutation_rate:
                    # Navigate to the parameter using dot notation
                    parts = param.split('.')
                    target = mutated_strategy
                    
                    # Navigate to the parent object
                    for part in parts[:-1]:
                        if part not in target:
                            target[part] = {}
                        target = target[part]
                    
                    # Mutate the parameter
                    param_name = parts[-1]
                    
                    # Handle different types of parameters
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer parameter
                        target[param_name] = random.randint(min_val, max_val)
                    elif isinstance(min_val, float) and isinstance(max_val, float):
                        # Float parameter
                        target[param_name] = min_val + random.random() * (max_val - min_val)
                    elif isinstance(min_val, bool) and isinstance(max_val, bool):
                        # Boolean parameter
                        target[param_name] = random.choice([True, False])
                    elif isinstance(min_val, str) and isinstance(max_val, list):
                        # Categorical parameter (string from list)
                        target[param_name] = random.choice(max_val)
            
            mutated_population.append(mutated_strategy)
            
        return mutated_population
    
    def evolve(self, population: List[Dict[str, Any]], 
              fitness_function: callable,
              parameter_ranges: Dict[str, Tuple]) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Perform one generation of evolution.
        
        Args:
            population: Current population of strategies
            fitness_function: Function to evaluate strategy fitness
            parameter_ranges: Dictionary of parameter ranges
            
        Returns:
            Tuple of (new_population, fitness_scores)
        """
        # Evaluate fitness
        fitness_scores = self.evaluate_fitness(population, fitness_function)
        
        # Select parents
        parents = self.select_parents(population, fitness_scores)
        
        # Perform crossover
        offspring = self.crossover(parents)
        
        # Apply mutation
        new_population = self.mutate(offspring, parameter_ranges)
        
        # Evaluate fitness of new population
        new_fitness_scores = self.evaluate_fitness(new_population, fitness_function)
        
        return new_population, new_fitness_scores
    
    def _deep_copy(self, obj: Any) -> Any:
        """Create a deep copy of an object."""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj
    
    def _crossover_strategies(self, parent1: Dict[str, Any], 
                           parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform crossover between two parent strategies.
        Uses a recursive approach to handle nested dictionaries.
        
        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            
        Returns:
            Tuple of (child1, child2)
        """
        child1 = {}
        child2 = {}
        
        # Get all keys from both parents
        all_keys = set(parent1.keys()) | set(parent2.keys())
        
        for key in all_keys:
            # Skip special keys
            if key in ['id', 'generation']:
                continue
                
            # Decide which parent to inherit from for this key
            if random.random() < 0.5:
                # Child1 inherits from parent1, Child2 inherits from parent2
                if key in parent1:
                    value1 = parent1[key]
                    child1[key] = self._deep_copy(value1)
                if key in parent2:
                    value2 = parent2[key]
                    child2[key] = self._deep_copy(value2)
            else:
                # Child1 inherits from parent2, Child2 inherits from parent1
                if key in parent2:
                    value2 = parent2[key]
                    child1[key] = self._deep_copy(value2)
                if key in parent1:
                    value1 = parent1[key]
                    child2[key] = self._deep_copy(value1)
                    
        return child1, child2

class StrategyEvolutionManager:
    """
    Manager for the complete strategy evolution process.
    Coordinates all agents and maintains the evolution pipeline.
    """
    def __init__(self, config_path: str = None):
        """
        Initialize the strategy evolution manager.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up directories
        self.output_dir = Path(self.config.get('output_dir', 'output'))
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        self.results_dir = Path(self.config.get('results_dir', 'results'))
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evolutionary algorithm
        self.evolution_algo = EvolutionaryAlgorithm(self.config.get('evolution', {}))
        
        # Initialize agents
        self._init_agents()
        
        # State tracking
        self.current_generation = 0
        self.best_strategy = None
        self.best_fitness = float('-inf')
        self.population = []
        self.fitness_scores = []
        self.generation_history = []
        
        # Automatic detection thresholds
        self.overfitting_threshold = self.config.get('overfitting_threshold', 0.5)
        self.convergence_threshold = self.config.get('convergence_threshold', 0.01)
        self.convergence_count = self.config.get('convergence_count', 5)
        
        # Counter for convergence detection
        self.convergence_counter = 0
        
        logger.info("Strategy Evolution Manager initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        config = {}
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.json'):
                        import json
                        config = json.load(f)
                    elif config_path.endswith(('.yaml', '.yml')):
                        import yaml
                        config = yaml.safe_load(f)
                    else:
                        logger.warning(f"Unsupported config file format: {config_path}")
                        
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
        
        # Set default values if not provided
        default_config = {
            'output_dir': 'output',
            'checkpoint_dir': 'checkpoints',
            'results_dir': 'results',
            'max_generations': 20,
            'population_size': 10,
            'overfitting_threshold': 0.5,
            'convergence_threshold': 0.01,
            'convergence_count': 5,
            'checkpoint_frequency': 5,
            'evolution': {
                'population_size': 10,
                'mutation_rate': 0.2,
                'crossover_rate': 0.7,
                'elitism_count': 2
            },
            'parameter_ranges': {
                'strategy.lookback_period': (10, 100),
                'strategy.threshold': (0.0, 0.5),
                'risk.stop_loss': (0.01, 0.1),
                'risk.take_profit': (0.02, 0.2),
                'risk.max_positions': (1, 10)
            }
        }
        
        # Merge default config with loaded config
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def _init_agents(self):
        """Initialize all agent components."""
        try:
            self.generation_agent = GenerationAgent(self.config.get('generation_agent', {}))
            self.backtesting_agent = BacktestingAgent(self.config.get('backtesting_agent', {}))
            self.risk_assessment_agent = RiskAssessmentAgent(self.config.get('risk_assessment_agent', {}))
            self.ranking_agent = RankingAgent(self.config.get('ranking_agent', {}))
            self.evolution_agent = EvolutionAgent(self.config.get('evolution_agent', {}))
            self.meta_review_agent = MetaReviewAgent(self.config.get('meta_review_agent', {}))
            
            logger.info("All agents initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    def generate_initial_population(self) -> List[Dict[str, Any]]:
        """
        Generate an initial population of strategies.
        
        Returns:
            List of strategy dictionaries
        """
        logger.info("Generating initial population of strategies")
        
        try:
            # Generate a template strategy using the generation agent
            template_result = self.generation_agent.execute(
                action="generate_template",
                goal=self.config.get('goal', 'Develop profitable trading strategies')
            )
            
            if 'strategy_template' in template_result:
                template_strategy = template_result['strategy_template']
            else:
                # Fallback template
                template_strategy = {
                    "name": "Base Strategy Template",
                    "description": "Template for generating strategies",
                    "strategy": {
                        "type": "momentum",
                        "lookback_period": 20,
                        "threshold": 0.1
                    },
                    "risk": {
                        "stop_loss": 0.05,
                        "take_profit": 0.1,
                        "max_positions": 5
                    }
                }
            
            # Get parameter ranges from config
            parameter_ranges = self.config.get('parameter_ranges', {})
            
            # Initialize population
            population = self.evolution_algo.initialize_population(
                template_strategy, parameter_ranges
            )
            
            logger.info(f"Generated initial population of {len(population)} strategies")
            
            return population
            
        except Exception as e:
            logger.error(f"Error generating initial population: {str(e)}")
            return []
    
    def evaluate_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a strategy by running backtests and risk assessment.
        
        Args:
            strategy: Strategy dictionary
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating strategy: {strategy.get('name', 'Unnamed')}")
        
        try:
            # Step 1: Run backtesting
            backtest_result = self.backtesting_agent.execute(
                action="backtest",
                strategy=strategy,
                split_data=True,  # Split data into in-sample and out-of-sample
                in_sample_ratio=self.config.get('in_sample_ratio', 0.7)
            )
            
            # Step 2: Run risk assessment
            risk_result = self.risk_assessment_agent.execute(
                action="assess_risk",
                strategy=strategy,
                backtest_result=backtest_result
            )
            
            # Step 3: Check for overfitting
            in_sample_metrics = backtest_result.get('in_sample_metrics', {})
            out_of_sample_metrics = backtest_result.get('out_of_sample_metrics', {})
            
            is_overfitted, overfitting_score = StrategyMetrics.detect_overfitting(
                in_sample_metrics, out_of_sample_metrics, self.overfitting_threshold
            )
            
            # Combine results
            evaluation_result = {
                'strategy_id': strategy.get('id', 'unknown'),
                'strategy_name': strategy.get('name', 'Unnamed'),
                'backtest_result': backtest_result,
                'risk_result': risk_result,
                'is_overfitted': is_overfitted,
                'overfitting_score': overfitting_score,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Strategy evaluation completed: is_overfitted={is_overfitted}, overfitting_score={overfitting_score:.2f}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating strategy: {str(e)}")
            return {
                'strategy_id': strategy.get('id', 'unknown'),
                'strategy_name': strategy.get('name', 'Unnamed'),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def calculate_fitness(self, strategy: Dict[str, Any]) -> float:
        """
        Calculate fitness score for a strategy, incorporating results
        from backtesting and risk assessment.
        
        Args:
            strategy: Strategy dictionary
            
        Returns:
            Fitness score (higher is better)
        """
        # Evaluate the strategy
        evaluation_result = self.evaluate_strategy(strategy)
        
        # Default fitness for failed evaluations
        if 'error' in evaluation_result:
            return float('-inf')
            
        # Extract metrics
        backtest_result = evaluation_result.get('backtest_result', {})
        risk_result = evaluation_result.get('risk_result', {})
        is_overfitted = evaluation_result.get('is_overfitted', True)
        
        # Get out-of-sample metrics (to prevent overfitting)
        metrics = backtest_result.get('out_of_sample_metrics', {})
        
        # Key metrics with weights
        sharpe_ratio = metrics.get('sharpe_ratio', 0.0) * 0.3
        sortino_ratio = metrics.get('sortino_ratio', 0.0) * 0.2
        max_drawdown = metrics.get('max_drawdown', 1.0) * -0.15  # Negative weight
        annual_return = metrics.get('annual_return', 0.0) * 0.15
        win_rate = metrics.get('win_rate', 0.0) * 0.1
        
        # Risk metrics
        risk_score = risk_result.get('risk_score', 0.0) * 0.1
        
        # Penalize overfitted strategies
        overfitting_penalty = -10.0 if is_overfitted else 0.0
        
        # Combined fitness score
        fitness = sharpe_ratio + sortino_ratio + max_drawdown + annual_return + win_rate + risk_score + overfitting_penalty
        
        # Store evaluation result with the strategy for later reference
        strategy['_evaluation_result'] = evaluation_result
        strategy['_fitness'] = fitness
        
        return fitness
    
    def run_evolution(self, max_generations: int = None) -> Dict[str, Any]:
        """
        Run the complete evolution process.
        
        Args:
            max_generations: Maximum number of generations to run
            
        Returns:
            Dictionary with evolution results
        """
        if max_generations is None:
            max_generations = self.config.get('max_generations', 20)
            
        logger.info(f"Starting evolution process for {max_generations} generations")
        
        # Generate initial population if not already done
        if not self.population:
            self.population = self.generate_initial_population()
            if not self.population:
                logger.error("Failed to generate initial population")
                return {"status": "error", "message": "Failed to generate initial population"}
        
        # Parameter ranges for evolution
        parameter_ranges = self.config.get('parameter_ranges', {})
        
        # Evolution loop
        for generation in range(self.current_generation, max_generations):
            logger.info(f"Starting generation {generation + 1}/{max_generations}")
            
            # Evolve the population
            new_population, fitness_scores = self.evolution_algo.evolve(
                self.population,
                self.calculate_fitness,
                parameter_ranges
            )
            
            # Update state
            self.population = new_population
            self.fitness_scores = fitness_scores
            self.current_generation = generation + 1
            
            # Save generation history
            generation_summary = self._summarize_generation()
            self.generation_history.append(generation_summary)
            
            # Update best strategy
            best_idx = fitness_scores.index(max(fitness_scores))
            best_fitness = fitness_scores[best_idx]
            
            if best_fitness > self.best_fitness:
                self.best_strategy = self.population[best_idx]
                self.best_fitness = best_fitness
                self.convergence_counter = 0
                
                logger.info(f"New best strategy found: {self.best_strategy.get('name', 'Unnamed')} with fitness {best_fitness:.4f}")
            else:
                # Check for convergence
                fitness_improvement = best_fitness - self.best_fitness
                
                if fitness_improvement < self.convergence_threshold:
                    self.convergence_counter += 1
                    logger.info(f"Convergence counter: {self.convergence_counter}/{self.convergence_count}")
                    
                    if self.convergence_counter >= self.convergence_count:
                        logger.info(f"Evolution converged after {generation + 1} generations")
                        break
                else:
                    self.convergence_counter = 0
            
            # Create checkpoint at specified frequency
            checkpoint_frequency = self.config.get('checkpoint_frequency', 5)
            if (generation + 1) % checkpoint_frequency == 0:
                self._save_checkpoint()
            
            # Meta-review of the generation
            self._run_meta_review(generation_summary)
            
            logger.info(f"Completed generation {generation + 1}")
            
        # Save final results
        self._save_results()
        
        logger.info(f"Evolution process completed after {self.current_generation} generations")
        
        # Return results summary
        return {
            "status": "success",
            "generations": self.current_generation,
            "best_strategy": self.best_strategy,
            "best_fitness": self.best_fitness,
            "final_population_size": len(self.population),
            "history_size": len(self.generation_history)
        }
    
    def _summarize_generation(self) -> Dict[str, Any]:
        """
        Create a summary of the current generation.
        
        Returns:
            Dictionary with generation summary
        """
        # Get non-empty fitness scores
        valid_scores = [score for score in self.fitness_scores if score != float('-inf')]
        
        # Calculate statistics
        avg_fitness = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        best_idx = self.fitness_scores.index(max(self.fitness_scores)) if self.fitness_scores else -1
        best_fitness = self.fitness_scores[best_idx] if best_idx >= 0 else 0
        best_strategy = self.population[best_idx] if best_idx >= 0 else None
        
        # Count overfitted strategies
        overfitted_count = 0
        for strategy in self.population:
            evaluation_result = strategy.get('_evaluation_result', {})
            if evaluation_result.get('is_overfitted', False):
                overfitted_count += 1
        
        # Create summary
        summary = {
            "generation": self.current_generation,
            "timestamp": datetime.now().isoformat(),
            "population_size": len(self.population),
            "average_fitness": avg_fitness,
            "best_fitness": best_fitness,
            "best_strategy_id": best_strategy.get('id', 'unknown') if best_strategy else 'none',
            "overfitted_count": overfitted_count,
            "overfitted_percentage": overfitted_count / len(self.population) * 100 if self.population else 0
        }
        
        return summary
    
    def _save_checkpoint(self) -> str:
        """
        Save a checkpoint of the current state.
        
        Returns:
            Path to checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"generation_{self.current_generation}_{timestamp}.json"
        
        checkpoint_data = {
            "generation": self.current_generation,
            "timestamp": datetime.now().isoformat(),
            "population": self.population,
            "fitness_scores": self.fitness_scores,
            "best_strategy": self.best_strategy,
            "best_fitness": self.best_fitness,
            "generation_history": self.generation_history
        }
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            return ""
    
    def _save_results(self) -> str:
        """
        Save final results of the evolution process.
        
        Returns:
            Path to results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f"evolution_results_{timestamp}.json"
        
        # Prepare results data
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "total_generations": self.current_generation,
            "final_population_size": len(self.population),
            "best_strategy": self.best_strategy,
            "best_fitness": self.best_fitness,
            "generation_history": self.generation_history,
            "convergence_reached": self.convergence_counter >= self.convergence_count
        }
        
        # Create visualizations
        self._generate_visualizations(timestamp)
        
        try:
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2)
                
            logger.info(f"Saved results to {results_path}")
            return str(results_path)
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return ""
    
    def _generate_visualizations(self, timestamp: str) -> None:
        """
        Generate visualizations of the evolution process.
        
        Args:
            timestamp: Timestamp string for filenames
        """
        # Create visualizations directory
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Fitness progression over generations
            generations = [summary["generation"] for summary in self.generation_history]
            avg_fitness = [summary["average_fitness"] for summary in self.generation_history]
            best_fitness = [summary["best_fitness"] for summary in self.generation_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(generations, avg_fitness, label='Average Fitness')
            plt.plot(generations, best_fitness, label='Best Fitness')
            plt.title('Fitness Progression Over Generations')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.legend()
            plt.grid(True)
            plt.savefig(viz_dir / f"fitness_progression_{timestamp}.png")
            plt.close()
            
            # 2. Overfitting percentage over generations
            overfitted_pct = [summary["overfitted_percentage"] for summary in self.generation_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(generations, overfitted_pct, marker='o')
            plt.title('Overfitting Percentage Over Generations')
            plt.xlabel('Generation')
            plt.ylabel('Overfitted Strategies (%)')
            plt.axhline(y=50, color='r', linestyle='--', label='50% Threshold')
            plt.grid(True)
            plt.legend()
            plt.savefig(viz_dir / f"overfitting_trend_{timestamp}.png")
            plt.close()
            
            # 3. Strategy performance metrics (for best strategy)
            if self.best_strategy:
                evaluation_result = self.best_strategy.get('_evaluation_result', {})
                backtest_result = evaluation_result.get('backtest_result', {})
                
                in_sample = backtest_result.get('in_sample_metrics', {})
                out_sample = backtest_result.get('out_of_sample_metrics', {})
                
                # Metrics to compare
                metrics = ['sharpe_ratio', 'sortino_ratio', 'annual_return', 'win_rate', 'max_drawdown']
                in_sample_values = [in_sample.get(m, 0) for m in metrics]
                out_sample_values = [out_sample.get(m, 0) for m in metrics]
                
                # Create DataFrame for plotting
                df = pd.DataFrame({
                    'Metric': metrics,
                    'In-Sample': in_sample_values,
                    'Out-of-Sample': out_sample_values
                })
                
                # Reshape for plotting
                df_plot = df.melt(id_vars=['Metric'], var_name='Dataset', value_name='Value')
                
                plt.figure(figsize=(12, 8))
                sns.barplot(data=df_plot, x='Metric', y='Value', hue='Dataset')
                plt.title('Best Strategy Performance: In-Sample vs Out-of-Sample')
                plt.ylabel('Value')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(viz_dir / f"best_strategy_metrics_{timestamp}.png")
                plt.close()
            
            logger.info(f"Generated visualizations in {viz_dir}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    def _run_meta_review(self, generation_summary: Dict[str, Any]) -> None:
        """
        Run meta-review of the evolution process to provide feedback.
        
        Args:
            generation_summary: Summary of the current generation
        """
        try:
            # Run meta-review
            meta_result = self.meta_review_agent.execute(
                action="review_generation",
                generation_summary=generation_summary,
                generation_history=self.generation_history,
                current_best_strategy=self.best_strategy
            )
            
            # Log insights
            if 'insights' in meta_result:
                for insight in meta_result['insights']:
                    logger.info(f"Meta-review insight: {insight}")
            
            # Apply recommendations if provided
            if 'recommendations' in meta_result:
                for rec in meta_result['recommendations']:
                    if 'parameter' in rec and 'adjustment' in rec:
                        param = rec['parameter']
                        adj = rec['adjustment']
                        
                        if param in self.config.get('parameter_ranges', {}):
                            current_range = self.config['parameter_ranges'][param]
                            
                            if adj == 'expand':
                                # Expand the range by 20%
                                min_val, max_val = current_range
                                range_size = max_val - min_val
                                new_min = min_val - range_size * 0.1
                                new_max = max_val + range_size * 0.1
                                
                                self.config['parameter_ranges'][param] = (new_min, new_max)
                                logger.info(f"Expanded parameter range for {param}: {current_range} -> {self.config['parameter_ranges'][param]}")
                                
                            elif adj == 'shift_up':
                                # Shift the range up by 10%
                                min_val, max_val = current_range
                                range_size = max_val - min_val
                                new_min = min_val + range_size * 0.1
                                new_max = max_val + range_size * 0.1
                                
                                self.config['parameter_ranges'][param] = (new_min, new_max)
                                logger.info(f"Shifted parameter range up for {param}: {current_range} -> {self.config['parameter_ranges'][param]}")
                                
                            elif adj == 'shift_down':
                                # Shift the range down by 10%
                                min_val, max_val = current_range
                                range_size = max_val - min_val
                                new_min = min_val - range_size * 0.1
                                new_max = max_val - range_size * 0.1
                                
                                self.config['parameter_ranges'][param] = (new_min, new_max)
                                logger.info(f"Shifted parameter range down for {param}: {current_range} -> {self.config['parameter_ranges'][param]}")
            
        except Exception as e:
            logger.error(f"Error running meta-review: {str(e)}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load a checkpoint to resume evolution.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                
            # Restore state
            self.current_generation = checkpoint_data.get('generation', 0)
            self.population = checkpoint_data.get('population', [])
            self.fitness_scores = checkpoint_data.get('fitness_scores', [])
            self.best_strategy = checkpoint_data.get('best_strategy', None)
            self.best_fitness = checkpoint_data.get('best_fitness', float('-inf'))
            self.generation_history = checkpoint_data.get('generation_history', [])
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            logger.info(f"Resuming from generation {self.current_generation} with {len(self.population)} strategies")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return False
    
    def run_autopilot(self, duration_hours: float = 24.0, 
                    checkpoint_interval_minutes: float = 30.0) -> Dict[str, Any]:
        """
        Run the evolution process in autopilot mode for a specified duration.
        
        Args:
            duration_hours: Duration to run in hours
            checkpoint_interval_minutes: Interval between checkpoints in minutes
            
        Returns:
            Dictionary with autopilot results
        """
        logger.info(f"Starting autopilot mode for {duration_hours} hours")
        
        # Convert duration to seconds
        duration_seconds = duration_hours * 3600
        checkpoint_interval_seconds = checkpoint_interval_minutes * 60
        
        # Record start time
        start_time = time.time()
        last_checkpoint_time = start_time
        
        # Results tracking
        autopilot_results = {
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "planned_duration_hours": duration_hours,
            "checkpoint_interval_minutes": checkpoint_interval_minutes,
            "checkpoints": [],
            "iterations": 0,
            "best_strategy": None,
            "best_fitness": float('-inf')
        }
        
        # Continue until time is up
        while (time.time() - start_time) < duration_seconds:
            # Check elapsed time
            elapsed_hours = (time.time() - start_time) / 3600
            remaining_hours = duration_hours - elapsed_hours
            
            logger.info(f"Autopilot progress: {elapsed_hours:.2f}h elapsed, {remaining_hours:.2f}h remaining")
            
            # Run one generation of evolution
            max_generations = self.current_generation + 1
            evolution_result = self.run_evolution(max_generations)
            
            # Update iterations count
            autopilot_results["iterations"] += 1
            
            # Check if checkpoint interval has passed
            if (time.time() - last_checkpoint_time) >= checkpoint_interval_seconds:
                # Create checkpoint
                checkpoint_path = self._save_checkpoint()
                
                # Record checkpoint
                autopilot_results["checkpoints"].append({
                    "timestamp": datetime.now().isoformat(),
                    "path": checkpoint_path,
                    "generation": self.current_generation,
                    "best_fitness": self.best_fitness
                })
                
                last_checkpoint_time = time.time()
            
            # Update best strategy overall
            if self.best_fitness > autopilot_results["best_fitness"]:
                autopilot_results["best_strategy"] = self.best_strategy
                autopilot_results["best_fitness"] = self.best_fitness
                
                logger.info(f"New best strategy found during autopilot: fitness={self.best_fitness:.4f}")
            
            # Check for convergence
            if self.convergence_counter >= self.convergence_count:
                logger.info("Evolution converged, restarting with new population")
                
                # Save the best strategy
                best_strategy = self.best_strategy
                
                # Generate new population
                self.population = self.generate_initial_population()
                
                # Add the best strategy to the new population (elitism across restarts)
                if best_strategy:
                    self.population[0] = best_strategy
                
                # Reset convergence counter
                self.convergence_counter = 0
        
        # Record end time
        end_time = time.time()
        actual_duration_hours = (end_time - start_time) / 3600
        
        # Update results
        autopilot_results["end_time"] = datetime.fromtimestamp(end_time).isoformat()
        autopilot_results["actual_duration_hours"] = actual_duration_hours
        
        # Save final results
        final_results_path = self._save_results()
        autopilot_results["final_results_path"] = final_results_path
        
        logger.info(f"Autopilot completed after {actual_duration_hours:.2f} hours and {autopilot_results['iterations']} iterations")
        
        return autopilot_results

def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run automated strategy evolution')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--autopilot', action='store_true', help='Run in autopilot mode')
    parser.add_argument('--duration', type=float, default=24.0, help='Duration in hours for autopilot mode')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations to evolve')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume from')
    
    args = parser.parse_args()
    
    try:
        # Initialize manager
        manager = StrategyEvolutionManager(args.config)
        
        # Load checkpoint if specified
        if args.checkpoint:
            success = manager.load_checkpoint(args.checkpoint)
            if not success:
                logger.error(f"Failed to load checkpoint from {args.checkpoint}, starting fresh")
        
        # Run evolution
        if args.autopilot:
            result = manager.run_autopilot(duration_hours=args.duration)
            logger.info(f"Autopilot completed with {result['iterations']} iterations")
        else:
            result = manager.run_evolution(max_generations=args.generations)
            logger.info(f"Evolution completed after {result['generations']} generations")
            
        # Print summary
        print("\nEvolution Results Summary")
        print("=" * 50)
        print(f"Total generations: {result.get('generations', 0)}")
        print(f"Best fitness: {result.get('best_fitness', 0):.4f}")
        if 'best_strategy' in result and result['best_strategy']:
            print(f"Best strategy name: {result['best_strategy'].get('name', 'Unnamed')}")
            print(f"Best strategy ID: {result['best_strategy'].get('id', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Error running strategy evolution: {str(e)}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())