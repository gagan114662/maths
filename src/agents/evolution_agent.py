"""
Evolution Agent for refining and improving trading strategies.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import random
import numpy as np

from .base_agent import BaseAgent
from ..utils.config import load_config

logger = logging.getLogger(__name__)

class EvolutionAgent(BaseAgent):
    """
    Agent for evolving trading strategies.
    
    Attributes:
        name: Agent identifier
        config: Configuration dictionary
        evolution_history: History of strategy evolution
        population: Current strategy population
    """
    
    def __init__(
        self,
        name: str = "evolution_agent",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize evolution agent."""
        from .base_agent import AgentType
        super().__init__(
            name=name,
            agent_type=AgentType.EVOLUTION,
            config=config,
            **kwargs
        )
        
        # Load evolution parameters
        self.evolution_params = self.config.get('evolution_params', {
            'population_size': 50,
            'elite_size': 5,
            'mutation_rate': 0.2,
            'crossover_rate': 0.7,
            'tournament_size': 5,
            'generations': 10
        })
        
        # Initialize population and history
        self.population = []
        self.evolution_history = []
        
        # Initialize metrics
        self.metrics.update({
            'generations_evolved': 0,
            'mutations_performed': 0,
            'crossovers_performed': 0,
            'improvement_rate': 0.0
        })
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process evolution request.
        
        Args:
            data: Request data including:
                - strategies: Strategies to evolve
                - market_data: Market data for evaluation
                - parameters: Evolution parameters
                
        Returns:
            Evolution results
        """
        try:
            # Validate request
            if not self._validate_request(data):
                raise ValueError("Invalid evolution request")
                
            # Initialize population
            self.population = data['strategies']
            
            # Run evolution
            evolved_population = await self._evolve_population(
                market_data=data['market_data'],
                generations=data.get('generations', self.evolution_params['generations'])
            )
            
            # Analyze results
            analysis = self._analyze_evolution(evolved_population)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analysis)
            
            # Store results
            evolution_id = self._store_evolution(
                evolved_population,
                analysis,
                recommendations,
                data
            )
            
            # Update metrics
            self._update_evolution_metrics(analysis)
            
            return {
                'status': 'success',
                'evolution_id': evolution_id,
                'evolved_strategies': evolved_population,
                'analysis': analysis,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Evolution failed: {str(e)}")
            self._log_error(e)
            raise
            
    async def _evolve_population(
        self,
        market_data: Dict[str, Any],
        generations: int
    ) -> List[Dict[str, Any]]:
        """
        Evolve population of strategies.
        
        Args:
            market_data: Market data for evaluation
            generations: Number of generations to evolve
            
        Returns:
            Evolved population
        """
        current_population = self.population
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = await self._evaluate_population(
                current_population,
                market_data
            )
            
            # Select parents
            parents = self._select_parents(
                current_population,
                fitness_scores
            )
            
            # Create new population
            new_population = []
            
            # Elitism - keep best strategies
            elite = self._get_elite(current_population, fitness_scores)
            new_population.extend(elite)
            
            # Generate offspring
            while len(new_population) < len(current_population):
                # Select parents
                parent1, parent2 = self._tournament_select(parents, fitness_scores)
                
                # Crossover
                if random.random() < self.evolution_params['crossover_rate']:
                    offspring1, offspring2 = self._crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                    
                # Mutation
                if random.random() < self.evolution_params['mutation_rate']:
                    offspring1 = self._mutate(offspring1)
                if random.random() < self.evolution_params['mutation_rate']:
                    offspring2 = self._mutate(offspring2)
                    
                new_population.extend([offspring1, offspring2])
                
            # Truncate to population size
            current_population = new_population[:len(self.population)]
            
            # Track progress
            self._track_generation(generation, current_population, fitness_scores)
            
        return current_population
        
    async def _evaluate_population(
        self,
        population: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate population fitness.
        
        Args:
            population: List of strategies
            market_data: Market data for evaluation
            
        Returns:
            Dictionary of fitness scores
        """
        fitness_scores = {}
        
        for strategy in population:
            # Request backtesting
            backtest_response = await self.send_message({
                'type': 'backtest_request',
                'data': {
                    'strategy': strategy,
                    'market_data': market_data
                },
                'timestamp': datetime.now().isoformat()
            })
            
            if backtest_response['status'] == 'success':
                # Calculate fitness score
                fitness_scores[strategy['id']] = self._calculate_fitness(
                    backtest_response['data']['analysis']
                )
                
        return fitness_scores
        
    def _select_parents(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Select parents for next generation.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores
            
        Returns:
            Selected parents
        """
        # Sort by fitness
        sorted_population = sorted(
            population,
            key=lambda x: fitness_scores.get(x['id'], 0),
            reverse=True
        )
        
        # Select top performers as parents
        num_parents = int(len(population) * 0.4)
        return sorted_population[:num_parents]
        
    def _tournament_select(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: Dict[str, float]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Select parents using tournament selection.
        
        Args:
            population: Population to select from
            fitness_scores: Fitness scores
            
        Returns:
            Two selected parents
        """
        def select_one():
            tournament = random.sample(
                population,
                min(self.evolution_params['tournament_size'], len(population))
            )
            return max(tournament, key=lambda x: fitness_scores.get(x['id'], 0))
            
        return select_one(), select_one()
        
    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform crossover between parents.
        
        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            
        Returns:
            Two offspring strategies
        """
        # Create copies
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Crossover parameters
        self._crossover_parameters(offspring1, offspring2)
        
        # Crossover rules
        self._crossover_rules(offspring1, offspring2)
        
        # Generate new IDs
        offspring1['id'] = f"evolved_{datetime.now().timestamp()}_1"
        offspring2['id'] = f"evolved_{datetime.now().timestamp()}_2"
        
        return offspring1, offspring2
        
    def _mutate(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate strategy.
        
        Args:
            strategy: Strategy to mutate
            
        Returns:
            Mutated strategy
        """
        mutated = strategy.copy()
        
        # Mutate parameters
        if 'parameters' in mutated:
            for param, value in mutated['parameters'].items():
                if isinstance(value, (int, float)):
                    # Add random noise
                    mutated['parameters'][param] = value * (1 + random.uniform(-0.1, 0.1))
                    
        # Mutate rules
        if 'entry_rules' in mutated:
            mutated['entry_rules'] = self._mutate_rules(mutated['entry_rules'])
        if 'exit_rules' in mutated:
            mutated['exit_rules'] = self._mutate_rules(mutated['exit_rules'])
            
        # Mutate position sizing
        if 'position_sizing' in mutated:
            mutated['position_sizing'] = self._mutate_position_sizing(
                mutated['position_sizing']
            )
            
        return mutated
        
    def _calculate_fitness(self, analysis: Dict[str, Any]) -> float:
        """
        Calculate strategy fitness score.
        
        Args:
            analysis: Strategy performance analysis
            
        Returns:
            Fitness score
        """
        # Combine multiple metrics
        sharpe_ratio = analysis['performance']['sharpe_ratio']
        sortino_ratio = analysis['performance']['sortino_ratio']
        max_drawdown = abs(analysis['performance']['max_drawdown'])
        win_rate = analysis['trades']['win_rate']
        
        # Weight the metrics
        fitness = (
            0.3 * sharpe_ratio +
            0.3 * sortino_ratio +
            0.2 * (1 - max_drawdown) +
            0.2 * win_rate
        )
        
        return max(0, fitness)  # Ensure non-negative
        
    def _get_elite(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Get elite strategies from population."""
        sorted_population = sorted(
            population,
            key=lambda x: fitness_scores.get(x['id'], 0),
            reverse=True
        )
        return sorted_population[:self.evolution_params['elite_size']]
        
    def _track_generation(
        self,
        generation: int,
        population: List[Dict[str, Any]],
        fitness_scores: Dict[str, float]
    ) -> None:
        """Track evolution progress."""
        best_fitness = max(fitness_scores.values()) if fitness_scores else 0
        avg_fitness = (
            sum(fitness_scores.values()) / len(fitness_scores)
            if fitness_scores else 0
        )
        
        self.evolution_history.append({
            'generation': generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'population_size': len(population),
            'timestamp': datetime.now().isoformat()
        })
        
    def _crossover_parameters(
        self,
        offspring1: Dict[str, Any],
        offspring2: Dict[str, Any]
    ) -> None:
        """Perform parameter crossover."""
        if 'parameters' in offspring1 and 'parameters' in offspring2:
            params1 = offspring1['parameters']
            params2 = offspring2['parameters']
            
            for param in set(params1.keys()) & set(params2.keys()):
                if random.random() < 0.5:
                    params1[param], params2[param] = params2[param], params1[param]
                    
    def _crossover_rules(
        self,
        offspring1: Dict[str, Any],
        offspring2: Dict[str, Any]
    ) -> None:
        """Perform trading rules crossover."""
        for rule_type in ['entry_rules', 'exit_rules']:
            if rule_type in offspring1 and rule_type in offspring2:
                rules1 = offspring1[rule_type]
                rules2 = offspring2[rule_type]
                
                if isinstance(rules1, list) and isinstance(rules2, list):
                    # Crossover point
                    point = random.randint(1, min(len(rules1), len(rules2)))
                    
                    # Swap rules
                    offspring1[rule_type] = rules1[:point] + rules2[point:]
                    offspring2[rule_type] = rules2[:point] + rules1[point:]
                    
    def _mutate_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mutate trading rules."""
        mutated = []
        
        for rule in rules:
            if random.random() < 0.2:  # 20% chance to modify rule
                rule = rule.copy()
                
                # Modify parameters
                if 'parameters' in rule:
                    for param, value in rule['parameters'].items():
                        if isinstance(value, (int, float)):
                            rule['parameters'][param] = value * (
                                1 + random.uniform(-0.1, 0.1)
                            )
                            
            mutated.append(rule)
            
        return mutated
        
    def _mutate_position_sizing(
        self,
        position_sizing: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mutate position sizing rules."""
        mutated = position_sizing.copy()
        
        # Modify size parameters
        if 'base_size' in mutated:
            mutated['base_size'] *= (1 + random.uniform(-0.1, 0.1))
            
        # Modify scaling factors
        if 'scaling_factors' in mutated:
            for factor, value in mutated['scaling_factors'].items():
                if isinstance(value, (int, float)):
                    mutated['scaling_factors'][factor] = value * (
                        1 + random.uniform(-0.1, 0.1)
                    )
                    
        return mutated
        
    def _analyze_evolution(
        self,
        evolved_population: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze evolution results.
        
        Args:
            evolved_population: Evolved strategies
            
        Returns:
            Evolution analysis
        """
        history = self.evolution_history
        
        return {
            'generations': len(history),
            'initial_fitness': {
                'best': history[0]['best_fitness'],
                'average': history[0]['avg_fitness']
            },
            'final_fitness': {
                'best': history[-1]['best_fitness'],
                'average': history[-1]['avg_fitness']
            },
            'improvement': {
                'best': history[-1]['best_fitness'] - history[0]['best_fitness'],
                'average': history[-1]['avg_fitness'] - history[0]['avg_fitness']
            },
            'convergence': self._analyze_convergence(history),
            'diversity': self._analyze_diversity(evolved_population)
        }
        
    def _analyze_convergence(
        self,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze evolution convergence."""
        fitness_history = [gen['best_fitness'] for gen in history]
        
        return {
            'converged': self._check_convergence(fitness_history),
            'generations_to_converge': self._generations_to_converge(fitness_history)
        }
        
    def _analyze_diversity(
        self,
        population: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze population diversity."""
        # Implement diversity analysis
        return {
            'parameter_diversity': 0.0,
            'rule_diversity': 0.0,
            'strategy_diversity': 0.0
        }
        
    def _check_convergence(self, fitness_history: List[float]) -> bool:
        """Check if evolution has converged."""
        if len(fitness_history) < 5:
            return False
            
        # Check if fitness improvement is minimal
        recent_improvements = [
            abs(fitness_history[i] - fitness_history[i-1])
            for i in range(-5, -1)
        ]
        
        return max(recent_improvements) < 0.01
        
    def _generations_to_converge(self, fitness_history: List[float]) -> int:
        """Calculate generations until convergence."""
        for i in range(5, len(fitness_history)):
            recent_improvements = [
                abs(fitness_history[j] - fitness_history[j-1])
                for j in range(i-4, i+1)
            ]
            if max(recent_improvements) < 0.01:
                return i
                
        return len(fitness_history)  # Did not converge