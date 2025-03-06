#!/usr/bin/env python3
"""
Multi-Objective Optimization Framework for Trading Strategies

This module implements a sophisticated multi-objective optimization framework that 
goes beyond single-metric optimization for trading strategies. It allows strategies
to be optimized across multiple competing objectives (return, risk, drawdown, etc.)
and implements stress testing to ensure robustness across various market conditions.

Key features:
- Pareto optimization for balancing multiple trading objectives
- Hypervolume indicator for measuring multi-objective performance
- Stress testing across different market regimes and scenarios
- Non-dominated sorting and crowding distance for maintaining diversity
- Adaptive constraint handling for realistic trading constraints
- Preference articulation for strategy selection based on user preferences
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import copy
import random
import itertools
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import norm, percentileofscore
import warnings

# Configure logging
logger = logging.getLogger(__name__)

class TradingObjective:
    """
    Class representing a trading strategy objective.
    
    This class defines a single objective that can be maximized or minimized
    during the optimization process.
    """
    
    def __init__(self, 
                name: str, 
                function: Callable[[pd.DataFrame], float], 
                direction: str = 'maximize',
                weight: float = 1.0,
                constraint_value: Optional[float] = None,
                constraint_type: Optional[str] = None):
        """
        Initialize a trading objective.
        
        Args:
            name: Name of the objective (e.g., 'return', 'sharpe_ratio', 'max_drawdown')
            function: Function that takes strategy results and returns a scalar value
            direction: Direction of optimization ('maximize' or 'minimize')
            weight: Weight of this objective when combining into a scalar fitness
            constraint_value: Optional constraint value (if this is a hard constraint)
            constraint_type: Type of constraint ('>=', '<=', '>', '<', '==', None)
        """
        self.name = name
        self.function = function
        
        if direction not in ['maximize', 'minimize']:
            raise ValueError("Direction must be 'maximize' or 'minimize'")
        self.direction = direction
        
        self.weight = weight
        self.constraint_value = constraint_value
        
        if constraint_type is not None and constraint_type not in ['>=', '<=', '>', '<', '==']:
            raise ValueError("Constraint type must be one of ['>=', '<=', '>', '<', '==', None]")
        self.constraint_type = constraint_type
        
        logger.info(f"Created objective: {name} ({direction}) with weight {weight}")
        if constraint_value is not None:
            logger.info(f"  Constraint: {constraint_type} {constraint_value}")
    
    def evaluate(self, strategy_results: pd.DataFrame) -> float:
        """
        Evaluate the objective for given strategy results.
        
        Args:
            strategy_results: DataFrame with strategy results
            
        Returns:
            Objective value
        """
        return self.function(strategy_results)
    
    def normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize objective value to [0, 1] range.
        
        Args:
            value: Raw objective value
            min_val: Minimum observed value
            max_val: Maximum observed value
            
        Returns:
            Normalized value in [0, 1] range
        """
        if np.isclose(min_val, max_val):
            return 0.5  # Avoid division by zero
            
        normalized = (value - min_val) / (max_val - min_val)
        
        # If minimizing, invert so higher is always better
        if self.direction == 'minimize':
            normalized = 1.0 - normalized
            
        return normalized
    
    def check_constraint(self, value: float) -> bool:
        """
        Check if the objective value satisfies its constraint.
        
        Args:
            value: Objective value
            
        Returns:
            True if constraint is satisfied or no constraint, False otherwise
        """
        if self.constraint_value is None or self.constraint_type is None:
            return True
            
        if self.constraint_type == '>=':
            return value >= self.constraint_value
        elif self.constraint_type == '<=':
            return value <= self.constraint_value
        elif self.constraint_type == '>':
            return value > self.constraint_value
        elif self.constraint_type == '<':
            return value < self.constraint_value
        elif self.constraint_type == '==':
            return np.isclose(value, self.constraint_value)
            
        return True  # Default case


class StressTest:
    """
    Class representing a stress test for trading strategies.
    
    This class defines a specific market scenario or condition used to
    test the robustness of trading strategies.
    """
    
    def __init__(self, 
                name: str, 
                condition_function: Callable[[pd.DataFrame], pd.DataFrame],
                description: str = "",
                weight: float = 1.0):
        """
        Initialize a stress test.
        
        Args:
            name: Name of the stress test
            condition_function: Function that takes market data and returns filtered/modified data
            description: Description of the stress test scenario
            weight: Weight of this stress test in the overall evaluation
        """
        self.name = name
        self.condition_function = condition_function
        self.description = description
        self.weight = weight
        
        logger.info(f"Created stress test: {name} with weight {weight}")
        if description:
            logger.info(f"  Description: {description}")
    
    def apply(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the stress test condition to market data.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Filtered/modified DataFrame representing the stress scenario
        """
        return self.condition_function(market_data)


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for trading strategies.
    
    This class implements various multi-objective optimization algorithms for
    finding Pareto-optimal trading strategies.
    """
    
    def __init__(self, 
                objectives: List[TradingObjective],
                stress_tests: Optional[List[StressTest]] = None,
                population_size: int = 50,
                max_generations: int = 30,
                crossover_rate: float = 0.8,
                mutation_rate: float = 0.2,
                tournament_size: int = 3,
                elitism_ratio: float = 0.1,
                seed: Optional[int] = None):
        """
        Initialize the multi-objective optimizer.
        
        Args:
            objectives: List of trading objectives
            stress_tests: List of stress tests (optional)
            population_size: Size of population in genetic algorithm
            max_generations: Maximum number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament selection
            elitism_ratio: Ratio of population preserved via elitism
            seed: Random seed for reproducibility
        """
        self.objectives = objectives
        self.stress_tests = stress_tests or []
        
        # Genetic algorithm parameters
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_ratio = elitism_ratio
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize storage for results
        self.population = []
        self.objective_values = []
        self.pareto_front = []
        self.pareto_history = []
        self.objective_ranges = {}
        
        logger.info(f"Initialized multi-objective optimizer with {len(objectives)} objectives "
                  f"and {len(stress_tests)} stress tests")
    
    def _evaluate_strategy(self, 
                          strategy: Any, 
                          evaluate_func: Callable[[Any, pd.DataFrame], pd.DataFrame], 
                          market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate a strategy against all objectives and stress tests.
        
        Args:
            strategy: Trading strategy instance
            evaluate_func: Function to evaluate strategy on market data
            market_data: Market data for evaluation
            
        Returns:
            Dictionary mapping objective names to their values
        """
        # Normal evaluation without stress testing
        results = evaluate_func(strategy, market_data)
        objective_values = {}
        
        # Calculate objective values
        for obj in self.objectives:
            value = obj.evaluate(results)
            objective_values[obj.name] = value
        
        # Apply stress tests if available
        if self.stress_tests:
            stress_results = {}
            for test in self.stress_tests:
                # Filter data according to stress condition
                stress_data = test.apply(market_data)
                
                # Only evaluate if we have enough data
                if len(stress_data) > 20:  # Arbitrary minimum number of data points
                    # Evaluate strategy on stress test data
                    test_results = evaluate_func(strategy, stress_data)
                    
                    # Calculate objective values for stress test
                    for obj in self.objectives:
                        value = obj.evaluate(test_results)
                        stress_results[f"{test.name}_{obj.name}"] = value
                else:
                    logger.warning(f"Stress test {test.name} has insufficient data: {len(stress_data)} points")
            
            # Combine with regular objective values
            objective_values.update(stress_results)
        
        return objective_values
    
    def _check_constraints(self, objective_values: Dict[str, float]) -> bool:
        """
        Check if strategy satisfies all constraints.
        
        Args:
            objective_values: Dictionary mapping objective names to their values
            
        Returns:
            True if all constraints are satisfied, False otherwise
        """
        for obj in self.objectives:
            if obj.constraint_type is not None and obj.constraint_value is not None:
                if not obj.check_constraint(objective_values[obj.name]):
                    return False
        return True
    
    def _calculate_dominance(self, 
                            objective_values_list: List[Dict[str, float]]) -> List[List[int]]:
        """
        Calculate dominance relationships between strategies.
        
        Args:
            objective_values_list: List of dictionaries with objective values
            
        Returns:
            List of indices of strategies dominated by each strategy
        """
        dominates = [[] for _ in range(len(objective_values_list))]
        
        for i in range(len(objective_values_list)):
            for j in range(len(objective_values_list)):
                if i == j:
                    continue
                    
                i_dominates_j = True
                j_dominates_i = True
                
                for obj in self.objectives:
                    i_val = objective_values_list[i][obj.name]
                    j_val = objective_values_list[j][obj.name]
                    
                    # For maximize objectives, higher is better
                    if obj.direction == 'maximize':
                        if i_val < j_val:
                            i_dominates_j = False
                        if i_val > j_val:
                            j_dominates_i = False
                    # For minimize objectives, lower is better
                    else:
                        if i_val > j_val:
                            i_dominates_j = False
                        if i_val < j_val:
                            j_dominates_i = False
                
                if i_dominates_j:
                    dominates[i].append(j)
                    
        return dominates
    
    def _fast_non_dominated_sort(self, 
                                objective_values_list: List[Dict[str, float]]) -> List[List[int]]:
        """
        Perform fast non-dominated sorting.
        
        Args:
            objective_values_list: List of dictionaries with objective values
            
        Returns:
            List of fronts, where each front is a list of indices
        """
        dominates = self._calculate_dominance(objective_values_list)
        dominated_count = [0] * len(objective_values_list)
        fronts = [[]]
        
        # Count how many solutions dominate each solution
        for i in range(len(objective_values_list)):
            for j in dominates[i]:
                dominated_count[j] += 1
                
            # If no solution dominates solution i, it's in the first front
            if dominated_count[i] == 0:
                fronts[0].append(i)
        
        # Find subsequent fronts
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            
            for i in fronts[front_idx]:
                for j in dominates[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            
            front_idx += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts
    
    def _calculate_crowding_distance(self, 
                                   front: List[int], 
                                   objective_values_list: List[Dict[str, float]]) -> List[float]:
        """
        Calculate crowding distance for each solution in a front.
        
        Args:
            front: List of indices forming a Pareto front
            objective_values_list: List of dictionaries with objective values
            
        Returns:
            List of crowding distances for each solution in the front
        """
        if len(front) <= 2:
            return [float('inf')] * len(front)
            
        distances = [0.0] * len(front)
        
        for obj in self.objectives:
            # Extract values for this objective
            values = [(i, objective_values_list[i][obj.name]) for i in front]
            
            # Sort by objective value
            values.sort(key=lambda x: x[1])
            
            # Set boundary distances to infinity
            idx_first, val_first = values[0]
            idx_last, val_last = values[-1]
            distances[0] = float('inf')
            distances[-1] = float('inf')
            
            # Calculate normalization factor
            norm = val_last - val_first
            if norm == 0:
                continue
                
            # Calculate distances for intermediate solutions
            for i in range(1, len(front) - 1):
                idx_prev, val_prev = values[i-1]
                idx_next, val_next = values[i+1]
                idx_curr, val_curr = values[i]
                
                # Add normalized distance contribution
                front_idx = front.index(idx_curr)
                distances[front_idx] += (val_next - val_prev) / norm
                
        return distances
    
    def _calculate_hypervolume(self, 
                             front: List[Dict[str, float]], 
                             reference_point: Dict[str, float]) -> float:
        """
        Calculate hypervolume indicator for a Pareto front.
        
        Args:
            front: List of dictionaries with objective values
            reference_point: Dictionary with reference point for each objective
            
        Returns:
            Hypervolume value
        """
        if not front:
            return 0.0
            
        # Extract objective values as arrays
        points = []
        for solution in front:
            point = []
            for obj in self.objectives:
                value = solution[obj.name]
                # Convert to standard form (higher is better)
                if obj.direction == 'minimize':
                    value = reference_point[obj.name] - value
                else:
                    value = value - reference_point[obj.name]
                    
                # Ensure non-negative
                value = max(0.0, value)
                point.append(value)
            points.append(point)
            
        # Use Monte Carlo estimation for hypervolume when dimensionality is high
        if len(self.objectives) > 3:
            return self._monte_carlo_hypervolume(points)
        else:
            return self._exact_hypervolume(points)
    
    def _exact_hypervolume(self, points: List[List[float]]) -> float:
        """
        Calculate exact hypervolume for low dimensional cases.
        
        Args:
            points: List of points forming the Pareto front
            
        Returns:
            Exact hypervolume
        """
        # For 1D case, return the maximum value
        if len(points[0]) == 1:
            return max(p[0] for p in points)
            
        # For 2D case, use the shoelace formula
        if len(points[0]) == 2:
            # Sort points by first objective (ascending)
            sorted_points = sorted(points, key=lambda p: p[0])
            
            # Calculate area using trapezoid rule
            area = 0.0
            for i in range(len(sorted_points) - 1):
                p1 = sorted_points[i]
                p2 = sorted_points[i + 1]
                area += (p2[0] - p1[0]) * p1[1]
                
            # Add final rectangle
            area += sorted_points[-1][0] * sorted_points[-1][1]
            
            return area
            
        # For 3D case, use a divide-and-conquer approach
        # This is a simplified implementation and may not be optimal
        if len(points[0]) == 3:
            # Use Monte Carlo for 3D as well for simplicity
            return self._monte_carlo_hypervolume(points)
        
        # Fallback for higher dimensions
        return self._monte_carlo_hypervolume(points)
    
    def _monte_carlo_hypervolume(self, 
                               points: List[List[float]], 
                               samples: int = 10000) -> float:
        """
        Estimate hypervolume using Monte Carlo sampling.
        
        Args:
            points: List of points forming the Pareto front
            samples: Number of Monte Carlo samples
            
        Returns:
            Estimated hypervolume
        """
        if not points:
            return 0.0
            
        # Determine bounds
        n_obj = len(points[0])
        upper_bounds = [max(p[i] for p in points) for i in range(n_obj)]
        
        # Generate random samples within bounds
        random_points = np.random.rand(samples, n_obj) * np.array(upper_bounds)
        
        # Count points dominated by Pareto front
        dominated_count = 0
        for sample in random_points:
            # Check if sample is dominated by any Pareto point
            for pareto_point in points:
                if all(sample[i] <= pareto_point[i] for i in range(n_obj)):
                    dominated_count += 1
                    break
                    
        # Calculate hypervolume
        volume = np.prod(upper_bounds)
        hypervolume = volume * dominated_count / samples
        
        return hypervolume
    
    def optimize(self, 
                strategy_generator: Callable[[], Any],
                strategy_mutator: Callable[[Any], Any],
                strategy_crossover: Callable[[Any, Any], Tuple[Any, Any]],
                strategy_evaluator: Callable[[Any, pd.DataFrame], pd.DataFrame],
                market_data: pd.DataFrame,
                callback: Optional[Callable[[int, List[Any], List[Dict[str, float]], List[int]], None]] = None) -> Tuple[List[Any], List[Dict[str, float]]]:
        """
        Perform multi-objective optimization.
        
        Args:
            strategy_generator: Function that generates a random strategy
            strategy_mutator: Function that mutates a strategy
            strategy_crossover: Function that creates offspring from two parent strategies
            strategy_evaluator: Function that evaluates a strategy on market data
            market_data: Market data for evaluation
            callback: Optional callback function called after each generation
            
        Returns:
            Tuple of (Pareto-optimal strategies, their objective values)
        """
        logger.info("Starting multi-objective optimization")
        
        # Initialize population
        self.population = [strategy_generator() for _ in range(self.population_size)]
        logger.info(f"Generated initial population of {self.population_size} strategies")
        
        # Evaluate initial population
        self.objective_values = []
        for i, strategy in enumerate(self.population):
            values = self._evaluate_strategy(strategy, strategy_evaluator, market_data)
            self.objective_values.append(values)
            if i % 10 == 0:
                logger.info(f"Evaluated {i}/{self.population_size} strategies")
                
        logger.info("Completed initial population evaluation")
                
        # Main optimization loop
        for generation in range(self.max_generations):
            logger.info(f"Generation {generation+1}/{self.max_generations}")
            
            # Perform non-dominated sorting
            fronts = self._fast_non_dominated_sort(self.objective_values)
            logger.info(f"Found {len(fronts)} Pareto fronts")
            
            # Store first front as current Pareto front
            self.pareto_front = [self.population[i] for i in fronts[0]]
            pareto_values = [self.objective_values[i] for i in fronts[0]]
            self.pareto_history.append((copy.deepcopy(self.pareto_front), 
                                      copy.deepcopy(pareto_values)))
            
            # Create new population
            new_population = []
            new_objective_values = []
            
            # Add elite individuals
            elite_count = max(1, int(self.elitism_ratio * self.population_size))
            elite_added = 0
            
            for front in fronts:
                if elite_added >= elite_count:
                    break
                    
                # Calculate crowding distance for this front
                distances = self._calculate_crowding_distance(front, self.objective_values)
                
                # Sort front by crowding distance (descending)
                sorted_front = [i for _, i in sorted(zip(distances, front), key=lambda x: -x[0])]
                
                # Add strategies from this front until elite_count is reached
                for i in sorted_front:
                    if elite_added >= elite_count:
                        break
                    new_population.append(self.population[i])
                    new_objective_values.append(self.objective_values[i])
                    elite_added += 1
                    
            logger.info(f"Added {elite_added} elite strategies to new population")
            
            # Fill the rest with offspring
            while len(new_population) < self.population_size:
                # Tournament selection for parents
                parent1 = self._tournament_selection(self.population, self.objective_values)
                parent2 = self._tournament_selection(self.population, self.objective_values)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = strategy_crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                    
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring1 = strategy_mutator(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = strategy_mutator(offspring2)
                    
                # Evaluate offspring
                offspring1_values = self._evaluate_strategy(offspring1, strategy_evaluator, market_data)
                offspring2_values = self._evaluate_strategy(offspring2, strategy_evaluator, market_data)
                
                # Add to new population if constraints satisfied
                if self._check_constraints(offspring1_values) and len(new_population) < self.population_size:
                    new_population.append(offspring1)
                    new_objective_values.append(offspring1_values)
                    
                if self._check_constraints(offspring2_values) and len(new_population) < self.population_size:
                    new_population.append(offspring2)
                    new_objective_values.append(offspring2_values)
            
            # Update population
            self.population = new_population
            self.objective_values = new_objective_values
            
            # Calculate hypervolume
            if len(self.pareto_front) > 0:
                # Determine reference point
                reference_point = {}
                for obj in self.objectives:
                    all_values = [values[obj.name] for values in self.objective_values]
                    if obj.direction == 'maximize':
                        reference_point[obj.name] = min(all_values) - 0.1 * (max(all_values) - min(all_values))
                    else:
                        reference_point[obj.name] = max(all_values) + 0.1 * (max(all_values) - min(all_values))
                
                hypervolume = self._calculate_hypervolume(pareto_values, reference_point)
                logger.info(f"Current hypervolume: {hypervolume:.4f}")
            
            # Call callback if provided
            if callback:
                callback(generation, self.pareto_front, pareto_values, fronts[0])
                
        # Final non-dominated sorting
        fronts = self._fast_non_dominated_sort(self.objective_values)
        logger.info(f"Final optimization complete. Found {len(fronts[0])} Pareto-optimal strategies")
        
        # Return Pareto front
        self.pareto_front = [self.population[i] for i in fronts[0]]
        pareto_values = [self.objective_values[i] for i in fronts[0]]
        
        return self.pareto_front, pareto_values
    
    def _tournament_selection(self, 
                            population: List[Any], 
                            objective_values: List[Dict[str, float]]) -> Any:
        """
        Perform tournament selection.
        
        Args:
            population: List of strategies
            objective_values: List of objective values for each strategy
            
        Returns:
            Selected strategy
        """
        indices = random.sample(range(len(population)), self.tournament_size)
        tournament = [population[i] for i in indices]
        tournament_values = [objective_values[i] for i in indices]
        
        # Calculate dominance relationships
        dominates = self._calculate_dominance(tournament_values)
        
        # Count how many solutions are dominated by each solution
        dominated_count = [len(dominates[i]) for i in range(len(tournament))]
        
        # Select the solution that dominates the most others
        best_idx = np.argmax(dominated_count)
        
        return tournament[best_idx]
    
    def select_preferred_solution(self, 
                                strategies: List[Any], 
                                objective_values: List[Dict[str, float]],
                                preferences: Dict[str, float] = None) -> Tuple[Any, Dict[str, float]]:
        """
        Select a preferred solution from the Pareto front.
        
        Args:
            strategies: List of Pareto-optimal strategies
            objective_values: Objective values for each strategy
            preferences: Optional dictionary of user preferences for each objective
            
        Returns:
            Tuple of (selected strategy, its objective values)
        """
        if not strategies:
            logger.warning("No strategies provided for selection")
            return None, {}
            
        if len(strategies) == 1:
            logger.info("Only one strategy in the Pareto front")
            return strategies[0], objective_values[0]
            
        # If no preferences, use equal weights
        if preferences is None:
            preferences = {obj.name: obj.weight for obj in self.objectives}
            
        # Normalize objective values
        normalized_values = []
        for values in objective_values:
            normalized = {}
            for obj in self.objectives:
                all_values = [values[obj.name] for values in objective_values]
                min_val, max_val = min(all_values), max(all_values)
                normalized[obj.name] = obj.normalize_value(values[obj.name], min_val, max_val)
            normalized_values.append(normalized)
            
        # Calculate weighted sum
        weighted_sums = []
        for normalized in normalized_values:
            weighted_sum = 0.0
            for obj_name, weight in preferences.items():
                if obj_name in normalized:
                    weighted_sum += normalized[obj_name] * weight
            weighted_sums.append(weighted_sum)
            
        # Select strategy with highest weighted sum
        best_idx = np.argmax(weighted_sums)
        
        logger.info(f"Selected strategy with weighted sum: {weighted_sums[best_idx]:.4f}")
        
        return strategies[best_idx], objective_values[best_idx]
    
    def cluster_pareto_front(self, 
                            strategies: List[Any], 
                            objective_values: List[Dict[str, float]],
                            n_clusters: int = 3) -> List[List[int]]:
        """
        Cluster the Pareto front into distinct strategy groups.
        
        Args:
            strategies: List of Pareto-optimal strategies
            objective_values: Objective values for each strategy
            n_clusters: Number of clusters to create
            
        Returns:
            List of lists containing indices of strategies in each cluster
        """
        if not strategies:
            logger.warning("No strategies provided for clustering")
            return []
            
        if len(strategies) <= n_clusters:
            logger.info(f"Number of strategies ({len(strategies)}) <= number of clusters ({n_clusters})")
            return [[i] for i in range(len(strategies))]
            
        # Prepare data for clustering
        data = []
        for values in objective_values:
            point = []
            for obj in self.objectives:
                value = values[obj.name]
                if obj.direction == 'minimize':
                    value = -value  # Convert to "higher is better" form
                point.append(value)
            data.append(point)
            
        # Normalize data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data_scaled)
        
        # Group indices by cluster
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
            
        # Log cluster sizes
        for i, cluster in enumerate(clusters):
            logger.info(f"Cluster {i}: {len(cluster)} strategies")
            
        return clusters
    
    def plot_pareto_front(self, 
                         objective_values: List[Dict[str, float]],
                         highlight_indices: List[int] = None,
                         title: str = "Pareto Front",
                         figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot the Pareto front.
        
        Args:
            objective_values: Objective values for each strategy in the Pareto front
            highlight_indices: Indices of strategies to highlight
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not objective_values:
            logger.warning("No objective values provided for plotting")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', fontsize=14)
            ax.set_title(title)
            return fig
            
        # If more than 2 objectives, use parallel coordinates
        if len(self.objectives) > 2:
            return self._plot_parallel_coordinates(objective_values, highlight_indices, title, figsize)
        else:
            # Use scatter plot for 2 objectives
            return self._plot_2d_pareto(objective_values, highlight_indices, title, figsize)
    
    def _plot_2d_pareto(self, 
                       objective_values: List[Dict[str, float]],
                       highlight_indices: List[int] = None,
                       title: str = "Pareto Front",
                       figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot 2D Pareto front.
        
        Args:
            objective_values: Objective values for each strategy
            highlight_indices: Indices of strategies to highlight
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if len(self.objectives) != 2:
            logger.warning(f"Expected 2 objectives, got {len(self.objectives)}. Using first two.")
            
        obj1, obj2 = self.objectives[:2]
        
        # Extract objective values
        x_values = [values[obj1.name] for values in objective_values]
        y_values = [values[obj2.name] for values in objective_values]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine axis directions
        x_direction = 1 if obj1.direction == 'maximize' else -1
        y_direction = 1 if obj2.direction == 'maximize' else -1
        
        # Plot points
        scatter = ax.scatter(x_values, y_values, alpha=0.7, s=50, c='blue')
        
        # Highlight specific solutions if requested
        if highlight_indices:
            highlighted_x = [x_values[i] for i in highlight_indices]
            highlighted_y = [y_values[i] for i in highlight_indices]
            ax.scatter(highlighted_x, highlighted_y, color='red', s=100, 
                      edgecolor='black', alpha=0.8)
        
        # Add labels and title
        x_label = f"{obj1.name} ({'Higher is better' if obj1.direction == 'maximize' else 'Lower is better'})"
        y_label = f"{obj2.name} ({'Higher is better' if obj2.direction == 'maximize' else 'Lower is better'})"
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        
        # Add grid
        ax.grid(alpha=0.3)
        
        # Set axis directions
        if x_direction == -1:
            ax.invert_xaxis()
        if y_direction == -1:
            ax.invert_yaxis()
            
        # Add colorbar
        # cb = plt.colorbar(scatter, ax=ax)
        # cb.set_label('Strategy Index')
        
        plt.tight_layout()
        
        return fig
    
    def _plot_parallel_coordinates(self, 
                                 objective_values: List[Dict[str, float]],
                                 highlight_indices: List[int] = None,
                                 title: str = "Pareto Front",
                                 figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot parallel coordinates for multi-dimensional Pareto front.
        
        Args:
            objective_values: Objective values for each strategy
            highlight_indices: Indices of strategies to highlight
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for parallel coordinates
        data = {}
        for i, obj in enumerate(self.objectives):
            values = [vals[obj.name] for vals in objective_values]
            # If minimizing, invert so higher is always better
            if obj.direction == 'minimize':
                min_val, max_val = min(values), max(values)
                values = [max_val + min_val - val for val in values]
            data[obj.name] = values
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Plot parallel coordinates
        pd.plotting.parallel_coordinates(df, class_column=None, ax=ax, alpha=0.5)
        
        # Highlight specific solutions if requested
        if highlight_indices:
            highlighted_data = df.iloc[highlight_indices]
            pd.plotting.parallel_coordinates(highlighted_data, class_column=None, 
                                           ax=ax, color='red', alpha=0.8, linewidth=2)
        
        # Add title
        ax.set_title(title)
        
        # Clean up x-axis labels
        ax.set_xticklabels([obj.name for obj in self.objectives], rotation=30, fontsize=10)
        
        # Add direction indicators to y-axis labels
        for i, obj in enumerate(self.objectives):
            direction = "↑" if obj.direction == 'maximize' else "↓"
            ax.text(i, ax.get_ylim()[1], direction, ha='center', va='bottom', fontsize=14)
            
        plt.tight_layout()
        
        return fig
    
    def run_stress_tests(self, 
                       strategy: Any, 
                       evaluate_func: Callable[[Any, pd.DataFrame], pd.DataFrame], 
                       market_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Run all stress tests on a strategy.
        
        Args:
            strategy: Strategy to test
            evaluate_func: Function to evaluate strategy on market data
            market_data: Market data
            
        Returns:
            Dictionary mapping test names to dictionaries of objective values
        """
        stress_results = {}
        
        # Add baseline performance (no stress test)
        base_results = evaluate_func(strategy, market_data)
        baseline = {}
        for obj in self.objectives:
            baseline[obj.name] = obj.evaluate(base_results)
        stress_results["baseline"] = baseline
        
        # Run each stress test
        for test in self.stress_tests:
            logger.info(f"Running stress test: {test.name}")
            
            # Filter data according to stress condition
            stress_data = test.apply(market_data)
            
            # Only evaluate if we have enough data
            if len(stress_data) > 20:  # Arbitrary minimum number of data points
                # Evaluate strategy on stress test data
                test_results = evaluate_func(strategy, stress_data)
                
                # Calculate objective values for stress test
                test_objectives = {}
                for obj in self.objectives:
                    test_objectives[obj.name] = obj.evaluate(test_results)
                
                stress_results[test.name] = test_objectives
                
                # Log results
                logger.info(f"  Stress test {test.name} results:")
                for obj_name, value in test_objectives.items():
                    baseline_value = baseline[obj_name]
                    change = value - baseline_value
                    pct_change = change / abs(baseline_value) * 100 if baseline_value != 0 else float('inf')
                    
                    direction = self.objectives[next(i for i, o in enumerate(self.objectives) if o.name == obj_name)].direction
                    is_better = (change > 0 and direction == 'maximize') or (change < 0 and direction == 'minimize')
                    
                    logger.info(f"    {obj_name}: {value:.4f} (Baseline: {baseline_value:.4f}, "
                              f"Change: {pct_change:+.2f}% {'better' if is_better else 'worse'})")
            else:
                logger.warning(f"Stress test {test.name} has insufficient data: {len(stress_data)} points")
        
        return stress_results
    
    def plot_stress_test_results(self, 
                               stress_results: Dict[str, Dict[str, float]],
                               figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot stress test results.
        
        Args:
            stress_results: Results from run_stress_tests
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not stress_results:
            logger.warning("No stress test results to plot")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', fontsize=14)
            ax.set_title("Stress Test Results")
            return fig
            
        # Create figure
        fig, axes = plt.subplots(len(self.objectives), 1, figsize=figsize, sharex=True)
        if len(self.objectives) == 1:
            axes = [axes]
        
        # Get test names and baseline
        test_names = list(stress_results.keys())
        baseline = stress_results["baseline"]
        
        # Plot each objective
        for i, obj in enumerate(self.objectives):
            ax = axes[i]
            
            # Extract values for this objective
            values = [stress_results[test][obj.name] for test in test_names]
            baseline_value = baseline[obj.name]
            
            # Calculate changes from baseline
            pct_changes = [(val - baseline_value) / abs(baseline_value) * 100 if baseline_value != 0 else 0 
                           for val in values]
            
            # Create bar colors based on whether change is good or bad
            colors = []
            for j, change in enumerate(pct_changes):
                if obj.direction == 'maximize':
                    colors.append('green' if change >= 0 else 'red')
                else:
                    colors.append('green' if change <= 0 else 'red')
            
            # Create bar plot
            bars = ax.bar(test_names, pct_changes, color=colors, alpha=0.7)
            
            # Add baseline reference line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add value labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                value = values[j]
                ax.text(bar.get_x() + bar.get_width()/2., 
                       height + (5 if height >= 0 else -15),
                       f'{value:.4f}', ha='center', va='bottom', rotation=0, fontsize=8)
            
            # Add labels
            ax.set_ylabel(f"{obj.name} % Change\n({'↑ better' if obj.direction == 'maximize' else '↓ better'})")
            
            # Add grid
            ax.grid(axis='y', alpha=0.3)
            
        # Add common labels
        axes[-1].set_xlabel("Stress Tests")
        axes[-1].set_xticklabels(test_names, rotation=45, ha='right')
        
        # Add title
        fig.suptitle("Stress Test Results (% Change from Baseline)", fontsize=14)
        
        plt.tight_layout()
        
        return fig


# Common trading objectives
def annualized_return(results: pd.DataFrame) -> float:
    """Calculate annualized return."""
    if 'equity' not in results.columns or len(results) < 2:
        return 0.0
    
    initial = results['equity'].iloc[0]
    final = results['equity'].iloc[-1]
    days = (results.index[-1] - results.index[0]).days
    
    if days <= 0 or initial <= 0:
        return 0.0
    
    # Calculate annualized return
    years = days / 365.25
    cagr = (final / initial) ** (1 / years) - 1
    
    return cagr


def sharpe_ratio(results: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    if 'returns' not in results.columns or len(results) < 30:
        return 0.0
    
    returns = results['returns'].dropna()
    
    if len(returns) == 0:
        return 0.0
    
    # Assuming returns are already daily
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    # Handle NaN or inf
    if np.isnan(sharpe) or np.isinf(sharpe):
        return 0.0
    
    return sharpe


def maximum_drawdown(results: pd.DataFrame) -> float:
    """Calculate maximum drawdown percentage."""
    if 'equity' not in results.columns or len(results) < 2:
        return 0.0
    
    equity = results['equity']
    
    # Calculate running maximum
    running_max = equity.cummax()
    
    # Calculate drawdowns
    drawdowns = (equity - running_max) / running_max
    
    # Get maximum drawdown
    max_dd = drawdowns.min()
    
    return max_dd


def win_rate(results: pd.DataFrame) -> float:
    """Calculate win rate."""
    if 'trade_returns' not in results.columns or len(results) < 2:
        return 0.0
    
    trade_returns = results['trade_returns'].dropna()
    
    if len(trade_returns) == 0:
        return 0.0
    
    # Calculate win rate
    wins = (trade_returns > 0).sum()
    total_trades = len(trade_returns)
    
    return wins / total_trades


def profit_factor(results: pd.DataFrame) -> float:
    """Calculate profit factor."""
    if 'trade_returns' not in results.columns or len(results) < 2:
        return 1.0
    
    trade_returns = results['trade_returns'].dropna()
    
    if len(trade_returns) == 0:
        return 1.0
    
    # Calculate profit factor
    gross_profits = trade_returns[trade_returns > 0].sum()
    gross_losses = -trade_returns[trade_returns < 0].sum()
    
    if gross_losses == 0:
        return 10.0  # Arbitrary high value for no losses
    
    return gross_profits / gross_losses


def avg_gain_loss_ratio(results: pd.DataFrame) -> float:
    """Calculate average gain/loss ratio."""
    if 'trade_returns' not in results.columns or len(results) < 2:
        return 1.0
    
    trade_returns = results['trade_returns'].dropna()
    
    if len(trade_returns) == 0:
        return 1.0
    
    # Calculate average gain and loss
    avg_gain = trade_returns[trade_returns > 0].mean() if any(trade_returns > 0) else 0
    avg_loss = -trade_returns[trade_returns < 0].mean() if any(trade_returns < 0) else 1  # Avoid division by zero
    
    if avg_loss == 0:
        return 10.0  # Arbitrary high value for no losses
    
    return avg_gain / avg_loss


def calmar_ratio(results: pd.DataFrame) -> float:
    """Calculate Calmar ratio."""
    if 'equity' not in results.columns or len(results) < 252:  # Require at least a year of data
        return 0.0
    
    # Calculate annualized return
    cagr = annualized_return(results)
    
    # Calculate maximum drawdown
    mdd = maximum_drawdown(results)
    
    if mdd == 0:
        return 10.0  # Arbitrary high value for no drawdown
    
    return cagr / abs(mdd)


def ulcer_index(results: pd.DataFrame) -> float:
    """Calculate Ulcer Index (UI) for measuring downside risk."""
    if 'equity' not in results.columns or len(results) < 2:
        return 0.0
    
    equity = results['equity']
    
    # Calculate running maximum
    running_max = equity.cummax()
    
    # Calculate percentage drawdowns
    drawdowns = (equity - running_max) / running_max
    
    # Square the drawdowns
    squared_drawdowns = drawdowns ** 2
    
    # Calculate Ulcer Index
    ui = np.sqrt(squared_drawdowns.mean())
    
    return ui


def sortino_ratio(results: pd.DataFrame, target_return: float = 0.0, mar: float = 0.0) -> float:
    """Calculate Sortino ratio, using downside deviation."""
    if 'returns' not in results.columns or len(results) < 30:
        return 0.0
    
    returns = results['returns'].dropna()
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns - target_return / 252  # Daily target return
    
    # Calculate downside deviation (only returns below MAR)
    downside_returns = excess_returns[excess_returns < mar / 252]
    
    if len(downside_returns) == 0:
        return 10.0  # Arbitrary high value if no downside
    
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_deviation == 0:
        return 10.0  # Arbitrary high value if no downside deviation
    
    # Calculate Sortino ratio
    sortino = np.sqrt(252) * excess_returns.mean() / downside_deviation
    
    # Handle NaN or inf
    if np.isnan(sortino) or np.isinf(sortino):
        return 0.0
    
    return sortino


def kelly_criterion(results: pd.DataFrame) -> float:
    """Calculate Kelly Criterion percentage."""
    if 'trade_returns' not in results.columns or len(results) < 2:
        return 0.0
    
    trade_returns = results['trade_returns'].dropna()
    
    if len(trade_returns) == 0:
        return 0.0
    
    # Calculate win rate and average win/loss ratio
    wins = trade_returns > 0
    win_rate_val = wins.mean()
    
    if win_rate_val == 0:
        return 0.0
    
    if win_rate_val == 1.0:
        return 1.0  # Full Kelly if all trades are winners
    
    avg_win = trade_returns[wins].mean() if any(wins) else 0
    avg_loss = -trade_returns[~wins].mean() if any(~wins) else 1  # Avoid division by zero
    
    if avg_loss == 0:
        return 1.0  # Full Kelly if no losses
    
    # Calculate Kelly percentage
    kelly = win_rate_val - (1 - win_rate_val) / (avg_win / avg_loss)
    
    # Bound between 0 and 1
    kelly = max(0.0, min(1.0, kelly))
    
    return kelly


def max_consecutive_losses(results: pd.DataFrame) -> float:
    """Calculate maximum consecutive losses."""
    if 'trade_returns' not in results.columns or len(results) < 2:
        return 0.0
    
    trade_returns = results['trade_returns'].dropna()
    
    if len(trade_returns) == 0:
        return 0.0
    
    # Create a series of 1s and -1s for wins and losses
    wins_losses = np.where(trade_returns > 0, 1, -1)
    
    # Find runs of consecutive losses
    loss_runs = []
    current_run = 0
    
    for wl in wins_losses:
        if wl < 0:
            current_run += 1
        else:
            if current_run > 0:
                loss_runs.append(current_run)
            current_run = 0
    
    # Add the last run if it's a loss
    if current_run > 0:
        loss_runs.append(current_run)
    
    if not loss_runs:
        return 0.0
    
    return max(loss_runs)


# Common stress tests
def high_volatility_stress_test(market_data: pd.DataFrame) -> pd.DataFrame:
    """Filter market data to high volatility periods."""
    if 'Close' not in market_data.columns or len(market_data) < 30:
        return market_data
    
    # Calculate rolling volatility (annualized)
    returns = market_data['Close'].pct_change()
    volatility = returns.rolling(window=21).std() * np.sqrt(252)
    
    # Find high volatility periods (top 25%)
    high_vol_threshold = volatility.quantile(0.75)
    high_vol_periods = volatility >= high_vol_threshold
    
    # Filter market data
    high_vol_data = market_data.loc[high_vol_periods]
    
    return high_vol_data


def bear_market_stress_test(market_data: pd.DataFrame) -> pd.DataFrame:
    """Filter market data to bear market periods."""
    if 'Close' not in market_data.columns or len(market_data) < 100:
        return market_data
    
    # Calculate 200-day moving average
    ma200 = market_data['Close'].rolling(window=200).mean()
    
    # Define bear market as price below MA200
    bear_market = market_data['Close'] < ma200
    
    # Filter market data
    bear_data = market_data.loc[bear_market]
    
    return bear_data


def drawdown_periods_stress_test(market_data: pd.DataFrame) -> pd.DataFrame:
    """Filter market data to major drawdown periods."""
    if 'Close' not in market_data.columns or len(market_data) < 30:
        return market_data
    
    # Calculate running maximum
    running_max = market_data['Close'].cummax()
    
    # Calculate drawdowns
    drawdowns = (market_data['Close'] - running_max) / running_max
    
    # Find major drawdown periods (below -10%)
    major_dd_periods = drawdowns <= -0.10
    
    # Filter market data
    dd_data = market_data.loc[major_dd_periods]
    
    return dd_data


def low_liquidity_stress_test(market_data: pd.DataFrame) -> pd.DataFrame:
    """Filter market data to low liquidity periods."""
    if 'Volume' not in market_data.columns or len(market_data) < 30:
        return market_data
    
    # Calculate average volume
    avg_volume = market_data['Volume'].rolling(window=21).mean()
    
    # Find low liquidity periods (bottom 25%)
    low_liq_threshold = avg_volume.quantile(0.25)
    low_liq_periods = avg_volume <= low_liq_threshold
    
    # Filter market data
    low_liq_data = market_data.loc[low_liq_periods]
    
    return low_liq_data


def rising_rates_stress_test(market_data: pd.DataFrame) -> pd.DataFrame:
    """Filter market data to rising interest rate periods."""
    if 'Interest_Rate' not in market_data.columns or len(market_data) < 30:
        return market_data
    
    # Calculate rate changes
    rate_changes = market_data['Interest_Rate'].diff()
    
    # Find rising rate periods
    rising_rate_periods = rate_changes > 0
    
    # Filter market data
    rising_rate_data = market_data.loc[rising_rate_periods]
    
    return rising_rate_data


def sector_rotation_stress_test(market_data: pd.DataFrame) -> pd.DataFrame:
    """Filter market data to sector rotation periods."""
    sector_cols = [col for col in market_data.columns if col.startswith('Sector_')]
    
    if not sector_cols or len(market_data) < 30:
        return market_data
    
    # Calculate sector returns
    sector_returns = market_data[sector_cols].pct_change()
    
    # Calculate sector return dispersion (std dev across sectors)
    dispersion = sector_returns.std(axis=1)
    
    # Find high dispersion periods (top 25%)
    high_disp_threshold = dispersion.quantile(0.75)
    high_disp_periods = dispersion >= high_disp_threshold
    
    # Filter market data
    rotation_data = market_data.loc[high_disp_periods]
    
    return rotation_data


def flash_crash_stress_test(market_data: pd.DataFrame) -> pd.DataFrame:
    """Filter market data to flash crash-like periods."""
    if 'Close' not in market_data.columns or len(market_data) < 30:
        return market_data
    
    # Calculate daily returns
    returns = market_data['Close'].pct_change()
    
    # Find extreme negative return days (below -3%)
    crash_days = returns <= -0.03
    
    # Include the day before and after each crash day
    crash_indices = market_data.index[crash_days]
    all_indices = []
    
    for idx in crash_indices:
        # Find position of the index
        pos = market_data.index.get_loc(idx)
        
        # Add day before, crash day, and day after if they exist
        if pos > 0:
            all_indices.append(market_data.index[pos-1])
        all_indices.append(idx)
        if pos < len(market_data) - 1:
            all_indices.append(market_data.index[pos+1])
    
    # Remove duplicates and sort
    all_indices = sorted(set(all_indices))
    
    # Filter market data
    crash_data = market_data.loc[all_indices]
    
    return crash_data


def _black_swan_crash_test_data() -> pd.DataFrame:
    """Generate synthetic data for a black swan crash event."""
    # Create a date range for our synthetic data (60 days)
    dates = pd.date_range(start='2020-01-01', periods=60)
    
    # Initial price and normal daily returns
    initial_price = 100.0
    normal_returns = np.random.normal(0.0005, 0.01, 40)  # 40 days of normal returns
    crash_returns = np.array([-0.02, -0.08, -0.10, -0.15, -0.05, 0.03, 0.05, -0.03, -0.02, 0.04] +
                           list(np.random.normal(0.001, 0.02, 10)))  # 20 days of crash + aftermath
    
    # Combine returns
    all_returns = np.concatenate([normal_returns, crash_returns])
    
    # Calculate price series
    prices = initial_price * (1 + all_returns).cumprod()
    
    # Create volume data
    normal_volume = np.random.normal(1000000, 200000, 40)
    crash_volume = np.random.normal(3000000, 800000, 20)  # Higher volume during crash
    volumes = np.concatenate([normal_volume, crash_volume])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': prices * (1 - np.random.uniform(0, 0.01, 60)),
        'High': prices * (1 + np.random.uniform(0, 0.02, 60)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, 60)),
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    return df


def black_swan_stress_test(market_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a synthetic black swan crash scenario based on historical data patterns.
    
    This function identifies the worst drawdown period in the data (if any)
    and amplifies it to simulate a black swan event, or generates synthetic
    crash data if no significant drawdown is found.
    """
    if 'Close' not in market_data.columns or len(market_data) < 60:
        # Not enough data, use synthetic data
        return _black_swan_crash_test_data()
    
    # Calculate running maximum
    running_max = market_data['Close'].cummax()
    
    # Calculate drawdowns
    drawdowns = (market_data['Close'] - running_max) / running_max
    
    # Find worst drawdown
    worst_dd = drawdowns.min()
    worst_dd_idx = drawdowns.idxmin()
    
    if worst_dd > -0.05:
        # No significant drawdown found, use synthetic data
        return _black_swan_crash_test_data()
    
    # Find the start of the drawdown (last peak before worst point)
    peak_idx = drawdowns.loc[:worst_dd_idx][::-1].idxmax()
    
    # Find recovery period (if any)
    recovery_data = drawdowns.loc[worst_dd_idx:]
    recovery_idx = None
    
    for idx, dd in recovery_data.iteritems():
        if dd >= -0.02:  # Consider "recovered" when drawdown is less than 2%
            recovery_idx = idx
            break
    
    if recovery_idx is None:
        # No recovery found, use all data after worst point
        crash_period = market_data.loc[peak_idx:]
    else:
        # Use data from peak to recovery
        crash_period = market_data.loc[peak_idx:recovery_idx]
    
    # Ensure we have enough data (at least 10 days)
    if len(crash_period) < 10:
        # Not enough data, use synthetic data
        return _black_swan_crash_test_data()
    
    # Amplify the crash by making the drawdown twice as severe
    crash_data = crash_period.copy()
    peak_price = crash_data['Close'].iloc[0]
    worst_price = crash_data['Close'].min()
    
    # Calculate amplification factor to double the drawdown
    orig_dd = (worst_price - peak_price) / peak_price
    target_dd = orig_dd * 2  # Double the drawdown
    
    # Scale the prices during the crash
    for i in range(1, len(crash_data)):
        current_dd = (crash_data['Close'].iloc[i] - peak_price) / peak_price
        if current_dd < 0:
            # Amplify negative returns
            new_dd = current_dd * (target_dd / orig_dd)
            crash_data.loc[crash_data.index[i], 'Close'] = peak_price * (1 + new_dd)
            
            # Adjust other price columns
            scale_factor = crash_data.loc[crash_data.index[i], 'Close'] / crash_period.loc[crash_period.index[i], 'Close']
            crash_data.loc[crash_data.index[i], 'Open'] *= scale_factor
            crash_data.loc[crash_data.index[i], 'High'] *= scale_factor
            crash_data.loc[crash_data.index[i], 'Low'] *= scale_factor
    
    # Amplify volume during the crash
    crash_data['Volume'] = crash_data['Volume'] * 2
    
    return crash_data


def main():
    """Example usage of the multi-objective optimization framework."""
    # Create objectives
    objectives = [
        TradingObjective("annualized_return", annualized_return, direction="maximize", weight=1.0),
        TradingObjective("sharpe_ratio", sharpe_ratio, direction="maximize", weight=1.0),
        TradingObjective("max_drawdown", maximum_drawdown, direction="minimize", weight=1.0,
                        constraint_value=-0.25, constraint_type=">="),
        TradingObjective("win_rate", win_rate, direction="maximize", weight=0.5)
    ]
    
    # Create stress tests
    stress_tests = [
        StressTest("high_volatility", high_volatility_stress_test, 
                  "Periods of high market volatility (top 25%)"),
        StressTest("bear_market", bear_market_stress_test,
                  "Periods where price is below 200-day moving average"),
        StressTest("major_drawdowns", drawdown_periods_stress_test,
                  "Periods with market drawdowns of 10% or more")
    ]
    
    # Create optimizer
    optimizer = MultiObjectiveOptimizer(
        objectives=objectives,
        stress_tests=stress_tests,
        population_size=50,
        max_generations=30,
        crossover_rate=0.8,
        mutation_rate=0.2
    )
    
    # Print info about the optimizer
    print("Multi-Objective Optimizer initialized with:")
    print(f"- {len(objectives)} objectives")
    for obj in objectives:
        constraint_str = ""
        if obj.constraint_value is not None:
            constraint_str = f" [constraint: {obj.constraint_type} {obj.constraint_value}]"
        print(f"  - {obj.name} ({obj.direction}, weight={obj.weight}){constraint_str}")
    
    print(f"- {len(stress_tests)} stress tests")
    for test in stress_tests:
        print(f"  - {test.name}: {test.description}")
    
    print("\nThis module provides a comprehensive framework for multi-objective strategy optimization.")
    print("It allows optimization across multiple competing objectives like return, risk, and robustness.")
    print("The framework also includes stress testing capabilities to ensure strategies perform well")
    print("under various market conditions including high volatility, bear markets, and black swan events.")
    
    print("\nKey features:")
    print("- Pareto optimization using genetic algorithms")
    print("- Non-dominated sorting and crowding distance for diverse solution set")
    print("- Hypervolume indicator for performance measurement")
    print("- Constraint handling for realistic trading limitations")
    print("- Multiple stress testing scenarios")
    print("- Visualization of Pareto fronts and stress test results")
    
    print("\nTo use this module, you need to:")
    print("1. Define your trading objectives (what to optimize)")
    print("2. Create stress tests (what conditions to test against)")
    print("3. Implement strategy generation, mutation, and crossover functions")
    print("4. Provide a strategy evaluation function")
    print("5. Run the optimizer with your market data")
    
    print("\nMULTI-OBJECTIVE OPTIMIZATION FRAMEWORK IMPLEMENTATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()