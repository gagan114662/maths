#!/usr/bin/env python
"""
Genetic Algorithm Module

This module implements genetic algorithms for evolving trading strategies,
including crossover, mutation, and selection operations optimized for financial
applications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any, Optional
import random
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class Individual:
    """Represents a single trading strategy individual in the population."""
    
    def __init__(self, genes: Dict[str, float], fitness: float = None):
        """
        Initialize an individual.

        Args:
            genes: Dictionary of parameter names and values
            fitness: Fitness score (None if not evaluated)
        """
        self.genes = genes
        self.fitness = fitness
        
    def __str__(self):
        return f"Fitness: {self.fitness}, Genes: {self.genes}"

class GeneticAlgorithm:
    """Implements genetic algorithm for trading strategy evolution."""
    
    def __init__(self,
                parameter_ranges: Dict[str, Tuple[float, float]],
                fitness_function: Callable,
                population_size: int = 50,
                generations: int = 100,
                mutation_rate: float = 0.1,
                crossover_rate: float = 0.8,
                elite_size: int = 2,
                tournament_size: int = 5):
        """
        Initialize the genetic algorithm.

        Args:
            parameter_ranges: Dictionary of parameter names and their ranges
            fitness_function: Function to evaluate individual fitness
            population_size: Size of population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of best individuals to preserve
            tournament_size: Size of tournament for selection
        """
        self.parameter_ranges = parameter_ranges
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        self.population = []
        self.best_individual = None
        self.generation_stats = []
        
        self.logger = logging.getLogger(__name__)
        
    def initialize_population(self):
        """Initialize random population."""
        self.population = []
        
        for _ in range(self.population_size):
            genes = {}
            for param, (min_val, max_val) in self.parameter_ranges.items():
                genes[param] = random.uniform(min_val, max_val)
            
            individual = Individual(genes)
            self.population.append(individual)
            
    def evaluate_population(self):
        """Evaluate fitness for all individuals in parallel."""
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.fitness_function, individual.genes)
                for individual in self.population
            ]
            
            for individual, future in zip(self.population, futures):
                individual.fitness = future.result()
                
        # Update best individual
        current_best = max(self.population, key=lambda x: x.fitness)
        if (self.best_individual is None or 
            current_best.fitness > self.best_individual.fitness):
            self.best_individual = Individual(
                genes=current_best.genes.copy(),
                fitness=current_best.fitness
            )
            
    def tournament_selection(self) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return parent1, parent2
            
        child1_genes = {}
        child2_genes = {}
        
        # Perform uniform crossover
        for param in self.parameter_ranges.keys():
            if random.random() < 0.5:
                child1_genes[param] = parent1.genes[param]
                child2_genes[param] = parent2.genes[param]
            else:
                child1_genes[param] = parent2.genes[param]
                child2_genes[param] = parent1.genes[param]
                
        return (
            Individual(child1_genes),
            Individual(child2_genes)
        )
    
    def mutate(self, individual: Individual):
        """Perform mutation on an individual."""
        for param, (min_val, max_val) in self.parameter_ranges.items():
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation = np.random.normal(0, (max_val - min_val) * 0.1)
                individual.genes[param] = np.clip(
                    individual.genes[param] + mutation,
                    min_val,
                    max_val
                )
    
    def evolve(self) -> Dict[str, Any]:
        """
        Run the genetic algorithm evolution process.
        
        Returns:
            Dictionary containing evolution results
        """
        self.initialize_population()
        self.evaluate_population()
        
        for generation in range(self.generations):
            # Store statistics
            self._store_generation_stats(generation)
            
            # Create new population
            new_population = []
            
            # Elitism - preserve best individuals
            sorted_population = sorted(
                self.population,
                key=lambda x: x.fitness,
                reverse=True
            )
            new_population.extend(sorted_population[:self.elite_size])
            
            # Create rest of new population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                self.mutate(child1)
                self.mutate(child2)
                
                new_population.append(child1)
                new_population.append(child2)
                
            # Trim to population size
            self.population = new_population[:self.population_size]
            
            # Evaluate new population
            self.evaluate_population()
            
            # Log progress
            self.logger.info(
                f"Generation {generation + 1}/{self.generations}: "
                f"Best Fitness = {self.best_individual.fitness:.4f}"
            )
            
        return {
            'best_individual': self.best_individual,
            'final_population': self.population,
            'evolution_stats': self.generation_stats,
            'convergence_metrics': self._calculate_convergence_metrics()
        }
    
    def _store_generation_stats(self, generation: int):
        """Store statistics for current generation."""
        fitness_values = [ind.fitness for ind in self.population]
        
        self.generation_stats.append({
            'generation': generation,
            'best_fitness': max(fitness_values),
            'avg_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'diversity': self._calculate_diversity()
        })
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        param_values = {
            param: [] for param in self.parameter_ranges.keys()
        }
        
        for individual in self.population:
            for param, value in individual.genes.items():
                param_values[param].append(value)
                
        # Calculate normalized standard deviation for each parameter
        diversity_scores = []
        for param, values in param_values.items():
            param_range = (
                self.parameter_ranges[param][1] - 
                self.parameter_ranges[param][0]
            )
            normalized_std = np.std(values) / param_range
            diversity_scores.append(normalized_std)
            
        return np.mean(diversity_scores)
    
    def _calculate_convergence_metrics(self) -> Dict[str, float]:
        """Calculate convergence metrics."""
        if len(self.generation_stats) < 2:
            return {}
            
        recent_gens = self.generation_stats[-10:]
        
        return {
            'final_diversity': recent_gens[-1]['diversity'],
            'fitness_improvement_rate': (
                recent_gens[-1]['best_fitness'] - recent_gens[0]['best_fitness']
            ) / len(recent_gens),
            'avg_fitness_std': np.std([gen['avg_fitness'] for gen in recent_gens]),
            'population_stability': np.mean([gen['diversity'] for gen in recent_gens])
        }
    
    def plot_evolution(self, save_path: Optional[str] = None):
        """Plot evolution progress."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot fitness evolution
        generations = [stat['generation'] for stat in self.generation_stats]
        best_fitness = [stat['best_fitness'] for stat in self.generation_stats]
        avg_fitness = [stat['avg_fitness'] for stat in self.generation_stats]
        
        ax1.plot(generations, best_fitness, 'b-', label='Best Fitness')
        ax1.plot(generations, avg_fitness, 'r--', label='Average Fitness')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution')
        ax1.legend()
        ax1.grid(True)
        
        # Plot diversity
        diversity = [stat['diversity'] for stat in self.generation_stats]
        ax2.plot(generations, diversity, 'g-', label='Population Diversity')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Diversity')
        ax2.set_title('Population Diversity')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        plt.close()