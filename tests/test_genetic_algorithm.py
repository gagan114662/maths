#!/usr/bin/env python
"""Unit tests for genetic algorithm optimization."""

import unittest
import numpy as np
from src.optimization.genetic_algorithm import GeneticAlgorithm, Individual

class TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        # Define test parameter ranges
        self.parameter_ranges = {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0)
        }
        
        # Define test fitness function (negative quadratic)
        def fitness_function(params):
            x = params['x']
            y = params['y']
            return -(x**2 + y**2)  # Maximum at (0, 0)
        
        # Initialize GA
        self.ga = GeneticAlgorithm(
            parameter_ranges=self.parameter_ranges,
            fitness_function=fitness_function,
            population_size=20,
            generations=10,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=2,
            tournament_size=3
        )
        
    def test_initialization(self):
        """Test population initialization."""
        self.ga.initialize_population()
        
        # Check population size
        self.assertEqual(len(self.ga.population), self.ga.population_size)
        
        # Check that all individuals have valid genes
        for individual in self.ga.population:
            for param, (min_val, max_val) in self.parameter_ranges.items():
                self.assertGreaterEqual(individual.genes[param], min_val)
                self.assertLessEqual(individual.genes[param], max_val)
                
    def test_evaluation(self):
        """Test population evaluation."""
        self.ga.initialize_population()
        self.ga.evaluate_population()
        
        # Check that all individuals have fitness values
        for individual in self.ga.population:
            self.assertIsNotNone(individual.fitness)
            
        # Check that best individual is tracked
        self.assertIsNotNone(self.ga.best_individual)
        
    def test_tournament_selection(self):
        """Test tournament selection."""
        self.ga.initialize_population()
        self.ga.evaluate_population()
        
        selected = self.ga.tournament_selection()
        
        # Check that selected individual is from population
        self.assertIn(selected, self.ga.population)
        
        # Create biased population to test selection pressure
        best_individual = Individual({'x': 0, 'y': 0}, fitness=0)
        worst_individual = Individual({'x': 5, 'y': 5}, fitness=-50)
        
        self.ga.population = [worst_individual] * 18 + [best_individual] * 2
        
        # Run multiple tournaments
        selections = [self.ga.tournament_selection() for _ in range(100)]
        
        # Better individuals should be selected more often
        best_count = len([s for s in selections if s.fitness == 0])
        self.assertGreater(best_count, 20)  # Should select best more than random
        
    def test_crossover(self):
        """Test crossover operation."""
        parent1 = Individual({'x': 1.0, 'y': 1.0})
        parent2 = Individual({'x': -1.0, 'y': -1.0})
        
        # Force crossover by setting rate to 1.0
        self.ga.crossover_rate = 1.0
        child1, child2 = self.ga.crossover(parent1, parent2)
        
        # Check that children have valid genes
        for child in [child1, child2]:
            for param, (min_val, max_val) in self.parameter_ranges.items():
                self.assertGreaterEqual(child.genes[param], min_val)
                self.assertLessEqual(child.genes[param], max_val)
                
        # Check that at least one gene is different (crossover occurred)
        different_genes = False
        for param in self.parameter_ranges.keys():
            if (child1.genes[param] != parent1.genes[param] or
                child2.genes[param] != parent2.genes[param]):
                different_genes = True
                break
        self.assertTrue(different_genes)
        
    def test_mutation(self):
        """Test mutation operation."""
        # Create individual with known genes
        individual = Individual({'x': 0.0, 'y': 0.0})
        original_genes = individual.genes.copy()
        
        # Force mutation by setting rate to 1.0
        self.ga.mutation_rate = 1.0
        self.ga.mutate(individual)
        
        # Check that at least one gene changed
        different_genes = False
        for param in self.parameter_ranges.keys():
            if individual.genes[param] != original_genes[param]:
                different_genes = True
                break
        self.assertTrue(different_genes)
        
        # Check that mutated genes are within bounds
        for param, (min_val, max_val) in self.parameter_ranges.items():
            self.assertGreaterEqual(individual.genes[param], min_val)
            self.assertLessEqual(individual.genes[param], max_val)
            
    def test_evolution(self):
        """Test full evolution process."""
        results = self.ga.evolve()
        
        # Check results structure
        self.assertIn('best_individual', results)
        self.assertIn('final_population', results)
        self.assertIn('evolution_stats', results)
        self.assertIn('convergence_metrics', results)
        
        # Check that best individual improved
        initial_best = self.ga.generation_stats[0]['best_fitness']
        final_best = self.ga.generation_stats[-1]['best_fitness']
        self.assertGreater(final_best, initial_best)
        
        # Check that best solution is near optimum (0, 0)
        best_x = results['best_individual'].genes['x']
        best_y = results['best_individual'].genes['y']
        self.assertLess(abs(best_x), 1.0)
        self.assertLess(abs(best_y), 1.0)
        
    def test_diversity_calculation(self):
        """Test population diversity calculation."""
        self.ga.initialize_population()
        
        # Test with uniform population
        self.ga.population = [Individual({'x': 0.0, 'y': 0.0})] * self.ga.population_size
        diversity = self.ga._calculate_diversity()
        self.assertAlmostEqual(diversity, 0.0)
        
        # Test with diverse population
        diverse_population = []
        for i in range(self.ga.population_size):
            genes = {
                'x': np.random.uniform(-5, 5),
                'y': np.random.uniform(-5, 5)
            }
            diverse_population.append(Individual(genes))
        self.ga.population = diverse_population
        
        diversity = self.ga._calculate_diversity()
        self.assertGreater(diversity, 0.0)
        
    def test_convergence_metrics(self):
        """Test convergence metrics calculation."""
        # Run evolution
        self.ga.evolve()
        
        metrics = self.ga._calculate_convergence_metrics()
        
        # Check metrics structure
        self.assertIn('final_diversity', metrics)
        self.assertIn('fitness_improvement_rate', metrics)
        self.assertIn('avg_fitness_std', metrics)
        self.assertIn('population_stability', metrics)
        
        # Check metric ranges
        self.assertGreaterEqual(metrics['final_diversity'], 0)
        self.assertLessEqual(metrics['final_diversity'], 1)
        self.assertGreaterEqual(metrics['population_stability'], 0)

if __name__ == '__main__':
    unittest.main()