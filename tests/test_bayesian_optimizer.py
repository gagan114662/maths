#!/usr/bin/env python
"""Unit tests for Bayesian optimization."""

import unittest
import numpy as np
from src.optimization.bayesian_optimizer import BayesianOptimizer

class TestBayesianOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        # Define test objective function (negative quadratic)
        def objective_function(params):
            x = params['x']
            y = params['y']
            return -(x**2 + y**2)  # Maximum at (0, 0)
            
        # Define parameter space
        self.parameter_space = {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0)
        }
        
        # Initialize optimizer
        self.optimizer = BayesianOptimizer(
            parameter_space=self.parameter_space,
            objective_function=objective_function,
            n_initial_points=5,
            exploration_weight=0.1,
            max_iterations=20
        )
        
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.n_initial_points, 5)
        self.assertEqual(self.optimizer.max_iterations, 20)
        self.assertEqual(self.optimizer.parameter_space, self.parameter_space)
        
        # Check GP model initialization
        self.assertIsNotNone(self.optimizer.gp)
        
    def test_optimization(self):
        """Test optimization process."""
        results = self.optimizer.optimize()
        
        # Check results structure
        self.assertIn('best_params', results)
        self.assertIn('best_value', results)
        self.assertIn('optimization_history', results)
        self.assertIn('n_iterations', results)
        
        # Check that best value is reasonable (should be close to 0)
        self.assertGreater(results['best_value'], -1.0)
        
        # Check that parameters are within bounds
        for param, value in results['best_params'].items():
            bounds = self.parameter_space[param]
            self.assertGreaterEqual(value, bounds[0])
            self.assertLessEqual(value, bounds[1])
            
    def test_expected_improvement(self):
        """Test expected improvement calculation."""
        # Perform initial sampling
        self.optimizer._initial_sampling()
        
        # Fit GP model
        X = np.array(self.optimizer.X)
        y = np.array(self.optimizer.y)
        self.optimizer.gp.fit(X, y)
        
        # Calculate EI for a test point
        test_point = np.array([0.0, 0.0])
        ei = self.optimizer._expected_improvement(test_point)
        
        # Check that EI is non-negative
        self.assertGreaterEqual(ei, 0)
        
    def test_convergence(self):
        """Test convergence checking."""
        # Create synthetic optimization history
        self.optimizer.y = [1.0, 0.5, 0.2, 0.1, 0.05, 0.04, 0.039, 0.038, 0.037, 0.037]
        
        # Check convergence
        is_converged = self.optimizer._check_convergence(window=5, threshold=1e-3)
        self.assertTrue(is_converged)
        
        # Test non-convergence
        self.optimizer.y = [1.0, 0.5, 0.2, 0.1, 0.05]
        is_converged = self.optimizer._check_convergence(window=5, threshold=1e-3)
        self.assertFalse(is_converged)
        
    def test_optimization_stats(self):
        """Test optimization statistics calculation."""
        # Run optimization
        self.optimizer.optimize()
        
        # Get stats
        stats = self.optimizer.get_optimization_stats()
        
        # Check stats structure
        self.assertIn('total_iterations', stats)
        self.assertIn('initial_value', stats)
        self.assertIn('final_value', stats)
        self.assertIn('improvement_percentage', stats)
        self.assertIn('convergence_metrics', stats)
        self.assertIn('parameter_ranges', stats)
        
        # Check parameter ranges
        for param in self.parameter_space.keys():
            self.assertIn(param, stats['parameter_ranges'])
            param_stats = stats['parameter_ranges'][param]
            self.assertIn('min', param_stats)
            self.assertIn('max', param_stats)
            self.assertIn('final', param_stats)
            
    def test_different_kernels(self):
        """Test optimization with different GP kernels."""
        kernel_types = ['rbf', 'matern', 'constant']
        
        for kernel in kernel_types:
            optimizer = BayesianOptimizer(
                parameter_space=self.parameter_space,
                objective_function=lambda x: -(x['x']**2 + x['y']**2),
                kernel=kernel,
                max_iterations=10
            )
            
            results = optimizer.optimize()
            self.assertIn('best_params', results)
            self.assertIn('best_value', results)
            
    def test_optimization_with_noise(self):
        """Test optimization with noisy objective function."""
        def noisy_objective(params):
            x = params['x']
            y = params['y']
            noise = np.random.normal(0, 0.1)
            return -(x**2 + y**2) + noise
            
        optimizer = BayesianOptimizer(
            parameter_space=self.parameter_space,
            objective_function=noisy_objective,
            n_initial_points=10,
            max_iterations=30
        )
        
        results = optimizer.optimize()
        
        # Check that optimization still finds reasonable solution
        best_x = results['best_params']['x']
        best_y = results['best_params']['y']
        
        # Should be close to origin despite noise
        self.assertLess(abs(best_x), 1.0)
        self.assertLess(abs(best_y), 1.0)
        
    def test_parameter_bounds(self):
        """Test that optimization respects parameter bounds."""
        results = self.optimizer.optimize()
        
        # Check all points in optimization history
        X_array = np.array(self.optimizer.X)
        
        for i, (param, bounds) in enumerate(self.parameter_space.items()):
            self.assertTrue(np.all(X_array[:, i] >= bounds[0]))
            self.assertTrue(np.all(X_array[:, i] <= bounds[1]))

if __name__ == '__main__':
    unittest.main()