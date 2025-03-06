#!/usr/bin/env python
"""
Bayesian Optimization Module

This module implements Bayesian optimization for efficient parameter tuning of
trading strategies using Gaussian Process regression and expected improvement
acquisition function.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from typing import Dict, List, Tuple, Callable, Any, Optional
import logging
from datetime import datetime

class BayesianOptimizer:
    """
    Implements Bayesian optimization for trading strategy parameter tuning.
    """
    
    def __init__(self,
                parameter_space: Dict[str, Tuple[float, float]],
                objective_function: Callable,
                n_initial_points: int = 5,
                exploration_weight: float = 0.1,
                kernel: str = 'matern',
                max_iterations: int = 100):
        """
        Initialize the Bayesian optimizer.

        Args:
            parameter_space: Dictionary of parameter names and their ranges (min, max)
            objective_function: Function to optimize (e.g., Sharpe ratio)
            n_initial_points: Number of initial random points
            exploration_weight: Weight for exploration vs exploitation
            kernel: Type of GP kernel ('rbf', 'matern', or 'constant')
            max_iterations: Maximum number of optimization iterations
        """
        self.parameter_space = parameter_space
        self.objective_function = objective_function
        self.n_initial_points = n_initial_points
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations
        
        # Initialize Gaussian Process model
        self.gp = self._initialize_gp(kernel)
        
        # Storage for optimization history
        self.X = []  # Parameters
        self.y = []  # Objective values
        self.best_params = None
        self.best_value = float('-inf')
        
        self.logger = logging.getLogger(__name__)
        
    def optimize(self) -> Dict[str, Any]:
        """
        Run the Bayesian optimization process.
        
        Returns:
            Dictionary containing optimization results
        """
        # Initial random sampling
        self._initial_sampling()
        
        # Main optimization loop
        for i in range(self.max_iterations):
            # Fit GP model
            self.gp.fit(np.array(self.X), np.array(self.y))
            
            # Find next point to evaluate
            next_point = self._find_next_point()
            
            # Evaluate objective function
            value = self.objective_function(self._dict_from_point(next_point))
            
            # Update optimization history
            self.X.append(next_point)
            self.y.append(value)
            
            # Update best result
            if value > self.best_value:
                self.best_value = value
                self.best_params = self._dict_from_point(next_point)
                
            # Log progress
            self.logger.info(f"Iteration {i+1}: Best value = {self.best_value:.4f}")
            
            # Check convergence
            if i > 10 and self._check_convergence():
                self.logger.info("Optimization converged")
                break
                
        X_array = np.array(self.X)
        
        X_array = np.array(self.X)
        
        X_array = np.array(self.X)
        
        X_array = np.array(self.X)
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'optimization_history': {
                'parameters': self.X,
                'values': self.y
            },
            'n_iterations': len(self.X) - self.n_initial_points,
            'convergence': self._calculate_convergence_metrics()
        }
    
    def _initialize_gp(self, kernel_type: str) -> GaussianProcessRegressor:
        """Initialize Gaussian Process model with specified kernel."""
        if kernel_type == 'rbf':
            kernel = ConstantKernel() * RBF()
        elif kernel_type == 'matern':
            kernel = ConstantKernel() * Matern(nu=2.5)
        else:  # constant
            kernel = ConstantKernel()
            
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=42
        )
    
    def _initial_sampling(self):
        """Perform initial random sampling."""
        for _ in range(self.n_initial_points):
            # Sample random point in parameter space
            point = self._sample_random_point()
            
            # Evaluate objective function
            value = self.objective_function(self._dict_from_point(point))
            
            # Store results
            self.X.append(point)
            self.y.append(value)
            
            # Update best result
            if value > self.best_value:
                self.best_value = value
                self.best_params = self._dict_from_point(point)
    
    def _sample_random_point(self) -> np.ndarray:
        """Sample random point from parameter space."""
        point = []
        for param_range in self.parameter_space.values():
            point.append(np.random.uniform(param_range[0], param_range[1]))
        return np.array(point)
    
    def _expected_improvement(self,
                           X: np.ndarray,
                           xi: float = 0.01) -> float:
        """
        Calculate expected improvement acquisition function.
        
        Args:
            X: Points to evaluate
            xi: Exploration-exploitation trade-off parameter
            
        Returns:
            Expected improvement values
        """
        mu, sigma = self.gp.predict(X.reshape(1, -1), return_std=True)
        
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        mu_sample_opt = np.max(self.y)
        
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        return ei
    
    def _find_next_point(self) -> np.ndarray:
        """Find next point to evaluate using expected improvement."""
        bounds = list(self.parameter_space.values())
        
        # Define acquisition function to minimize
        def acquisition(x):
            return -self._expected_improvement(x, xi=self.exploration_weight)
        
        # Run several optimizations with different starting points
        best_x = None
        best_acquisition_value = float('inf')
        
        for _ in range(10):
            x0 = self._sample_random_point()
            
            result = minimize(
                acquisition,
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.fun < best_acquisition_value:
                best_acquisition_value = result.fun
                best_x = result.x
                
        return best_x
    
    def _dict_from_point(self, point: np.ndarray) -> Dict[str, float]:
        """Convert point array to parameter dictionary."""
        return {
            name: value for name, value in zip(self.parameter_space.keys(), point)
        }
    
    def _check_convergence(self, window: int = 10, threshold: float = 1e-4) -> bool:
        """Check if optimization has converged."""
        if len(self.y) < window:
            return False
            
        recent_values = self.y[-window:]
        improvement = np.abs(np.max(recent_values) - np.min(recent_values))
        
        return improvement < threshold
    
    def _calculate_convergence_metrics(self) -> Dict[str, float]:
        """Calculate convergence metrics."""
        if len(self.y) < 2:
            return {}
            
        return {
            'final_improvement': abs(self.y[-1] - self.y[-2]),
            'total_improvement': abs(self.y[-1] - self.y[0]),
            'convergence_rate': np.std(self.y[-10:]) if len(self.y) >= 10 else None
        }
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # Plot objective values
        plt.subplot(1, 2, 1)
        plt.plot(self.y, 'b-', label='Objective value')
        plt.plot(np.maximum.accumulate(self.y), 'r--', label='Best value')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Optimization History')
        plt.legend()
        
        # Plot parameter evolution
        plt.subplot(1, 2, 2)
        X_array = np.array(self.X)
        for i, param_name in enumerate(self.parameter_space.keys()):
            plt.plot(X_array[:, i], label=param_name)
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.title('Parameter Evolution')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        plt.close()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'total_iterations': len(self.X),
            'initial_value': self.y[0],
            'final_value': self.y[-1],
            'improvement_percentage': (self.y[-1] - self.y[0]) / abs(self.y[0]) * 100,
            'convergence_metrics': self._calculate_convergence_metrics(),
            'parameter_ranges': {
                name: {
                    'min': min(X_array[:, i]),
                    'max': max(X_array[:, i]),
                    'final': self.best_params[name]
                }
                for i, name in enumerate(self.parameter_space.keys())
            }
        }