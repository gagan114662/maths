# Core API Reference

## Overview

This document provides detailed API documentation for the core classes in the Enhanced Trading Strategy System.

## BaseAgent

Core base class for all agents in the system.

```python
class BaseAgent:
    """
    Base class for implementing trading agents.
    
    Attributes:
        name (str): Agent identifier
        config (Dict): Configuration parameters
        state (Dict): Current agent state
    """
    
    def __init__(self, name: str = None, config: Dict = None):
        """
        Initialize agent.
        
        Args:
            name: Agent identifier
            config: Configuration dictionary
        """
        pass
        
    def process(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process market data and generate insights.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary containing processing results
            
        Raises:
            ValidationError: If data validation fails
            ProcessingError: If processing fails
        """
        raise NotImplementedError
```

## BaseStrategy

Base class for implementing trading strategies.

```python
class BaseStrategy:
    """
    Base class for trading strategies.
    
    Attributes:
        name (str): Strategy identifier
        params (Dict): Strategy parameters
        position_limits (Dict): Trading limits
    """
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary containing:
            - signals: Trading signals (-1, 0, 1)
            - position_sizes: Position sizes
            - metrics: Strategy metrics
            
        Raises:
            ValidationError: If data validation fails
            StrategyError: If signal generation fails
        """
        raise NotImplementedError
```

## ModelTrainer

Handles model training and validation.

```python
class ModelTrainer:
    """
    Trainer for machine learning models.
    
    Attributes:
        config (Dict): Training configuration
        model: Model instance
        optimizer: Optimizer instance
    """
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame = None,
        callbacks: List = None
    ) -> Dict[str, Any]:
        """
        Train model with data.
        
        Args:
            train_data: Training data
            val_data: Validation data
            callbacks: List of callback functions
            
        Returns:
            Dictionary containing:
            - history: Training history
            - metrics: Performance metrics
            - model: Trained model
            
        Raises:
            TrainingError: If training fails
            ValidationError: If validation fails
        """
        pass
```

## DataPipeline

Handles data processing and transformation.

```python
class DataPipeline:
    """
    Data processing pipeline.
    
    Attributes:
        validators (List): Data validators
        transformers (List): Data transformers
        config (Dict): Pipeline configuration
    """
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process data through pipeline.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Processed DataFrame
            
        Raises:
            ValidationError: If validation fails
            ProcessingError: If processing fails
        """
        pass
```

## StrategyEvaluator

Evaluates trading strategy performance.

```python
class StrategyEvaluator:
    """
    Evaluates trading strategies.
    
    Attributes:
        metrics (Dict): Evaluation metrics
        risk_limits (Dict): Risk management limits
    """
    
    def evaluate_strategy(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        returns: pd.DataFrame,
        trades: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate strategy performance.
        
        Args:
            predictions: Strategy predictions
            actuals: Actual market data
            returns: Strategy returns
            trades: Trade information
            
        Returns:
            Dictionary containing:
            - ranking_metrics: Strategy ranking metrics
            - portfolio_metrics: Portfolio performance
            - error_metrics: Prediction errors
            - ethical_compliance: Compliance checks
            
        Raises:
            EvaluationError: If evaluation fails
        """
        pass
```

## HyperparameterTuner

Optimizes model hyperparameters.

```python
class HyperparameterTuner:
    """
    Tunes model hyperparameters.
    
    Attributes:
        config (Dict): Tuning configuration
        strategy (str): Optimization strategy
    """
    
    def optimize(
        self,
        model,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        param_grid: Dict
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters.
        
        Args:
            model: Model to optimize
            train_data: Training data
            val_data: Validation data
            param_grid: Parameter search space
            
        Returns:
            Dictionary containing:
            - best_params: Optimal parameters
            - best_score: Best validation score
            - search_results: All trials
            
        Raises:
            OptimizationError: If optimization fails
        """
        pass
```

## ErrorHandling

Common error types used in the system.

```python
class ValidationError(Exception):
    """Raised when data validation fails."""
    pass

class ProcessingError(Exception):
    """Raised when data processing fails."""
    pass

class TrainingError(Exception):
    """Raised when model training fails."""
    pass

class EvaluationError(Exception):
    """Raised when strategy evaluation fails."""
    pass

class OptimizationError(Exception):
    """Raised when optimization fails."""
    pass
```

## Utility Functions

### Data Validation

```python
def validate_data(
    data: pd.DataFrame,
    required_columns: List[str] = None,
    checks: List[str] = None
) -> bool:
    """
    Validate DataFrame.
    
    Args:
        data: DataFrame to validate
        required_columns: Required column names
        checks: List of validation checks to perform
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    pass
```

### Performance Metrics

```python
def calculate_metrics(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate performance metrics.
    
    Args:
        predictions: Model predictions
        actuals: Actual values
        
    Returns:
        Dictionary of metrics
    """
    pass
```

## Configuration

Example configuration structure:

```yaml
model:
  type: "lstm"
  params:
    input_size: 64
    hidden_size: 128
    num_layers: 2

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001

validation:
  split_ratio: 0.2
  early_stopping:
    patience: 10

data:
  features:
    - "close"
    - "volume"
    - "volatility"
  targets:
    - "returns"
  window_size: 20
```

## Additional Resources

- [Example Usage](../examples/index.md)
- [Tutorials](../tutorials/index.md)
- [FAQ](../faq.md)