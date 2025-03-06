# Advanced Model Training and Optimization

This guide covers advanced topics in model training, optimization, and hyperparameter tuning using FinTSB integration.

## Overview

Learn how to:
- Train advanced models
- Tune hyperparameters
- Optimize performance
- Validate results

## Prerequisites

```python
from src.training import ModelTrainer, HyperparameterTuner
from src.utils.config import load_config
from src.data_processors import DataPipeline
```

## 1. Advanced Model Configuration

### LSTM Configuration
```yaml
# configs/advanced_lstm.yaml
model:
  type: "lstm"
  params:
    input_size: 64
    hidden_size: 128
    num_layers: 3
    dropout: 0.2
    bidirectional: true
    attention: true
    
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  scheduler:
    type: "ReduceLROnPlateau"
    patience: 5
    factor: 0.5
  
validation:
  split_ratio: 0.2
  early_stopping:
    patience: 10
    min_delta: 0.001
    
optimization:
  gradient_clipping: 1.0
  weight_decay: 0.0001
```

## 2. Advanced Training Pipeline

```python
class AdvancedTrainer:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.trainer = ModelTrainer(self.config)
        self.pipeline = DataPipeline()
        
    def train_with_validation(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame
    ):
        """Train model with advanced validation."""
        # Preprocess data
        train_processed = self.pipeline.process(train_data)
        val_processed = self.pipeline.process(val_data)
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(
                patience=self.config['validation']['early_stopping']['patience'],
                min_delta=self.config['validation']['early_stopping']['min_delta']
            ),
            ModelCheckpoint(
                filepath='models/best_model.pt',
                save_best_only=True
            ),
            LearningRateScheduler(
                scheduler_type=self.config['training']['scheduler']['type'],
                **self.config['training']['scheduler']
            ),
            TensorBoard(log_dir='logs/training')
        ]
        
        # Train model
        history = self.trainer.train(
            train_data=train_processed,
            val_data=val_processed,
            callbacks=callbacks
        )
        
        return history

```

## 3. Hyperparameter Tuning

### Setup Tuning Configuration
```python
tuning_config = {
    'model_params': {
        'hidden_size': [64, 128, 256],
        'num_layers': [1, 2, 3],
        'dropout': [0.1, 0.2, 0.3]
    },
    'training_params': {
        'batch_size': [16, 32, 64],
        'learning_rate': [0.001, 0.0005, 0.0001]
    }
}

class HyperparameterSearch:
    def __init__(self, base_config: Dict, tuning_config: Dict):
        self.base_config = base_config
        self.tuning_config = tuning_config
        self.tuner = HyperparameterTuner()
        
    def optimize(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Run hyperparameter optimization."""
        def objective(trial):
            # Generate parameters
            params = self._generate_params(trial)
            
            # Create trainer with parameters
            trainer = ModelTrainer(params)
            
            # Train and evaluate
            history = trainer.train(train_data, val_data)
            
            return history['val_loss'][-1]
            
        # Run optimization
        best_params = self.tuner.optimize(
            objective,
            n_trials=100,
            timeout=3600  # 1 hour
        )
        
        return best_params
```

## 4. Advanced Validation

```python
class ModelValidator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def validate(self, val_data: pd.DataFrame):
        """Comprehensive model validation."""
        results = {}
        
        # Performance metrics
        results['metrics'] = self._calculate_metrics(val_data)
        
        # Stability analysis
        results['stability'] = self._analyze_stability(val_data)
        
        # Error analysis
        results['errors'] = self._analyze_errors(val_data)
        
        return results
        
    def _calculate_metrics(self, data):
        """Calculate comprehensive metrics."""
        predictions = self.model.predict(data)
        
        return {
            'mse': mean_squared_error(data['target'], predictions),
            'mae': mean_absolute_error(data['target'], predictions),
            'r2': r2_score(data['target'], predictions),
            'ic': information_coefficient(data['target'], predictions)
        }
```

## 5. Performance Optimization

```python
class ModelOptimizer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def optimize_inference(self):
        """Optimize model for inference."""
        # Quantization
        self.model = self._quantize_model()
        
        # Pruning
        self.model = self._prune_model()
        
        # Compilation
        self.model = self._compile_model()
        
        return self.model
        
    def _quantize_model(self):
        """Quantize model for faster inference."""
        return torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
```

## 6. Model Analysis

```python
class ModelAnalyzer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
    def analyze_feature_importance(self):
        """Analyze feature importance."""
        importances = []
        
        # Permutation importance
        for feature in self.data.columns:
            importance = self._calculate_permutation_importance(feature)
            importances.append((feature, importance))
            
        return sorted(importances, key=lambda x: x[1], reverse=True)
        
    def analyze_prediction_confidence(self, predictions):
        """Analyze prediction confidence."""
        return {
            'mean_confidence': predictions['confidence'].mean(),
            'confidence_distribution': predictions['confidence'].hist(),
            'low_confidence_predictions': predictions[
                predictions['confidence'] < 0.5
            ]
        }
```

## Best Practices

1. Data Preparation
```python
# Standardize features
from src.utils.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Handle missing values
from src.utils.preprocessing import handle_missing
data_clean = handle_missing(data, strategy='forward_fill')
```

2. Model Selection
```python
# Cross-validation for model selection
from src.utils.validation import cross_validate_models

models = {
    'lstm': LSTMModel(config),
    'transformer': TransformerModel(config),
    'gru': GRUModel(config)
}

results = cross_validate_models(models, data, n_splits=5)
```

3. Error Analysis
```python
# Analyze prediction errors
from src.utils.analysis import analyze_errors

error_analysis = analyze_errors(
    predictions=model.predict(test_data),
    actuals=test_data['target']
)
```

## Common Issues

1. Overfitting
```python
# Add regularization
config['model']['params']['weight_decay'] = 0.0001
config['model']['params']['dropout'] = 0.3

# Use early stopping
trainer.add_callback(EarlyStopping(patience=10))
```

2. Underfitting
```python
# Increase model capacity
config['model']['params']['hidden_size'] *= 2
config['model']['params']['num_layers'] += 1

# Adjust learning rate
config['training']['learning_rate'] *= 0.1
```

3. Memory Issues
```python
# Use gradient accumulation
config['training']['gradient_accumulation_steps'] = 4

# Enable mixed precision training
config['training']['use_amp'] = True
```

## Next Steps

1. Explore [Advanced Architectures](architectures.md)
2. Learn about [Custom Loss Functions](loss_functions.md)
3. Study [Ensemble Methods](ensembles.md)
4. Review [Deployment Optimization](deployment.md)