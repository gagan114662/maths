# Configuration Reference

## Overview

This document details all available configuration options for the Enhanced Trading Strategy System.

## Core Configuration Structure

```yaml
# config.yaml

# Model Configuration
model:
  type: "lstm"  # Options: lstm, transformer, gru, attention
  params:
    input_size: 64
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
    bidirectional: true
    attention: true  # Only for attention-based models
    
# Training Configuration
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  optimizer:
    type: "adam"  # Options: adam, sgd, rmsprop
    weight_decay: 0.0001
  scheduler:
    type: "reduce_on_plateau"  # Options: reduce_on_plateau, cosine, step
    patience: 5
    factor: 0.5
  device: "cuda"  # Options: cuda, cpu
  gradient_clipping: 1.0
  
# Validation Configuration
validation:
  split_ratio: 0.2
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001
  cross_validation:
    enabled: false
    folds: 5
    
# Data Configuration
data:
  features:
    - close
    - volume
    - volatility
  target: returns
  window_size: 20
  stride: 1
  normalization:
    type: "standard"  # Options: standard, minmax, robust
    per_channel: true
  augmentation:
    enabled: false
    methods:
      - type: "noise"
        std: 0.1
      - type: "timeshift"
        max_shift: 5
        
# Risk Management Configuration
risk:
  position_limits:
    max_position: 0.1  # 10% of portfolio
    max_leverage: 1.0
  drawdown:
    max_drawdown: 0.2  # 20% maximum drawdown
    evaluation_window: 252  # trading days
  var:
    confidence: 0.95
    horizon: 1
  volatility:
    max_vol: 0.3
    vol_window: 20
    
# Performance Configuration
performance:
  metrics:
    - sharpe_ratio
    - sortino_ratio
    - max_drawdown
    - win_rate
  targets:
    sharpe_ratio: 1.0
    max_drawdown: -0.2
    win_rate: 0.55
    
# Agent Configuration
agents:
  volatility:
    enabled: true
    window_size: 20
    threshold: 0.1
  signal:
    enabled: true
    smoothing_window: 5
  strategy:
    enabled: true
    position_sizing: dynamic
    
# Strategy Configuration
strategy:
  type: "momentum"  # Options: momentum, mean_reversion, ml_based
  parameters:
    entry_threshold: 0.5
    exit_threshold: -0.2
    holding_period: 5
  filters:
    min_volume: 1000000
    min_price: 5.0
    
# System Configuration
system:
  logging:
    level: "INFO"
    file: "logs/system.log"
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  cache:
    enabled: true
    directory: "cache/"
    max_size: "1GB"
  debugging:
    verbose: false
    profile: false
    
# API Configuration
api:
  eastmoney:
    base_url: "https://quantapi.eastmoney.com/api/v1"
    rate_limit: 60  # requests per minute
    timeout: 30
  database:
    type: "postgresql"
    host: "localhost"
    port: 5432
    name: "trading_db"
```

## Configuration Categories

### 1. Model Configuration
Controls model architecture and parameters:
```yaml
model:
  type: "lstm"
  params:
    input_size: 64
    hidden_size: 128
```

### 2. Training Configuration
Defines training behavior:
```yaml
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
```

### 3. Data Configuration
Specifies data processing:
```yaml
data:
  features: [close, volume]
  window_size: 20
```

### 4. Risk Management
Sets risk limits and controls:
```yaml
risk:
  position_limits:
    max_position: 0.1
```

## Environment Variables

Required environment variables:
```bash
# API Credentials
EASTMONEY_API_KEY=your_key
EASTMONEY_API_SECRET=your_secret

# Database Connection
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=user
DB_PASSWORD=password
```

## Configuration Files Location

Default configuration locations:
```
project_root/
├── configs/
│   ├── default.yaml     # Default configuration
│   ├── production.yaml  # Production settings
│   └── development.yaml # Development settings
```

## Configuration Validation

Configuration is validated using:
```python
from src.utils.config import validate_config

# Validate configuration
config = validate_config('configs/default.yaml')
```

## Loading Configuration

```python
from src.utils.config import load_config

# Load with environment overrides
config = load_config('configs/default.yaml', env_prefix='TRADING_')

# Load with multiple files
config = load_config([
    'configs/default.yaml',
    'configs/local.yaml'
])
```

## Modifying Configuration

### 1. Runtime Updates
```python
from src.utils.config import update_config

# Update specific values
update_config(config, {
    'training.batch_size': 64,
    'model.params.dropout': 0.3
})
```

### 2. Environment Overrides
```bash
# Override via environment variables
export TRADING_TRAINING_BATCH_SIZE=64
export TRADING_MODEL_DROPOUT=0.3
```

## Best Practices

1. Version Control
   - Keep configurations in version control
   - Use environment variables for secrets
   - Document all changes

2. Environment Separation
   - Use different configs for dev/prod
   - Keep sensitive data separate
   - Use environment overrides

3. Validation
   - Validate all configurations
   - Check for required fields
   - Verify value ranges

4. Documentation
   - Document all options
   - Provide example values
   - Include validation rules

## Common Issues

1. Missing Fields
```python
# Check for required fields
from src.utils.config import check_required_fields
check_required_fields(config, required_fields)
```

2. Invalid Values
```python
# Validate value ranges
from src.utils.config import validate_ranges
validate_ranges(config, valid_ranges)
```

3. Type Mismatches
```python
# Convert types
from src.utils.config import convert_types
config = convert_types(config, type_mapping)