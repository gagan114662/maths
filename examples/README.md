# Trading Strategy Agent System Examples

This directory contains example scripts and configurations for using the Enhanced Trading Strategy System.

## Quick Start

1. Run the example system:
```bash
./run_agent_system.py
```

2. Run with custom configuration:
```bash
./run_agent_system.py --config config/example_config.yaml
```

## Configuration

### Example Configuration
A complete example configuration is provided in `config/example_config.yaml`. This includes:
- System settings
- Market configuration
- Strategy parameters
- Risk management settings
- Agent configurations
- Pipeline settings
- Monitoring parameters

### Customization
To customize the system:
1. Copy `config/example_config.yaml` to `config/custom_config.yaml`
2. Modify parameters as needed
3. Run with your custom configuration

## Components

### Agent Pipeline
The example demonstrates a complete agent pipeline:
1. Generation Agent: Creates trading strategies
2. Backtesting Agent: Evaluates strategies
3. Risk Assessment Agent: Analyzes risks
4. Ranking Agent: Compares strategies
5. Evolution Agent: Improves strategies
6. Meta-Review Agent: Provides system insights

### Strategy Templates
Pre-defined strategy templates include:
- Momentum strategies
- Mean reversion strategies
- Trend following strategies
- Volatility-based strategies

## Output

The system generates:
1. Strategy evaluations
2. Performance metrics
3. Risk assessments
4. System recommendations
5. Analysis reports

### Results Directory
Results are saved in the `results` directory:
```
results/
├── strategies/       # Generated strategies
├── backtests/       # Backtest results
├── analysis/        # System analysis
├── metrics/         # Performance metrics
└── reports/         # Generated reports
```

## Configuration Options

### System Settings
```yaml
system:
  name: "trading_system_example"
  mode: "development"
  log_level: "INFO"
```

### Market Configuration
```yaml
market:
  symbols: ["AAPL", "MSFT", "GOOGL"]
  timeframe: "1d"
  lookback: 90
```

### Risk Management
```yaml
risk:
  max_drawdown: 0.2
  var_limit: 0.05
  position_limit: 0.1
```

### Agent Settings
```yaml
agents:
  generation:
    max_strategies: 20
    batch_size: 5
  
  backtesting:
    initial_capital: 100000
    trading_costs: true
```

## Examples

### Basic Usage
```python
from src.agents import factory, DEFAULT_PIPELINE

# Create agent pipeline
pipeline_agents = factory.create_agent_pipeline(DEFAULT_PIPELINE)

# Process data
results = await factory.process_pipeline(pipeline_agents, input_data)
```

### Custom Pipeline
```python
pipeline_config = {
    'agents': [
        {
            'type': 'generation',
            'name': 'custom_generator',
            'config': {'max_strategies': 5}
        },
        {
            'type': 'backtesting',
            'name': 'custom_tester',
            'config': {'initial_capital': 50000}
        }
    ]
}

pipeline_agents = factory.create_agent_pipeline(pipeline_config)
```

### Strategy Development
```python
# Run complete strategy development cycle
results = await run_strategy_development()

# Extract best strategies
strategies = results['strategies']

# Analyze results
analysis = results['analysis']
```

## Monitoring

### Performance Metrics
Monitor system performance:
```bash
tail -f logs/performance.log
```

### Resource Usage
Monitor resource usage:
```bash
./monitor_system.py --resources
```

### Strategy Evolution
Track strategy evolution:
```bash
./monitor_system.py --evolution
```

## Troubleshooting

### Common Issues

1. Memory Usage
```yaml
system:
  max_memory: "4G"  # Increase if needed
```

2. Performance
```yaml
pipeline:
  parallel_execution: true
  checkpointing: true
```

3. Error Handling
```yaml
pipeline:
  error_tolerance: 0.1
  retry_attempts: 3
```

### Debug Mode
Enable debug mode:
```yaml
development:
  debug: true
  profile: true
```

## Best Practices

1. Start Small
   - Begin with few strategies
   - Use shorter timeframes
   - Limited symbol set

2. Iterative Development
   - Test configuration changes
   - Monitor performance
   - Gradual optimization

3. Resource Management
   - Enable checkpointing
   - Monitor memory usage
   - Use parallel execution

4. Risk Control
   - Set conservative limits
   - Enable all safety checks
   - Monitor continuously

## Support

For issues and questions:
1. Check documentation
2. Review error logs
3. Open GitHub issue
4. Contact development team