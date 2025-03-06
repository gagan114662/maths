# Testing Documentation

## Overview

This directory contains the test suite for the Enhanced Trading Strategy System. The tests are organized by component and include unit tests, integration tests, and end-to-end tests.

## Test Structure

```
tests/
├── conftest.py           # Common test fixtures and utilities
├── requirements-test.txt # Test-specific dependencies
├── README.md            # This file
├── agents/             # Tests for agent components
├── core/              # Tests for core system components
├── data/              # Tests for data handling
├── execution/         # Tests for execution system
├── monitoring/        # Tests for monitoring system
├── strategies/        # Tests for trading strategies
├── web/              # Tests for web interface
└── integration/      # Integration tests
```

## Running Tests

### Quick Start

1. Install test dependencies:
```bash
pip install -r tests/requirements-test.txt
```

2. Run all tests:
```bash
pytest
```

3. Run specific test categories:
```bash
pytest tests/agents/          # Run agent tests
pytest tests/integration/     # Run integration tests
pytest -m "not slow"         # Skip slow tests
```

### Test Categories

- **Unit Tests**: Basic component testing
- **Integration Tests**: Component interaction testing
- **End-to-End Tests**: Complete system workflow testing
- **Performance Tests**: System performance and scalability testing

### Test Markers

- `@pytest.mark.slow`: Tests that take a long time to run
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.web`: Web interface tests
- `@pytest.mark.monitoring`: Monitoring system tests
- `@pytest.mark.strategy`: Strategy-related tests
- `@pytest.mark.security`: Security-related tests

### Environment Variables

- `TEST_MODE`: Set to 'true' for test environment
- `CONFIG_PATH`: Path to test configuration file
- `LOG_LEVEL`: Logging level for tests

### Configuration

Test configuration is managed through:
1. `pytest.ini`: Global test settings
2. `tests/config/test_config.yaml`: Test-specific configuration
3. `conftest.py`: Common fixtures and utilities

## Writing Tests

### Guidelines

1. **Test Organization**
   - Place tests in appropriate directories
   - Use descriptive test names
   - Follow the AAA pattern (Arrange, Act, Assert)

2. **Fixtures**
   - Use fixtures for common setup
   - Keep fixtures focused and minimal
   - Document fixture purpose and usage

3. **Mocking**
   - Mock external dependencies
   - Use `pytest-mock` for mocking
   - Document mock behavior

4. **Assertions**
   - Use specific assertions
   - Include meaningful error messages
   - Test both positive and negative cases

### Example Test

```python
@pytest.mark.integration
async def test_strategy_generation(agent_factory, sample_market_data):
    """Test strategy generation workflow."""
    # Arrange
    generator = agent_factory.create_agent('generation')
    
    # Act
    strategy = await generator.generate_strategy({
        'type': 'momentum',
        'parameters': {'lookback': 20}
    })
    
    # Assert
    assert strategy is not None
    assert strategy.type == 'momentum'
    assert strategy.parameters['lookback'] == 20
```

## Test Coverage

Coverage reports are generated automatically when running tests. View the report:

```bash
pytest --cov=src --cov-report=html
```

The report will be available in `htmlcov/index.html`.

## Continuous Integration

Tests are automatically run in CI/CD pipeline:

1. On pull requests
2. On merges to main branch
3. Nightly for long-running tests

### CI Configuration

- All tests must pass before merge
- Coverage must not decrease
- Performance tests run only on scheduled builds

## Troubleshooting

### Common Issues

1. **Test Database Issues**
   - Ensure test database is clean
   - Check database permissions
   - Verify connection settings

2. **Async Test Issues**
   - Use proper async fixtures
   - Handle event loop correctly
   - Check for unhandled coroutines

3. **Mock Issues**
   - Verify mock configuration
   - Check mock return values
   - Ensure proper cleanup

### Debug Tools

1. **Logging**
   - Set `LOG_LEVEL=DEBUG`
   - Check `logs/test.log`
   - Use `pytest -vv` for verbose output

2. **Debugger**
   - Use `pytest --pdb`
   - Set breakpoints with `breakpoint()`
   - Use VS Code debugging

## Contributing

1. Add tests for new features
2. Maintain or improve coverage
3. Follow test naming conventions
4. Update test documentation

## Support

For questions or issues:
1. Check existing test documentation
2. Review test examples
3. Create GitHub issue
4. Contact development team