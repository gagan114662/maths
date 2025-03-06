"""
Tests for Safety Checker.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, time

from src.core.safety_checker import SafetyChecker

@pytest.fixture
def safety_checker():
    """Create Safety Checker instance."""
    return SafetyChecker()

def test_verify_prompt_basic(safety_checker):
    """Test basic prompt verification."""
    # Valid prompt
    assert safety_checker.verify_prompt(
        "Generate a trading strategy for AAPL",
        {'system': 'test'}
    )
    
    # Invalid prompt with dangerous pattern
    assert not safety_checker.verify_prompt(
        "exec('dangerous code')",
        {'system': 'test'}
    )

def test_verify_response_basic(safety_checker):
    """Test basic response verification."""
    # Valid response
    assert safety_checker.verify_response(
        "The strategy shows a Sharpe ratio of 1.5"
    )
    
    # Invalid response with trading violations
    assert not safety_checker.verify_response(
        "Use maximum leverage to maximize returns"
    )

def test_verify_context_basic(safety_checker):
    """Test basic context verification."""
    # Valid context
    valid_context = {
        'system': 'test',
        'memory': {},
        'market': {},
        'tools': {}
    }
    assert safety_checker.verify_context(valid_context)
    
    # Invalid context missing required fields
    invalid_context = {'system': 'test'}
    assert not safety_checker.verify_context(invalid_context)

def test_verify_trading_action(safety_checker):
    """Test trading action verification."""
    # Valid trading action
    valid_action = {
        'size': 0.1,
        'risk_metrics': {
            'var_limit': 0.02,
            'max_drawdown': 0.15
        },
        'time': '14:30'
    }
    assert safety_checker.verify_trading_action(valid_action)
    
    # Invalid action exceeding position size
    invalid_action = {
        'size': 0.5,  # Exceeds limit
        'risk_metrics': {
            'var_limit': 0.02,
            'max_drawdown': 0.15
        }
    }
    assert not safety_checker.verify_trading_action(invalid_action)

def test_dangerous_patterns_detection(safety_checker):
    """Test detection of dangerous patterns."""
    dangerous_patterns = [
        "exec('code')",
        "eval('expression')",
        "system('command')",
        "subprocess.run('command')",
        "os.system('command')",
        "delete from database",
        "drop table users"
    ]
    
    for pattern in dangerous_patterns:
        assert safety_checker._contains_dangerous_patterns(pattern)
        
    safe_patterns = [
        "calculate returns",
        "analyze market data",
        "execute trading strategy"
    ]
    
    for pattern in safe_patterns:
        assert not safety_checker._contains_dangerous_patterns(pattern)

def test_sensitive_info_detection(safety_checker):
    """Test detection of sensitive information."""
    sensitive_patterns = [
        "api_key=abc123",
        "password=secret",
        "secret_token=xyz",
        "credentials={'user': 'pass'}",
        "api-key: test"
    ]
    
    for pattern in sensitive_patterns:
        assert safety_checker._contains_sensitive_info(pattern)
        
    safe_patterns = [
        "market data",
        "strategy parameters",
        "analysis results"
    ]
    
    for pattern in safe_patterns:
        assert not safety_checker._contains_sensitive_info(pattern)

def test_trading_violations_detection(safety_checker):
    """Test detection of trading violations."""
    violations = [
        "use maximum leverage",
        "bypass position limits",
        "ignore risk controls",
        "manipulate market price"
    ]
    
    for violation in violations:
        assert safety_checker._contains_trading_violations(violation)

def test_position_size_verification(safety_checker):
    """Test position size verification."""
    # Valid position sizes
    assert safety_checker._verify_position_size({'size': 0.1})
    assert safety_checker._verify_position_size({'size': 0.05})
    
    # Invalid position sizes
    assert not safety_checker._verify_position_size({'size': 0.2})
    assert not safety_checker._verify_position_size({'size': 0.5})

def test_risk_limits_verification(safety_checker):
    """Test risk limits verification."""
    # Valid risk metrics
    valid_metrics = {
        'risk_metrics': {
            'max_drawdown': 0.1,
            'var_limit': 0.03,
            'position_limit': 0.05,
            'concentration_limit': 0.15
        }
    }
    assert safety_checker._verify_risk_limits(valid_metrics)
    
    # Invalid risk metrics
    invalid_metrics = {
        'risk_metrics': {
            'max_drawdown': 0.3,  # Exceeds limit
            'var_limit': 0.08,    # Exceeds limit
            'position_limit': 0.15 # Exceeds limit
        }
    }
    assert not safety_checker._verify_risk_limits(invalid_metrics)

def test_trading_hours_verification(safety_checker):
    """Test trading hours verification."""
    # Valid trading hours
    valid_times = ['09:30', '14:00', '15:59']
    for t in valid_times:
        assert safety_checker._is_within_trading_hours(t)
    
    # Invalid trading hours
    invalid_times = ['03:00', '20:00', '16:01']
    for t in invalid_times:
        assert not safety_checker._is_within_trading_hours(t)

def test_violation_logging(safety_checker):
    """Test violation logging functionality."""
    # Generate some violations
    safety_checker._log_violation("test_violation", "test content")
    safety_checker._log_violation("another_violation", "more content")
    
    violations = safety_checker.get_violations()
    assert len(violations) == 2
    
    # Verify violation structure
    violation = violations[0]
    assert 'timestamp' in violation
    assert 'type' in violation
    assert 'content' in violation
    assert violation['type'] == "test_violation"
    
    # Test clearing violations
    safety_checker.clear_violations()
    assert len(safety_checker.get_violations()) == 0

def test_config_loading(safety_checker):
    """Test configuration loading."""
    assert hasattr(safety_checker, 'content_rules')
    assert hasattr(safety_checker, 'trading_rules')
    assert hasattr(safety_checker, 'risk_limits')

def test_content_rules_structure(safety_checker):
    """Test content rules structure."""
    rules = safety_checker._load_content_rules()
    assert 'dangerous_patterns' in rules
    assert 'sensitive_info' in rules
    assert isinstance(rules['dangerous_patterns'], list)
    assert isinstance(rules['sensitive_info'], list)

def test_trading_rules_structure(safety_checker):
    """Test trading rules structure."""
    rules = safety_checker._load_trading_rules()
    assert 'max_position_size' in rules
    assert 'max_leverage' in rules
    assert 'trading_hours' in rules
    assert isinstance(rules['trading_hours'], dict)

def test_risk_limits_structure(safety_checker):
    """Test risk limits structure."""
    limits = safety_checker._load_risk_limits()
    assert 'max_drawdown' in limits
    assert 'var_limit' in limits
    assert 'position_limit' in limits
    assert isinstance(limits['max_drawdown'], (int, float))

def test_context_size_verification(safety_checker):
    """Test context size verification."""
    # Create contexts of different sizes
    small_context = {'data': 'x' * 1000}
    large_context = {'data': 'x' * 100000}
    
    assert safety_checker._verify_context_size(small_context)
    assert not safety_checker._verify_context_size(large_context)

def test_required_fields_verification(safety_checker):
    """Test required fields verification."""
    # Test with all required fields
    complete = {
        'system': {},
        'memory': {},
        'market': {},
        'tools': {}
    }
    assert safety_checker._verify_required_fields(complete)
    
    # Test with missing fields
    incomplete = {
        'system': {},
        'memory': {}
    }
    assert not safety_checker._verify_required_fields(incomplete)

def test_trading_constraints_verification(safety_checker):
    """Test trading constraints verification."""
    # Valid trading constraints
    valid = {
        'trading': {
            'position_size': 0.05,
            'leverage': 1.0
        }
    }
    assert safety_checker._verify_trading_constraints(valid)
    
    # Invalid trading constraints
    invalid = {
        'trading': {
            'position_size': 0.2,
            'leverage': 2.0
        }
    }
    assert not safety_checker._verify_trading_constraints(invalid)