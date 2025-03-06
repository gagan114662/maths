"""
Tests for the credential management system.
"""
import pytest
from pathlib import Path
import os
from src.utils.credentials import CredentialManager, setup_credentials
from unittest.mock import patch

@pytest.fixture
def temp_env_file(tmp_path):
    """Create a temporary environment file."""
    env_file = tmp_path / ".env"
    return env_file

@pytest.fixture
def credential_manager(temp_env_file):
    """Create a CredentialManager instance with temporary env file."""
    return CredentialManager(temp_env_file)

@pytest.fixture
def sample_credentials():
    """Sample valid credentials."""
    return {
        'EASTMONEY_API_KEY': 'a' * 32,  # 32-character API key
        'EASTMONEY_API_SECRET': 'b' * 64  # 64-character API secret
    }

def test_initialization(credential_manager, temp_env_file):
    """Test CredentialManager initialization."""
    assert credential_manager.env_file == temp_env_file
    assert isinstance(credential_manager.REQUIRED_VARS, dict)
    assert 'EASTMONEY_API_KEY' in credential_manager.REQUIRED_VARS

def test_create_example_env(credential_manager):
    """Test creation of example environment file."""
    credential_manager.create_example_env()
    example_file = credential_manager.env_file.with_suffix('.env.example')
    
    assert example_file.exists()
    content = example_file.read_text()
    
    # Check that all required variables are included
    for var in credential_manager.REQUIRED_VARS:
        assert var in content
        assert f"{var}=your_{var.lower()}_here" in content

def test_set_and_get_credentials(credential_manager, sample_credentials):
    """Test setting and getting credentials."""
    # Set credentials
    for key, value in sample_credentials.items():
        credential_manager.set_credential(key, value)
    
    # Get credentials
    credentials = credential_manager.get_credentials()
    
    # Verify credentials
    assert credentials == sample_credentials

def test_invalid_credential_key(credential_manager):
    """Test handling of invalid credential keys."""
    with pytest.raises(ValueError):
        credential_manager.set_credential('INVALID_KEY', 'value')

def test_missing_credentials(credential_manager):
    """Test handling of missing credentials."""
    with pytest.raises(ValueError):
        credential_manager.get_credentials()

def test_credential_validation(credential_manager, sample_credentials):
    """Test credential format validation."""
    # Set valid credentials
    for key, value in sample_credentials.items():
        credential_manager.set_credential(key, value)
    
    # Validate credentials
    validation = credential_manager.validate_credentials()
    assert all(validation.values())
    
    # Test invalid API key format
    credential_manager.set_credential('EASTMONEY_API_KEY', 'too_short')
    validation = credential_manager.validate_credentials()
    assert not validation['EASTMONEY_API_KEY']

def test_check_credentials(credential_manager, sample_credentials):
    """Test credential availability checking."""
    assert not credential_manager.check_credentials()
    
    # Set credentials
    for key, value in sample_credentials.items():
        credential_manager.set_credential(key, value)
    
    assert credential_manager.check_credentials()

@patch('builtins.input')
def test_interactive_setup(mock_input, temp_env_file, sample_credentials):
    """Test interactive credential setup."""
    # Mock user input
    mock_input.side_effect = sample_credentials.values()
    
    # Run interactive setup
    manager = setup_credentials(interactive=True)
    
    # Verify credentials were set
    assert manager.check_credentials()
    credentials = manager.get_credentials()
    assert credentials == sample_credentials

def test_non_interactive_setup(temp_env_file):
    """Test non-interactive credential setup."""
    with pytest.raises(ValueError) as exc_info:
        setup_credentials(interactive=False)
    
    assert "Missing credentials" in str(exc_info.value)
    assert temp_env_file.with_suffix('.env.example').exists()

def test_env_file_persistence(credential_manager, sample_credentials):
    """Test that credentials persist in .env file."""
    # Set credentials
    for key, value in sample_credentials.items():
        credential_manager.set_credential(key, value)
    
    # Create new manager instance with same file
    new_manager = CredentialManager(credential_manager.env_file)
    
    # Verify credentials are loaded
    assert new_manager.check_credentials()
    assert new_manager.get_credentials() == sample_credentials

def test_update_existing_credential(credential_manager, sample_credentials):
    """Test updating existing credentials."""
    # Set initial credentials
    key = 'EASTMONEY_API_KEY'
    credential_manager.set_credential(key, sample_credentials[key])
    
    # Update credential
    new_value = 'c' * 32
    credential_manager.set_credential(key, new_value)
    
    # Verify update
    assert credential_manager.get_credentials()[key] == new_value

def test_credential_security(credential_manager, sample_credentials):
    """Test credential security measures."""
    # Set credentials
    for key, value in sample_credentials.items():
        credential_manager.set_credential(key, value)
    
    # Check file permissions
    env_file = credential_manager.env_file
    if os.name != 'nt':  # Skip on Windows
        assert oct(env_file.stat().st_mode)[-3:] in ('600', '640', '644')
    
    # Check content security
    content = env_file.read_text()
    assert 'export' not in content  # Shouldn't expose credentials as exports
    assert '#' in content  # Should include comments