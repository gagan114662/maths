"""
Tests for the project setup validation script.
"""
import pytest
from pathlib import Path
import sys
import yaml
import shutil
from unittest.mock import patch, MagicMock
import validate_setup

@pytest.fixture
def temp_project_root(tmp_path):
    """Create temporary project structure."""
    # Create main directories
    (tmp_path / "data/ibkr/1d").mkdir(parents=True)
    (tmp_path / "data/kraken/1d").mkdir(parents=True)
    (tmp_path / "data/stocksymbolslists").mkdir(parents=True)
    (tmp_path / "FinTSB/src").mkdir(parents=True)
    (tmp_path / "mathematricks/src").mkdir(parents=True)
    
    # Create config files
    eastmoney_config = {
        'api': {'base_url': 'https://test.com'},
        'categories': [{'name': 'test', 'datasets': 1}]
    }
    with open(tmp_path / "FinTSB/data/eastmoney_config.yaml", 'w') as f:
        yaml.dump(eastmoney_config, f)
        
    lstm_config = {
        'model': {'type': 'lstm'},
        'training': {'epochs': 10}
    }
    with open(tmp_path / "FinTSB/configs/fintsb_lstm.yaml", 'w') as f:
        yaml.dump(lstm_config, f)
        
    # Create requirements.txt
    with open(tmp_path / "requirements.txt", 'w') as f:
        f.write("pytest>=6.0.0\npandas>=1.0.0\n")
    
    return tmp_path

def test_check_python_version():
    """Test Python version checking."""
    with patch('sys.version_info', (3, 8, 0)):
        assert validate_setup.check_python_version()
        
    with patch('sys.version_info', (3, 7, 0)):
        assert not validate_setup.check_python_version()

def test_check_dependencies(temp_project_root):
    """Test dependency checking."""
    with patch('pkg_resources.require') as mock_require:
        # Test when all dependencies are met
        mock_require.return_value = True
        assert validate_setup.check_dependencies()
        
        # Test with missing dependency
        mock_require.side_effect = [True, Exception("Missing package")]
        assert not validate_setup.check_dependencies()

def test_check_frameworks(temp_project_root):
    """Test framework availability checking."""
    fintsb_ok, mathematricks_ok = validate_setup.check_frameworks()
    assert fintsb_ok
    assert mathematricks_ok
    
    # Test with missing framework
    shutil.rmtree(temp_project_root / "FinTSB")
    fintsb_ok, mathematricks_ok = validate_setup.check_frameworks()
    assert not fintsb_ok
    assert mathematricks_ok

def test_check_data_directories(temp_project_root):
    """Test data directory checking."""
    assert validate_setup.check_data_directories()
    
    # Test with missing directory
    shutil.rmtree(temp_project_root / "data/ibkr")
    assert not validate_setup.check_data_directories()

def test_check_configurations(temp_project_root):
    """Test configuration file validation."""
    assert validate_setup.check_configurations()
    
    # Test with invalid YAML
    with open(temp_project_root / "FinTSB/data/eastmoney_config.yaml", 'w') as f:
        f.write("invalid: yaml: :")
    assert not validate_setup.check_configurations()

@patch('src.utils.credentials.CredentialManager')
def test_check_credentials(mock_credential_manager):
    """Test credential validation."""
    # Test with valid credentials
    instance = mock_credential_manager.return_value
    instance.check_credentials.return_value = True
    assert validate_setup.check_credentials()
    
    # Test with missing credentials
    instance.check_credentials.return_value = False
    assert not validate_setup.check_credentials()
    
    # Test with error
    instance.check_credentials.side_effect = Exception("Error")
    assert not validate_setup.check_credentials()

def test_run_tests():
    """Test running project tests."""
    with patch('subprocess.run') as mock_run:
        # Test successful test run
        mock_run.return_value = MagicMock(returncode=0)
        assert validate_setup.run_tests()
        
        # Test failed test run
        mock_run.return_value = MagicMock(returncode=1)
        assert not validate_setup.run_tests()
        
        # Test subprocess error
        mock_run.side_effect = Exception("Error")
        assert not validate_setup.run_tests()

def test_main_success(temp_project_root):
    """Test main function with all checks passing."""
    with patch.multiple(
        validate_setup,
        check_python_version=MagicMock(return_value=True),
        check_dependencies=MagicMock(return_value=True),
        check_frameworks=MagicMock(return_value=(True, True)),
        check_data_directories=MagicMock(return_value=True),
        check_configurations=MagicMock(return_value=True),
        check_credentials=MagicMock(return_value=True),
        run_tests=MagicMock(return_value=True)
    ):
        with pytest.raises(SystemExit) as exc_info:
            validate_setup.main()
        assert exc_info.value.code == 0

def test_main_failure(temp_project_root):
    """Test main function with some checks failing."""
    with patch.multiple(
        validate_setup,
        check_python_version=MagicMock(return_value=True),
        check_dependencies=MagicMock(return_value=False),  # Failing check
        check_frameworks=MagicMock(return_value=(True, True)),
        check_data_directories=MagicMock(return_value=True),
        check_configurations=MagicMock(return_value=True),
        check_credentials=MagicMock(return_value=True),
        run_tests=MagicMock(return_value=True)
    ):
        with pytest.raises(SystemExit) as exc_info:
            validate_setup.main()
        assert exc_info.value.code == 1

def test_logging_output(caplog):
    """Test logging output."""
    with patch.multiple(
        validate_setup,
        check_python_version=MagicMock(return_value=True),
        check_dependencies=MagicMock(return_value=True),
        check_frameworks=MagicMock(return_value=(True, True)),
        check_data_directories=MagicMock(return_value=True),
        check_configurations=MagicMock(return_value=True),
        check_credentials=MagicMock(return_value=True),
        run_tests=MagicMock(return_value=True)
    ):
        try:
            validate_setup.main()
        except SystemExit:
            pass
            
        assert "Starting project validation..." in caplog.text
        assert "All checks passed!" in caplog.text