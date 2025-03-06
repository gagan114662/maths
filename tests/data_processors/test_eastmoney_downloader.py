"""
Tests for the EastMoney data downloader.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import responses
from unittest.mock import Mock, patch
from FinTSB.data.eastmoney_downloader import EastMoneyDataDownloader, DataValidationError

@pytest.fixture
def config_data():
    """Sample configuration data."""
    return {
        'api': {
            'base_url': 'https://quantapi.eastmoney.com/api/v1',
            'version': 'v1',
            'rate_limit': 60,
            'retry_attempts': 3,
            'timeout': 30
        },
        'categories': [
            {'name': 'extreme', 'datasets': 5},
            {'name': 'fall', 'datasets': 5},
            {'name': 'fluctuation', 'datasets': 5},
            {'name': 'rise', 'datasets': 5}
        ],
        'download': {
            'batch_size': 1,
            'delay_between_requests': 0.1,
            'max_retries': 3,
            'retry_delay': 1,
            'timeout': 30
        },
        'validation': {
            'check_data_quality': True,
            'required_columns': ['open', 'high', 'low', 'close', 'volume'],
            'min_rows': 100,
            'date_range': {
                'start': '2010-01-01',
                'end': '2024-12-31'
            }
        },
        'preprocessing': {
            'normalize': True,
            'fill_missing': True,
            'remove_outliers': True,
            'outlier_std_threshold': 3.0
        },
        'storage': {
            'format': 'pkl',
            'compression': None
        },
        'logging': {
            'level': 'INFO',
            'file': 'test_download.log',
            'max_size': '1MB',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'errors': {
            'max_consecutive_failures': 3,
            'alert_threshold': 0.2
        }
    }

@pytest.fixture
def config_file(tmp_path, config_data):
    """Create temporary config file."""
    config_path = tmp_path / "eastmoney_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    return config_path

@pytest.fixture
def sample_data():
    """Create sample market data."""
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.normal(100, 10, 200),
        'high': np.random.normal(102, 10, 200),
        'low': np.random.normal(98, 10, 200),
        'close': np.random.normal(101, 10, 200),
        'volume': np.random.randint(1000, 10000, 200)
    })
    return data

@pytest.fixture
def downloader(config_file):
    """Create EastMoney downloader instance."""
    return EastMoneyDataDownloader(config_file, api_key="test_key")

def test_initialization(downloader, config_data):
    """Test downloader initialization."""
    assert downloader.api_key == "test_key"
    assert downloader.config['api']['base_url'] == config_data['api']['base_url']
    assert downloader.consecutive_failures == 0

@responses.activate
def test_successful_download(downloader, sample_data):
    """Test successful data download."""
    # Mock API response
    responses.add(
        responses.GET,
        f"{downloader.config['api']['base_url']}/stock/data/extreme/dataset_1",
        json=sample_data.to_dict(orient='records'),
        status=200
    )
    
    data = downloader.download_dataset('extreme', 1)
    assert isinstance(data, pd.DataFrame)
    assert len(data) == len(sample_data)
    assert all(col in data.columns for col in downloader.config['validation']['required_columns'])

def test_data_validation(downloader, sample_data):
    """Test data validation."""
    # Test valid data
    assert downloader.validate_data(sample_data)
    
    # Test missing columns
    invalid_data = sample_data.drop(columns=['volume'])
    with pytest.raises(DataValidationError):
        downloader.validate_data(invalid_data)
    
    # Test insufficient rows
    invalid_data = sample_data.head(10)
    with pytest.raises(DataValidationError):
        downloader.validate_data(invalid_data)

def test_data_preprocessing(downloader, sample_data):
    """Test data preprocessing."""
    processed_data = downloader.preprocess_data(sample_data.copy())
    
    # Check normalization
    numeric_cols = ['open', 'high', 'low', 'close']
    for col in numeric_cols:
        assert abs(processed_data[col].mean()) < 1e-10
        assert abs(processed_data[col].std() - 1.0) < 1e-10
    
    # Check outlier removal
    threshold = downloader.config['preprocessing']['outlier_std_threshold']
    for col in numeric_cols:
        assert all(abs(processed_data[col]) <= threshold)

@responses.activate
def test_error_handling(downloader):
    """Test error handling and retries."""
    # Mock failed API response
    responses.add(
        responses.GET,
        f"{downloader.config['api']['base_url']}/stock/data/extreme/dataset_1",
        status=500
    )
    
    with pytest.raises(Exception) as exc_info:
        for _ in range(downloader.config['errors']['max_consecutive_failures'] + 1):
            downloader.download_dataset('extreme', 1)
    
    assert "Too many consecutive download failures" in str(exc_info.value)

def test_save_dataset(downloader, sample_data, tmp_path):
    """Test dataset saving."""
    # Test successful save
    success = downloader.save_dataset(sample_data, 'extreme', 1, tmp_path)
    assert success
    assert (tmp_path / 'extreme' / 'dataset_1.pkl').exists()
    
    # Test failed save (invalid path)
    invalid_path = Path('/nonexistent/path')
    success = downloader.save_dataset(sample_data, 'extreme', 1, invalid_path)
    assert not success

@responses.activate
def test_download_all(downloader, sample_data, tmp_path):
    """Test downloading all datasets."""
    # Mock API responses for all datasets
    for category in ['extreme', 'fall', 'fluctuation', 'rise']:
        for dataset_num in range(1, 6):
            responses.add(
                responses.GET,
                f"{downloader.config['api']['base_url']}/stock/data/{category}/dataset_{dataset_num}",
                json=sample_data.to_dict(orient='records'),
                status=200
            )
    
    success_counts = downloader.download_all(tmp_path)
    
    # Check success counts
    assert all(count == 5 for count in success_counts.values())
    
    # Check saved files
    for category in ['extreme', 'fall', 'fluctuation', 'rise']:
        category_dir = tmp_path / category
        assert category_dir.exists()
        assert len(list(category_dir.glob('*.pkl'))) == 5

def test_rate_limiting(downloader):
    """Test rate limiting functionality."""
    with patch('time.sleep') as mock_sleep:
        downloader.download_all(Path('/tmp'))
        assert mock_sleep.called
        assert mock_sleep.call_args[0][0] == downloader.config['download']['delay_between_requests']

def test_logging(downloader, tmp_path, caplog):
    """Test logging functionality."""
    log_file = tmp_path / 'test.log'
    downloader.config['logging']['file'] = str(log_file)
    downloader._setup_logger()
    
    # Trigger some logs
    downloader.download_dataset('extreme', 1)
    
    # Check log file exists and contains entries
    assert log_file.exists()
    assert len(caplog.records) > 0