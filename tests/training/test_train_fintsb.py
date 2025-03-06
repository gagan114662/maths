"""
Tests for the FinTSB training pipeline.
"""
import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from src.training.train_fintsb import EnhancedFinTSBTrainer

@pytest.fixture
def sample_config(tmp_path):
    """Create a sample configuration for testing."""
    config = {
        'data': {
            'base_path': str(tmp_path / 'data'),
            'categories': ['extreme', 'fall', 'fluctuation', 'rise'],
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'sequence_length': 10,
            'prediction_length': 5,
            'batch_size': 32,
            'num_workers': 0,
            'shuffle': True
        },
        'model': {
            'params': {
                'input_size': 64,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2
            }
        },
        'training': {
            'device': 'cpu',
            'epochs': 1
        },
        'logging': {
            'level': 'INFO',
            'save_path': str(tmp_path / 'train.log')
        },
        'output': {
            'model_path': str(tmp_path / 'model.bin')
        }
    }
    
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    return config_path

@pytest.fixture
def sample_data(tmp_path):
    """Create sample training data."""
    data_dir = tmp_path / 'data'
    for category in ['extreme', 'fall', 'fluctuation', 'rise']:
        category_dir = data_dir / category
        category_dir.mkdir(parents=True)
        
        # Create sample datasets
        for i in range(1, 6):
            data = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, 100),
                'feature_2': np.random.normal(0, 1, 100),
                'target': np.random.normal(0, 1, 100)
            })
            
            file_path = category_dir / f'dataset_{i}.pkl'
            data.to_pickle(file_path)
    
    return data_dir

def test_trainer_initialization(sample_config):
    """Test trainer initialization."""
    trainer = EnhancedFinTSBTrainer(sample_config)
    assert trainer is not None
    assert trainer.device == torch.device('cpu')

def test_config_loading(sample_config):
    """Test configuration loading."""
    trainer = EnhancedFinTSBTrainer(sample_config)
    assert trainer.config['data']['train_ratio'] == 0.7
    assert trainer.config['model']['params']['hidden_size'] == 128

@pytest.mark.requires_data
def test_data_loading(sample_config, sample_data):
    """Test data loading functionality."""
    trainer = EnhancedFinTSBTrainer(sample_config)
    datasets = trainer._load_datasets()
    
    assert len(datasets) == 4  # extreme, fall, fluctuation, rise
    assert all(isinstance(data, pd.DataFrame) for data in datasets.values())

@pytest.mark.requires_data
def test_data_loader_creation(sample_config, sample_data):
    """Test creation of data loaders."""
    trainer = EnhancedFinTSBTrainer(sample_config)
    datasets = trainer._load_datasets()
    data_loaders = trainer._prepare_data_loaders(datasets)
    
    assert len(data_loaders) == 4  # extreme, fall, fluctuation, rise
    for category_loaders in data_loaders.values():
        assert 'train' in category_loaders
        assert 'val' in category_loaders
        assert 'test' in category_loaders

@pytest.mark.slow
@pytest.mark.requires_data
def test_training_execution(sample_config, sample_data):
    """Test complete training execution."""
    trainer = EnhancedFinTSBTrainer(sample_config)
    
    try:
        trainer.train()
        assert True  # Training completed without errors
    except Exception as e:
        pytest.fail(f"Training failed with error: {str(e)}")

def test_results_saving(sample_config, tmp_path):
    """Test saving of results."""
    trainer = EnhancedFinTSBTrainer(sample_config)
    
    # Create sample results
    predictions = pd.DataFrame({
        'pred': np.random.normal(0, 1, 100)
    })
    
    evaluation_results = {
        'ranking_metrics': {'ic_mean': 0.1},
        'portfolio_metrics': {'sharpe_ratio': 1.5},
        'error_metrics': {'rmse': 0.1},
        'ethical_compliance': {'no_market_manipulation': True},
        'overall_score': 0.8
    }
    
    # Test saving
    trainer._save_results('test_category', evaluation_results, predictions)
    
    # Check if files were created
    output_dir = Path(trainer.config['output']['model_path']).parent
    saved_files = list(output_dir.glob('*'))
    assert len(saved_files) > 0

def test_metric_logging(sample_config, caplog):
    """Test metric logging functionality."""
    trainer = EnhancedFinTSBTrainer(sample_config)
    
    evaluation_results = {
        'ranking_metrics': {'ic_mean': 0.1},
        'portfolio_metrics': {'sharpe_ratio': 1.5},
        'error_metrics': {'rmse': 0.1},
        'ethical_compliance': {'no_market_manipulation': True},
        'overall_score': 0.8
    }
    
    trainer._log_metrics('test_category', evaluation_results)
    
    # Check if metrics were logged
    assert 'Evaluation results for test_category' in caplog.text
    assert 'ic_mean: 0.1000' in caplog.text
    assert 'sharpe_ratio: 1.5000' in caplog.text

def test_error_handling(sample_config):
    """Test error handling in training pipeline."""
    trainer = EnhancedFinTSBTrainer(sample_config)
    
    # Test with non-existent data path
    trainer.config['data']['base_path'] = 'non_existent_path'
    datasets = trainer._load_datasets()
    assert len(datasets) == 0  # Should handle missing data gracefully