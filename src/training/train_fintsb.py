"""
Training pipeline integrating FinTSB with enhanced evaluation metrics.
"""
import sys
import os
from pathlib import Path
import yaml
import torch
import logging
from typing import Dict, Any
import pandas as pd
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Add FinTSB to path
FINTSB_PATH = PROJECT_ROOT / "FinTSB"
sys.path.append(str(FINTSB_PATH))

# Import FinTSB modules
from FinTSB.src.models.LSTM import LSTM as LSTMModel
from FinTSB.src.dataset import Dataset as TimeSeriesDataset
from FinTSB import train as fintsb_trainer

# Import our evaluation system
from src.strategies.evaluator import StrategyEvaluator

class EnhancedFinTSBTrainer:
    def __init__(self, config_path: str):
        """
        Initialize enhanced FinTSB trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.evaluator = StrategyEvaluator()
        self.device = torch.device(self.config['training']['device'])
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("EnhancedTrainer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(self.config['logging']['level'])
            
            # Add file handler
            fh = logging.FileHandler(self.config['logging']['save_path'])
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
        return logger

    def _load_datasets(self):
        """Load and prepare datasets from all categories."""
        datasets = {}
        base_path = Path(self.config['data']['base_path'])
        
        for category in self.config['data']['categories']:
            category_data = []
            category_path = base_path / category
            
            for i in range(1, 6):  # 5 datasets per category
                file_path = category_path / f"dataset_{i}.pkl"
                try:
                    with open(file_path, 'rb') as f:
                        data = pd.read_pickle(f)
                    category_data.append(data)
                except Exception as e:
                    self.logger.error(f"Error loading {file_path}: {str(e)}")
                    
            if category_data:
                datasets[category] = pd.concat(category_data, axis=0)
                
        return datasets

    def _prepare_data_loaders(self, datasets: Dict[str, pd.DataFrame]):
        """Prepare data loaders for training."""
        data_loaders = {}
        
        for category, data in datasets.items():
            # Create TimeSeriesDataset
            dataset = TimeSeriesDataset(
                data,
                sequence_length=self.config['data']['sequence_length'],
                prediction_length=self.config['data']['prediction_length']
            )
            
            # Split dataset
            train_size = int(len(dataset) * self.config['data']['train_ratio'])
            val_size = int(len(dataset) * self.config['data']['val_ratio'])
            test_size = len(dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )
            
            # Create data loaders
            data_loaders[category] = {
                'train': torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.config['data']['batch_size'],
                    shuffle=self.config['data']['shuffle'],
                    num_workers=self.config['data']['num_workers']
                ),
                'val': torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=self.config['data']['batch_size'],
                    shuffle=False,
                    num_workers=self.config['data']['num_workers']
                ),
                'test': torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=self.config['data']['batch_size'],
                    shuffle=False,
                    num_workers=self.config['data']['num_workers']
                )
            }
            
        return data_loaders

    def train(self):
        """Execute training pipeline with enhanced evaluation."""
        self.logger.info("Starting enhanced training pipeline")
        
        # Load datasets
        self.logger.info("Loading datasets")
        datasets = self._load_datasets()
        
        # Prepare data loaders
        self.logger.info("Preparing data loaders")
        data_loaders = self._prepare_data_loaders(datasets)
        
        # Initialize model
        model = LSTMModel(**self.config['model']['params']).to(self.device)
        
        # Training loop for each category
        for category, loaders in data_loaders.items():
            self.logger.info(f"Training on {category} dataset")
            
            # Train using FinTSB trainer
            trainer = fintsb_trainer.Trainer(
                model=model,
                train_loader=loaders['train'],
                val_loader=loaders['val'],
                test_loader=loaders['test'],
                config=self.config
            )
            
            # Train and get predictions
            predictions, actuals = trainer.train()
            
            # Calculate returns (simplified for example)
            returns = self._calculate_returns(predictions, actuals)
            
            # Generate trade information
            trades = self._generate_trades(predictions, returns)
            
            # Enhanced evaluation
            evaluation_results = self.evaluator.evaluate_strategy(
                predictions=predictions,
                actuals=actuals,
                returns=returns,
                trades=trades
            )
            
            # Save results
            self._save_results(category, evaluation_results, predictions)
            
            # Log metrics
            self._log_metrics(category, evaluation_results)
            
        self.logger.info("Training pipeline completed")

    def _calculate_returns(self, predictions: pd.DataFrame, 
                         actuals: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy returns based on predictions."""
        # Implementation depends on specific strategy logic
        returns = pd.DataFrame(index=predictions.index)
        
        # Simple long-short strategy based on predictions
        signals = (predictions > 0).astype(float)
        returns['strategy'] = signals * actuals.shift(-1)
        
        return returns

    def _generate_trades(self, predictions: pd.DataFrame, 
                        returns: pd.DataFrame) -> pd.DataFrame:
        """Generate trade information for evaluation."""
        trades = pd.DataFrame()
        
        # Extract trade entry/exit points
        signals = (predictions > 0).astype(float)
        trades['entry_price'] = predictions[signals == 1]
        trades['exit_price'] = predictions[signals == 0]
        trades['returns'] = returns['strategy']
        trades['timestamp'] = predictions.index
        
        return trades

    def _save_results(self, category: str, evaluation_results: Dict[str, Any], 
                     predictions: pd.DataFrame):
        """Save evaluation results and predictions."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config['output']['model_path']).parent
        
        # Save predictions
        predictions_path = output_dir / f"pred_{category}_{timestamp}.pkl"
        predictions.to_pickle(predictions_path)
        
        # Save evaluation results
        results_path = output_dir / f"evaluation_{category}_{timestamp}.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(evaluation_results, f)
            
        self.logger.info(f"Results saved to {output_dir}")

    def _log_metrics(self, category: str, evaluation_results: Dict[str, Any]):
        """Log evaluation metrics."""
        self.logger.info(f"\nEvaluation results for {category}:")
        
        # Log ranking metrics
        self.logger.info("\nRanking Metrics:")
        for metric, value in evaluation_results['ranking_metrics'].items():
            self.logger.info(f"{metric}: {value:.4f}")
            
        # Log portfolio metrics
        self.logger.info("\nPortfolio Metrics:")
        for metric, value in evaluation_results['portfolio_metrics'].items():
            self.logger.info(f"{metric}: {value:.4f}")
            
        # Log error metrics
        self.logger.info("\nError Metrics:")
        for metric, value in evaluation_results['error_metrics'].items():
            self.logger.info(f"{metric}: {value:.4f}")
            
        # Log compliance status
        self.logger.info("\nCompliance Status:")
        for check, status in evaluation_results['ethical_compliance'].items():
            self.logger.info(f"{check}: {'Pass' if status else 'Fail'}")
            
        # Log overall score
        self.logger.info(f"\nOverall Strategy Score: {evaluation_results['overall_score']:.4f}")

def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python train_fintsb.py <config_path>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    trainer = EnhancedFinTSBTrainer(config_path)
    trainer.train()

if __name__ == "__main__":
    main()