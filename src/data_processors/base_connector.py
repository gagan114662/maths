"""
Base data connector class for handling different data sources.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd
from pathlib import Path

class BaseDataConnector(ABC):
    def __init__(self, data_path: Path):
        """
        Initialize the data connector.
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = data_path
        self.current_data = None
        self._validate_data_path()

    def _validate_data_path(self) -> None:
        """Validate that the data path exists."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

    @abstractmethod
    def load_data(self, 
                  symbols: Union[str, List[str]], 
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  timeframe: str = "1d") -> pd.DataFrame:
        """
        Load data for given symbols and date range.
        
        Args:
            symbols: Single symbol or list of symbols to load
            start_date: Start date for data loading (YYYY-MM-DD)
            end_date: End date for data loading (YYYY-MM-DD)
            timeframe: Data timeframe (e.g., "1d", "1h", "5m")
            
        Returns:
            DataFrame with the loaded data
        """
        pass

    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data according to FinTSB standards.
        
        Args:
            data: Raw data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        pass

    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data at the stock dimension for each trading day.
        
        Args:
            data: DataFrame to normalize
            
        Returns:
            Normalized DataFrame
        """
        # Implement standard normalization across all data sources
        result = data.copy()
        
        # Get numeric columns for normalization
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        
        # Normalize each numeric column
        for col in numeric_cols:
            mean = result[col].mean()
            std = result[col].std()
            if std != 0:  # Avoid division by zero
                result[col] = (result[col] - mean) / std
                
        return result

    def assess_data_quality(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Assess data quality using sequence-based metrics.
        
        Args:
            data: DataFrame to assess
            
        Returns:
            Dictionary containing quality metrics
        """
        metrics = {}
        
        # Missing data percentage
        metrics['missing_pct'] = data.isnull().mean().mean() * 100
        
        # Data staleness (percentage of unchanged values)
        for col in data.select_dtypes(include=['float64', 'int64']).columns:
            metrics[f'{col}_staleness'] = (data[col].pct_change() == 0).mean() * 100
            
        # Basic statistical metrics
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        metrics['skewness'] = numeric_data.skew().mean()
        metrics['kurtosis'] = numeric_data.kurtosis().mean()
        
        return metrics

    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in the data.
        
        Args:
            data: DataFrame to check for anomalies
            
        Returns:
            DataFrame with anomaly flags
        """
        result = data.copy()
        
        # Add anomaly detection column
        result['is_anomaly'] = False
        
        for col in data.select_dtypes(include=['float64', 'int64']).columns:
            # Calculate z-scores
            z_scores = (data[col] - data[col].mean()) / data[col].std()
            
            # Mark values beyond 3 standard deviations as anomalies
            result.loc[abs(z_scores) > 3, 'is_anomaly'] = True
            
        return result

    def categorize_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize data into movement patterns.
        
        Args:
            data: DataFrame to categorize
            
        Returns:
            DataFrame with pattern categories
        """
        result = data.copy()
        
        # Calculate daily returns
        if 'close' in result.columns:
            result['returns'] = result['close'].pct_change()
            
            # Define pattern categories
            result['pattern'] = 'normal'
            
            # Uptrend: Positive returns exceeding 2%
            result.loc[result['returns'] > 0.02, 'pattern'] = 'uptrend'
            
            # Downtrend: Negative returns exceeding -2%
            result.loc[result['returns'] < -0.02, 'pattern'] = 'downtrend'
            
            # High volatility: Returns exceeding ±3%
            result.loc[abs(result['returns']) > 0.03, 'pattern'] = 'volatile'
            
            # Black swan: Returns exceeding ±10%
            result.loc[abs(result['returns']) > 0.10, 'pattern'] = 'black_swan'
            
        return result

    def get_available_timeframes(self) -> List[str]:
        """Get list of available timeframes in the data directory."""
        return [d.name for d in self.data_path.iterdir() if d.is_dir()]

    def get_available_symbols(self, timeframe: str = "1d") -> List[str]:
        """Get list of available symbols for a given timeframe."""
        timeframe_path = self.data_path / timeframe
        if not timeframe_path.exists():
            return []
        return [f.stem for f in timeframe_path.glob("*.csv")]