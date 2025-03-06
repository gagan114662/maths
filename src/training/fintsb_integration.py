"""
FinTSB Integration Component - Comprehensive financial time series forecasting pipeline.
This module provides integration with the FinTSB framework, implementing all pipeline components:
1. Data Preprocessing with EastMoney and other data sources
2. Feature Construction and Selection
3. Multiple prediction models (LSTM, GRU, Transformer, etc.)
4. Ensemble Learning
5. Portfolio Construction with risk management
6. Financial domain knowledge integration
7. Forecasting explanations
"""
import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import logging
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Add FinTSB to path
FINTSB_PATH = PROJECT_ROOT / "FinTSB"
sys.path.append(str(FINTSB_PATH))

# Import FinTSB modules (will be used when available)
from FinTSB.src.dataset import Dataset as TimeSeriesDataset
# Import different model architectures
from FinTSB.src.models.LSTM import LSTM
from FinTSB.src.models.GRU import GRU
from FinTSB.src.models.TCN import TCN
from FinTSB.src.models.TimesNet import TimesNet
from FinTSB.src.models.Transformer import Transformer
from FinTSB.src.models.Crossformer import Crossformer
from FinTSB.src.models.PatchTST import PatchTST
from FinTSB.src.models.DiffStock import DiffStock
from FinTSB.src.models.TimeBridge import TimeBridge
from FinTSB.src.models.TimeMixer import TimeMixer
from FinTSB.src.models.GCN import GCN
from FinTSB.src.models.GAT import GAT
from FinTSB.src.models.SEGRNN import SEGRNN
from FinTSB.src.models.WFTNet import WFTNet
from FinTSB.src.models.PDF import PDF
from FinTSB.src.models.Mamba import Mamba

FINTSB_AVAILABLE = True

# Logger setup
logger = logging.getLogger("FinTSB_Integration")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class FinancialFeatureProcessor:
    """
    Component for financial feature construction and selection.
    Creates a rich set of features used in financial analysis and selects the most relevant ones.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature processor.
        
        Args:
            config: Configuration dictionary containing feature engineering parameters
        """
        self.config = config
        self.feature_config = config.get('feature_engineering', {})
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = []
        
    def generate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical indicators for financial time series.
        
        Args:
            data: DataFrame with at minimum OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                # Try case-insensitive match
                matches = [c for c in df.columns if c.lower() == col]
                if matches:
                    df[col] = df[matches[0]]
                else:
                    logger.warning(f"Column {col} not found. Some indicators may not be calculated.")
        
        # If we have all the price data
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Moving Averages
            for window in self.feature_config.get('ma_windows', [5, 10, 20, 50, 200]):
                df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}']
            
            # Exponential Moving Averages
            for window in self.feature_config.get('ema_windows', [5, 10, 20, 50, 200]):
                df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
                df[f'ema_ratio_{window}'] = df['close'] / df[f'ema_{window}']
                
            # RSI (Relative Strength Index)
            for window in self.feature_config.get('rsi_windows', [14, 28]):
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=window).mean()
                avg_loss = loss.rolling(window=window).mean()
                
                rs = avg_gain / avg_loss
                df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            ema_fast = df['close'].ewm(span=12, adjust=False).mean()
            ema_slow = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            for window in self.feature_config.get('bb_windows', [20]):
                df[f'bb_middle_{window}'] = df['close'].rolling(window=window).mean()
                df[f'bb_std_{window}'] = df['close'].rolling(window=window).std()
                df[f'bb_upper_{window}'] = df[f'bb_middle_{window}'] + 2 * df[f'bb_std_{window}']
                df[f'bb_lower_{window}'] = df[f'bb_middle_{window}'] - 2 * df[f'bb_std_{window}']
                df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
                df[f'bb_percent_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
            
            # ATR (Average True Range)
            for window in self.feature_config.get('atr_windows', [14]):
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                df[f'atr_{window}'] = true_range.rolling(window=window).mean()
                df[f'atr_ratio_{window}'] = df[f'atr_{window}'] / df['close']
                
            # Stochastic Oscillator
            for window in self.feature_config.get('stoch_windows', [14]):
                low_min = df['low'].rolling(window=window).min()
                high_max = df['high'].rolling(window=window).max()
                
                df[f'stoch_k_{window}'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
                df[f'stoch_d_{window}'] = df[f'stoch_k_{window}'].rolling(window=3).mean()
                
            # ADX (Average Directional Index)
            for window in self.feature_config.get('adx_windows', [14]):
                df[f'adx_{window}'] = self._calculate_adx(df, window)
                
            # Price returns / momentum
            for period in self.feature_config.get('return_periods', [1, 2, 3, 5, 10, 21, 63]):
                df[f'return_{period}d'] = df['close'].pct_change(period)
                
            # Volatility
            for window in self.feature_config.get('vol_windows', [5, 10, 20, 60]):
                df[f'volatility_{window}d'] = df['close'].pct_change().rolling(window=window).std()
                
        # If we have volume data
        if 'volume' in df.columns:
            # Volume indicators
            df['volume_change'] = df['volume'].pct_change()
            
            for window in self.feature_config.get('vol_ma_windows', [5, 10, 20]):
                df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
                df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']
                
            # On-Balance Volume (OBV)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
        # Replace infinities and NaN values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Forward fill and then backward fill remaining NaNs
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate ADX for the given window."""
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        
        pos_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        neg_dm = ((down_move > up_move) & (down_move > 0)) * down_move
        
        # Smooth DM and TR
        tr_smooth = tr.rolling(window).mean()
        pos_dm_smooth = pos_dm.rolling(window).mean()
        neg_dm_smooth = neg_dm.rolling(window).mean()
        
        # Directional Indicators
        pdi = 100 * (pos_dm_smooth / tr_smooth)
        ndi = 100 * (neg_dm_smooth / tr_smooth)
        
        # Directional Index
        dx = 100 * ((pdi - ndi).abs() / (pdi + ndi))
        
        # Average Directional Index
        adx = dx.rolling(window).mean()
        
        return adx
        
    def select_features(self, data: pd.DataFrame, target_col: str, 
                        n_features: Optional[int] = None) -> pd.DataFrame:
        """
        Select the most important features using statistical methods.
        
        Args:
            data: DataFrame with features
            target_col: Target column name for feature selection
            n_features: Number of features to select (default: from config)
            
        Returns:
            DataFrame with selected features
        """
        if n_features is None:
            n_features = self.feature_config.get('n_features', min(50, data.shape[1]))
            
        # Remove the target from features
        features = data.drop(columns=[target_col])
        target = data[target_col]
        
        # Use SelectKBest for feature selection
        self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
        self.feature_selector.fit(features, target)
        
        # Get selected feature names
        feature_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = features.columns[feature_indices].tolist()
        
        # Alternative approach with Random Forest for feature importance
        if self.feature_config.get('use_rf_importance', True):
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(features, target)
            
            # Get feature importances
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Select top features
            top_feature_indices = indices[:n_features]
            rf_selected_features = features.columns[top_feature_indices].tolist()
            
            # Combine both selection methods
            combined_features = list(set(self.selected_features) | set(rf_selected_features))
            if len(combined_features) > n_features:
                combined_features = combined_features[:n_features]
                
            self.selected_features = combined_features
            
        logger.info(f"Selected {len(self.selected_features)} features")
        
        return data[self.selected_features + [target_col]]
    
    def normalize_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize features using StandardScaler.
        
        Args:
            data: DataFrame with features
            fit: Whether to fit the scaler (True for train, False for test/val)
            
        Returns:
            DataFrame with normalized features
        """
        # Copy the data
        df = data.copy()
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            # Fit and transform
            self.scaler.fit(df[numeric_cols])
            
        # Transform the data
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def process_features(self, data: pd.DataFrame, target_col: str, 
                         is_training: bool = True) -> pd.DataFrame:
        """
        Complete feature processing pipeline.
        
        Args:
            data: Input DataFrame with at least OHLCV data
            target_col: Target column name
            is_training: Whether this is training data (to fit transformers)
            
        Returns:
            Processed DataFrame with engineered features
        """
        # Generate technical indicators
        df = self.generate_technical_indicators(data)
        
        # Select features if training
        if is_training:
            df = self.select_features(df, target_col)
        elif self.selected_features:
            # For non-training data, use previously selected features
            df = df[self.selected_features + [target_col]]
            
        # Normalize features
        df = self.normalize_features(df, fit=is_training)
        
        return df

class FinancialDataProcessor:
    """
    Component for financial data preprocessing, including downloading and cleaning.
    Integrates with EastMoney and other data sources.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.cache_dir = Path(self.data_config.get('cache_dir', 'data_cache'))
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_data(self, symbols: List[str], 
                      source: str = 'yahoo',
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Download financial data from the specified source.
        
        Args:
            symbols: List of symbols to download
            source: Data source ('yahoo', 'eastmoney', 'alphavantage', 'ibkr')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary of DataFrames keyed by symbol
        """
        if not start_date:
            start_date = self.data_config.get('default_start_date', '2013-01-01')
        if not end_date:
            end_date = self.data_config.get('default_end_date', datetime.now().strftime('%Y-%m-%d'))
            
        logger.info(f"Downloading data for {len(symbols)} symbols from {source}")
        
        data_dict = {}
        
        if source == 'yahoo':
            try:
                import yfinance as yf
                
                for symbol in symbols:
                    cache_path = self.cache_dir / f"{symbol}.csv"
                    
                    # Check if we have cached data
                    if cache_path.exists() and not self.data_config.get('force_download', False):
                        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                        logger.info(f"Loaded {symbol} from cache")
                    else:
                        # Download from Yahoo Finance
                        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                        
                        # Rename columns to lowercase
                        df.columns = [col.lower() for col in df.columns]
                        
                        # Save to cache
                        df.to_csv(cache_path)
                        logger.info(f"Downloaded {symbol} from Yahoo Finance")
                    
                    data_dict[symbol] = df
                    
            except ImportError:
                logger.error("yfinance package not installed. Please install with: pip install yfinance")
                
        elif source == 'eastmoney':
            # Use FinTSB's EastMoney downloader
            try:
                sys.path.append(str(FINTSB_PATH / 'data'))
                from eastmoney_downloader import EastMoneyDownloader
                
                downloader = EastMoneyDownloader(self.data_config.get('eastmoney_config', 'FinTSB/data/eastmoney_config.yaml'))
                
                for symbol in symbols:
                    cache_path = self.cache_dir / f"{symbol}_eastmoney.csv"
                    
                    # Check if we have cached data
                    if cache_path.exists() and not self.data_config.get('force_download', False):
                        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                        logger.info(f"Loaded {symbol} from cache (EastMoney)")
                    else:
                        # Download from EastMoney
                        df = downloader.download_stock_data(symbol, start_date, end_date)
                        
                        # Save to cache
                        df.to_csv(cache_path)
                        logger.info(f"Downloaded {symbol} from EastMoney")
                    
                    data_dict[symbol] = df
                    
            except (ImportError, FileNotFoundError) as e:
                logger.error(f"EastMoney downloader error: {str(e)}")
                
        elif source == 'ibkr':
            try:
                from src.data_processors.ibkr_connector import IBKRConnector
                
                # Initialize IBKR connector
                connector = IBKRConnector(
                    config_path=self.data_config.get('ibkr_config', 'config/ibkr_config.yaml')
                )
                
                for symbol in symbols:
                    cache_path = self.cache_dir / f"{symbol}_ibkr.csv"
                    
                    # Check if we have cached data
                    if cache_path.exists() and not self.data_config.get('force_download', False):
                        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                        logger.info(f"Loaded {symbol} from cache (IBKR)")
                    else:
                        # Download from IBKR
                        df = connector.get_historical_data(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            timeframe=self.data_config.get('timeframe', 'daily')
                        )
                        
                        # Save to cache
                        df.to_csv(cache_path)
                        logger.info(f"Downloaded {symbol} from IBKR")
                    
                    data_dict[symbol] = df
                    
            except ImportError:
                logger.error("IBKR connector not available")
                
        elif source == 'alphavantage':
            try:
                from alpha_vantage.timeseries import TimeSeries
                
                api_key = self.data_config.get('alphavantage_api_key', '')
                if not api_key:
                    logger.error("AlphaVantage API key not provided")
                    return data_dict
                
                ts = TimeSeries(key=api_key, output_format='pandas')
                
                for symbol in symbols:
                    cache_path = self.cache_dir / f"{symbol}_av.csv"
                    
                    # Check if we have cached data
                    if cache_path.exists() and not self.data_config.get('force_download', False):
                        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                        logger.info(f"Loaded {symbol} from cache (AlphaVantage)")
                    else:
                        # Download from Alpha Vantage
                        df, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
                        
                        # Filter by date
                        df = df.loc[start_date:end_date]
                        
                        # Rename columns
                        df.columns = ['open', 'high', 'low', 'close', 'volume']
                        
                        # Save to cache
                        df.to_csv(cache_path)
                        logger.info(f"Downloaded {symbol} from AlphaVantage")
                    
                    data_dict[symbol] = df
                    
            except ImportError:
                logger.error("alpha_vantage package not installed. Please install with: pip install alpha_vantage")
        
        else:
            logger.error(f"Unsupported data source: {source}")
            
        return data_dict
    
    def clean_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Clean and preprocess financial data.
        
        Args:
            data: DataFrame or Dict of DataFrames to clean
            
        Returns:
            Cleaned DataFrame or Dict of DataFrames
        """
        if isinstance(data, dict):
            # Process each DataFrame in the dictionary
            cleaned_data = {}
            for symbol, df in data.items():
                cleaned_data[symbol] = self._clean_single_dataframe(df)
            return cleaned_data
        else:
            # Process a single DataFrame
            return self._clean_single_dataframe(data)
    
    def _clean_single_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean a single DataFrame of financial data."""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure index is DatetimeIndex if it contains dates
        if not isinstance(df.index, pd.DatetimeIndex) and df.index.dtype == object:
            try:
                df.index = pd.to_datetime(df.index)
            except:
                logger.warning("Could not convert index to DatetimeIndex")
        
        # Convert column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Handle missing values
        if self.data_config.get('handle_missing', True):
            # Forward fill price data (carrying the last value forward)
            df = df.fillna(method='ffill')
            
            # For volume and other non-price data, fill with zeros
            if 'volume' in df.columns:
                df['volume'] = df['volume'].fillna(0)
                
        # Remove outliers if specified
        if self.data_config.get('remove_outliers', False):
            # For each numeric column, apply outlier detection
            for col in df.select_dtypes(include=[np.number]).columns:
                # Calculate Q1, Q3, and IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Replace outliers with NaN
                df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan
                
            # Fill NaN values with forward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Sort index if necessary
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            
        # Drop duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Create returns column if desired
        if self.data_config.get('calculate_returns', True) and 'close' in df.columns:
            df['returns'] = df['close'].pct_change()
            df['returns'] = df['returns'].fillna(0)
            
        return df
    
    def process_data(self, symbols: List[str], 
                     source: str = 'yahoo',
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Complete data processing pipeline.
        
        Args:
            symbols: List of symbols to process
            source: Data source
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary of processed DataFrames keyed by symbol
        """
        # Download data
        data_dict = self.download_data(symbols, source, start_date, end_date)
        
        # Clean data
        cleaned_dict = self.clean_data(data_dict)
        
        return cleaned_dict
    
    def create_market_dataset(self, data_dict: Dict[str, pd.DataFrame], 
                             category: str = 'general') -> pd.DataFrame:
        """
        Create a market dataset combining multiple symbols.
        Categorizes data based on market characteristics.
        
        Args:
            data_dict: Dictionary of DataFrames keyed by symbol
            category: Market category ('general', 'rising', 'falling', 'volatile', 'stable')
            
        Returns:
            Combined DataFrame with market data
        """
        all_dfs = []
        
        for symbol, df in data_dict.items():
            # Add symbol column
            temp_df = df.copy()
            temp_df['symbol'] = symbol
            all_dfs.append(temp_df)
            
        # Combine all DataFrames
        combined_df = pd.concat(all_dfs, axis=0)
        
        # Store in appropriate category folder
        category_dir = Path(self.data_config.get('category_dir', 'FinTSB')) / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset
        dataset_path = category_dir / f"dataset_{len(os.listdir(category_dir)) + 1}.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump(combined_df, f)
            
        logger.info(f"Created market dataset in category {category}: {dataset_path}")
        
        return combined_df

class FinTSBModelManager:
    """
    Manager for FinTSB time series forecasting models.
    Implements different model architectures and ensemble learning.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.device = torch.device(self.model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.models = {}
        self.ensemble_weights = None
        
    def _get_model_by_name(self, model_name: str) -> Any:
        """Get a model class by name."""
        model_mapping = {
            'lstm': LSTM,
            'gru': GRU,
            'tcn': TCN,
            'timesnet': TimesNet,
            'transformer': Transformer,
            'crossformer': Crossformer,
            'patchtst': PatchTST,
            'diffstock': DiffStock,
            'timebridge': TimeBridge,
            'timemixer': TimeMixer,
            'gcn': GCN,
            'gat': GAT,
            'segrnn': SEGRNN,
            'wftnet': WFTNet,
            'pdf': PDF,
            'mamba': Mamba
        }
        
        return model_mapping.get(model_name.lower())
    
    def create_model(self, model_name: str, model_params: Dict[str, Any]) -> Any:
        """
        Create a FinTSB model.
        
        Args:
            model_name: Name of the model architecture
            model_params: Model parameters
            
        Returns:
            Model instance
        """
        # Models are now always available since we've removed the fallback mechanism
        
        model_class = self._get_model_by_name(model_name)
        if not model_class:
            logger.error(f"Model {model_name} not supported")
            return None
        
        try:
            model = model_class(**model_params).to(self.device)
            logger.info(f"Created {model_name} model")
            return model
        except Exception as e:
            logger.error(f"Error creating {model_name} model: {str(e)}")
            return None
    
    def create_dataset(self, data: pd.DataFrame, 
                      sequence_length: int, 
                      prediction_length: int,
                      target_col: str = 'close') -> Any:
        """
        Create a TimeSeriesDataset for model training.
        
        Args:
            data: DataFrame with features
            sequence_length: Input sequence length
            prediction_length: Prediction horizon
            target_col: Target column name
            
        Returns:
            TimeSeriesDataset instance
        """
        # Dataset is now always available since we've removed the fallback mechanism
        
        try:
            dataset = TimeSeriesDataset(
                data,
                sequence_length=sequence_length,
                prediction_length=prediction_length,
                target_column=target_col
            )
            return dataset
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            return None
    
    def train_model(self, model_name: str, 
                   train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader,
                   model_params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model architecture
            train_loader: Training data loader
            val_loader: Validation data loader
            model_params: Model parameters
            
        Returns:
            Tuple of (trained model, training results)
        """
        if model_params is None:
            model_params = self.model_config.get('params', {})
        
        # Create model
        model = self.create_model(model_name, model_params)
        if model is None:
            return None, {}
        
        # Set up optimizer
        optimizer_name = self.model_config.get('optimizer', 'adam')
        learning_rate = self.model_config.get('learning_rate', 0.001)
        
        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
        # Set up loss function
        loss_name = self.model_config.get('loss', 'mse')
        if loss_name.lower() == 'mse':
            criterion = torch.nn.MSELoss()
        elif loss_name.lower() == 'mae':
            criterion = torch.nn.L1Loss()
        else:
            criterion = torch.nn.MSELoss()
            
        # Training loop
        num_epochs = self.model_config.get('epochs', 100)
        early_stopping_patience = self.model_config.get('early_stopping_patience', 10)
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
        # Load best model
        model.load_state_dict(best_model)
        
        # Save model
        if self.model_config.get('save_model', True):
            save_dir = Path(self.model_config.get('save_dir', 'models'))
            save_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = save_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save(model.state_dict(), model_path)
            
        # Store model
        self.models[model_name] = model
        
        # Return results
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs': len(train_losses)
        }
        
        return model, results
    
    def train_ensemble(self, 
                      train_loader: torch.utils.data.DataLoader,
                      val_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Train an ensemble of models.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary with ensemble results
        """
        ensemble_config = self.model_config.get('ensemble', {})
        model_configs = ensemble_config.get('models', [
            {'name': 'lstm', 'weight': 1.0},
            {'name': 'gru', 'weight': 1.0},
            {'name': 'transformer', 'weight': 1.0}
        ])
        
        ensemble_results = {}
        model_performances = []
        
        # Train each model in the ensemble
        for model_config in model_configs:
            model_name = model_config['name']
            logger.info(f"Training {model_name} for ensemble")
            
            model, results = self.train_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader
            )
            
            if model is not None:
                # Store results
                ensemble_results[model_name] = results
                
                # Track performance for weight calculation
                model_performances.append({
                    'name': model_name,
                    'val_loss': results['best_val_loss'],
                    'initial_weight': model_config.get('weight', 1.0)
                })
        
        # Calculate ensemble weights
        if ensemble_config.get('weight_method', 'inverse_loss') == 'inverse_loss':
            # Inverse of validation loss (better models get higher weights)
            inverse_losses = [1.0 / max(perf['val_loss'], 1e-10) for perf in model_performances]
            total_inverse = sum(inverse_losses)
            
            if total_inverse > 0:
                self.ensemble_weights = {
                    perf['name']: (1.0 / max(perf['val_loss'], 1e-10)) / total_inverse * perf['initial_weight']
                    for perf, inv in zip(model_performances, inverse_losses)
                }
            else:
                # Equal weights as fallback
                self.ensemble_weights = {perf['name']: perf['initial_weight'] for perf in model_performances}
                
        elif ensemble_config.get('weight_method', 'inverse_loss') == 'equal':
            # Equal weights
            self.ensemble_weights = {perf['name']: perf['initial_weight'] for perf in model_performances}
            
        else:
            # Use initial weights
            self.ensemble_weights = {perf['name']: perf['initial_weight'] for perf in model_performances}
            
        logger.info(f"Ensemble weights: {self.ensemble_weights}")
        
        # Store ensemble results
        ensemble_results['weights'] = self.ensemble_weights
        
        return ensemble_results
    
    def predict(self, inputs: torch.Tensor, model_name: Optional[str] = None) -> torch.Tensor:
        """
        Generate predictions from a single model or ensemble.
        
        Args:
            inputs: Input tensor
            model_name: Name of the model to use (None for ensemble)
            
        Returns:
            Predictions tensor
        """
        if model_name is not None:
            # Single model prediction
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found")
                return None
                
            model = self.models[model_name]
            model.eval()
            
            with torch.no_grad():
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                
            return outputs
        else:
            # Ensemble prediction
            if not self.ensemble_weights or not self.models:
                logger.error("No ensemble weights or models available")
                return None
                
            ensemble_outputs = None
            total_weight = 0
            
            for model_name, weight in self.ensemble_weights.items():
                if model_name in self.models:
                    model = self.models[model_name]
                    model.eval()
                    
                    with torch.no_grad():
                        inputs = inputs.to(self.device)
                        outputs = model(inputs) * weight
                        
                    if ensemble_outputs is None:
                        ensemble_outputs = outputs
                    else:
                        ensemble_outputs += outputs
                        
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_outputs /= total_weight
                
            return ensemble_outputs

class FinancialExplainer:
    """
    Component for explaining financial forecasts and strategy decisions.
    Provides insights into why a particular forecast was made.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the financial explainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.explainer_config = config.get('explainer', {})
        
    def calculate_feature_importance(self, model, 
                                    features: pd.DataFrame, 
                                    predictions: pd.DataFrame,
                                    feature_names: List[str]) -> Dict[str, float]:
        """
        Calculate feature importance for model predictions.
        
        Args:
            model: Trained model
            features: Feature DataFrame
            predictions: Prediction DataFrame
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.explainer_config.get('method', 'shap') == 'shap':
            try:
                import shap
                
                # Create explainer
                explainer = shap.DeepExplainer(model, torch.tensor(features.values).float())
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(torch.tensor(features.values).float())
                
                # Calculate average importance
                importance_dict = {}
                for i, feature in enumerate(feature_names):
                    importance_dict[feature] = np.abs(shap_values[i]).mean()
                    
                return importance_dict
                
            except ImportError:
                logger.warning("SHAP not installed. Using fallback importance method.")
                
        # Fallback to permutation importance
        importance_dict = {}
        
        # Convert to numpy for easier manipulation
        X = features.values
        y_pred_orig = predictions.values.flatten()
        
        # Calculate baseline error
        baseline_error = np.mean((y_pred_orig - y_pred_orig)**2)  # Should be 0
        
        # For each feature, permute and calculate importance
        for i, feature in enumerate(feature_names):
            # Make a copy of the feature matrix
            X_permuted = X.copy()
            
            # Permute the feature
            np.random.shuffle(X_permuted[:, i])
            
            # Generate predictions with permuted feature
            if hasattr(model, 'predict'):
                y_pred_permuted = model.predict(X_permuted).flatten()
            else:
                # For PyTorch models
                model.eval()
                with torch.no_grad():
                    y_pred_permuted = model(torch.tensor(X_permuted).float()).numpy().flatten()
            
            # Calculate error with permuted feature
            permuted_error = np.mean((y_pred_orig - y_pred_permuted)**2)
            
            # Importance is the increase in error
            importance_dict[feature] = permuted_error - baseline_error
            
        return importance_dict
    
    def explain_prediction(self, prediction: float, 
                          feature_importance: Dict[str, float],
                          features: pd.DataFrame,
                          timestamp: Any) -> Dict[str, Any]:
        """
        Generate explanation for a specific prediction.
        
        Args:
            prediction: Prediction value
            feature_importance: Feature importance dictionary
            features: Feature values
            timestamp: Timestamp for the prediction
            
        Returns:
            Dictionary with explanation details
        """
        # Sort features by importance
        sorted_importance = sorted(
            feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # Take top n features
        top_n = self.explainer_config.get('top_n_features', 5)
        top_features = sorted_importance[:top_n]
        
        # Determine direction of prediction (up or down)
        direction = "up" if prediction > 0 else "down"
        
        # Create explanation
        explanation = {
            'timestamp': timestamp,
            'prediction': prediction,
            'direction': direction,
            'confidence': abs(prediction), # Simple proxy for confidence
            'top_factors': [
                {
                    'feature': feature, 
                    'importance': importance,
                    'value': features.loc[timestamp, feature] if feature in features.columns else None,
                    'impact': "positive" if importance > 0 else "negative"
                }
                for feature, importance in top_features
            ],
            'market_conditions': self._analyze_market_conditions(features, timestamp)
        }
        
        return explanation
    
    def _analyze_market_conditions(self, features: pd.DataFrame, 
                                 timestamp: Any) -> Dict[str, Any]:
        """Analyze overall market conditions for explanation context."""
        # Extract relevant market features
        market_conditions = {}
        
        trend_indicators = ['ma_20', 'ma_50', 'ma_200', 'ema_20']
        momentum_indicators = ['rsi_14', 'macd']
        volatility_indicators = ['bb_width_20', 'atr_14', 'volatility_20d']
        
        # Analyze trend
        trend_values = []
        for indicator in trend_indicators:
            if indicator in features.columns:
                if indicator.startswith('ma_') or indicator.startswith('ema_'):
                    # Compare current price to moving average
                    if 'close' in features.columns:
                        current_price = features.loc[timestamp, 'close']
                        ma_value = features.loc[timestamp, indicator]
                        trend_values.append(1 if current_price > ma_value else -1)
        
        market_conditions['trend'] = 'bullish' if np.mean(trend_values) > 0 else 'bearish' if np.mean(trend_values) < 0 else 'neutral'
        
        # Analyze momentum
        momentum_values = []
        if 'rsi_14' in features.columns:
            rsi = features.loc[timestamp, 'rsi_14']
            momentum_values.append(1 if rsi > 60 else -1 if rsi < 40 else 0)
            
        if 'macd' in features.columns and 'macd_signal' in features.columns:
            macd = features.loc[timestamp, 'macd']
            signal = features.loc[timestamp, 'macd_signal']
            momentum_values.append(1 if macd > signal else -1)
            
        market_conditions['momentum'] = 'strong' if np.mean(momentum_values) > 0.5 else 'weak' if np.mean(momentum_values) < -0.5 else 'neutral'
        
        # Analyze volatility
        volatility_values = []
        if 'bb_width_20' in features.columns:
            bb_width = features.loc[timestamp, 'bb_width_20']
            # Compare to recent average
            recent_width = features.loc[:timestamp, 'bb_width_20'].tail(20).mean()
            volatility_values.append(1 if bb_width > recent_width else -1)
            
        market_conditions['volatility'] = 'high' if np.mean(volatility_values) > 0 else 'low'
        
        return market_conditions
    
    def generate_text_explanation(self, explanation: Dict[str, Any]) -> str:
        """
        Generate human-readable text explanation.
        
        Args:
            explanation: Explanation dictionary
            
        Returns:
            Text explanation
        """
        text = f"Forecast: {explanation['direction'].upper()} with {explanation['confidence']:.2f} confidence\n\n"
        
        text += "Key factors:\n"
        for factor in explanation['top_factors']:
            impact = "↑" if factor['impact'] == "positive" else "↓"
            text += f"- {factor['feature']}: {impact} (importance: {factor['importance']:.4f}, value: {factor['value']:.4f})\n"
            
        text += f"\nMarket conditions: {explanation['market_conditions']['trend']} trend, "
        text += f"{explanation['market_conditions']['momentum']} momentum, "
        text += f"{explanation['market_conditions']['volatility']} volatility"
        
        return text
    
    def generate_visualization(self, feature_importance: Dict[str, float], 
                             save_path: str) -> None:
        """
        Generate visualization of feature importance.
        
        Args:
            feature_importance: Feature importance dictionary
            save_path: Path to save the visualization
        """
        try:
            # Sort importance values
            sorted_importance = sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            # Take top 10 features
            top_n = min(10, len(sorted_importance))
            top_features = sorted_importance[:top_n]
            
            features = [f[0] for f in top_features]
            importance = [f[1] for f in top_features]
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, 6))
            plt.barh(features, importance)
            plt.xlabel('Importance')
            plt.title('Feature Importance')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")

class PortfolioConstructor:
    """
    Component for constructing and managing portfolios based on financial forecasts.
    Implements risk management measures and portfolio optimization.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the portfolio constructor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.portfolio_config = config.get('portfolio', {})
        
    def allocate_weights(self, predictions: Dict[str, float], 
                        risk_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Allocate portfolio weights based on predictions and risk metrics.
        
        Args:
            predictions: Dictionary of predictions by symbol
            risk_metrics: Dictionary of risk metrics by symbol
            
        Returns:
            Dictionary of weights by symbol
        """
        # Default to equal weight
        equal_weight = 1.0 / max(1, len(predictions))
        weights = {symbol: equal_weight for symbol in predictions}
        
        # Choose allocation method
        allocation_method = self.portfolio_config.get('allocation_method', 'equal_weight')
        
        if allocation_method == 'equal_weight':
            # Already set to equal weight
            pass
            
        elif allocation_method == 'prediction_weighted':
            # Weight by prediction magnitude
            total_pred = sum(abs(pred) for pred in predictions.values())
            if total_pred > 0:
                weights = {
                    symbol: abs(pred) / total_pred 
                    for symbol, pred in predictions.items()
                }
                
        elif allocation_method == 'sharpe_weighted':
            # Weight by Sharpe ratio
            sharpe_ratios = {
                symbol: metrics.get('sharpe_ratio', 0)
                for symbol, metrics in risk_metrics.items()
            }
            
            total_sharpe = sum(max(0, sharpe) for sharpe in sharpe_ratios.values())
            if total_sharpe > 0:
                weights = {
                    symbol: max(0, sharpe) / total_sharpe
                    for symbol, sharpe in sharpe_ratios.items()
                }
                
        elif allocation_method == 'minimum_variance':
            try:
                # Get covariance matrix
                symbols = list(predictions.keys())
                returns = np.array([risk_metrics[symbol].get('returns', [0]) for symbol in symbols])
                cov_matrix = np.cov(returns) if len(returns) > 1 else np.array([[1]])
                
                # Inverse variance weights
                if cov_matrix.shape[0] > 1:
                    inv_diag = 1 / np.diag(cov_matrix)
                    total_inv = sum(inv_diag)
                    if total_inv > 0:
                        for i, symbol in enumerate(symbols):
                            weights[symbol] = inv_diag[i] / total_inv
                            
            except Exception as e:
                logger.error(f"Error in minimum variance allocation: {str(e)}")
                
        # Apply position constraints
        min_position = self.portfolio_config.get('min_position', 0.01)
        max_position = self.portfolio_config.get('max_position', 0.25)
        
        # Adjust weights to be within bounds
        for symbol in weights:
            weights[symbol] = max(min_position, min(max_position, weights[symbol]))
            
        # Normalize to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
            
        return weights
    
    def apply_risk_management(self, weights: Dict[str, float], 
                             risk_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Apply risk management constraints to portfolio weights.
        
        Args:
            weights: Initial portfolio weights
            risk_metrics: Risk metrics for each symbol
            
        Returns:
            Adjusted portfolio weights
        """
        # Copy weights
        adjusted_weights = weights.copy()
        
        # Check maximum volatility constraint
        max_portfolio_vol = self.portfolio_config.get('max_portfolio_volatility', 0.2)
        
        # Calculate portfolio volatility
        symbols = list(weights.keys())
        vols = np.array([risk_metrics[symbol].get('volatility', 0.2) for symbol in symbols])
        
        # Simple volatility calculation (ignoring correlations)
        weighted_vols = np.array([weights[symbol] * vols[i] for i, symbol in enumerate(symbols)])
        portfolio_vol = np.sqrt(np.sum(weighted_vols**2))
        
        # If portfolio volatility exceeds max, scale down weights
        if portfolio_vol > max_portfolio_vol:
            scale_factor = max_portfolio_vol / portfolio_vol
            adjusted_weights = {symbol: weight * scale_factor for symbol, weight in adjusted_weights.items()}
            
        # Check maximum drawdown constraint
        max_portfolio_dd = self.portfolio_config.get('max_portfolio_drawdown', 0.25)
        
        # Calculate expected portfolio drawdown
        drawdowns = np.array([risk_metrics[symbol].get('max_drawdown', 0.2) for symbol in symbols])
        weighted_dd = np.sum([weights[symbol] * drawdowns[i] for i, symbol in enumerate(symbols)])
        
        # If expected drawdown exceeds max, scale down weights
        if weighted_dd > max_portfolio_dd:
            scale_factor = max_portfolio_dd / weighted_dd
            adjusted_weights = {symbol: weight * scale_factor for symbol, weight in adjusted_weights.items()}
            
        # Rebalance to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {symbol: weight / total_weight for symbol, weight in adjusted_weights.items()}
            
        return adjusted_weights
    
    def generate_portfolio_signals(self, weights: Dict[str, float], 
                                 predictions: Dict[str, float],
                                 current_portfolio: Optional[Dict[str, float]] = None) -> Dict[str, str]:
        """
        Generate trading signals to achieve target portfolio weights.
        
        Args:
            weights: Target portfolio weights
            predictions: Predictions by symbol
            current_portfolio: Current portfolio weights (empty = cash)
            
        Returns:
            Dictionary of trading signals by symbol
        """
        signals = {}
        
        # If no current portfolio, assume all cash
        if current_portfolio is None:
            current_portfolio = {symbol: 0.0 for symbol in weights}
            
        # Determine buy/sell signals
        for symbol, target_weight in weights.items():
            current_weight = current_portfolio.get(symbol, 0.0)
            
            # Calculate weight difference
            weight_diff = target_weight - current_weight
            
            # Apply threshold to avoid tiny trades
            min_trade_size = self.portfolio_config.get('min_trade_size', 0.005)
            
            if abs(weight_diff) < min_trade_size:
                signals[symbol] = "HOLD"
            elif weight_diff > 0:
                # Direction check - only buy if prediction is positive
                if predictions.get(symbol, 0) > 0:
                    signals[symbol] = "BUY"
                else:
                    signals[symbol] = "HOLD"
            else:
                # Direction check - only sell if prediction is negative
                if predictions.get(symbol, 0) < 0:
                    signals[symbol] = "SELL"
                else:
                    signals[symbol] = "HOLD"
                    
        return signals
    
    def optimize_portfolio(self, predictions: Dict[str, float], 
                         risk_metrics: Dict[str, Dict[str, float]],
                         current_portfolio: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Complete portfolio optimization pipeline.
        
        Args:
            predictions: Predictions by symbol
            risk_metrics: Risk metrics by symbol
            current_portfolio: Current portfolio weights
            
        Returns:
            Dictionary with portfolio allocation results
        """
        # Initial allocation
        weights = self.allocate_weights(predictions, risk_metrics)
        
        # Apply risk management
        adjusted_weights = self.apply_risk_management(weights, risk_metrics)
        
        # Generate signals
        signals = self.generate_portfolio_signals(
            adjusted_weights, predictions, current_portfolio
        )
        
        # Calculate expected portfolio metrics
        symbols = list(adjusted_weights.keys())
        
        # Expected return
        expected_returns = np.array([predictions.get(symbol, 0) for symbol in symbols])
        weighted_returns = np.sum([adjusted_weights[symbol] * expected_returns[i] for i, symbol in enumerate(symbols)])
        
        # Expected volatility (simplified)
        volatilities = np.array([risk_metrics[symbol].get('volatility', 0.2) for symbol in symbols])
        weighted_vol = np.sqrt(np.sum([adjusted_weights[symbol]**2 * volatilities[i]**2 for i, symbol in enumerate(symbols)]))
        
        # Expected Sharpe
        risk_free = self.portfolio_config.get('risk_free_rate', 0.02)
        expected_sharpe = (weighted_returns - risk_free) / weighted_vol if weighted_vol > 0 else 0
        
        # Result dictionary
        result = {
            'weights': adjusted_weights,
            'signals': signals,
            'expected_return': weighted_returns,
            'expected_volatility': weighted_vol,
            'expected_sharpe': expected_sharpe,
            'allocation_method': self.portfolio_config.get('allocation_method', 'equal_weight')
        }
        
        return result

class FinTSBPipeline:
    """
    Complete FinTSB pipeline integrating all components:
    - Data preprocessing
    - Feature engineering
    - Model training
    - Forecasting
    - Explanation
    - Portfolio construction
    """
    def __init__(self, config_path: str):
        """
        Initialize the complete pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_processor = FinancialDataProcessor(self.config)
        self.feature_processor = FinancialFeatureProcessor(self.config)
        self.model_manager = FinTSBModelManager(self.config)
        self.explainer = FinancialExplainer(self.config)
        self.portfolio_constructor = PortfolioConstructor(self.config)
        
        # Set up logging
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_logging(self) -> None:
        """Set up logging for the pipeline."""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('file', 'fintsb_pipeline.log')
        
        logger.setLevel(getattr(logging, log_level))
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Returns:
            Results dictionary
        """
        # Step 1: Download and preprocess data
        logger.info("Step 1: Downloading and preprocessing data")
        symbols = self.config['data']['symbols']
        data_source = self.config['data'].get('source', 'yahoo')
        start_date = self.config['data'].get('start_date')
        end_date = self.config['data'].get('end_date')
        
        data_dict = self.data_processor.process_data(
            symbols=symbols,
            source=data_source,
            start_date=start_date,
            end_date=end_date
        )
        
        # Step 2: Feature engineering for each symbol
        logger.info("Step 2: Feature engineering")
        feature_dicts = {}
        for symbol, df in data_dict.items():
            logger.info(f"Processing features for {symbol}")
            
            # Process features
            processed_df = self.feature_processor.process_features(
                data=df,
                target_col='close',
                is_training=True
            )
            
            feature_dicts[symbol] = processed_df
            
        # Step 3: Prepare datasets and data loaders
        logger.info("Step 3: Preparing datasets and data loaders")
        datasets = {}
        data_loaders = {}
        
        for symbol, df in feature_dicts.items():
            # Create TimeSeriesDataset
            dataset = self.model_manager.create_dataset(
                data=df,
                sequence_length=self.config['data']['sequence_length'],
                prediction_length=self.config['data']['prediction_length'],
                target_col='close'
            )
            
            if dataset is not None:
                datasets[symbol] = dataset
                
                # Split dataset
                train_size = int(len(dataset) * self.config['data']['train_ratio'])
                val_size = int(len(dataset) * self.config['data']['val_ratio'])
                test_size = len(dataset) - train_size - val_size
                
                train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                    dataset, [train_size, val_size, test_size]
                )
                
                # Create data loaders
                data_loaders[symbol] = {
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
                
        # Step 4: Train models
        logger.info("Step 4: Training models")
        training_method = self.config['model'].get('training_method', 'ensemble')
        models = {}
        predictions = {}
        
        if training_method == 'single':
            # Train a single model for each symbol
            model_name = self.config['model'].get('name', 'lstm')
            
            for symbol, loaders in data_loaders.items():
                logger.info(f"Training {model_name} for {symbol}")
                
                model, results = self.model_manager.train_model(
                    model_name=model_name,
                    train_loader=loaders['train'],
                    val_loader=loaders['val']
                )
                
                if model is not None:
                    models[symbol] = model
                    
                    # Generate predictions
                    all_predictions = []
                    all_inputs = []
                    
                    with torch.no_grad():
                        for inputs, _ in loaders['test']:
                            preds = self.model_manager.predict(inputs, model_name)
                            all_predictions.append(preds.cpu().numpy())
                            all_inputs.append(inputs.cpu().numpy())
                            
                    # Combine predictions
                    combined_preds = np.concatenate(all_predictions)
                    # Store predictions for this symbol
                    predictions[symbol] = combined_preds
                    
        elif training_method == 'ensemble':
            # Train an ensemble for each symbol
            for symbol, loaders in data_loaders.items():
                logger.info(f"Training ensemble for {symbol}")
                
                ensemble_results = self.model_manager.train_ensemble(
                    train_loader=loaders['train'],
                    val_loader=loaders['val']
                )
                
                # Generate predictions
                all_predictions = []
                
                with torch.no_grad():
                    for inputs, _ in loaders['test']:
                        preds = self.model_manager.predict(inputs)
                        all_predictions.append(preds.cpu().numpy())
                        
                # Combine predictions
                combined_preds = np.concatenate(all_predictions)
                # Store predictions for this symbol
                predictions[symbol] = combined_preds
                
        # Step 5: Generate forecasts and explanations
        logger.info("Step 5: Generating forecasts and explanations")
        forecasts = {}
        explanations = {}
        
        for symbol in predictions:
            # Calculate average prediction
            avg_pred = np.mean(predictions[symbol])
            
            # Create simple forecast dictionary
            forecasts[symbol] = {
                'prediction': avg_pred,
                'direction': 'up' if avg_pred > 0 else 'down',
                'magnitude': abs(avg_pred)
            }
            
            # Generate explanation if we have feature data
            if symbol in feature_dicts:
                # Calculate feature importance
                feature_importance = {}
                feature_names = feature_dicts[symbol].columns.tolist()
                
                # Use a simple approximation for feature importance
                for feature in feature_names:
                    # Random importance for demonstration
                    importance = np.random.random()
                    feature_importance[feature] = importance
                    
                # Generate explanation
                explanations[symbol] = self.explainer.explain_prediction(
                    prediction=avg_pred,
                    feature_importance=feature_importance,
                    features=feature_dicts[symbol],
                    timestamp=feature_dicts[symbol].index[-1]
                )
                
                # Generate text explanation
                text_explanation = self.explainer.generate_text_explanation(explanations[symbol])
                explanations[symbol]['text'] = text_explanation
                
        # Step 6: Portfolio construction
        logger.info("Step 6: Portfolio construction")
        
        # Calculate risk metrics
        risk_metrics = {}
        for symbol, df in data_dict.items():
            if 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized
                
                # Calculate max drawdown
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.cummax()
                drawdown = (cumulative_returns / running_max) - 1
                max_drawdown = drawdown.min()
                
                # Calculate Sharpe ratio
                risk_free = self.config['portfolio'].get('risk_free_rate', 0.02) / 252
                sharpe = (returns.mean() - risk_free) / returns.std() * np.sqrt(252)
                
                risk_metrics[symbol] = {
                    'volatility': volatility,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe,
                    'returns': returns.values
                }
                
        # Summarize predictions for portfolio construction
        pred_summary = {symbol: forecast['prediction'] for symbol, forecast in forecasts.items()}
        
        # Optimize portfolio
        portfolio_result = self.portfolio_constructor.optimize_portfolio(
            predictions=pred_summary,
            risk_metrics=risk_metrics
        )
        
        # Step 7: Prepare final results
        logger.info("Step 7: Preparing results")
        
        results = {
            'forecasts': forecasts,
            'explanations': explanations,
            'portfolio': portfolio_result,
            'risk_metrics': risk_metrics,
            'timestamps': {
                'start': start_date,
                'end': end_date,
                'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Save results
        if self.config.get('save_results', True):
            output_dir = Path(self.config.get('output_dir', 'output'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = output_dir / f"fintsb_results_{timestamp}.json"
            
            with open(results_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = json.dumps(results, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                f.write(json_results)
                
            logger.info(f"Results saved to {results_path}")
            
        return results

# Example usage:
def main(config_path: str):
    """Run the FinTSB pipeline with the given configuration."""
    pipeline = FinTSBPipeline(config_path)
    results = pipeline.run()
    
    # Log summary of results
    logger.info(f"Pipeline completed successfully")
    logger.info(f"Generated forecasts for {len(results['forecasts'])} symbols")
    logger.info(f"Portfolio allocation: {len(results['portfolio']['weights'])} symbols")
    
    # Display sample forecast
    if results['forecasts']:
        symbol = next(iter(results['forecasts']))
        forecast = results['forecasts'][symbol]
        logger.info(f"Sample forecast for {symbol}: {forecast['direction']} ({forecast['prediction']:.4f})")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FinTSB pipeline')
    parser.add_argument('config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    main(args.config)