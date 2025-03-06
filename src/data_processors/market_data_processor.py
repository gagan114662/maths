"""
Market Data Processor Module.
Provides comprehensive functionality for downloading, processing, and caching market data.
Supports multiple data sources and ensures data quality.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import pickle
import json
from datetime import datetime, timedelta
import requests
import gzip
import shutil
import time
import random

# Import AlternativeDataProcessor
from src.data_processors.alternative_data_processor import AlternativeDataProcessor

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

class MarketDataProcessor:
    """
    Comprehensive market data processor that handles downloading, cleaning,
    caching, and augmenting financial market data from multiple sources.
    """
    
    def __init__(self, data_dir: str = None, cache_dir: str = None, 
                config_file: str = None, alt_data_config_path: str = None):
        """
        Initialize the market data processor.
        
        Args:
            data_dir: Directory for storing downloaded data
            cache_dir: Directory for caching processed data
            config_file: Path to configuration file
            alt_data_config_path: Path to alternative data configuration file
        """
        # Set up directories
        self.data_dir = Path(data_dir) if data_dir else Path(PROJECT_ROOT) / "data"
        self.cache_dir = Path(cache_dir) if cache_dir else Path(PROJECT_ROOT) / "data_cache"
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration if provided
        self.config = {}
        if config_file:
            self._load_config(config_file)
            
        # Set up data source directories
        sources = ['yahoo', 'ibkr', 'alphavantage', 'kraken', 'eastmoney', 'custom', 'alternative']
        for source in sources:
            source_dir = self.data_dir / source
            source_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize data quality metrics
        self.quality_metrics = {}
        
        # Initialize alternative data processor
        alt_data_cache_dir = str(self.cache_dir / "alternative_data")
        self.alt_data_processor = AlternativeDataProcessor(
            cache_dir=alt_data_cache_dir,
            config_path=alt_data_config_path
        )
        
        logger.info(f"Market Data Processor initialized with data_dir={self.data_dir}, cache_dir={self.cache_dir}")
    
    def _load_config(self, config_file: str):
        """Load configuration from file."""
        try:
            config_path = Path(config_file)
            
            if config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                import yaml
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported config file format: {config_path.suffix}")
                
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            self.config = {}
    
    def download_data(self, symbols: List[str], source: str = 'yahoo',
                     start_date: str = None, end_date: str = None,
                     timeframe: str = 'daily', force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download market data for the specified symbols.
        
        Args:
            symbols: List of symbols to download
            source: Data source (yahoo, ibkr, alphavantage, kraken, eastmoney)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            timeframe: Data timeframe (daily, hourly, minute)
            force_download: Whether to force download even if cached data exists
            
        Returns:
            Dictionary of DataFrames keyed by symbol
        """
        # Set default dates if not provided (10 years of data)
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        if not start_date:
            # Default to 10 years of data
            start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
            
        logger.info(f"Downloading data for {len(symbols)} symbols from {source} ({start_date} to {end_date})")
        
        # Create dictionary to store downloaded data
        data_dict = {}
        
        # Directory for this data source
        source_dir = self.data_dir / source
        
        # Use appropriate downloader based on source
        if source == 'yahoo':
            data_dict = self._download_from_yahoo(symbols, start_date, end_date, source_dir, force_download)
        elif source == 'ibkr':
            data_dict = self._download_from_ibkr(symbols, start_date, end_date, timeframe, source_dir, force_download)
        elif source == 'alphavantage':
            data_dict = self._download_from_alphavantage(symbols, start_date, end_date, timeframe, source_dir, force_download)
        elif source == 'kraken':
            data_dict = self._download_from_kraken(symbols, start_date, end_date, timeframe, source_dir, force_download)
        elif source == 'eastmoney':
            data_dict = self._download_from_eastmoney(symbols, start_date, end_date, timeframe, source_dir, force_download)
        else:
            logger.error(f"Unsupported data source: {source}")
            
        # Check data quality for all downloaded data
        for symbol, df in data_dict.items():
            self.quality_metrics[symbol] = self._check_data_quality(df)
            
        return data_dict
    
    def _download_from_yahoo(self, symbols: List[str], start_date: str, 
                           end_date: str, source_dir: Path, 
                           force_download: bool) -> Dict[str, pd.DataFrame]:
        """Download data from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed. Please install with 'pip install yfinance'")
            return {}
            
        data_dict = {}
        
        for symbol in symbols:
            # Construct file path for cached data
            file_path = source_dir / f"{symbol}.csv"
            
            # Check if we already have the data cached
            if file_path.exists() and not force_download:
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    logger.info(f"Loaded {symbol} from cache ({source_dir})")
                    data_dict[symbol] = df
                    continue
                except Exception as e:
                    logger.warning(f"Error loading cached data for {symbol}: {str(e)}")
            
            # Download the data
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                # Check if data is empty
                if df.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Standardize column names (lowercase)
                df.columns = [col.lower() for col in df.columns]
                
                # Save to cache
                df.to_csv(file_path)
                logger.info(f"Downloaded and cached {symbol} data")
                
                data_dict[symbol] = df
                
            except Exception as e:
                logger.error(f"Error downloading {symbol} from Yahoo Finance: {str(e)}")
        
        return data_dict
    
    def _download_from_ibkr(self, symbols: List[str], start_date: str, 
                          end_date: str, timeframe: str, source_dir: Path, 
                          force_download: bool) -> Dict[str, pd.DataFrame]:
        """Download data from Interactive Brokers."""
        # Try to import IBKR connector
        try:
            from src.data_processors.ibkr_connector import IBKRConnector
        except ImportError:
            logger.error("IBKR connector not found or cannot be imported")
            return {}
            
        data_dict = {}
        
        try:
            # Initialize the connector (this would use configuration settings)
            connector = IBKRConnector(config_path=self.config.get('ibkr_config', None))
            
            for symbol in symbols:
                # Construct file path for cached data
                file_path = source_dir / f"{symbol}_{timeframe}.csv"
                
                # Check if we already have the data cached
                if file_path.exists() and not force_download:
                    try:
                        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                        logger.info(f"Loaded {symbol} from cache ({source_dir})")
                        data_dict[symbol] = df
                        continue
                    except Exception as e:
                        logger.warning(f"Error loading cached data for {symbol}: {str(e)}")
                
                # Download the data
                try:
                    df = connector.get_historical_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe=timeframe
                    )
                    
                    # Check if data is empty
                    if df.empty:
                        logger.warning(f"No data available for {symbol}")
                        continue
                    
                    # Save to cache
                    df.to_csv(file_path)
                    logger.info(f"Downloaded and cached {symbol} data from IBKR")
                    
                    data_dict[symbol] = df
                    
                except Exception as e:
                    logger.error(f"Error downloading {symbol} from IBKR: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error initializing IBKR connector: {str(e)}")
            
        return data_dict
    
    def _download_from_alphavantage(self, symbols: List[str], start_date: str, 
                                  end_date: str, timeframe: str, source_dir: Path, 
                                  force_download: bool) -> Dict[str, pd.DataFrame]:
        """Download data from Alpha Vantage."""
        try:
            from alpha_vantage.timeseries import TimeSeries
        except ImportError:
            logger.error("alpha_vantage not installed. Please install with 'pip install alpha_vantage'")
            return {}
            
        # Check for API key
        api_key = self.config.get('alpha_vantage_api_key', os.environ.get('ALPHA_VANTAGE_API_KEY'))
        if not api_key:
            logger.error("Alpha Vantage API key not found. Please set ALPHA_VANTAGE_API_KEY environment variable or in config")
            return {}
            
        data_dict = {}
        
        # Initialize API client
        ts = TimeSeries(key=api_key, output_format='pandas')
        
        for symbol in symbols:
            # Construct file path for cached data
            file_path = source_dir / f"{symbol}_{timeframe}.csv"
            
            # Check if we already have the data cached
            if file_path.exists() and not force_download:
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    logger.info(f"Loaded {symbol} from cache ({source_dir})")
                    data_dict[symbol] = df
                    continue
                except Exception as e:
                    logger.warning(f"Error loading cached data for {symbol}: {str(e)}")
            
            # Download the data
            try:
                # Choose function based on timeframe
                if timeframe == 'daily':
                    df, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
                elif timeframe == 'intraday':
                    df, meta_data = ts.get_intraday(symbol=symbol, interval='60min', outputsize='full')
                else:
                    logger.warning(f"Unsupported timeframe for Alpha Vantage: {timeframe}")
                    continue
                
                # Filter by date
                df = df.loc[start_date:end_date]
                
                # Check if data is empty
                if df.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Standardize column names
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                
                # Save to cache
                df.to_csv(file_path)
                logger.info(f"Downloaded and cached {symbol} data from Alpha Vantage")
                
                data_dict[symbol] = df
                
                # Alpha Vantage has rate limits, so we need to pause between requests
                time.sleep(15)  # Basic rate limit: 5 calls per minute (75 per day)
                
            except Exception as e:
                logger.error(f"Error downloading {symbol} from Alpha Vantage: {str(e)}")
        
        return data_dict
    
    def _download_from_kraken(self, symbols: List[str], start_date: str, 
                            end_date: str, timeframe: str, source_dir: Path, 
                            force_download: bool) -> Dict[str, pd.DataFrame]:
        """Download data from Kraken cryptocurrency exchange."""
        # Parse dates to timestamps for Kraken API
        try:
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        except ValueError as e:
            logger.error(f"Error parsing dates: {str(e)}")
            return {}
            
        data_dict = {}
        
        # Map timeframe to Kraken interval parameter
        timeframe_map = {
            'minute': 1,
            'hourly': 60,
            'daily': 1440,
            'weekly': 10080,
            'monthly': 43200
        }
        
        interval = timeframe_map.get(timeframe, 1440)  # Default to daily
        
        for symbol in symbols:
            # For Kraken, convert traditional symbol format to Kraken format
            kraken_symbol = symbol.replace('/', '')  # e.g., BTC/USD -> BTCUSD
            
            # Construct file path for cached data
            file_path = source_dir / f"{symbol}_{timeframe}.csv"
            
            # Check if we already have the data cached
            if file_path.exists() and not force_download:
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    logger.info(f"Loaded {symbol} from cache ({source_dir})")
                    data_dict[symbol] = df
                    continue
                except Exception as e:
                    logger.warning(f"Error loading cached data for {symbol}: {str(e)}")
            
            # Download the data from Kraken API
            try:
                # Kraken API endpoint for OHLC data
                endpoint = f"https://api.kraken.com/0/public/OHLC"
                
                # Parameters for the request
                params = {
                    'pair': kraken_symbol,
                    'interval': interval,
                    'since': start_timestamp
                }
                
                # Make the request
                response = requests.get(endpoint, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'result' in data and kraken_symbol in data['result']:
                        # Extract OHLC data
                        ohlc_data = data['result'][kraken_symbol]
                        
                        # Create DataFrame
                        columns = ['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
                        df = pd.DataFrame(ohlc_data, columns=columns)
                        
                        # Convert timestamp to datetime and set as index
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        df.set_index('timestamp', inplace=True)
                        
                        # Filter by date
                        df = df[start_date:end_date]
                        
                        # Convert string values to float
                        for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
                            df[col] = df[col].astype(float)
                            
                        # Save to cache
                        df.to_csv(file_path)
                        logger.info(f"Downloaded and cached {symbol} data from Kraken")
                        
                        data_dict[symbol] = df
                    else:
                        logger.warning(f"No data available for {symbol} from Kraken")
                else:
                    logger.error(f"Error from Kraken API: {response.text}")
                    
                # Sleep to respect rate limits (public API: 1 request per 3 seconds)
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Error downloading {symbol} from Kraken: {str(e)}")
        
        return data_dict
    
    def _download_from_eastmoney(self, symbols: List[str], start_date: str, 
                               end_date: str, timeframe: str, source_dir: Path, 
                               force_download: bool) -> Dict[str, pd.DataFrame]:
        """Download data from EastMoney (Chinese market data)."""
        # Try to import EastMoney downloader
        try:
            from FinTSB.data.eastmoney_downloader import EastMoneyDownloader
        except ImportError:
            logger.error("EastMoney downloader not found or cannot be imported")
            return {}
            
        data_dict = {}
        
        try:
            # Initialize the downloader
            config_path = self.config.get('eastmoney_config', 'FinTSB/data/eastmoney_config.yaml')
            downloader = EastMoneyDownloader(config_path)
            
            for symbol in symbols:
                # Construct file path for cached data
                file_path = source_dir / f"{symbol}_{timeframe}.csv"
                
                # Check if we already have the data cached
                if file_path.exists() and not force_download:
                    try:
                        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                        logger.info(f"Loaded {symbol} from cache ({source_dir})")
                        data_dict[symbol] = df
                        continue
                    except Exception as e:
                        logger.warning(f"Error loading cached data for {symbol}: {str(e)}")
                
                # Download the data
                try:
                    df = downloader.download_stock_data(symbol, start_date, end_date)
                    
                    # Check if data is empty
                    if df.empty:
                        logger.warning(f"No data available for {symbol}")
                        continue
                    
                    # Save to cache
                    df.to_csv(file_path)
                    logger.info(f"Downloaded and cached {symbol} data from EastMoney")
                    
                    data_dict[symbol] = df
                    
                except Exception as e:
                    logger.error(f"Error downloading {symbol} from EastMoney: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error initializing EastMoney downloader: {str(e)}")
            
        return data_dict
    
    def clean_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                 filter_outliers: bool = True, handle_missing: bool = True,
                 fill_gaps: bool = True) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Clean and preprocess market data.
        
        Args:
            data: Single DataFrame or dictionary of DataFrames
            filter_outliers: Whether to filter out outliers
            handle_missing: Whether to handle missing values
            fill_gaps: Whether to fill gaps in time series
            
        Returns:
            Cleaned data in the same format as input
        """
        if isinstance(data, dict):
            # Process each DataFrame in the dictionary
            cleaned_data = {}
            for symbol, df in data.items():
                cleaned_data[symbol] = self._clean_single_dataframe(
                    df, filter_outliers, handle_missing, fill_gaps
                )
                
                # Update quality metrics
                self.quality_metrics[symbol] = self._check_data_quality(cleaned_data[symbol])
                
            return cleaned_data
        else:
            # Process a single DataFrame
            cleaned_df = self._clean_single_dataframe(
                data, filter_outliers, handle_missing, fill_gaps
            )
            
            # Update quality metrics
            self.quality_metrics["single_df"] = self._check_data_quality(cleaned_df)
            
            return cleaned_df
    
    def _clean_single_dataframe(self, df: pd.DataFrame, filter_outliers: bool,
                              handle_missing: bool, fill_gaps: bool) -> pd.DataFrame:
        """Clean a single DataFrame of market data."""
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Ensure index is DatetimeIndex
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            try:
                df_clean.index = pd.to_datetime(df_clean.index)
            except:
                logger.warning("Could not convert index to DatetimeIndex")
        
        # Standardize column names to lowercase
        df_clean.columns = [col.lower() for col in df_clean.columns]
        
        # Handle missing values
        if handle_missing:
            # For price columns, use forward fill (carry last value forward)
            price_cols = [col for col in df_clean.columns if col in ['open', 'high', 'low', 'close', 'adj close']]
            df_clean[price_cols] = df_clean[price_cols].fillna(method='ffill')
            
            # For volume, fill with zeros or median
            if 'volume' in df_clean.columns:
                # Check if we should use median instead of zero (for stocks with consistent volume)
                if df_clean['volume'].median() > 1000:
                    df_clean['volume'] = df_clean['volume'].fillna(df_clean['volume'].median())
                else:
                    df_clean['volume'] = df_clean['volume'].fillna(0)
        
        # Fill gaps in time series (missing days/periods)
        if fill_gaps and isinstance(df_clean.index, pd.DatetimeIndex):
            # Determine frequency
            if len(df_clean) > 1:
                # Try to infer frequency
                freq = pd.infer_freq(df_clean.index)
                
                if freq is None:
                    # If can't infer, calculate most common frequency
                    diff = df_clean.index.to_series().diff().dropna()
                    if not diff.empty:
                        most_common_diff = diff.value_counts().index[0]
                        if most_common_diff.days == 1:
                            freq = 'D'  # Daily
                        elif most_common_diff.seconds == 3600:
                            freq = 'H'  # Hourly
                        elif most_common_diff.seconds == 60:
                            freq = 'T'  # Minute
                
                if freq is not None:
                    # Create new index with regular frequency
                    new_index = pd.date_range(start=df_clean.index.min(), end=df_clean.index.max(), freq=freq)
                    
                    # Reindex DataFrame
                    df_clean = df_clean.reindex(new_index)
                    
                    # Fill missing values
                    df_clean = df_clean.fillna(method='ffill')
            
        # Filter outliers
        if filter_outliers:
            # For each price column, detect and handle outliers
            price_cols = [col for col in df_clean.columns if col in ['open', 'high', 'low', 'close', 'adj close']]
            
            for col in price_cols:
                if col in df_clean.columns:
                    # Calculate rolling median and standard deviation
                    roll_median = df_clean[col].rolling(window=20, min_periods=5).median()
                    roll_std = df_clean[col].rolling(window=20, min_periods=5).std()
                    
                    # Identify outliers (beyond 4 standard deviations)
                    outlier_mask = (df_clean[col] < (roll_median - 4 * roll_std)) | (df_clean[col] > (roll_median + 4 * roll_std))
                    
                    # Replace outliers with rolling median
                    df_clean.loc[outlier_mask, col] = roll_median[outlier_mask]
        
        # Sort index
        df_clean = df_clean.sort_index()
        
        # Remove duplicates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # Calculate returns if close price is available
        if 'close' in df_clean.columns and 'returns' not in df_clean.columns:
            df_clean['returns'] = df_clean['close'].pct_change()
            df_clean['returns'] = df_clean['returns'].fillna(0)
        
        return df_clean
    
    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Check the quality of market data."""
        quality = {}
        
        # Check for missing values
        quality['missing_percentage'] = df.isna().mean().mean() * 100
        
        # Check for data completeness (time coverage)
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            # Calculate expected number of periods
            expected_periods = (df.index.max() - df.index.min()).days + 1
            actual_periods = len(df)
            
            # Adjust for weekends if daily data
            if pd.infer_freq(df.index) == 'D':
                # Approximate number of business days (5/7 of total days)
                expected_periods = expected_periods * 5 // 7
                
            quality['completeness'] = min(100, (actual_periods / expected_periods) * 100)
        else:
            quality['completeness'] = 0
        
        # Check for outliers
        if 'close' in df.columns:
            # Calculate rolling median and standard deviation
            roll_median = df['close'].rolling(window=20, min_periods=5).median()
            roll_std = df['close'].rolling(window=20, min_periods=5).std()
            
            # Identify potential outliers
            outliers = (df['close'] < (roll_median - 3 * roll_std)) | (df['close'] > (roll_median + 3 * roll_std))
            quality['outlier_percentage'] = outliers.mean() * 100
            
        # Check for volatility
        if 'returns' in df.columns:
            quality['volatility'] = df['returns'].std() * 100
            
        # Overall quality score (higher is better)
        quality['overall_score'] = max(0, 100 - quality.get('missing_percentage', 0) - 
                                   quality.get('outlier_percentage', 0))
                                   
        return quality
    
    def augment_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                   augmentation_method: str = 'bootstrap',
                   scenarios: List[str] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Augment market data with synthetic samples or scenarios.
        
        Args:
            data: Original market data
            augmentation_method: Method to use for augmentation (bootstrap, extreme, custom)
            scenarios: List of scenarios to generate (market_crash, high_volatility, etc.)
            
        Returns:
            Augmented data in the same format as input
        """
        if isinstance(data, dict):
            # Process each DataFrame in the dictionary
            augmented_data = {}
            for symbol, df in data.items():
                augmented_data[symbol] = self._augment_single_dataframe(
                    df, augmentation_method, scenarios
                )
            return augmented_data
        else:
            # Process a single DataFrame
            return self._augment_single_dataframe(data, augmentation_method, scenarios)
    
    def _augment_single_dataframe(self, df: pd.DataFrame, 
                               augmentation_method: str,
                               scenarios: List[str]) -> pd.DataFrame:
        """Augment a single DataFrame of market data."""
        # Make a copy of the original data
        augmented_df = df.copy()
        
        # Apply appropriate augmentation method
        if augmentation_method == 'bootstrap':
            augmented_df = self._bootstrap_augmentation(augmented_df)
        elif augmentation_method == 'extreme':
            augmented_df = self._extreme_scenarios_augmentation(augmented_df, scenarios)
        elif augmentation_method == 'custom':
            augmented_df = self._custom_augmentation(augmented_df, scenarios)
        else:
            logger.warning(f"Unsupported augmentation method: {augmentation_method}")
            
        return augmented_df
    
    def _bootstrap_augmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bootstrap augmentation: resampling returns with replacement.
        
        This method preserves the statistical properties of the original data
        while generating new synthetic samples.
        """
        if 'returns' not in df.columns and 'close' in df.columns:
            # Calculate returns if not available
            df['returns'] = df['close'].pct_change().fillna(0)
            
        if 'returns' in df.columns:
            # Get the original returns
            returns = df['returns'].values
            
            # Create dictionary to store augmented data
            aug_data = {col: df[col].values.copy() for col in df.columns}
            
            # Number of samples to generate (50% of original data)
            n_samples = len(df) // 2
            
            # Starting point for augmentation
            start_idx = len(df)
            
            # Last actual close price
            last_close = df['close'].iloc[-1]
            
            # Bootstrap: resample returns with replacement
            bootstrap_returns = np.random.choice(returns, size=n_samples)
            
            # Generate new prices and returns
            new_returns = bootstrap_returns
            new_closes = [last_close]
            
            for ret in new_returns:
                new_close = new_closes[-1] * (1 + ret)
                new_closes.append(new_close)
                
            new_closes = new_closes[1:]  # Remove the seed value
            
            # Generate new dates
            last_date = df.index[-1]
            new_dates = [last_date + pd.Timedelta(days=i+1) for i in range(n_samples)]
            
            # Create new DataFrame with augmented data
            aug_df = pd.DataFrame({
                'close': new_closes,
                'returns': new_returns
            }, index=new_dates)
            
            # Generate other OHLC values based on close prices and volatility
            avg_volatility = df['close'].pct_change().std()
            
            aug_df['open'] = aug_df['close'].shift(1) * (1 + np.random.normal(0, avg_volatility, len(aug_df)))
            aug_df['high'] = aug_df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, avg_volatility, len(aug_df))))
            aug_df['low'] = aug_df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, avg_volatility, len(aug_df))))
            
            # Forward fill NaN values
            aug_df = aug_df.fillna(method='ffill')
            
            # If volume column exists, generate synthetic volume data
            if 'volume' in df.columns:
                avg_volume = df['volume'].mean()
                vol_std = df['volume'].std()
                aug_df['volume'] = np.abs(np.random.normal(avg_volume, vol_std, len(aug_df)))
            
            # Combine original and augmented data
            combined_df = pd.concat([df, aug_df])
            
            # Mark augmented data
            combined_df['is_augmented'] = False
            combined_df.loc[aug_df.index, 'is_augmented'] = True
            
            return combined_df
            
        return df  # Return original if can't augment
    
    def _extreme_scenarios_augmentation(self, df: pd.DataFrame, scenarios: List[str]) -> pd.DataFrame:
        """
        Generate extreme market scenarios for stress testing.
        
        Scenarios:
        - market_crash: Simulate a sharp market decline
        - high_volatility: Simulate a period of high volatility
        - low_liquidity: Simulate reduced trading volume
        - v_recovery: Simulate a V-shaped recovery after a crash
        """
        if not scenarios:
            scenarios = ['market_crash', 'high_volatility']
            
        # Create a copy of the original data
        augmented_df = df.copy()
        
        # Add an indicator column for the original data
        augmented_df['scenario'] = 'original'
        
        # Last date in the original data
        last_date = augmented_df.index[-1]
        
        # For each requested scenario
        for scenario in scenarios:
            scenario_df = None
            
            if scenario == 'market_crash':
                # Simulate a market crash: -5% to -15% over 10 days
                crash_magnitude = np.random.uniform(0.05, 0.15)
                n_days = 10
                
                # Daily returns during crash
                crash_returns = np.linspace(-crash_magnitude/n_days*2, -crash_magnitude/n_days*0.5, n_days)
                np.random.shuffle(crash_returns)  # Randomize the sequence
                
                # Create new dates
                new_dates = [last_date + pd.Timedelta(days=i+1) for i in range(n_days)]
                
                # Starting price
                start_price = augmented_df['close'].iloc[-1]
                
                # Generate prices
                prices = [start_price]
                for ret in crash_returns:
                    prices.append(prices[-1] * (1 + ret))
                
                prices = prices[1:]  # Remove the seed value
                
                # Create scenario DataFrame
                scenario_df = pd.DataFrame({
                    'open': prices,
                    'high': prices,
                    'low': prices,
                    'close': prices,
                    'returns': crash_returns,
                    'scenario': 'market_crash'
                }, index=new_dates)
                
                # Add some variability to OHLC
                volatility = augmented_df['close'].pct_change().std() * 1.5  # Higher volatility during crash
                
                scenario_df['open'] = scenario_df['close'].shift(1) * (1 + np.random.normal(0, volatility, len(scenario_df)))
                scenario_df['high'] = scenario_df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, volatility/2, len(scenario_df))))
                scenario_df['low'] = scenario_df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, volatility, len(scenario_df))))
                
                # Fill NaN values
                scenario_df = scenario_df.fillna(method='ffill')
                
                # Add volume if present in original data
                if 'volume' in augmented_df.columns:
                    avg_volume = augmented_df['volume'].mean() * 1.5  # Higher volume during crash
                    vol_std = augmented_df['volume'].std() * 1.5
                    scenario_df['volume'] = np.abs(np.random.normal(avg_volume, vol_std, len(scenario_df)))
                
            elif scenario == 'high_volatility':
                # Simulate high volatility: same average returns but 2-3x standard deviation
                n_days = 20
                
                # Get original return statistics
                orig_mean = augmented_df['returns'].mean()
                orig_std = augmented_df['returns'].std() * np.random.uniform(2, 3)  # 2-3x more volatile
                
                # Generate volatile returns
                volatile_returns = np.random.normal(orig_mean, orig_std, n_days)
                
                # Create new dates
                new_dates = [last_date + pd.Timedelta(days=i+1) for i in range(n_days)]
                
                # Starting price
                start_price = augmented_df['close'].iloc[-1]
                
                # Generate prices
                prices = [start_price]
                for ret in volatile_returns:
                    prices.append(prices[-1] * (1 + ret))
                
                prices = prices[1:]  # Remove the seed value
                
                # Create scenario DataFrame
                scenario_df = pd.DataFrame({
                    'close': prices,
                    'returns': volatile_returns,
                    'scenario': 'high_volatility'
                }, index=new_dates)
                
                # Generate OHLC data with high volatility
                scenario_df['open'] = scenario_df['close'].shift(1) * (1 + np.random.normal(0, orig_std, len(scenario_df)))
                
                # High-Low range is wider in volatile markets
                hl_range = orig_std * 3
                scenario_df['high'] = scenario_df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, hl_range, len(scenario_df))))
                scenario_df['low'] = scenario_df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, hl_range, len(scenario_df))))
                
                # Fill NaN values
                scenario_df = scenario_df.fillna(method='ffill')
                
                # Add volume if present in original data
                if 'volume' in augmented_df.columns:
                    avg_volume = augmented_df['volume'].mean() * 1.2  # Slightly higher volume
                    vol_std = augmented_df['volume'].std() * 2  # Much more variable
                    scenario_df['volume'] = np.abs(np.random.normal(avg_volume, vol_std, len(scenario_df)))
            
            elif scenario == 'v_recovery':
                # Simulate a V-shaped recovery: sharp decline followed by sharp recovery
                n_days = 20  # Total days for the V shape
                crash_days = n_days // 2
                recovery_days = n_days - crash_days
                
                # Crash and recovery magnitude
                magnitude = np.random.uniform(0.1, 0.25)  # 10-25% movement
                
                # Daily returns
                crash_returns = np.linspace(-magnitude/crash_days*1.5, -magnitude/crash_days*0.5, crash_days)
                recovery_returns = np.linspace(magnitude/recovery_days*0.5, magnitude/recovery_days*1.5, recovery_days)
                v_returns = np.concatenate([crash_returns, recovery_returns])
                
                # Add some noise
                v_returns += np.random.normal(0, 0.005, n_days)
                
                # Create new dates
                new_dates = [last_date + pd.Timedelta(days=i+1) for i in range(n_days)]
                
                # Starting price
                start_price = augmented_df['close'].iloc[-1]
                
                # Generate prices
                prices = [start_price]
                for ret in v_returns:
                    prices.append(prices[-1] * (1 + ret))
                
                prices = prices[1:]  # Remove the seed value
                
                # Create scenario DataFrame
                scenario_df = pd.DataFrame({
                    'close': prices,
                    'returns': v_returns,
                    'scenario': 'v_recovery'
                }, index=new_dates)
                
                # Generate OHLC
                volatility = augmented_df['close'].pct_change().std() * 1.5  # Higher volatility during crash/recovery
                
                scenario_df['open'] = scenario_df['close'].shift(1) * (1 + np.random.normal(0, volatility, len(scenario_df)))
                scenario_df['high'] = scenario_df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, volatility/2, len(scenario_df))))
                scenario_df['low'] = scenario_df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, volatility, len(scenario_df))))
                
                # Fill NaN values
                scenario_df = scenario_df.fillna(method='ffill')
                
                # Add volume if present in original data
                if 'volume' in augmented_df.columns:
                    # Volume usually increases during both crash and recovery
                    avg_volume = augmented_df['volume'].mean() * 1.8
                    vol_std = augmented_df['volume'].std() * 1.5
                    scenario_df['volume'] = np.abs(np.random.normal(avg_volume, vol_std, len(scenario_df)))
            
            # Add the scenario data to the augmented DataFrame
            if scenario_df is not None:
                augmented_df = pd.concat([augmented_df, scenario_df])
                
        return augmented_df
    
    def _custom_augmentation(self, df: pd.DataFrame, scenarios: List[str]) -> pd.DataFrame:
        """Custom data augmentation based on provided scenarios."""
        # Create a copy of the original data
        augmented_df = df.copy()
        
        # Add an indicator column for the original data
        augmented_df['scenario'] = 'original'
        
        # Check if custom scenarios are defined in config
        custom_scenarios = self.config.get('custom_scenarios', {})
        
        # Apply each requested scenario that is defined
        for scenario in scenarios or []:
            if scenario in custom_scenarios:
                scenario_config = custom_scenarios[scenario]
                
                # Extract scenario parameters
                n_days = scenario_config.get('days', 20)
                return_mean = scenario_config.get('return_mean', 0)
                return_std = scenario_config.get('return_std', df['returns'].std() if 'returns' in df.columns else 0.01)
                volume_factor = scenario_config.get('volume_factor', 1.0)
                
                # Last date in the original data
                last_date = augmented_df.index[-1]
                
                # Create new dates
                new_dates = [last_date + pd.Timedelta(days=i+1) for i in range(n_days)]
                
                # Generate returns from distribution
                scenario_returns = np.random.normal(return_mean, return_std, n_days)
                
                # Starting price
                start_price = augmented_df['close'].iloc[-1]
                
                # Generate prices
                prices = [start_price]
                for ret in scenario_returns:
                    prices.append(prices[-1] * (1 + ret))
                
                prices = prices[1:]  # Remove the seed value
                
                # Create scenario DataFrame
                scenario_df = pd.DataFrame({
                    'close': prices,
                    'returns': scenario_returns,
                    'scenario': scenario
                }, index=new_dates)
                
                # Generate OHLC
                scenario_df['open'] = scenario_df['close'].shift(1) * (1 + np.random.normal(0, return_std, len(scenario_df)))
                scenario_df['high'] = scenario_df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, return_std/2, len(scenario_df))))
                scenario_df['low'] = scenario_df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, return_std, len(scenario_df))))
                
                # Fill NaN values
                scenario_df = scenario_df.fillna(method='ffill')
                
                # Add volume if present in original data
                if 'volume' in augmented_df.columns:
                    avg_volume = augmented_df['volume'].mean() * volume_factor
                    vol_std = augmented_df['volume'].std() * volume_factor
                    scenario_df['volume'] = np.abs(np.random.normal(avg_volume, vol_std, len(scenario_df)))
                
                # Add the scenario data to the augmented DataFrame
                augmented_df = pd.concat([augmented_df, scenario_df])
                
            else:
                logger.warning(f"Custom scenario '{scenario}' not defined in configuration")
                
        return augmented_df
    
    def save_to_cache(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                    key: str, compress: bool = True) -> str:
        """
        Save data to cache for later retrieval.
        
        Args:
            data: Data to cache
            key: Cache key/identifier
            compress: Whether to compress the data
            
        Returns:
            Path to cached file
        """
        # Create cache filename
        cache_filename = f"{key}_{int(time.time())}.pkl"
        cache_path = self.cache_dir / cache_filename
        
        # Save the data
        if compress:
            # Save compressed pickle
            with gzip.open(cache_path.with_suffix('.pkl.gz'), 'wb') as f:
                pickle.dump(data, f)
            cache_path = cache_path.with_suffix('.pkl.gz')
        else:
            # Save regular pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                
        logger.info(f"Saved data to cache: {cache_path}")
        
        return str(cache_path)
    
    def load_from_cache(self, cache_path: str) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load data from cache.
        
        Args:
            cache_path: Path to cached file
            
        Returns:
            Cached data
        """
        cache_path = Path(cache_path)
        
        try:
            # Check if file is compressed
            if cache_path.suffix == '.gz':
                with gzip.open(cache_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    
            logger.info(f"Loaded data from cache: {cache_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from cache: {str(e)}")
            return None
    
    def get_market_data(self, symbols: List[str], 
                      start_date: str = None, end_date: str = None,
                      source: str = 'yahoo', timeframe: str = 'daily',
                      clean: bool = True, augment: bool = False,
                      use_cache: bool = True, force_download: bool = False,
                      include_alternative_data: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Complete pipeline for getting market data, with caching and preprocessing.
        
        Args:
            symbols: List of symbols to get data for
            start_date: Start date (default: 10 years ago)
            end_date: End date (default: today)
            source: Data source (yahoo, ibkr, alphavantage, kraken, eastmoney)
            timeframe: Data timeframe (daily, hourly, minute)
            clean: Whether to clean the data
            augment: Whether to augment the data with synthetic samples
            use_cache: Whether to use cached data if available
            force_download: Whether to force download even if cached data exists
            include_alternative_data: Whether to include alternative data sources
            
        Returns:
            Dictionary of DataFrames with market data
        """
        # Generate cache key
        cache_key = f"market_data_{'-'.join(symbols)}_{source}_{timeframe}"
        if start_date and end_date:
            cache_key += f"_{start_date}_to_{end_date}"
        if include_alternative_data:
            cache_key += "_with_alt_data"
            
        cache_key = cache_key.replace("/", "_").replace(":", "_")
        
        # Check if cached data exists
        cached_files = list(self.cache_dir.glob(f"{cache_key}_*.pkl*"))
        cached_data = None
        
        if use_cache and cached_files and not force_download:
            # Use the most recent cache file
            most_recent = sorted(cached_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            cached_data = self.load_from_cache(most_recent)
            
        if cached_data is not None:
            logger.info(f"Using cached market data for {len(symbols)} symbols")
            return cached_data
            
        # Download data
        data_dict = self.download_data(symbols, source, start_date, end_date, timeframe, force_download)
        
        # Clean data if requested
        if clean:
            data_dict = self.clean_data(data_dict)
            
        # Augment data if requested
        if augment:
            data_dict = self.augment_data(data_dict)
            
        # Include alternative data if requested
        if include_alternative_data:
            data_dict = self._integrate_alternative_data(data_dict, symbols)
            
        # Cache the processed data
        if use_cache:
            self.save_to_cache(data_dict, cache_key)
            
        return data_dict
        
    def _integrate_alternative_data(self, data_dict: Dict[str, pd.DataFrame], 
                                  symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Integrate alternative data sources with market data.
        
        Args:
            data_dict: Dictionary of DataFrames with market data
            symbols: List of symbols
            
        Returns:
            Dictionary of DataFrames with integrated alternative data
        """
        logger.info("Integrating alternative data sources")
        
        # Process each symbol
        for symbol in symbols:
            if symbol in data_dict:
                try:
                    # Integrate alternative data
                    enhanced_df = self.alt_data_processor.integrate_alternative_data(
                        market_data=data_dict[symbol],
                        symbol=symbol,
                        include_news=True,
                        include_social=True,
                        include_satellite=False,  # Satellite data is more resource-intensive
                        include_macro=True,
                        include_events=True
                    )
                    
                    # Generate additional features
                    enhanced_df = self.alt_data_processor.generate_alternative_data_features(
                        market_data=enhanced_df,
                        symbol=symbol
                    )
                    
                    # Update the data dictionary
                    data_dict[symbol] = enhanced_df
                    
                    logger.info(f"Successfully integrated alternative data for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error integrating alternative data for {symbol}: {str(e)}")
        
        return data_dict
    
    def get_data_quality_report(self) -> Dict[str, Dict[str, float]]:
        """Get a report on data quality metrics."""
        return self.quality_metrics
    
    def export_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                  export_dir: str, format: str = 'csv') -> Dict[str, str]:
        """
        Export data to various formats.
        
        Args:
            data: Data to export
            export_dir: Directory to export to
            format: Export format (csv, json, excel, parquet)
            
        Returns:
            Dictionary mapping symbols to export paths
        """
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        export_paths = {}
        
        if isinstance(data, dict):
            # Export each DataFrame in the dictionary
            for symbol, df in data.items():
                file_path = self._export_single_dataframe(df, export_path, symbol, format)
                export_paths[symbol] = file_path
        else:
            # Export a single DataFrame
            file_path = self._export_single_dataframe(data, export_path, "data", format)
            export_paths["data"] = file_path
            
        return export_paths
    
    def _export_single_dataframe(self, df: pd.DataFrame, export_path: Path, 
                               name: str, format: str) -> str:
        """Export a single DataFrame to the specified format."""
        # Clean name for filename
        safe_name = name.replace("/", "_").replace(":", "_")
        
        try:
            if format.lower() == 'csv':
                file_path = export_path / f"{safe_name}.csv"
                df.to_csv(file_path)
            elif format.lower() == 'json':
                file_path = export_path / f"{safe_name}.json"
                df.to_json(file_path, orient='records', date_format='iso')
            elif format.lower() == 'excel':
                file_path = export_path / f"{safe_name}.xlsx"
                df.to_excel(file_path)
            elif format.lower() == 'parquet':
                file_path = export_path / f"{safe_name}.parquet"
                df.to_parquet(file_path)
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
            logger.info(f"Exported {name} to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error exporting {name}: {str(e)}")
            return None

    def analyze_alternative_data_importance(self, symbols: List[str], 
                                    target_column: str = 'returns', 
                                    lookback_days: int = 180) -> Dict[str, Dict[str, float]]:
        """
        Analyze the importance of alternative data features for predicting market returns.
        
        Args:
            symbols: List of symbols to analyze
            target_column: Target column for prediction (typically 'returns')
            lookback_days: Number of days of data to use for analysis
            
        Returns:
            Dictionary mapping symbols to feature importance scores
        """
        logger.info(f"Analyzing alternative data importance for {len(symbols)} symbols")
        
        # Get today's date
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate start date based on lookback period
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        # Get market data with alternative data
        data_dict = self.get_market_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            clean=True,
            include_alternative_data=True
        )
        
        # Dictionary to store importance scores
        importance_dict = {}
        
        # Analyze each symbol
        for symbol in symbols:
            if symbol in data_dict and not data_dict[symbol].empty:
                df = data_dict[symbol]
                
                # Create target variable (next day's returns)
                target = df[target_column].shift(-1).dropna()
                
                # Align features with target (drop last row where target is NaN)
                features = df.iloc[:-1].copy()
                
                # Calculate feature importance
                try:
                    importance_scores = self.alt_data_processor.get_alternative_data_importance(
                        features=features,
                        target=target
                    )
                    
                    importance_dict[symbol] = importance_scores
                    logger.info(f"Calculated feature importance for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error calculating feature importance for {symbol}: {str(e)}")
                    importance_dict[symbol] = {}
        
        return importance_dict

# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = MarketDataProcessor()
    
    # Download data for some symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = processor.get_market_data(symbols, start_date='2020-01-01', end_date='2023-01-01')
    
    # Check data quality
    quality_report = processor.get_data_quality_report()
    for symbol, metrics in quality_report.items():
        print(f"\nData quality for {symbol}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")
    
    # Example of using alternative data (commented out for normal usage)
    # alt_data = processor.get_market_data(symbols, start_date='2022-01-01', end_date='2023-01-01', include_alternative_data=True)
    # importance = processor.analyze_alternative_data_importance(symbols)
    
    # Export the data
    processor.export_data(data, "exported_data", format='csv')