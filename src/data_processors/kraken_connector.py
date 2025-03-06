"""
Kraken data connector for handling Kraken cryptocurrency data.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from .base_connector import BaseDataConnector
from ..utils.config import KRAKEN_DATA_PATH

logger = logging.getLogger(__name__)

class KrakenDataConnector(BaseDataConnector):
    def __init__(self, data_path: Path = KRAKEN_DATA_PATH):
        """
        Initialize the Kraken data connector.
        
        Args:
            data_path: Path to Kraken data directory
        """
        super().__init__(data_path)
        self.required_columns = ['open', 'high', 'low', 'close', 'volume', 'trades', 'vwap']
        self.timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        self.supported_pairs = self._load_supported_pairs()
        
    def _load_supported_pairs(self) -> List[str]:
        """Load list of supported trading pairs."""
        try:
            pairs_file = self.data_path / "kraken_symbols_universe.csv"
            if pairs_file.exists():
                df = pd.read_csv(pairs_file)
                if 'symbol' in df.columns:
                    return df['symbol'].tolist()
            
            # If file doesn't exist or doesn't have symbol column, search for pairs
            return self._discover_available_pairs()
        except Exception as e:
            logger.error(f"Error loading supported pairs: {str(e)}")
            return self._discover_available_pairs()
    
    def _discover_available_pairs(self) -> List[str]:
        """Discover available pairs from data directory."""
        pairs = set()
        
        for timeframe in self.timeframes:
            timeframe_dir = self.data_path / timeframe
            if not timeframe_dir.exists():
                continue
                
            for file_path in timeframe_dir.glob("*.csv"):
                # Extract pair name from filename
                pair = file_path.stem.split("_")[0]
                pairs.add(pair)
        
        return list(pairs)
    
    def load_data(self,
                 symbols: Union[str, List[str]],
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 timeframe: str = "1d") -> pd.DataFrame:
        """
        Load Kraken data for specified symbols and date range.
        
        Args:
            symbols: Single symbol or list of symbols to load
            start_date: Start date for data loading (YYYY-MM-DD)
            end_date: End date for data loading (YYYY-MM-DD)
            timeframe: Data timeframe (e.g., "1d", "1h")
            
        Returns:
            DataFrame with the loaded data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Convert dates to datetime for filtering
        start_dt = pd.to_datetime(start_date) if start_date else None
        end_dt = pd.to_datetime(end_date) if end_date else None
        
        all_data = []
        timeframe_path = self.data_path / timeframe
        
        if not timeframe_path.exists():
            raise ValueError(f"Timeframe '{timeframe}' data directory not found")
        
        for symbol in symbols:
            try:
                # Standardize symbol name
                symbol = symbol.upper()
                
                # Try different filename patterns
                symbol_files = list(timeframe_path.glob(f"{symbol}*.csv"))
                symbol_files.extend(list(timeframe_path.glob(f"{symbol.lower()}*.csv")))
                
                if not symbol_files:
                    logger.warning(f"No data files found for symbol: {symbol}")
                    continue
                
                # Take the most recent file
                symbol_file = sorted(symbol_files)[-1]
                
                # Load the data
                df = pd.read_csv(symbol_file)
                
                # Check required columns
                missing_cols = [col for col in self.required_columns if col not in df.columns]
                if missing_cols:
                    # Try to map column names if not found directly
                    df = self._map_columns(df)
                    
                    # Recheck after mapping
                    still_missing = [col for col in self.required_columns if col not in df.columns]
                    if still_missing:
                        raise ValueError(f"Required columns {still_missing} not found in {symbol_file}")
                
                # Ensure datetime format for 'date' column
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                elif 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp'])
                else:
                    raise ValueError(f"No date/timestamp column found in {symbol_file}")
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Filter by date range if specified
                if start_dt:
                    df = df[df['date'] >= start_dt]
                if end_dt:
                    df = df[df['date'] <= end_dt]
                
                # Skip if no data after filtering
                if df.empty:
                    logger.warning(f"No data in date range for symbol: {symbol}")
                    continue
                    
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError(f"No data loaded for any of the symbols: {symbols}")
        
        # Combine all data
        combined_data = pd.concat(all_data, axis=0, ignore_index=True)
        
        # Sort by date and symbol
        combined_data = combined_data.sort_values(['date', 'symbol'])
        
        return combined_data
    
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map columns to standard names if possible."""
        # Common mappings
        column_mappings = {
            'time': 'date',
            'timestamp': 'date',
            'price': 'close',
            'vol': 'volume',
            'last': 'close',
            'vw': 'vwap',
            'num_trades': 'trades'
        }
        
        # Create a copy to avoid modifying original
        result = df.copy()
        
        # Map columns
        for src, dst in column_mappings.items():
            if src in result.columns and dst not in result.columns:
                result[dst] = result[src]
        
        # Infer missing OHLC if possible
        if 'close' in result.columns:
            if 'open' not in result.columns:
                result['open'] = result['close']
            if 'high' not in result.columns:
                result['high'] = result['close']
            if 'low' not in result.columns:
                result['low'] = result['close']
        
        return result
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Kraken data with crypto-specific features.
        
        Args:
            data: Raw Kraken data DataFrame
            
        Returns:
            Preprocessed DataFrame with additional features
        """
        result = data.copy()
        
        # Calculate returns
        result['returns'] = result.groupby('symbol')['close'].pct_change()
        result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
        
        # Calculate volatility
        result['volatility_daily'] = result.groupby('symbol')['returns'].rolling(window=24).std().reset_index(0, drop=True)
        
        # Calculate moving averages
        result['ma_20'] = result.groupby('symbol')['close'].rolling(window=20).mean().reset_index(0, drop=True)
        result['ma_50'] = result.groupby('symbol')['close'].rolling(window=50).mean().reset_index(0, drop=True)
        result['ma_200'] = result.groupby('symbol')['close'].rolling(window=200).mean().reset_index(0, drop=True)
        
        # Calculate MACD
        result['ema_12'] = result.groupby('symbol')['close'].ewm(span=12).mean().reset_index(0, drop=True)
        result['ema_26'] = result.groupby('symbol')['close'].ewm(span=26).mean().reset_index(0, drop=True)
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result.groupby('symbol')['macd'].ewm(span=9).mean().reset_index(0, drop=True)
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # Calculate RSI
        delta = result.groupby('symbol')['close'].diff()
        gain = (delta.where(delta > 0, 0)).groupby(result['symbol'])
        loss = (-delta.where(delta < 0, 0)).groupby(result['symbol'])
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate volume-based metrics
        result['volume_ma'] = result.groupby('symbol')['volume'].rolling(window=20).mean().reset_index(0, drop=True)
        result['volume_ratio'] = result['volume'] / result['volume_ma']
        
        # Calculate crypto-specific metrics
        
        # VWAP-based metrics (if available)
        if 'vwap' in result.columns:
            result['price_to_vwap'] = result['close'] / result['vwap']
            result['vwap_delta'] = (result['close'] - result['vwap']) / result['vwap']
        
        # NVT (Network Value to Transactions) ratio approximation
        # This normally uses on-chain data, but we can approximate with volume
        result['nvt_proxy'] = result['close'] / (result['volume'] / result['close'])
        
        # Handle missing values
        result = result.fillna(method='ffill')  # Forward fill
        result = result.fillna(0)  # Fill remaining NaNs with 0
        
        return result
    
    def get_available_timeframes(self) -> List[str]:
        """
        Get list of available timeframes in the data directory.
        
        Returns:
            List of available timeframe strings
        """
        available = []
        for timeframe in self.timeframes:
            if (self.data_path / timeframe).exists():
                available.append(timeframe)
        return available
    
    def get_symbol_universe(self) -> pd.DataFrame:
        """
        Get the universe of available symbols with metadata.
        
        Returns:
            DataFrame with symbol information
        """
        # Try to load from universe file
        universe_file = self.data_path / "kraken_symbols_universe.csv"
        if universe_file.exists():
            return pd.read_csv(universe_file)
        
        # Otherwise construct universe from available files
        symbols = []
        for timeframe in self.get_available_timeframes():
            timeframe_dir = self.data_path / timeframe
            for file_path in timeframe_dir.glob("*.csv"):
                symbol = file_path.stem.split("_")[0].upper()
                symbols.append({
                    'symbol': symbol,
                    'source': 'kraken',
                    'data_file': str(file_path.name),
                    'timeframe': timeframe
                })
        
        if not symbols:
            raise ValueError("No symbols found in Kraken data directory")
            
        return pd.DataFrame(symbols).drop_duplicates(subset=['symbol'])
    
    def get_latest_data_info(self) -> Dict[str, datetime]:
        """
        Get the latest data timestamp for each symbol.
        
        Returns:
            Dictionary mapping symbols to their latest data timestamp
        """
        latest_dates = {}
        
        for timeframe in self.get_available_timeframes():
            timeframe_dir = self.data_path / timeframe
            for file_path in timeframe_dir.glob("*.csv"):
                try:
                    # Extract symbol from filename
                    symbol = file_path.stem.split("_")[0].upper()
                    
                    # Read the last few rows to get the latest date
                    df = pd.read_csv(file_path, nrows=100, skiprows=lambda x: x > 0 and x < max(0, sum(1 for _ in open(file_path)) - 100))
                    
                    date_col = 'date' if 'date' in df.columns else 'timestamp'
                    if date_col in df.columns:
                        latest_date = pd.to_datetime(df[date_col]).max()
                        
                        if symbol not in latest_dates or latest_date > latest_dates[symbol]:
                            latest_dates[symbol] = latest_date
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {str(e)}")
                    continue
        
        return latest_dates