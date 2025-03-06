"""
IBKR data connector for handling local IBKR data files.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
from .base_connector import BaseDataConnector
from ..utils.config import IBKR_DATA_PATH

class IBKRDataConnector(BaseDataConnector):
    def __init__(self, data_path: Path = IBKR_DATA_PATH):
        """
        Initialize the IBKR data connector.
        
        Args:
            data_path: Path to IBKR data directory
        """
        super().__init__(data_path)
        self.required_columns = ['open', 'high', 'low', 'close', 'volume', 'average', 'barCount']

    def load_data(self,
                 symbols: Union[str, List[str]],
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 timeframe: str = "1d") -> pd.DataFrame:
        """
        Load IBKR data for specified symbols and date range.
        
        Args:
            symbols: Single symbol or list of symbols to load
            start_date: Start date for data loading (YYYY-MM-DD)
            end_date: End date for data loading (YYYY-MM-DD)
            timeframe: Data timeframe (e.g., "1d", "1m")
            
        Returns:
            DataFrame with the loaded data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date) if start_date else None
        end_dt = pd.to_datetime(end_date) if end_date else None

        all_data = []
        timeframe_path = self.data_path / timeframe

        for symbol in symbols:
            try:
                # Handle different CSV filename patterns
                symbol_files = list(timeframe_path.glob(f"{symbol}*.csv"))
                
                if not symbol_files:
                    continue

                # Use the most recent file if multiple exist
                symbol_file = sorted(symbol_files)[-1]
                
                # Read the CSV file
                df = pd.read_csv(symbol_file)
                
                # Ensure required columns exist
                for col in self.required_columns:
                    if col not in df.columns:
                        raise ValueError(f"Required column {col} not found in {symbol_file}")

                # Convert date column
                df['date'] = pd.to_datetime(df['date'])
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Filter by date range if specified
                if start_dt:
                    df = df[df['date'] >= start_dt]
                if end_dt:
                    df = df[df['date'] <= end_dt]
                
                all_data.append(df)
                
            except Exception as e:
                print(f"Error loading data for {symbol}: {str(e)}")
                continue

        if not all_data:
            raise ValueError(f"No data loaded for symbols: {symbols}")

        # Combine all data
        combined_data = pd.concat(all_data, axis=0, ignore_index=True)
        
        # Sort by date and symbol
        combined_data = combined_data.sort_values(['date', 'symbol'])
        
        return combined_data

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess IBKR data according to FinTSB standards.
        
        Args:
            data: Raw IBKR data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        result = data.copy()
        
        # Calculate additional features
        result['returns'] = result.groupby('symbol')['close'].pct_change()
        result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
        
        # Calculate volatility
        result['volatility'] = result.groupby('symbol')['returns'].rolling(window=20).std().reset_index(0, drop=True)
        
        # Calculate moving averages
        result['ma_20'] = result.groupby('symbol')['close'].rolling(window=20).mean().reset_index(0, drop=True)
        result['ma_50'] = result.groupby('symbol')['close'].rolling(window=50).mean().reset_index(0, drop=True)
        
        # Calculate RSI
        delta = result.groupby('symbol')['close'].diff()
        gain = (delta.where(delta > 0, 0)).groupby(result['symbol'])
        loss = (-delta.where(delta < 0, 0)).groupby(result['symbol'])
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate trading activity features
        result['volume_ma'] = result.groupby('symbol')['volume'].rolling(window=20).mean().reset_index(0, drop=True)
        result['volume_ratio'] = result['volume'] / result['volume_ma']
        
        # Handle missing values
        result = result.fillna(method='ffill')  # Forward fill
        result = result.fillna(0)  # Fill remaining NaNs with 0
        
        return result

    def _is_valid_csv(self, file_path: Path) -> bool:
        """
        Check if a CSV file is valid and contains required columns.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Boolean indicating if the file is valid
        """
        try:
            df = pd.read_csv(file_path, nrows=1)
            return all(col in df.columns for col in self.required_columns)
        except Exception:
            return False

    def get_latest_data_info(self) -> Dict[str, datetime]:
        """
        Get the latest data timestamp for each symbol.
        
        Returns:
            Dictionary mapping symbols to their latest data timestamp
        """
        latest_dates = {}
        
        for timeframe in self.get_available_timeframes():
            timeframe_path = self.data_path / timeframe
            
            for file_path in timeframe_path.glob("*.csv"):
                if not self._is_valid_csv(file_path):
                    continue
                    
                symbol = file_path.stem.split()[0]  # Handle spaces in filenames
                
                try:
                    df = pd.read_csv(file_path)
                    latest_date = pd.to_datetime(df['date']).max()
                    
                    if symbol not in latest_dates or latest_date > latest_dates[symbol]:
                        latest_dates[symbol] = latest_date
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
                    
        return latest_dates