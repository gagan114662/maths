#!/usr/bin/env python
"""
QuantConnect Market Regime Detection Module

This module implements advanced market regime detection for adaptive strategy behavior:
1. Identifies different market regimes (bull, bear, sideways, high-volatility)
2. Provides regime-specific strategy optimization
3. Enables adaptive position sizing based on current regime
4. Supports dynamic parameter adjustment for different regimes
5. Tracks regime transitions for advanced reporting
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
from hmmlearn import hmm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qc_market_regimes.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class MarketRegimeClassifier:
    """Class for market regime classification and detection."""
    
    REGIMES = {
        0: "Bull Market",
        1: "Bear Market",
        2: "Sideways Market",
        3: "High Volatility",
        4: "Low Volatility Bull",
        5: "Recovery"
    }
    
    def __init__(self, lookback_period=252, short_lookback=20, num_regimes=4):
        """
        Initialize the market regime classifier.
        
        Args:
            lookback_period (int): Period for long-term regime detection (default: 252 trading days)
            short_lookback (int): Period for short-term feature calculation (default: 20 trading days)
            num_regimes (int): Number of market regimes to identify (default: 4)
        """
        self.lookback_period = lookback_period
        self.short_lookback = short_lookback
        self.num_regimes = num_regimes
        self.hmm_model = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.regime_transitions = []
        self.current_regime = None
        self.regime_metrics = {}
        
    def calculate_features(self, price_data):
        """
        Calculate features for regime detection.
        
        Args:
            price_data (DataFrame): Historical price data with at least OHLC columns
            
        Returns:
            DataFrame: Features for regime detection
        """
        # Make a copy to avoid modifying the original data
        df = price_data.copy()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate volatility features
        df['volatility'] = df['returns'].rolling(window=self.short_lookback).std() * np.sqrt(252)
        df['atr'] = self._calculate_atr(df)
        df['atr_pct'] = df['atr'] / df['close']
        
        # Calculate trend features
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        df['trend_20_50'] = df['sma_20'] / df['sma_50'] - 1
        df['trend_50_200'] = df['sma_50'] / df['sma_200'] - 1
        
        # Calculate momentum features
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['mom_20'] = df['close'] / df['close'].shift(20) - 1
        df['mom_60'] = df['close'] / df['close'].shift(60) - 1
        
        # Calculate mean reversion features
        df['mean_dev_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['bollinger_width'] = self._calculate_bollinger_width(df)
        
        # Calculate volume features (if volume is available)
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        else:
            df['volume_ratio'] = 1.0
        
        # Calculate drawdown
        df['rolling_max'] = df['close'].rolling(window=self.lookback_period, min_periods=1).max()
        df['drawdown'] = df['close'] / df['rolling_max'] - 1
        
        # Calculate autocorrelation features
        df['autocorr_5'] = df['returns'].rolling(window=20).apply(
            lambda x: x.autocorr(lag=5) if len(x.dropna()) > 5 else np.nan
        )
        
        # Drop NaN values
        df = df.dropna()
        
        # Select features for training
        features = df[[
            'volatility', 'atr_pct', 'trend_20_50', 'trend_50_200',
            'rsi_14', 'mom_20', 'mom_60', 'mean_dev_20',
            'bollinger_width', 'volume_ratio', 'drawdown', 'autocorr_5'
        ]]
        
        return features, df
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        
        ema_up = up.ewm(com=period-1, adjust=False).mean()
        ema_down = down.ewm(com=period-1, adjust=False).mean()
        
        rs = ema_up / ema_down
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_width(self, df, period=20, num_std=2):
        """Calculate Bollinger Band width."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        middle_band = typical_price.rolling(window=period).mean()
        std_dev = typical_price.rolling(window=period).std()
        
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        bollinger_width = (upper_band - lower_band) / middle_band
        
        return bollinger_width
    
    def fit(self, price_data, method="kmeans"):
        """
        Fit the market regime classifier to historical data.
        
        Args:
            price_data (DataFrame): Historical price data with at least OHLC columns
            method (str): Classification method ('kmeans' or 'hmm')
            
        Returns:
            self: Fitted classifier
        """
        # Calculate features
        features, price_df = self.calculate_features(price_data)
        
        # Normalize features
        scaled_features = self.scaler.fit_transform(features)
        
        if method == "kmeans":
            # Fit KMeans clustering
            self.kmeans_model = KMeans(n_clusters=self.num_regimes, random_state=42, n_init=10)
            self.kmeans_model.fit(scaled_features)
            
            # Assign regimes
            regimes = self.kmeans_model.predict(scaled_features)
            
            # Map cluster numbers to meaningful regime labels
            self._map_kmeans_clusters_to_regimes(features, regimes, price_df)
            
        elif method == "hmm":
            # Fit Hidden Markov Model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.hmm_model = hmm.GaussianHMM(
                    n_components=self.num_regimes, 
                    covariance_type="full", 
                    n_iter=1000,
                    random_state=42
                )
                self.hmm_model.fit(scaled_features)
            
            # Predict regimes
            regimes = self.hmm_model.predict(scaled_features)
            
            # Map HMM states to meaningful regime labels
            self._map_hmm_states_to_regimes(features, regimes, price_df)
        
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'kmeans' or 'hmm'.")
        
        # Detect regime transitions
        self._detect_regime_transitions(regimes, price_df.index)
        
        # Set current regime
        if len(regimes) > 0:
            self.current_regime = regimes[-1]
        
        return self
    
    def _map_kmeans_clusters_to_regimes(self, features, clusters, price_df):
        """Map KMeans clusters to meaningful regime labels."""
        # Calculate average feature values for each cluster
        cluster_data = pd.DataFrame({
            'cluster': clusters,
            'return': price_df['returns'].values,
            'volatility': features['volatility'].values,
            'trend': features['trend_20_50'].values,
            'mom': features['mom_20'].values,
            'drawdown': features['drawdown'].values,
            'price': price_df['close'].values
        })
        
        cluster_stats = cluster_data.groupby('cluster').agg({
            'return': 'mean',
            'volatility': 'mean',
            'trend': 'mean',
            'mom': 'mean',
            'drawdown': 'mean',
            'price': ['mean', 'std']
        })
        
        # Map clusters to regimes
        regime_map = {}
        
        # Identify bull market: positive returns, positive trend, lower volatility
        bull_cluster = cluster_stats['return'].idxmax()
        regime_map[bull_cluster] = 0  # Bull Market
        
        # Identify bear market: negative returns, negative trend, higher volatility
        bear_cluster = cluster_stats['return'].idxmin()
        regime_map[bear_cluster] = 1  # Bear Market
        
        # Identify sideways market: low absolute returns, low trend
        sideways_candidates = cluster_stats.drop([bull_cluster, bear_cluster])
        if not sideways_candidates.empty:
            sideways_cluster = sideways_candidates['trend'].abs().idxmin()
            regime_map[sideways_cluster] = 2  # Sideways Market
        
        # Identify high volatility: highest volatility cluster among remaining
        remaining_clusters = set(range(self.num_regimes)) - set(regime_map.keys())
        if remaining_clusters:
            high_vol_cluster = cluster_stats.loc[list(remaining_clusters), 'volatility'].idxmax()
            regime_map[high_vol_cluster] = 3  # High Volatility
        
        # Map any remaining clusters to additional regimes or existing ones
        remaining_clusters = set(range(self.num_regimes)) - set(regime_map.keys())
        regime_counter = 4
        for cluster in remaining_clusters:
            if regime_counter < len(self.REGIMES):
                regime_map[cluster] = regime_counter
                regime_counter += 1
            else:
                # Assign to closest existing regime
                regime_map[cluster] = 0  # Default to bull market
        
        # Store regime statistics
        self.regime_metrics = {}
        for cluster, regime in regime_map.items():
            stats = cluster_stats.loc[cluster]
            self.regime_metrics[regime] = {
                'return': stats['return'],
                'volatility': stats['volatility'],
                'trend': stats['trend'],
                'momentum': stats['mom'],
                'drawdown': stats['drawdown'],
                'mean_price': stats[('price', 'mean')],
                'price_std': stats[('price', 'std')]
            }
        
        # Map clusters to regimes
        self.cluster_to_regime = regime_map
        
        return regime_map
    
    def _map_hmm_states_to_regimes(self, features, states, dates):
        """Map HMM states to meaningful regime labels."""
        # Calculate average feature values for each state
        state_data = pd.DataFrame({
            'state': states,
            'volatility': features['volatility'].values,
            'trend': features['trend_20_50'].values,
            'mom': features['mom_20'].values,
            'drawdown': features['drawdown'].values,
            'atr_pct': features['atr_pct'].values
        })
        
        state_stats = state_data.groupby('state').agg({
            'volatility': 'mean',
            'trend': 'mean',
            'mom': 'mean',
            'drawdown': 'mean',
            'atr_pct': 'mean'
        })
        
        # Map states to regimes
        regime_map = {}
        
        # Identify bull market: positive trend, positive momentum
        bull_candidates = state_stats[(state_stats['trend'] > 0) & (state_stats['mom'] > 0)]
        if not bull_candidates.empty:
            bull_state = bull_candidates['volatility'].idxmin()  # Lower volatility bull market
            regime_map[bull_state] = 0  # Bull Market
        else:
            # If no clear bull market, use the state with highest trend
            bull_state = state_stats['trend'].idxmax()
            regime_map[bull_state] = 0  # Bull Market
        
        # Identify bear market: negative trend, negative momentum, larger drawdowns
        bear_candidates = state_stats[(state_stats['trend'] < 0) & (state_stats['mom'] < 0)]
        if not bear_candidates.empty:
            bear_state = bear_candidates['drawdown'].idxmin()  # Largest drawdown
            regime_map[bear_state] = 1  # Bear Market
        else:
            # If no clear bear market, use the state with lowest trend
            bear_state = state_stats['trend'].idxmin()
            regime_map[bear_state] = 1  # Bear Market
        
        # Identify high volatility: highest volatility state among remaining
        remaining_states = set(range(self.num_regimes)) - set(regime_map.keys())
        if remaining_states:
            vol_ranking = state_stats.loc[list(remaining_states), 'volatility'].sort_values(ascending=False)
            if not vol_ranking.empty:
                high_vol_state = vol_ranking.index[0]
                regime_map[high_vol_state] = 3  # High Volatility
                remaining_states.remove(high_vol_state)
        
        # Assign sideways market to state with lowest absolute trend among remaining
        if remaining_states:
            remaining_trend_abs = state_stats.loc[list(remaining_states), 'trend'].abs()
            if not remaining_trend_abs.empty:
                sideways_state = remaining_trend_abs.idxmin()
                regime_map[sideways_state] = 2  # Sideways Market
                remaining_states.remove(sideways_state)
        
        # Map any remaining states to additional regimes or existing ones
        regime_counter = 4
        for state in remaining_states:
            if regime_counter < len(self.REGIMES):
                regime_map[state] = regime_counter
                regime_counter += 1
            else:
                # Assign to closest existing regime
                regime_map[state] = 0  # Default to bull market
        
        # Store regime statistics
        self.regime_metrics = {}
        for state, regime in regime_map.items():
            stats = state_stats.loc[state]
            self.regime_metrics[regime] = {
                'volatility': stats['volatility'],
                'trend': stats['trend'],
                'momentum': stats['mom'],
                'drawdown': stats['drawdown'],
                'atr_pct': stats['atr_pct']
            }
        
        # Map clusters to regimes
        self.state_to_regime = regime_map
        
        return regime_map
    
    def _detect_regime_transitions(self, regimes, dates):
        """Detect and record regime transitions."""
        if len(regimes) < 2:
            return []
        
        transitions = []
        current_regime = regimes[0]
        start_date = dates[0]
        
        for i in range(1, len(regimes)):
            if regimes[i] != current_regime:
                # Record the transition
                transitions.append({
                    'from_regime': int(current_regime),
                    'to_regime': int(regimes[i]),
                    'from_regime_name': self.REGIMES.get(int(current_regime), "Unknown"),
                    'to_regime_name': self.REGIMES.get(int(regimes[i]), "Unknown"),
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': dates[i].strftime('%Y-%m-%d'),
                    'duration_days': (dates[i] - start_date).days
                })
                
                # Update current regime and start date
                current_regime = regimes[i]
                start_date = dates[i]
        
        # Record the last regime period
        transitions.append({
            'from_regime': int(current_regime),
            'to_regime': None,
            'from_regime_name': self.REGIMES.get(int(current_regime), "Unknown"),
            'to_regime_name': "Current",
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': dates[-1].strftime('%Y-%m-%d'),
            'duration_days': (dates[-1] - start_date).days
        })
        
        self.regime_transitions = transitions
        return transitions
    
    def predict(self, price_data, method="kmeans"):
        """
        Predict market regimes for new data.
        
        Args:
            price_data (DataFrame): Price data with at least OHLC columns
            method (str): Classification method ('kmeans' or 'hmm')
            
        Returns:
            ndarray: Predicted regimes
        """
        # Check if model is trained
        if method == "kmeans" and self.kmeans_model is None:
            raise ValueError("KMeans model not trained. Call fit() first.")
        elif method == "hmm" and self.hmm_model is None:
            raise ValueError("HMM model not trained. Call fit() first.")
        
        # Calculate features
        features, _ = self.calculate_features(price_data)
        
        # Normalize features
        scaled_features = self.scaler.transform(features)
        
        # Predict regimes
        if method == "kmeans":
            clusters = self.kmeans_model.predict(scaled_features)
            regimes = np.array([self.cluster_to_regime.get(cluster, 0) for cluster in clusters])
        else:  # method == "hmm"
            states = self.hmm_model.predict(scaled_features)
            regimes = np.array([self.state_to_regime.get(state, 0) for state in states])
        
        # Update current regime
        if len(regimes) > 0:
            self.current_regime = regimes[-1]
        
        return regimes
    
    def get_regime_metrics(self):
        """
        Get metrics for each identified regime.
        
        Returns:
            dict: Metrics for each regime
        """
        if not self.regime_metrics:
            logger.warning("No regime metrics available. Call fit() first.")
            return {}
        
        result = {}
        for regime_id, metrics in self.regime_metrics.items():
            regime_name = self.REGIMES.get(regime_id, f"Regime {regime_id}")
            result[regime_name] = metrics
        
        return result
    
    def get_regime_transitions(self):
        """
        Get regime transitions history.
        
        Returns:
            list: Regime transitions
        """
        return self.regime_transitions
    
    def get_current_regime(self):
        """
        Get current market regime.
        
        Returns:
            dict: Current regime information
        """
        if self.current_regime is None:
            return {"regime_id": None, "regime_name": "Unknown", "metrics": {}}
        
        regime_id = self.current_regime
        regime_name = self.REGIMES.get(regime_id, f"Regime {regime_id}")
        metrics = self.regime_metrics.get(regime_id, {})
        
        return {
            "regime_id": regime_id,
            "regime_name": regime_name,
            "metrics": metrics
        }
    
    def plot_regimes(self, price_data, regimes=None, title="Market Regimes"):
        """
        Plot market regimes with price data.
        
        Args:
            price_data (DataFrame): Price data with close prices
            regimes (ndarray, optional): Predicted regimes. If None, predictions will be generated.
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if regimes is None:
            # Generate predictions if not provided
            features, _ = self.calculate_features(price_data)
            scaled_features = self.scaler.transform(features)
            
            if hasattr(self, 'kmeans_model') and self.kmeans_model is not None:
                clusters = self.kmeans_model.predict(scaled_features)
                regimes = np.array([self.cluster_to_regime.get(cluster, 0) for cluster in clusters])
            elif hasattr(self, 'hmm_model') and self.hmm_model is not None:
                states = self.hmm_model.predict(scaled_features)
                regimes = np.array([self.state_to_regime.get(state, 0) for state in states])
            else:
                raise ValueError("No trained model available for prediction")
        
        # Create figure and axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price data
        ax1.plot(price_data.index, price_data['close'], color='black', linewidth=1.5)
        ax1.set_title(title)
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        # Plot regimes as background colors
        if len(regimes) > 0:
            # Get start index for regimes (may be different from price data due to feature calculation)
            start_idx = max(0, len(price_data) - len(regimes))
            regime_dates = price_data.index[start_idx:start_idx+len(regimes)]
            
            # Define colors for regimes
            colors = ['#90EE90', '#FF9999', '#FFCC99', '#FF99CC', '#99CCFF', '#CC99FF']
            
            # Plot each regime period
            regime_changes = np.where(np.diff(np.append([-1], regimes)))[0]
            for i in range(len(regime_changes)):
                start = regime_changes[i]
                end = regime_changes[i+1] if i < len(regime_changes) - 1 else len(regimes)
                regime = regimes[start]
                
                if start < len(regime_dates) and end <= len(regime_dates):
                    # Add colored background for the regime period
                    ax1.axvspan(regime_dates[start], regime_dates[end-1], 
                                alpha=0.2, color=colors[regime % len(colors)])
                    
                    # Add regime label in the middle of the period
                    mid_point = start + (end - start) // 2
                    if mid_point < len(regime_dates):
                        ax1.text(regime_dates[mid_point], ax1.get_ylim()[1] * 0.95,
                                self.REGIMES.get(regime, f"Regime {regime}"),
                                ha='center', va='top', backgroundcolor='white', alpha=0.7)
            
            # Plot regime index in the second subplot
            ax2.plot(regime_dates, regimes, color='blue', linewidth=1.5)
            ax2.set_ylabel('Regime')
            ax2.set_yticks(list(self.REGIMES.keys()))
            ax2.set_yticklabels([self.REGIMES.get(i, f"Regime {i}") for i in self.REGIMES.keys()])
            ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.xlabel('Date')
        fig.autofmt_xdate()
        plt.tight_layout()
        
        return fig
    
    def save_model(self, filepath):
        """
        Save the trained model to file.
        
        Args:
            filepath (str): Path to save the model
            
        Returns:
            bool: True if saved successfully
        """
        # Create model data
        model_data = {
            'lookback_period': self.lookback_period,
            'short_lookback': self.short_lookback,
            'num_regimes': self.num_regimes,
            'regime_metrics': self.regime_metrics,
            'regime_transitions': self.regime_transitions,
            'current_regime': self.current_regime
        }
        
        try:
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from file.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            MarketRegimeClassifier: Loaded model
        """
        try:
            # Load from file
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            # Create new instance
            instance = cls(
                lookback_period=model_data.get('lookback_period', 252),
                short_lookback=model_data.get('short_lookback', 20),
                num_regimes=model_data.get('num_regimes', 4)
            )
            
            # Set model attributes
            instance.regime_metrics = model_data.get('regime_metrics', {})
            instance.regime_transitions = model_data.get('regime_transitions', [])
            instance.current_regime = model_data.get('current_regime')
            
            logger.info(f"Model loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

class RegimeAdaptiveStrategyOptimizer:
    """Class for regime-specific strategy optimization."""
    
    def __init__(self, regime_classifier=None):
        """
        Initialize the regime-adaptive strategy optimizer.
        
        Args:
            regime_classifier (MarketRegimeClassifier, optional): Trained regime classifier
        """
        self.regime_classifier = regime_classifier or MarketRegimeClassifier()
        self.regime_optimizers = {}
        self.default_parameters = {}
        self.regime_performance = {}
    
    def get_regime_parameters(self, regime_id=None):
        """
        Get optimized parameters for a specific regime.
        
        Args:
            regime_id (int, optional): Regime ID. If None, current regime is used.
            
        Returns:
            dict: Optimized parameters for the regime
        """
        if regime_id is None:
            current_regime = self.regime_classifier.get_current_regime()
            regime_id = current_regime['regime_id']
        
        # Return parameters for the specified regime if available
        if regime_id in self.regime_optimizers:
            return self.regime_optimizers[regime_id]
        
        # Fall back to default parameters
        return self.default_parameters
    
    def optimize_for_regime(self, regime_id, price_data, default_parameters, parameter_ranges, 
                           optimization_criterion='sharpe_ratio', method='grid_search'):
        """
        Optimize strategy parameters for a specific market regime.
        
        Args:
            regime_id (int): Regime ID to optimize for
            price_data (DataFrame): Historical price data with regime labels
            default_parameters (dict): Default strategy parameters
            parameter_ranges (dict): Ranges for parameter optimization
            optimization_criterion (str): Criterion to optimize (sharpe_ratio, return, etc.)
            method (str): Optimization method (grid_search, bayesian, genetic)
            
        Returns:
            dict: Optimized parameters for the regime
        """
        # Store default parameters
        self.default_parameters = default_parameters
        
        # Filter data for the specific regime
        regime_data = self._filter_data_for_regime(price_data, regime_id)
        
        if regime_data.empty:
            logger.warning(f"No data available for regime {regime_id}. Using default parameters.")
            self.regime_optimizers[regime_id] = default_parameters
            return default_parameters
        
        # Perform optimization based on the specified method
        if method == 'grid_search':
            optimized_params = self._grid_search_optimization(
                regime_data, default_parameters, parameter_ranges, optimization_criterion
            )
        elif method == 'bayesian':
            optimized_params = self._bayesian_optimization(
                regime_data, default_parameters, parameter_ranges, optimization_criterion
            )
        elif method == 'genetic':
            optimized_params = self._genetic_optimization(
                regime_data, default_parameters, parameter_ranges, optimization_criterion
            )
        else:
            logger.warning(f"Unsupported optimization method: {method}. Using grid search.")
            optimized_params = self._grid_search_optimization(
                regime_data, default_parameters, parameter_ranges, optimization_criterion
            )
        
        # Store optimized parameters for this regime
        self.regime_optimizers[regime_id] = optimized_params
        
        return optimized_params
    
    def _filter_data_for_regime(self, price_data, regime_id):
        """Filter price data for a specific regime."""
        if self.regime_classifier is None:
            logger.warning("No regime classifier available. Cannot filter data by regime.")
            return price_data
        
        # Get features and regimes
        features, df = self.regime_classifier.calculate_features(price_data)
        
        # Predict regimes if not already in the data
        if 'regime' not in df.columns:
            if hasattr(self.regime_classifier, 'kmeans_model') and self.regime_classifier.kmeans_model is not None:
                # Use KMeans model
                clusters = self.regime_classifier.kmeans_model.predict(self.regime_classifier.scaler.transform(features))
                df['regime'] = [self.regime_classifier.cluster_to_regime.get(c, 0) for c in clusters]
            elif hasattr(self.regime_classifier, 'hmm_model') and self.regime_classifier.hmm_model is not None:
                # Use HMM model
                states = self.regime_classifier.hmm_model.predict(self.regime_classifier.scaler.transform(features))
                df['regime'] = [self.regime_classifier.state_to_regime.get(s, 0) for s in states]
            else:
                logger.warning("No trained model in regime classifier. Cannot filter data by regime.")
                return price_data
        
        # Filter data for the specified regime
        regime_data = df[df['regime'] == regime_id]
        
        if regime_data.empty:
            logger.warning(f"No data available for regime {regime_id}.")
        else:
            logger.info(f"Filtered {len(regime_data)} samples for regime {regime_id}.")
        
        return regime_data
    
    def _grid_search_optimization(self, price_data, default_parameters, parameter_ranges, optimization_criterion):
        """
        Perform grid search optimization.
        
        This is a placeholder implementation. In a real system, you would:
        1. Generate a grid of parameter combinations
        2. Run backtests for each combination
        3. Evaluate based on the optimization criterion
        4. Return the best parameters
        """
        # Placeholder implementation - would be replaced with actual grid search
        logger.info("Grid search optimization not fully implemented. Using default parameters with slight adjustments.")
        
        # Create a copy of default parameters with slight adjustments
        optimized_params = default_parameters.copy()
        
        # Adjust some parameters based on data characteristics
        if 'volatility' in price_data.columns:
            avg_volatility = price_data['volatility'].mean()
            
            # Adjust position sizing based on volatility
            if 'position_size' in optimized_params:
                if avg_volatility > 0.2:  # High volatility
                    optimized_params['position_size'] *= 0.8  # Reduce position size
                elif avg_volatility < 0.1:  # Low volatility
                    optimized_params['position_size'] *= 1.2  # Increase position size
            
            # Adjust stop loss based on volatility
            if 'stop_loss_atr' in optimized_params:
                if avg_volatility > 0.2:  # High volatility
                    optimized_params['stop_loss_atr'] *= 1.2  # Wider stop loss
                elif avg_volatility < 0.1:  # Low volatility
                    optimized_params['stop_loss_atr'] *= 0.8  # Tighter stop loss
        
        # Adjust entry/exit thresholds based on trend strength
        if 'trend_20_50' in price_data.columns:
            avg_trend = price_data['trend_20_50'].mean()
            
            # Adjust entry threshold based on trend strength
            if 'entry_threshold' in optimized_params:
                if avg_trend > 0.02:  # Strong trend
                    optimized_params['entry_threshold'] *= 0.9  # More aggressive entries
                elif avg_trend < 0.005:  # Weak trend
                    optimized_params['entry_threshold'] *= 1.1  # More conservative entries
        
        return optimized_params
    
    def _bayesian_optimization(self, price_data, default_parameters, parameter_ranges, optimization_criterion):
        """
        Perform Bayesian optimization.
        
        This is a placeholder implementation. In a real system, you would use:
        1. A Bayesian optimization library (e.g., scikit-optimize)
        2. Define an objective function that runs a backtest and returns the criterion
        3. Run the optimization to find the best parameters
        """
        # Placeholder - would be replaced with actual Bayesian optimization
        logger.info("Bayesian optimization not implemented. Using default parameters.")
        return default_parameters
    
    def _genetic_optimization(self, price_data, default_parameters, parameter_ranges, optimization_criterion):
        """
        Perform genetic algorithm optimization.
        
        This is a placeholder implementation. In a real system, you would use:
        1. A genetic algorithm library
        2. Define a fitness function based on the backtest performance
        3. Run the genetic algorithm to evolve optimal parameters
        """
        # Placeholder - would be replaced with actual genetic algorithm
        logger.info("Genetic optimization not implemented. Using default parameters.")
        return default_parameters

class QuantConnectRegimeIntegration:
    """
    Class for integrating market regime detection with QuantConnect.
    
    This class provides:
    1. Regime-aware parameter optimization for QuantConnect algorithms
    2. Regime transition detection for adaptive strategy behavior
    3. Regime-specific performance analysis
    4. Regime visualization and reporting
    """
    
    def __init__(self, market_data=None):
        """
        Initialize QuantConnect regime integration.
        
        Args:
            market_data (DataFrame, optional): Historical market data
        """
        self.market_data = market_data
        self.regime_classifier = MarketRegimeClassifier()
        self.optimizer = RegimeAdaptiveStrategyOptimizer(self.regime_classifier)
        self.regime_transitions = []
        self.regime_performance = {}
    
    def train_regime_model(self, market_data=None, method="kmeans", num_regimes=4):
        """
        Train the market regime classifier.
        
        Args:
            market_data (DataFrame, optional): Historical market data
            method (str): Classification method ('kmeans' or 'hmm')
            num_regimes (int): Number of regimes to identify
            
        Returns:
            MarketRegimeClassifier: Trained classifier
        """
        # Use provided market data or stored data
        data = market_data if market_data is not None else self.market_data
        
        if data is None:
            logger.error("No market data provided for training")
            return None
        
        # Set number of regimes
        self.regime_classifier.num_regimes = num_regimes
        
        # Train the model
        self.regime_classifier.fit(data, method=method)
        
        # Update the optimizer with the trained classifier
        self.optimizer.regime_classifier = self.regime_classifier
        
        # Extract regime transitions
        self.regime_transitions = self.regime_classifier.get_regime_transitions()
        
        return self.regime_classifier
    
    def optimize_strategy_by_regime(self, parameter_template, parameter_ranges):
        """
        Optimize strategy parameters for each detected market regime.
        
        Args:
            parameter_template (dict): Default strategy parameters
            parameter_ranges (dict): Ranges for parameter optimization
            
        Returns:
            dict: Optimized parameters for each regime
        """
        if self.market_data is None:
            logger.error("No market data available for optimization")
            return {}
        
        # Ensure regime classifier is trained
        if self.regime_classifier.current_regime is None:
            logger.info("Training regime classifier before optimization")
            self.train_regime_model(self.market_data)
        
        # Optimize for each regime
        regime_params = {}
        for regime_id in range(self.regime_classifier.num_regimes):
            try:
                optimized_params = self.optimizer.optimize_for_regime(
                    regime_id, 
                    self.market_data,
                    parameter_template,
                    parameter_ranges,
                    optimization_criterion='sharpe_ratio',
                    method='grid_search'
                )
                
                regime_name = self.regime_classifier.REGIMES.get(regime_id, f"Regime {regime_id}")
                regime_params[regime_name] = optimized_params
                
                logger.info(f"Optimized parameters for {regime_name}")
            except Exception as e:
                logger.error(f"Error optimizing for regime {regime_id}: {e}")
                regime_name = self.regime_classifier.REGIMES.get(regime_id, f"Regime {regime_id}")
                regime_params[regime_name] = parameter_template.copy()
        
        return regime_params
    
    def get_current_regime(self):
        """
        Get the current market regime.
        
        Returns:
            dict: Current regime information
        """
        return self.regime_classifier.get_current_regime()
    
    def get_regime_parameters(self):
        """
        Get optimized parameters for the current market regime.
        
        Returns:
            dict: Optimized parameters
        """
        current_regime = self.get_current_regime()
        return self.optimizer.get_regime_parameters(current_regime['regime_id'])
    
    def analyze_regime_performance(self, backtest_results):
        """
        Analyze strategy performance across different market regimes.
        
        Args:
            backtest_results (dict): Backtest results including equity curve and trades
            
        Returns:
            dict: Performance metrics by regime
        """
        if self.market_data is None:
            logger.error("No market data available for performance analysis")
            return {}
        
        # Ensure regime classifier is trained
        if self.regime_classifier.current_regime is None:
            logger.info("Training regime classifier before performance analysis")
            self.train_regime_model(self.market_data)
        
        # Extract equity curve from backtest results
        if 'equity_curve' not in backtest_results:
            logger.error("No equity curve in backtest results")
            return {}
        
        equity_curve = backtest_results['equity_curve']
        
        # Align equity curve dates with market data
        aligned_data = pd.DataFrame({
            'equity': equity_curve
        })
        
        # Get regime classifications
        features, _ = self.regime_classifier.calculate_features(self.market_data)
        regimes = self.regime_classifier.predict(self.market_data)
        
        # Calculate daily returns
        aligned_data['returns'] = aligned_data['equity'].pct_change()
        
        # Add regime labels
        aligned_data['regime'] = regimes[-len(aligned_data):] if len(regimes) >= len(aligned_data) else regimes
        
        # Calculate performance metrics by regime
        regime_performance = {}
        
        for regime_id in range(self.regime_classifier.num_regimes):
            regime_name = self.regime_classifier.REGIMES.get(regime_id, f"Regime {regime_id}")
            regime_data = aligned_data[aligned_data['regime'] == regime_id]
            
            if regime_data.empty:
                logger.warning(f"No data available for {regime_name}")
                continue
            
            # Calculate performance metrics
            total_return = (1 + regime_data['returns']).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(regime_data)) - 1
            volatility = regime_data['returns'].std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(regime_data['equity'])
            
            # Store metrics
            regime_performance[regime_name] = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_days': len(regime_data)
            }
            
            logger.info(f"Performance metrics calculated for {regime_name}")
        
        # Store results
        self.regime_performance = regime_performance
        
        return regime_performance
    
    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown from equity curve."""
        # Calculate drawdown
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve / peak) - 1
        
        # Calculate maximum drawdown
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def generate_regime_report(self, output_dir=None):
        """
        Generate a comprehensive report of market regimes and strategy performance.
        
        Args:
            output_dir (str, optional): Directory to save the report
            
        Returns:
            dict: Report data and file paths
        """
        if self.market_data is None:
            logger.error("No market data available for report generation")
            return {}
        
        # Ensure regime classifier is trained
        if self.regime_classifier.current_regime is None:
            logger.info("Training regime classifier before report generation")
            self.train_regime_model(self.market_data)
        
        # Create output directory if specified
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate regime visualization
        try:
            fig = self.regime_classifier.plot_regimes(self.market_data, title="Market Regimes")
            
            # Save plot if output directory is specified
            if output_dir is not None:
                plot_path = os.path.join(output_dir, "market_regimes.png")
                fig.savefig(plot_path)
                plt.close(fig)
            else:
                plot_path = None
        except Exception as e:
            logger.error(f"Error generating regime visualization: {e}")
            fig = None
            plot_path = None
        
        # Compile report data
        report = {
            'current_regime': self.get_current_regime(),
            'regime_transitions': self.regime_transitions,
            'regime_metrics': self.regime_classifier.get_regime_metrics(),
            'regime_performance': self.regime_performance,
            'visualization_path': plot_path
        }
        
        # Save report as JSON if output directory is specified
        if output_dir is not None:
            try:
                report_path = os.path.join(output_dir, "regime_report.json")
                with open(report_path, 'w') as f:
                    # Convert numpy types to native Python types for JSON serialization
                    json_report = {}
                    for key, value in report.items():
                        if key == 'visualization_path':
                            json_report[key] = value
                        else:
                            json_report[key] = self._convert_to_json_serializable(value)
                    
                    json.dump(json_report, f, indent=2)
                
                report['report_path'] = report_path
                logger.info(f"Regime report saved to {report_path}")
            except Exception as e:
                logger.error(f"Error saving regime report: {e}")
        
        return report
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return self._convert_to_json_serializable(obj.tolist())
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Market Regime Detection and Adaptation")
    parser.add_argument("--data", required=True, help="Path to market data CSV file")
    parser.add_argument("--method", choices=["kmeans", "hmm"], default="kmeans", help="Classification method")
    parser.add_argument("--regimes", type=int, default=4, help="Number of regimes to identify")
    parser.add_argument("--output", default="regime_output", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Load market data
    try:
        data = pd.read_csv(args.data, parse_dates=True, index_col=0)
        logger.info(f"Loaded market data from {args.data} with {len(data)} rows")
    except Exception as e:
        logger.error(f"Error loading market data: {e}")
        return 1
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return 1
    
    # Create output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Create and train market regime classifier
    integration = QuantConnectRegimeIntegration(data)
    integration.train_regime_model(method=args.method, num_regimes=args.regimes)
    
    # Generate report
    report = integration.generate_regime_report(args.output)
    
    # Print summary
    current_regime = report['current_regime']
    print(f"\nCurrent Market Regime: {current_regime['regime_name']}")
    
    if 'regime_transitions' in report and report['regime_transitions']:
        print("\nRegime Transitions:")
        for i, transition in enumerate(report['regime_transitions'][-3:]):
            print(f"  {i+1}. {transition['from_regime_name']}: {transition['start_date']} to {transition['end_date']} ({transition['duration_days']} days)")
    
    print(f"\nReport and visualizations saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())