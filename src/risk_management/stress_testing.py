"""
Stress testing module for backtesting strategies against historical market regimes.

This module provides functionality to:
1. Identify historical market regimes using unsupervised learning
2. Perform stress tests on strategies against different market regimes
3. Generate comprehensive stress test reports
4. Analyze strategy robustness across different market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from enum import Enum
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import os
import json
from concurrent.futures import ProcessPoolExecutor
import statsmodels.api as sm
from scipy.stats import norm

logger = logging.getLogger(__name__)

class RegimeType(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RECOVERY = "recovery"
    CRASH = "crash"
    SIDEWAYS = "sideways"
    EXPANSION = "expansion"
    CONTRACTION = "contraction"
    CUSTOM = "custom"

class DetectionMethod(Enum):
    """Regime detection methods."""
    KMEANS = "kmeans"
    THRESHOLD = "threshold"
    HMM = "hmm"  # Hidden Markov Model
    CLASSIFICATION = "classification"
    VOLATILITY_BANDS = "volatility_bands"
    TREND_MOMENTUM = "trend_momentum"
    PRE_DEFINED = "pre_defined"

class StressTestType(Enum):
    """Types of stress tests."""
    HISTORICAL = "historical"  # Historical market regimes
    SYNTHETIC = "synthetic"    # Synthetic market scenarios
    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulations
    FACTOR_SHOCK = "factor_shock"  # Factor shock scenarios
    TAIL_RISK = "tail_risk"    # Tail risk scenarios
    LIQUIDITY_CRISIS = "liquidity_crisis"  # Liquidity crisis scenarios
    CUSTOM = "custom"         # Custom stress scenarios

class MarketRegimeDetector:
    """
    Detects market regimes using various detection methods.
    
    This class implements multiple approaches to identify and classify
    different market regimes (bull, bear, high volatility, etc.).
    """
    
    def __init__(self, 
                 method: DetectionMethod = DetectionMethod.KMEANS,
                 n_regimes: int = 4,
                 lookback_window: int = 252,
                 features: List[str] = None,
                 custom_regime_func: callable = None,
                 regime_labels: Dict[int, RegimeType] = None):
        """
        Initialize the market regime detector.
        
        Args:
            method: Method for regime detection
            n_regimes: Number of regimes to detect
            lookback_window: Window for calculating regime features
            features: Features to use for regime detection
            custom_regime_func: Custom function for regime detection
            regime_labels: Mapping of cluster indices to regime types
        """
        self.method = method
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.custom_regime_func = custom_regime_func
        self.regime_labels = regime_labels or {}
        
        # Default features if none specified
        self.features = features or [
            'returns', 'volatility', 'trend', 'momentum', 
            'drawdown', 'volume_change', 'correlation'
        ]
        
        # Storage for data and results
        self.market_data = None
        self.feature_data = None
        self.regimes = None
        self.regime_stats = None
        self.model = None
        
        logger.info(f"Initialized MarketRegimeDetector with method: {method.value}")
    
    def load_market_data(self, 
                       market_data: pd.DataFrame,
                       volume_data: pd.DataFrame = None,
                       benchmark_correlation_data: pd.DataFrame = None) -> None:
        """
        Load market data for regime detection.
        
        Args:
            market_data: DataFrame with market price data
            volume_data: Optional DataFrame with volume data
            benchmark_correlation_data: Optional DataFrame with correlation data
        """
        self.market_data = market_data
        self.volume_data = volume_data
        self.benchmark_correlation_data = benchmark_correlation_data
        
        logger.info(f"Loaded market data with {len(market_data)} data points")
    
    def calculate_features(self) -> pd.DataFrame:
        """
        Calculate features for regime detection.
        
        Returns:
            DataFrame with calculated features
        """
        if self.market_data is None:
            logger.error("No market data available for feature calculation")
            return None
            
        # Check if we already have features calculated
        if self.feature_data is not None:
            return self.feature_data
            
        # Initialize features DataFrame
        features_df = pd.DataFrame(index=self.market_data.index)
        
        # Calculate returns if not already available
        if 'returns' in self.features:
            if 'returns' in self.market_data.columns:
                features_df['returns'] = self.market_data['returns']
            elif 'close' in self.market_data.columns:
                features_df['returns'] = self.market_data['close'].pct_change()
            else:
                # Use first numeric column
                numeric_cols = self.market_data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    features_df['returns'] = self.market_data[numeric_cols[0]].pct_change()
                else:
                    logger.warning("No suitable price data for returns calculation")
        
        # Calculate volatility
        if 'volatility' in self.features and 'returns' in features_df.columns:
            # Rolling volatility with specified window
            features_df['volatility'] = features_df['returns'].rolling(window=self.lookback_window // 4).std() * np.sqrt(252)
            
        # Calculate trend
        if 'trend' in self.features and 'close' in self.market_data.columns:
            # Simple trend indicator: current price vs moving average
            ma = self.market_data['close'].rolling(window=self.lookback_window // 2).mean()
            features_df['trend'] = (self.market_data['close'] / ma - 1) * 100
            
        # Calculate momentum
        if 'momentum' in self.features and 'close' in self.market_data.columns:
            # Momentum as 3-month return
            features_df['momentum'] = self.market_data['close'].pct_change(periods=63)
            
        # Calculate drawdown
        if 'drawdown' in self.features and 'close' in self.market_data.columns:
            # Running maximum
            running_max = self.market_data['close'].expanding().max()
            # Drawdown percentage
            features_df['drawdown'] = (self.market_data['close'] / running_max - 1) * 100
            
        # Calculate volume change if volume data is available
        if 'volume_change' in self.features and self.volume_data is not None:
            if 'volume' in self.volume_data.columns:
                # Calculate percentage change in volume
                features_df['volume_change'] = self.volume_data['volume'].pct_change(periods=5)
            else:
                # Use first numeric column
                numeric_cols = self.volume_data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    features_df['volume_change'] = self.volume_data[numeric_cols[0]].pct_change(periods=5)
                    
        # Calculate correlation if benchmark data is available
        if 'correlation' in self.features and self.benchmark_correlation_data is not None:
            # If benchmark data is a DataFrame with many assets, use the first column
            if isinstance(self.benchmark_correlation_data, pd.DataFrame) and len(self.benchmark_correlation_data.columns) > 1:
                benchmark_returns = self.benchmark_correlation_data.iloc[:, 0]
            else:
                benchmark_returns = self.benchmark_correlation_data
                
            # Ensure we have returns for correlation calculation
            if 'returns' in features_df.columns and isinstance(benchmark_returns, pd.Series):
                # Calculate rolling correlation
                aligned_data = pd.DataFrame({
                    'market': features_df['returns'],
                    'benchmark': benchmark_returns
                }).dropna()
                
                # Calculate rolling correlation with appropriate window
                window = min(63, len(aligned_data) // 4)  # Use at most 63 days or 1/4 of data
                if window > 10:  # Require at least 10 data points
                    features_df['correlation'] = aligned_data['market'].rolling(window=window).corr(aligned_data['benchmark'])
        
        # Drop rows with NaN values
        features_df = features_df.dropna()
        
        # Store feature data
        self.feature_data = features_df
        
        logger.info(f"Calculated {len(features_df.columns)} features for regime detection")
        return features_df
    
    def detect_regimes(self) -> pd.Series:
        """
        Detect market regimes using the specified method.
        
        Returns:
            Series with regime labels for each date
        """
        # Calculate features if not already done
        if self.feature_data is None:
            self.calculate_features()
            
        if self.feature_data is None or self.feature_data.empty:
            logger.error("No feature data available for regime detection")
            return None
            
        # Detect regimes using the specified method
        if self.method == DetectionMethod.KMEANS:
            regimes = self._detect_regimes_kmeans()
        elif self.method == DetectionMethod.THRESHOLD:
            regimes = self._detect_regimes_threshold()
        elif self.method == DetectionMethod.HMM:
            regimes = self._detect_regimes_hmm()
        elif self.method == DetectionMethod.CLASSIFICATION:
            regimes = self._detect_regimes_classification()
        elif self.method == DetectionMethod.VOLATILITY_BANDS:
            regimes = self._detect_regimes_volatility_bands()
        elif self.method == DetectionMethod.TREND_MOMENTUM:
            regimes = self._detect_regimes_trend_momentum()
        elif self.method == DetectionMethod.PRE_DEFINED:
            regimes = self._detect_regimes_pre_defined()
        elif self.method == DetectionMethod.CUSTOM and self.custom_regime_func is not None:
            # Use custom function for regime detection
            regimes = self.custom_regime_func(self.feature_data)
        else:
            logger.warning(f"Unknown regime detection method: {self.method}. Using K-Means.")
            regimes = self._detect_regimes_kmeans()
            
        # Store detected regimes
        self.regimes = regimes
        
        # Calculate regime statistics
        self._calculate_regime_statistics()
        
        logger.info(f"Detected {len(regimes.unique())} market regimes")
        return regimes
    
    def _detect_regimes_kmeans(self) -> pd.Series:
        """
        Detect market regimes using K-Means clustering.
        
        Returns:
            Series with regime labels for each date
        """
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.feature_data)
        
        # Use PCA to reduce dimensionality if we have many features
        if scaled_features.shape[1] > 3:
            pca = PCA(n_components=min(3, scaled_features.shape[1]))
            scaled_features = pca.fit_transform(scaled_features)
            
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Create Series with cluster labels
        regimes = pd.Series(cluster_labels, index=self.feature_data.index)
        
        # Store model for later use
        self.model = kmeans
        
        # Map numeric cluster labels to regime types if mapping is provided
        if self.regime_labels:
            regimes = regimes.map(self.regime_labels).fillna(regimes)
            
        return regimes
    
    def _detect_regimes_threshold(self) -> pd.Series:
        """
        Detect market regimes using simple thresholds.
        
        Returns:
            Series with regime labels for each date
        """
        # This approach uses thresholds on trend and volatility
        
        # Ensure we have the required features
        required_features = ['trend', 'volatility']
        if not all(feature in self.feature_data.columns for feature in required_features):
            logger.warning("Missing required features for threshold-based detection")
            return self._detect_regimes_kmeans()
            
        # Initialize regimes
        regimes = pd.Series(index=self.feature_data.index, data=None)
        
        # Define thresholds
        trend_threshold = 5.0  # 5% above/below moving average
        vol_threshold = 20.0   # 20% above/below median volatility
        
        # Calculate median volatility
        median_vol = self.feature_data['volatility'].median()
        high_vol = median_vol * (1 + vol_threshold/100)
        low_vol = median_vol * (1 - vol_threshold/100)
        
        # Assign regimes based on thresholds
        for date, row in self.feature_data.iterrows():
            trend = row['trend']
            vol = row['volatility']
            
            # Determine regime
            if trend > trend_threshold and vol <= high_vol:
                # Strong uptrend with moderate volatility
                regimes[date] = RegimeType.BULL.value
            elif trend > trend_threshold and vol > high_vol:
                # Uptrend with high volatility
                regimes[date] = RegimeType.RECOVERY.value
            elif trend < -trend_threshold and vol > high_vol:
                # Downtrend with high volatility
                regimes[date] = RegimeType.CRASH.value
            elif trend < -trend_threshold and vol <= high_vol:
                # Downtrend with moderate volatility
                regimes[date] = RegimeType.BEAR.value
            elif abs(trend) <= trend_threshold and vol <= low_vol:
                # Sideways market with low volatility
                regimes[date] = RegimeType.SIDEWAYS.value
            elif vol > high_vol:
                # High volatility regime
                regimes[date] = RegimeType.HIGH_VOLATILITY.value
            elif vol < low_vol:
                # Low volatility regime
                regimes[date] = RegimeType.LOW_VOLATILITY.value
            else:
                # Default case
                regimes[date] = RegimeType.SIDEWAYS.value
                
        return regimes
    
    def _detect_regimes_hmm(self) -> pd.Series:
        """
        Detect market regimes using Hidden Markov Model.
        
        Returns:
            Series with regime labels for each date
        """
        # This is a placeholder for HMM-based regime detection
        
        # Try to import hmmlearn
        try:
            from hmmlearn import hmm
        except ImportError:
            logger.warning("hmmlearn not installed, falling back to K-Means")
            return self._detect_regimes_kmeans()
            
        # Prepare data
        if 'returns' not in self.feature_data.columns:
            logger.warning("Returns not available for HMM, falling back to K-Means")
            return self._detect_regimes_kmeans()
            
        # Reshape returns for HMM
        returns = self.feature_data['returns'].values.reshape(-1, 1)
        
        # Create and fit HMM model
        model = hmm.GaussianHMM(n_components=self.n_regimes, random_state=42)
        
        try:
            model.fit(returns)
            
            # Predict hidden states
            hidden_states = model.predict(returns)
            
            # Create Series with regime labels
            regimes = pd.Series(hidden_states, index=self.feature_data.index)
            
            # Store model for later use
            self.model = model
            
            # Map numeric state labels to regime types if mapping is provided
            if self.regime_labels:
                regimes = regimes.map(self.regime_labels).fillna(regimes)
                
            return regimes
                
        except Exception as e:
            logger.error(f"Error fitting HMM model: {str(e)}")
            return self._detect_regimes_kmeans()
    
    def _detect_regimes_classification(self) -> pd.Series:
        """
        Detect market regimes using classification.
        
        Returns:
            Series with regime labels for each date
        """
        # This is a placeholder for classification-based regime detection
        logger.warning("Classification-based regime detection not implemented, falling back to K-Means")
        return self._detect_regimes_kmeans()
    
    def _detect_regimes_volatility_bands(self) -> pd.Series:
        """
        Detect market regimes using volatility bands.
        
        Returns:
            Series with regime labels for each date
        """
        # This approach uses volatility bands to identify regimes
        
        # Ensure we have volatility
        if 'volatility' not in self.feature_data.columns:
            logger.warning("Volatility not available for detection, falling back to K-Means")
            return self._detect_regimes_kmeans()
            
        # Calculate volatility percentiles
        vol = self.feature_data['volatility']
        vol_low = vol.quantile(0.25)
        vol_high = vol.quantile(0.75)
        
        # Determine trend if available
        has_trend = 'trend' in self.feature_data.columns
        if has_trend:
            trend = self.feature_data['trend']
            trend_low = trend.quantile(0.25)
            trend_high = trend.quantile(0.75)
            
        # Initialize regimes
        regimes = pd.Series(index=self.feature_data.index, data=None)
        
        # Assign regimes based on volatility and trend
        for date, row in self.feature_data.iterrows():
            vol_value = row['volatility']
            
            if has_trend:
                trend_value = row['trend']
                
                # Assign regime based on volatility and trend
                if vol_value > vol_high and trend_value > trend_high:
                    regimes[date] = RegimeType.RECOVERY.value
                elif vol_value > vol_high and trend_value < trend_low:
                    regimes[date] = RegimeType.CRASH.value
                elif vol_value <= vol_high and trend_value > trend_high:
                    regimes[date] = RegimeType.BULL.value
                elif vol_value <= vol_high and trend_value < trend_low:
                    regimes[date] = RegimeType.BEAR.value
                elif vol_value < vol_low:
                    regimes[date] = RegimeType.LOW_VOLATILITY.value
                else:
                    regimes[date] = RegimeType.SIDEWAYS.value
            else:
                # Assign regime based only on volatility
                if vol_value > vol_high:
                    regimes[date] = RegimeType.HIGH_VOLATILITY.value
                elif vol_value < vol_low:
                    regimes[date] = RegimeType.LOW_VOLATILITY.value
                else:
                    regimes[date] = RegimeType.SIDEWAYS.value
                    
        return regimes
    
    def _detect_regimes_trend_momentum(self) -> pd.Series:
        """
        Detect market regimes using trend and momentum indicators.
        
        Returns:
            Series with regime labels for each date
        """
        # Ensure we have required features
        required_features = ['trend', 'momentum']
        if not all(feature in self.feature_data.columns for feature in required_features):
            logger.warning("Missing required features for trend-momentum detection")
            return self._detect_regimes_kmeans()
            
        # Initialize regimes
        regimes = pd.Series(index=self.feature_data.index, data=None)
        
        # Calculate percentiles
        trend = self.feature_data['trend']
        momentum = self.feature_data['momentum']
        
        trend_high = trend.quantile(0.7)
        trend_low = trend.quantile(0.3)
        mom_high = momentum.quantile(0.7)
        mom_low = momentum.quantile(0.3)
        
        # Assign regimes based on trend and momentum
        for date, row in self.feature_data.iterrows():
            t = row['trend']
            m = row['momentum']
            
            # Determine regime
            if t > trend_high and m > mom_high:
                # Strong uptrend with high momentum
                regimes[date] = RegimeType.BULL.value
            elif t < trend_low and m < mom_low:
                # Strong downtrend with negative momentum
                regimes[date] = RegimeType.BEAR.value
            elif t < trend_low and m > mom_high:
                # Downtrend with improving momentum (potential recovery)
                regimes[date] = RegimeType.RECOVERY.value
            elif t > trend_high and m < mom_low:
                # Uptrend with deteriorating momentum (potential reversal)
                regimes[date] = RegimeType.CONTRACTION.value
            elif abs(t) <= (trend_high - trend_low) / 2:
                # Sideways market
                regimes[date] = RegimeType.SIDEWAYS.value
            elif t > 0:
                # Moderate uptrend
                regimes[date] = RegimeType.EXPANSION.value
            else:
                # Moderate downtrend
                regimes[date] = RegimeType.CONTRACTION.value
                
        return regimes
    
    def _detect_regimes_pre_defined(self) -> pd.Series:
        """
        Use pre-defined regime periods.
        
        Returns:
            Series with regime labels for each date
        """
        # This method assumes that regime_labels is a dictionary mapping
        # datetime ranges to regime types
        
        if not isinstance(self.regime_labels, dict):
            logger.warning("No pre-defined regimes provided")
            return self._detect_regimes_kmeans()
            
        # Initialize regimes
        regimes = pd.Series(index=self.feature_data.index, data=None)
        
        # Assign regimes based on pre-defined periods
        for regime_range, regime_type in self.regime_labels.items():
            if isinstance(regime_range, tuple) and len(regime_range) == 2:
                start_date, end_date = regime_range
                mask = (regimes.index >= start_date) & (regimes.index <= end_date)
                regimes[mask] = regime_type
                
        # Fill any gaps with default regime
        regimes = regimes.fillna(RegimeType.SIDEWAYS.value)
        
        return regimes
    
    def _calculate_regime_statistics(self) -> Dict:
        """
        Calculate statistics for each detected regime.
        
        Returns:
            Dictionary with statistics for each regime
        """
        if self.regimes is None:
            logger.warning("No regimes detected for statistics calculation")
            return {}
            
        # Group data by regime
        regime_stats = {}
        
        # Check if we have return data
        if 'returns' in self.feature_data.columns:
            # Calculate statistics for each regime
            for regime in self.regimes.unique():
                # Get dates for this regime
                regime_dates = self.regimes[self.regimes == regime].index
                
                # Get returns for this regime
                regime_returns = self.feature_data.loc[regime_dates, 'returns']
                
                if len(regime_returns) > 0:
                    # Calculate basic statistics
                    stats = {
                        'count': len(regime_returns),
                        'mean_return': regime_returns.mean(),
                        'median_return': regime_returns.median(),
                        'std_return': regime_returns.std(),
                        'min_return': regime_returns.min(),
                        'max_return': regime_returns.max(),
                        'annualized_return': regime_returns.mean() * 252,
                        'annualized_volatility': regime_returns.std() * np.sqrt(252),
                        'sharpe_ratio': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                        'positive_days': (regime_returns > 0).mean(),
                        'negative_days': (regime_returns < 0).mean(),
                    }
                    
                    # Add drawdown if we have close prices
                    if 'close' in self.market_data.columns:
                        regime_prices = self.market_data.loc[regime_dates, 'close']
                        if len(regime_prices) > 0:
                            # Calculate drawdown
                            peak = regime_prices.expanding().max()
                            drawdown = (regime_prices / peak - 1) * 100
                            stats['max_drawdown'] = drawdown.min()
                            
                    # Add other available features
                    for feature in self.feature_data.columns:
                        if feature != 'returns':
                            feature_values = self.feature_data.loc[regime_dates, feature]
                            if len(feature_values) > 0:
                                stats[f'mean_{feature}'] = feature_values.mean()
                                stats[f'median_{feature}'] = feature_values.median()
                                stats[f'std_{feature}'] = feature_values.std()
                    
                    regime_stats[regime] = stats
        
        # Store regime statistics
        self.regime_stats = regime_stats
        
        return regime_stats
    
    def plot_regimes(self, figsize=(12, 10)) -> plt.Figure:
        """
        Plot detected regimes with market data.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.regimes is None:
            logger.warning("No regimes detected for plotting")
            return None
            
        if 'close' not in self.market_data.columns:
            logger.warning("No close prices available for plotting")
            return None
            
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot market data
        axes[0].plot(self.market_data.index, self.market_data['close'], 
                   label='Market Price', color='black')
        
        # Add colored background for regimes
        regimes = self.regimes.unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(regimes)))
        regime_colors = {regime: colors[i] for i, regime in enumerate(regimes)}
        
        ylim = axes[0].get_ylim()
        
        for regime in regimes:
            regime_periods = self.regimes[self.regimes == regime].index
            if len(regime_periods) > 0:
                # Group consecutive dates
                breaks = np.where(np.diff(regime_periods) > pd.Timedelta(days=1))[0]
                start_idx = 0
                
                for break_idx in list(breaks) + [len(regime_periods) - 1]:
                    period_start = regime_periods[start_idx]
                    period_end = regime_periods[break_idx]
                    
                    axes[0].axvspan(period_start, period_end, 
                                  alpha=0.2, color=regime_colors[regime])
                    
                    start_idx = break_idx + 1
        
        # Plot features
        if self.feature_data is not None:
            if 'volatility' in self.feature_data.columns:
                axes[1].plot(self.feature_data.index, self.feature_data['volatility'], 
                           label='Volatility', color='red')
                axes[1].set_ylabel('Volatility')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
            if 'trend' in self.feature_data.columns:
                axes[2].plot(self.feature_data.index, self.feature_data['trend'], 
                           label='Trend', color='blue')
                axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                axes[2].set_ylabel('Trend')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
        # Add regime legend
        handles = [plt.Rectangle((0, 0), 1, 1, fc=regime_colors[r], alpha=0.2) for r in regimes]
        labels = [str(r) for r in regimes]
        axes[0].legend(handles, labels, loc='upper left')
        
        axes[0].set_title('Market Regimes')
        axes[0].set_ylabel('Price')
        axes[0].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Date')
        
        plt.tight_layout()
        
        return fig
    
    def generate_regime_report(self, output_dir: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive report on detected regimes.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Dictionary with report data
        """
        if self.regimes is None:
            logger.warning("No regimes detected for report generation")
            return {}
            
        if self.regime_stats is None:
            self._calculate_regime_statistics()
            
        # Prepare report data
        report = {
            'detected_regimes': list(self.regimes.unique()),
            'regime_stats': self.regime_stats,
            'detection_method': self.method.value,
            'n_regimes': self.n_regimes,
            'feature_names': list(self.feature_data.columns)
        }
        
        # Add sample dates for each regime
        regime_dates = {}
        for regime in self.regimes.unique():
            regime_indices = self.regimes[self.regimes == regime].index
            if len(regime_indices) > 0:
                regime_dates[str(regime)] = {
                    'start_date': str(regime_indices[0]),
                    'end_date': str(regime_indices[-1]),
                    'total_days': len(regime_indices)
                }
                
        report['regime_dates'] = regime_dates
        
        # Generate and save report if output directory is specified
        if output_dir is not None:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save report as JSON
            report_path = os.path.join(output_dir, 'regime_report.json')
            
            # Prepare JSON-serializable report
            json_report = {
                'detected_regimes': [str(r) for r in report['detected_regimes']],
                'detection_method': report['detection_method'],
                'n_regimes': report['n_regimes'],
                'feature_names': report['feature_names'],
                'regime_dates': report['regime_dates']
            }
            
            # Add regime statistics
            json_report['regime_stats'] = {}
            for regime, stats in self.regime_stats.items():
                json_report['regime_stats'][str(regime)] = {
                    k: float(v) if isinstance(v, (np.floating, float)) else v 
                    for k, v in stats.items()
                }
                
            # Save JSON report
            with open(report_path, 'w') as f:
                json.dump(json_report, f, indent=2)
                
            # Generate and save plots
            fig = self.plot_regimes()
            if fig is not None:
                plot_path = os.path.join(output_dir, 'regime_plot.png')
                fig.savefig(plot_path)
                plt.close(fig)
                
                report['plot_path'] = plot_path
                
            report['report_path'] = report_path
            
            logger.info(f"Saved regime report to {report_path}")
            
        return report
    
    def get_regime_for_date(self, date: datetime) -> str:
        """
        Get the regime for a specific date.
        
        Args:
            date: Date to get regime for
            
        Returns:
            Regime label for the specified date
        """
        if self.regimes is None:
            logger.warning("No regimes detected")
            return None
            
        # Convert date to pandas Timestamp if necessary
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
            
        # Check if date is in regimes index
        if date in self.regimes.index:
            return self.regimes[date]
            
        # Find closest date if exact date not available
        closest_date = self.regimes.index[self.regimes.index <= date]
        if len(closest_date) > 0:
            return self.regimes[closest_date[-1]]
            
        return None
    
    def get_similar_regimes(self, current_date: datetime, n_periods: int = 3) -> List[Tuple[datetime, datetime]]:
        """
        Find historical periods with similar market regimes.
        
        Args:
            current_date: Current date
            n_periods: Number of similar periods to find
            
        Returns:
            List of (start_date, end_date) tuples for similar regime periods
        """
        if self.regimes is None:
            logger.warning("No regimes detected")
            return []
            
        # Get current regime
        current_regime = self.get_regime_for_date(current_date)
        if current_regime is None:
            return []
            
        # Find all periods with the same regime
        same_regime = self.regimes[self.regimes == current_regime]
        
        # Group consecutive dates into periods
        periods = []
        start_date = None
        prev_date = None
        
        for date in same_regime.index:
            if start_date is None:
                # Start a new period
                start_date = date
                prev_date = date
            elif (date - prev_date) > pd.Timedelta(days=7):
                # Gap of more than 7 days, end current period and start new one
                periods.append((start_date, prev_date))
                start_date = date
            
            prev_date = date
            
        # Add the last period
        if start_date is not None:
            periods.append((start_date, prev_date))
            
        # Sort periods by length (longest first)
        periods.sort(key=lambda x: (x[1] - x[0]).days, reverse=True)
        
        # Return top N periods
        return periods[:n_periods]

class StressTester:
    """
    Stress tests strategies against historical and synthetic market regimes.
    
    This class provides functionality to test strategy performance
    under various market conditions to assess robustness.
    """
    
    def __init__(self, 
                 strategy_func: callable,
                 market_data: pd.DataFrame,
                 regime_detector: Optional[MarketRegimeDetector] = None,
                 custom_regimes: Optional[pd.Series] = None,
                 monte_carlo_sims: int = 1000,
                 parallel_jobs: int = None):
        """
        Initialize the stress tester.
        
        Args:
            strategy_func: Function that takes market data and returns strategy returns
            market_data: DataFrame with market price data
            regime_detector: Optional regime detector instance
            custom_regimes: Optional custom regime labels
            monte_carlo_sims: Number of Monte Carlo simulations
            parallel_jobs: Number of parallel jobs for simulations
        """
        self.strategy_func = strategy_func
        self.market_data = market_data
        self.regime_detector = regime_detector
        self.custom_regimes = custom_regimes
        self.monte_carlo_sims = monte_carlo_sims
        self.parallel_jobs = parallel_jobs or max(1, os.cpu_count() - 1)
        
        # Storage for results
        self.regimes = None
        self.historical_results = None
        self.synthetic_results = None
        self.monte_carlo_results = None
        self.factor_shock_results = None
        self.tail_risk_results = None
        
        # Initialize regimes
        if regime_detector is not None and hasattr(regime_detector, 'regimes') and regime_detector.regimes is not None:
            self.regimes = regime_detector.regimes
        elif custom_regimes is not None:
            self.regimes = custom_regimes
        else:
            # Create a new regime detector
            self.regime_detector = MarketRegimeDetector()
            self.regime_detector.load_market_data(market_data)
            self.regimes = self.regime_detector.detect_regimes()
            
        logger.info("Initialized StressTester")
    
    def run_stress_tests(self, test_types: List[StressTestType] = None) -> Dict:
        """
        Run a comprehensive suite of stress tests.
        
        Args:
            test_types: List of stress test types to run
            
        Returns:
            Dictionary with stress test results
        """
        # Default to historical and Monte Carlo tests
        if test_types is None:
            test_types = [StressTestType.HISTORICAL, StressTestType.MONTE_CARLO]
            
        # Run selected tests
        results = {}
        
        for test_type in test_types:
            if test_type == StressTestType.HISTORICAL:
                results['historical'] = self.run_historical_stress_test()
            elif test_type == StressTestType.SYNTHETIC:
                results['synthetic'] = self.run_synthetic_stress_test()
            elif test_type == StressTestType.MONTE_CARLO:
                results['monte_carlo'] = self.run_monte_carlo_stress_test()
            elif test_type == StressTestType.FACTOR_SHOCK:
                results['factor_shock'] = self.run_factor_shock_stress_test()
            elif test_type == StressTestType.TAIL_RISK:
                results['tail_risk'] = self.run_tail_risk_stress_test()
            elif test_type == StressTestType.LIQUIDITY_CRISIS:
                results['liquidity_crisis'] = self.run_liquidity_crisis_stress_test()
            elif test_type == StressTestType.CUSTOM:
                results['custom'] = self.run_custom_stress_test()
                
        logger.info(f"Completed {len(results)} stress test types")
        
        # Store results
        self.stress_test_results = results
        
        return results
    
    def run_historical_stress_test(self, benchmark_data: pd.DataFrame = None, 
                              regime_specific_params: Dict[str, Dict] = None,
                              include_drawdown_paths: bool = True,
                              include_regime_transitions: bool = True) -> Dict:
        """
        Run advanced historical stress test against detected market regimes.
        
        This method tests a strategy across different historical market regimes
        to evaluate its robustness and identify potential vulnerabilities.
        
        Args:
            benchmark_data: Optional benchmark returns for relative performance
            regime_specific_params: Optional parameters to override for specific regimes
            include_drawdown_paths: Whether to include drawdown path analysis
            include_regime_transitions: Whether to analyze regime transitions
            
        Returns:
            Dictionary with comprehensive historical stress test results
        """
        if self.regimes is None:
            logger.warning("No regimes available for historical stress test")
            return {}
            
        # Run strategy on full dataset
        logger.info("Running strategy on full dataset for historical stress test")
        full_returns = self.strategy_func(self.market_data)
        
        if full_returns is None or len(full_returns) == 0:
            logger.error("Strategy returned no results")
            return {}
            
        # Align regimes and returns
        common_index = self.regimes.index.intersection(full_returns.index)
        if len(common_index) == 0:
            logger.error("No common dates between regimes and strategy returns")
            return {}
            
        regimes = self.regimes.loc[common_index]
        returns = full_returns.loc[common_index]
        
        # Initialize results structure
        regime_performance = {}
        regime_drawdowns = {}
        regime_transitions = {}
        regime_relative_performance = {}
        regime_statistical_significance = {}
        
        # Process benchmark data if provided
        benchmark_returns = None
        if benchmark_data is not None:
            # Extract benchmark returns
            if isinstance(benchmark_data, pd.DataFrame) and 'returns' in benchmark_data.columns:
                benchmark_returns = benchmark_data['returns']
            elif isinstance(benchmark_data, pd.DataFrame) and 'close' in benchmark_data.columns:
                benchmark_returns = benchmark_data['close'].pct_change()
            elif isinstance(benchmark_data, pd.Series):
                benchmark_returns = benchmark_data
            
            # Align benchmark returns with strategy returns
            if benchmark_returns is not None:
                common_benchmark_index = benchmark_returns.index.intersection(common_index)
                if len(common_benchmark_index) > 0:
                    benchmark_returns = benchmark_returns.loc[common_benchmark_index]
                    returns = returns.loc[common_benchmark_index]
                    regimes = regimes.loc[common_benchmark_index]
                else:
                    logger.warning("No common dates between benchmark and strategy returns")
                    benchmark_returns = None
        
        logger.info(f"Analyzing performance across {len(regimes.unique())} distinct market regimes")
        
        # Calculate performance metrics for each regime
        for regime in regimes.unique():
            # Get returns for this regime
            regime_mask = regimes == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 0:
                # Calculate standard performance metrics
                performance = self._calculate_performance_metrics(regime_returns)
                regime_performance[str(regime)] = performance
                
                # Calculate statistical significance (t-test on returns)
                if len(regime_returns) > 10:  # Require minimum sample size
                    tstat, pvalue = sm.stats.ttest_1samp(regime_returns.values, 0.0)
                    regime_statistical_significance[str(regime)] = {
                        't_statistic': float(tstat),
                        'p_value': float(pvalue),
                        'significant_05': bool(pvalue < 0.05),
                        'significant_01': bool(pvalue < 0.01),
                        'sample_size': len(regime_returns)
                    }
                
                # Calculate drawdown paths if requested
                if include_drawdown_paths:
                    cum_returns = (1 + regime_returns).cumprod()
                    peak = cum_returns.expanding().max()
                    drawdown = (cum_returns / peak - 1) * 100
                    
                    # Identify worst drawdown period
                    worst_dd = drawdown.min()
                    worst_dd_idx = drawdown.idxmin()
                    
                    # Find start and end of this drawdown period
                    peak_idx = peak.loc[:worst_dd_idx].idxmax()
                    recovery_mask = (drawdown.loc[worst_dd_idx:] == 0)
                    
                    # Check if recovery occurred within this regime
                    recovery_idx = None
                    if recovery_mask.any():
                        recovery_idx = recovery_mask.idxmax()
                    
                    # Store drawdown information
                    regime_drawdowns[str(regime)] = {
                        'worst_drawdown': float(worst_dd),
                        'start_date': str(peak_idx.date()),
                        'worst_date': str(worst_dd_idx.date()),
                        'recovery_date': str(recovery_idx.date()) if recovery_idx is not None else None,
                        'drawdown_length': (worst_dd_idx - peak_idx).days,
                        'recovery_length': (recovery_idx - worst_dd_idx).days if recovery_idx is not None else None,
                        'total_length': (recovery_idx - peak_idx).days if recovery_idx is not None else None,
                        'path': list(zip(drawdown.index.strftime('%Y-%m-%d').tolist(), drawdown.round(2).tolist()))
                    }
                
                # Calculate relative performance if benchmark available
                if benchmark_returns is not None:
                    benchmark_regime_returns = benchmark_returns[regime_mask]
                    if len(benchmark_regime_returns) > 0:
                        # Calculate benchmark performance
                        benchmark_performance = self._calculate_performance_metrics(benchmark_regime_returns)
                        
                        # Calculate relative performance metrics
                        relative_perf = {}
                        for metric in ['annualized_return', 'annualized_volatility', 'sharpe_ratio', 'max_drawdown']:
                            if metric in performance and metric in benchmark_performance:
                                relative_perf[f'relative_{metric}'] = performance[metric] - benchmark_performance[metric]
                        
                        # Calculate alpha and beta
                        if len(regime_returns) > 10:
                            # Prepare data for regression
                            data = pd.DataFrame({
                                'strategy': regime_returns,
                                'benchmark': benchmark_regime_returns
                            }).dropna()
                            
                            if len(data) > 10:  # Ensure enough data points after removing NaNs
                                # Run regression
                                X = sm.add_constant(data['benchmark'])
                                model = sm.OLS(data['strategy'], X).fit()
                                
                                # Extract alpha and beta
                                alpha = model.params[0]
                                beta = model.params[1]
                                alpha_tstat = model.tvalues[0]
                                alpha_pval = model.pvalues[0]
                                r_squared = model.rsquared
                                
                                # Annualize alpha
                                alpha_annualized = alpha * 252
                                
                                relative_perf.update({
                                    'alpha': float(alpha),
                                    'alpha_annualized': float(alpha_annualized),
                                    'beta': float(beta),
                                    'alpha_t_statistic': float(alpha_tstat),
                                    'alpha_p_value': float(alpha_pval),
                                    'alpha_significant_05': bool(alpha_pval < 0.05),
                                    'r_squared': float(r_squared),
                                    'correlation': float(data['strategy'].corr(data['benchmark']))
                                })
                        
                        # Store relative performance
                        regime_relative_performance[str(regime)] = {
                            'strategy_performance': performance,
                            'benchmark_performance': benchmark_performance,
                            'relative_metrics': relative_perf
                        }
        
        # Analyze regime transitions if requested
        if include_regime_transitions:
            transitions = {}
            regime_changes = []
            
            # Identify regime change points
            prev_regime = None
            for date, regime in zip(regimes.index, regimes.values):
                if prev_regime is not None and regime != prev_regime:
                    # Found a transition
                    transition_key = f"{prev_regime}_to_{regime}"
                    
                    # Count transitions
                    if transition_key in transitions:
                        transitions[transition_key]['count'] += 1
                    else:
                        transitions[transition_key] = {
                            'from_regime': str(prev_regime),
                            'to_regime': str(regime),
                            'count': 1,
                            'dates': []
                        }
                    
                    # Store date of transition
                    transitions[transition_key]['dates'].append(str(date.date()))
                    
                    # Record this transition point
                    regime_changes.append({
                        'date': str(date.date()),
                        'from_regime': str(prev_regime),
                        'to_regime': str(regime)
                    })
                
                prev_regime = regime
            
            # Analyze performance during transition periods
            for change in regime_changes:
                date = pd.Timestamp(change['date'])
                
                # Define transition period (1 month before and after)
                start_date = date - pd.Timedelta(days=30)
                end_date = date + pd.Timedelta(days=30)
                
                # Get returns during transition
                mask = (returns.index >= start_date) & (returns.index <= end_date)
                transition_returns = returns[mask]
                
                if len(transition_returns) > 10:  # Require minimum sample size
                    # Calculate performance during transition
                    transition_performance = self._calculate_performance_metrics(transition_returns)
                    
                    # Add to change record
                    change['performance'] = transition_performance
            
            # Store transition analysis
            regime_transitions = {
                'transitions': transitions,
                'regime_changes': regime_changes,
                'total_transitions': len(regime_changes)
            }
        
        # Calculate overall performance
        overall_performance = self._calculate_performance_metrics(returns)
        
        # Calculate statistical significance of overall performance
        overall_significance = {}
        if len(returns) > 10:
            tstat, pvalue = sm.stats.ttest_1samp(returns.values, 0.0)
            overall_significance = {
                't_statistic': float(tstat),
                'p_value': float(pvalue),
                'significant_05': bool(pvalue < 0.05),
                'significant_01': bool(pvalue < 0.01),
                'sample_size': len(returns)
            }
        
        # Calculate overall alpha and beta if benchmark available
        overall_rel_perf = {}
        if benchmark_returns is not None and len(benchmark_returns) > 10:
            # Prepare data for regression
            data = pd.DataFrame({
                'strategy': returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            if len(data) > 10:
                # Run regression
                X = sm.add_constant(data['benchmark'])
                model = sm.OLS(data['strategy'], X).fit()
                
                # Extract alpha and beta
                alpha = model.params[0]
                beta = model.params[1]
                alpha_tstat = model.tvalues[0]
                alpha_pval = model.pvalues[0]
                r_squared = model.rsquared
                
                # Annualize alpha
                alpha_annualized = alpha * 252
                
                overall_rel_perf = {
                    'alpha': float(alpha),
                    'alpha_annualized': float(alpha_annualized),
                    'beta': float(beta),
                    'alpha_t_statistic': float(alpha_tstat),
                    'alpha_p_value': float(alpha_pval),
                    'alpha_significant_05': bool(alpha_pval < 0.05),
                    'r_squared': float(r_squared),
                    'correlation': float(data['strategy'].corr(data['benchmark'])),
                    'tracking_error': float((data['strategy'] - data['benchmark']).std() * np.sqrt(252)),
                    'information_ratio': float((data['strategy'].mean() - data['benchmark'].mean()) / 
                                            (data['strategy'] - data['benchmark']).std() * np.sqrt(252))
                }
        
        # Calculate strategy consistency scores across regimes
        regime_metrics = {}
        for metric in ['annualized_return', 'sharpe_ratio', 'max_drawdown']:
            values = [perf[metric] for perf in regime_performance.values() if metric in perf]
            if values:
                regime_metrics[metric] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'std': float(np.std(values)),
                    'range': float(np.max(values) - np.min(values)),
                    'consistency_score': float(1.0 - np.std(values) / (abs(np.mean(values)) + 1e-6))
                }
        
        # Calculate overall consistency score (higher is better)
        if 'sharpe_ratio' in regime_metrics:
            sharpe_consistency = regime_metrics['sharpe_ratio']['consistency_score']
            sharpe_mean = regime_metrics['sharpe_ratio']['mean']
            overall_consistency = sharpe_consistency * (0.5 + 0.5 * (1.0 / (1.0 + np.exp(-sharpe_mean))))
        else:
            overall_consistency = 0.0
        
        # Store all results
        self.historical_results = {
            'regime_performance': regime_performance,
            'overall_performance': overall_performance,
            'regime_dates': {
                str(regime): {
                    'dates': list(regimes[regimes == regime].index.strftime('%Y-%m-%d')),
                    'count': len(regimes[regimes == regime]),
                    'percentage': float(len(regimes[regimes == regime]) / len(regimes))
                } for regime in regimes.unique()
            },
            'regime_metrics': regime_metrics,
            'overall_consistency': float(overall_consistency),
            'statistical_significance': {
                'overall': overall_significance,
                'by_regime': regime_statistical_significance
            }
        }
        
        # Add optional results if calculated
        if include_drawdown_paths and regime_drawdowns:
            self.historical_results['regime_drawdowns'] = regime_drawdowns
            
        if include_regime_transitions and regime_transitions:
            self.historical_results['regime_transitions'] = regime_transitions
            
        if benchmark_returns is not None:
            self.historical_results['benchmark_comparison'] = {
                'overall': overall_rel_perf,
                'by_regime': regime_relative_performance
            }
        
        logger.info(f"Completed enhanced historical stress test across {len(regime_performance)} regimes")
        
        return self.historical_results
    
    def run_synthetic_stress_test(self, 
                                 duration: int = 252,  # 1 year
                                 repeats: int = 3) -> Dict:
        """
        Run stress test on synthetic market scenarios.
        
        Args:
            duration: Duration of each synthetic scenario in days
            repeats: Number of repeats for each regime
            
        Returns:
            Dictionary with synthetic stress test results
        """
        if self.regimes is None:
            logger.warning("No regimes available for synthetic stress test")
            return {}
            
        if self.regime_detector is None or not hasattr(self.regime_detector, 'regime_stats'):
            logger.warning("No regime statistics available for synthetic stress test")
            return {}
            
        # Generate synthetic data for each regime
        synthetic_performance = {}
        
        for regime in self.regimes.unique():
            # Skip if no statistics available for this regime
            if str(regime) not in self.regime_detector.regime_stats:
                continue
                
            regime_stats = self.regime_detector.regime_stats[str(regime)]
            
            # Skip if return statistics not available
            if 'mean_return' not in regime_stats or 'std_return' not in regime_stats:
                continue
                
            # Parameters for synthetic returns
            mean_return = regime_stats['mean_return']
            std_return = regime_stats['std_return']
            
            # Generate multiple synthetic scenarios
            scenario_results = []
            
            for i in range(repeats):
                # Generate synthetic returns
                synthetic_returns = np.random.normal(mean_return, std_return, duration)
                synthetic_returns = pd.Series(synthetic_returns)
                
                # Run strategy on synthetic data
                # For simplicity, we'll use the synthetic returns directly
                # In practice, you would generate synthetic price data and run the full strategy
                
                # Calculate performance metrics
                performance = self._calculate_performance_metrics(synthetic_returns)
                scenario_results.append(performance)
                
            # Average results across scenarios
            avg_performance = {}
            for metric in scenario_results[0].keys():
                values = [result[metric] for result in scenario_results]
                avg_performance[metric] = np.mean(values)
                
            synthetic_performance[str(regime)] = {
                'avg_performance': avg_performance,
                'scenario_results': scenario_results
            }
            
        # Store results
        self.synthetic_results = {
            'synthetic_performance': synthetic_performance,
            'parameters': {
                'duration': duration,
                'repeats': repeats
            }
        }
        
        logger.info(f"Completed synthetic stress test with {len(synthetic_performance)} regimes")
        
        return self.synthetic_results
    
    def run_monte_carlo_stress_test(self, 
                                  n_simulations: int = None, 
                                  confidence_level: float = 0.95) -> Dict:
        """
        Run Monte Carlo stress test simulations.
        
        Args:
            n_simulations: Number of simulations (default: self.monte_carlo_sims)
            confidence_level: Confidence level for VaR and CVaR
            
        Returns:
            Dictionary with Monte Carlo stress test results
        """
        n_simulations = n_simulations or self.monte_carlo_sims
        
        # Run strategy on full dataset
        full_returns = self.strategy_func(self.market_data)
        
        if full_returns is None or len(full_returns) == 0:
            logger.error("Strategy returned no results")
            return {}
            
        # Calculate mean and covariance matrix of returns
        mean_return = full_returns.mean()
        std_return = full_returns.std()
        
        # Generate Monte Carlo simulations
        simulated_returns = []
        
        for i in range(n_simulations):
            # Generate random returns
            sim_returns = np.random.normal(mean_return, std_return, len(full_returns))
            simulated_returns.append(sim_returns)
            
        # Calculate performance metrics for each simulation
        sim_performance = []
        
        with ProcessPoolExecutor(max_workers=self.parallel_jobs) as executor:
            # Submit all simulations
            futures = [executor.submit(self._calculate_performance_metrics, 
                                   pd.Series(sim_returns)) 
                     for sim_returns in simulated_returns]
            
            # Get results
            for future in futures:
                sim_performance.append(future.result())
        
        # Calculate statistics across simulations
        metrics = list(sim_performance[0].keys())
        metrics_stats = {}
        
        for metric in metrics:
            values = np.array([perf[metric] for perf in sim_performance])
            
            metrics_stats[metric] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'q5': float(np.percentile(values, 5)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'q95': float(np.percentile(values, 95)),
            }
            
        # Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
        returns = np.array([perf['total_return'] for perf in sim_performance])
        var = float(np.percentile(returns, 100 * (1 - confidence_level)))
        cvar = float(np.mean(returns[returns <= var]))
        
        # Store results
        self.monte_carlo_results = {
            'metrics_stats': metrics_stats,
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level,
            'n_simulations': n_simulations
        }
        
        logger.info(f"Completed Monte Carlo stress test with {n_simulations} simulations")
        
        return self.monte_carlo_results
    
    def run_factor_shock_stress_test(self, 
                                   factor_shocks: Optional[Dict[str, float]] = None) -> Dict:
        """
        Run stress test with factor shocks.
        
        Args:
            factor_shocks: Dictionary mapping factors to shock magnitudes
            
        Returns:
            Dictionary with factor shock stress test results
        """
        # Default factor shocks if none provided
        if factor_shocks is None:
            factor_shocks = {
                'market': -0.10,  # 10% market drop
                'volatility': 2.0,  # 2x volatility
                'correlation': 0.2,  # 0.2 increase in correlations
                'liquidity': -0.5,  # 50% decrease in liquidity
            }
            
        # This is a placeholder implementation
        # In practice, would apply factor shocks to market data and run strategy
        
        logger.warning("Factor shock stress test is a placeholder implementation")
        
        # Store results
        self.factor_shock_results = {
            'factor_shocks': factor_shocks,
            'placeholder': True
        }
        
        return self.factor_shock_results
    
    def run_tail_risk_stress_test(self, 
                                confidence_levels: List[float] = None) -> Dict:
        """
        Run tail risk stress test.
        
        Args:
            confidence_levels: List of confidence levels for VaR and CVaR
            
        Returns:
            Dictionary with tail risk stress test results
        """
        # Default confidence levels if none provided
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
            
        # Run strategy on full dataset
        full_returns = self.strategy_func(self.market_data)
        
        if full_returns is None or len(full_returns) == 0:
            logger.error("Strategy returned no results")
            return {}
            
        # Calculate tail risk metrics
        var_metrics = {}
        cvar_metrics = {}
        
        for level in confidence_levels:
            # Calculate VaR
            var = float(np.percentile(full_returns, 100 * (1 - level)))
            var_metrics[f"{level:.2f}"] = var
            
            # Calculate CVaR
            cvar = float(np.mean(full_returns[full_returns <= var]))
            cvar_metrics[f"{level:.2f}"] = cvar
            
        # Calculate extreme value metrics
        min_return = float(full_returns.min())
        max_return = float(full_returns.max())
        worst_drawdown = self._calculate_max_drawdown(full_returns)
        
        # Store results
        self.tail_risk_results = {
            'var_metrics': var_metrics,
            'cvar_metrics': cvar_metrics,
            'min_return': min_return,
            'max_return': max_return,
            'worst_drawdown': worst_drawdown,
            'confidence_levels': confidence_levels
        }
        
        logger.info(f"Completed tail risk stress test with {len(confidence_levels)} confidence levels")
        
        return self.tail_risk_results
    
    def run_liquidity_crisis_stress_test(self) -> Dict:
        """
        Run liquidity crisis stress test.
        
        Returns:
            Dictionary with liquidity crisis stress test results
        """
        # This is a placeholder implementation
        # In practice, would simulate liquidity crisis scenarios
        
        logger.warning("Liquidity crisis stress test is a placeholder implementation")
        
        # Store results
        self.liquidity_crisis_results = {
            'placeholder': True
        }
        
        return self.liquidity_crisis_results
    
    def run_custom_stress_test(self, custom_scenarios: Optional[Dict] = None) -> Dict:
        """
        Run custom stress test with user-defined scenarios.
        
        Args:
            custom_scenarios: Dictionary with custom stress scenarios
            
        Returns:
            Dictionary with custom stress test results
        """
        # This is a placeholder implementation
        # In practice, would run strategy on custom scenarios
        
        logger.warning("Custom stress test is a placeholder implementation")
        
        # Store results
        self.custom_stress_results = {
            'custom_scenarios': custom_scenarios,
            'placeholder': True
        }
        
        return self.custom_stress_results
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate performance metrics for a return series.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with performance metrics
        """
        if len(returns) == 0:
            return {}
            
        # Calculate basic statistics
        mean_return = float(returns.mean())
        median_return = float(returns.median())
        std_return = float(returns.std())
        min_return = float(returns.min())
        max_return = float(returns.max())
        
        # Calculate performance metrics
        total_return = float((1 + returns).prod() - 1)
        annualized_return = float((1 + total_return) ** (252 / len(returns)) - 1) if len(returns) > 0 else 0
        annualized_volatility = float(std_return * np.sqrt(252))
        sharpe_ratio = float(annualized_return / annualized_volatility) if annualized_volatility > 0 else 0
        
        # Calculate drawdown
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # Calculate additional metrics
        positive_days = float((returns > 0).mean())
        negative_days = float((returns < 0).mean())
        win_loss_ratio = float(abs(returns[returns > 0].mean() / returns[returns < 0].mean())) if len(returns[returns < 0]) > 0 else float('inf')
        
        # Compile metrics
        metrics = {
            'mean_return': mean_return,
            'median_return': median_return,
            'std_return': std_return,
            'min_return': min_return,
            'max_return': max_return,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'positive_days': positive_days,
            'negative_days': negative_days,
            'win_loss_ratio': win_loss_ratio,
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown from returns.
        
        Args:
            returns: Series of returns
            
        Returns:
            Maximum drawdown as a positive percentage
        """
        # Convert returns to cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cum_returns / running_max - 1)
        
        # Return maximum drawdown as a positive value
        return float(abs(drawdown.min()))
    
    def plot_stress_test_results(self, 
                               test_type: StressTestType = StressTestType.HISTORICAL,
                               figsize=(12, 10)) -> plt.Figure:
        """
        Plot stress test results.
        
        Args:
            test_type: Type of stress test to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if test_type == StressTestType.HISTORICAL:
            return self._plot_historical_stress_test(figsize)
        elif test_type == StressTestType.MONTE_CARLO:
            return self._plot_monte_carlo_stress_test(figsize)
        elif test_type == StressTestType.TAIL_RISK:
            return self._plot_tail_risk_stress_test(figsize)
        else:
            logger.warning(f"No plotting implementation for {test_type.value} stress test")
            return None
    
    def _plot_historical_stress_test(self, figsize=(15, 12)) -> plt.Figure:
        """
        Plot enhanced historical stress test results with detailed analysis.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with comprehensive visualization of stress test results
        """
        if self.historical_results is None:
            logger.warning("No historical stress test results to plot")
            return None
            
        # Create figure with subplots (3x2 grid for more detailed analysis)
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # Get regime performance data
        regime_performance = self.historical_results['regime_performance']
        
        # Get regime names, replacing numeric IDs with readable labels if available
        regimes = list(regime_performance.keys())
        regime_labels = []
        for r in regimes:
            if r in RegimeType.__members__:
                regime_labels.append(RegimeType[r].value.capitalize())
            elif r.isdigit() and int(r) < len(RegimeType):
                regime_labels.append(list(RegimeType)[int(r)].value.capitalize())
            else:
                regime_labels.append(f"Regime {r}")
        
        # Extract performance metrics
        returns = [perf['annualized_return'] * 100 for perf in regime_performance.values()]
        volatilities = [perf['annualized_volatility'] * 100 for perf in regime_performance.values()]
        sharpe_ratios = [perf['sharpe_ratio'] for perf in regime_performance.values()]
        drawdowns = [perf['max_drawdown'] * 100 for perf in regime_performance.values()]
        win_rates = [perf.get('positive_days', 0) * 100 for perf in regime_performance.values()]
        
        # 1. Plot returns by regime (sorted)
        return_data = sorted(zip(regime_labels, returns), key=lambda x: x[1])
        sorted_labels = [r for r, _ in return_data]
        sorted_returns = [v for _, v in return_data]
        
        bars = axes[0, 0].barh(sorted_labels, sorted_returns)
        
        # Color bars based on return
        for i, bar in enumerate(bars):
            if sorted_returns[i] >= 0:
                bar.set_color('#2ca02c')  # Green
            else:
                bar.set_color('#d62728')  # Red
                
        axes[0, 0].set_title('Annualized Return by Regime', fontweight='bold')
        axes[0, 0].set_xlabel('Return (%)')
        axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add statistical significance markers if available
        if 'statistical_significance' in self.historical_results and 'by_regime' in self.historical_results['statistical_significance']:
            sig_data = self.historical_results['statistical_significance']['by_regime']
            for i, label in enumerate(sorted_labels):
                regime_key = next((r for r in regimes if regime_labels[regimes.index(r)] == label), None)
                if regime_key and regime_key in sig_data and sig_data[regime_key].get('significant_05', False):
                    marker = '*' if sig_data[regime_key].get('significant_01', False) else '+'
                    axes[0, 0].text(sorted_returns[i], i, marker, fontsize=14, 
                                  va='center', ha='left' if sorted_returns[i] >= 0 else 'right',
                                  color='black')
        
        # 2. Plot Sharpe ratio by regime (sorted)
        sharpe_data = sorted(zip(regime_labels, sharpe_ratios), key=lambda x: x[1])
        sorted_labels = [r for r, _ in sharpe_data]
        sorted_sharpes = [v for _, v in sharpe_data]
        
        bars = axes[0, 1].barh(sorted_labels, sorted_sharpes)
        
        # Color bars based on Sharpe ratio
        for i, bar in enumerate(bars):
            # Color gradient based on Sharpe value
            if sorted_sharpes[i] >= 1.0:
                bar.set_color('#2ca02c')  # Good - Green
            elif sorted_sharpes[i] >= 0.5:
                bar.set_color('#7fba2c')  # Moderate - Light Green
            elif sorted_sharpes[i] >= 0:
                bar.set_color('#b8c62c')  # Marginal - Yellow-Green
            elif sorted_sharpes[i] >= -0.5:
                bar.set_color('#e6912c')  # Poor - Orange
            else:
                bar.set_color('#d62728')  # Bad - Red
                
        axes[0, 1].set_title('Sharpe Ratio by Regime', fontweight='bold')
        axes[0, 1].set_xlabel('Sharpe Ratio')
        axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Plot drawdown by regime (sorted)
        drawdown_data = sorted(zip(regime_labels, drawdowns), key=lambda x: x[1], reverse=True)
        sorted_labels = [r for r, _ in drawdown_data]
        sorted_drawdowns = [v for _, v in drawdown_data]
        
        bars = axes[1, 0].barh(sorted_labels, sorted_drawdowns)
        
        # Color bars based on drawdown severity
        for i, bar in enumerate(bars):
            # Gradient based on drawdown severity
            if sorted_drawdowns[i] <= 5:
                bar.set_color('#2ca02c')  # Excellent - Green
            elif sorted_drawdowns[i] <= 10:
                bar.set_color('#7fba2c')  # Good - Light Green
            elif sorted_drawdowns[i] <= 15:
                bar.set_color('#b8c62c')  # Moderate - Yellow-Green
            elif sorted_drawdowns[i] <= 20:
                bar.set_color('#e6912c')  # Concerning - Orange
            else:
                bar.set_color('#d62728')  # Severe - Red
                
        axes[1, 0].set_title('Maximum Drawdown by Regime', fontweight='bold')
        axes[1, 0].set_xlabel('Drawdown (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Plot win rate (% positive days) by regime
        win_rate_data = sorted(zip(regime_labels, win_rates), key=lambda x: x[1])
        sorted_labels = [r for r, _ in win_rate_data]
        sorted_win_rates = [v for _, v in win_rate_data]
        
        bars = axes[1, 1].barh(sorted_labels, sorted_win_rates)
        
        # Color based on win rate
        for i, bar in enumerate(bars):
            # Color gradient based on win rate
            if sorted_win_rates[i] >= 60:
                bar.set_color('#2ca02c')  # Excellent - Green
            elif sorted_win_rates[i] >= 55:
                bar.set_color('#7fba2c')  # Good - Light Green
            elif sorted_win_rates[i] >= 50:
                bar.set_color('#b8c62c')  # Fair - Yellow-Green
            elif sorted_win_rates[i] >= 45:
                bar.set_color('#e6912c')  # Poor - Orange
            else:
                bar.set_color('#d62728')  # Bad - Red
                
        axes[1, 1].set_title('Win Rate (% Positive Days) by Regime', fontweight='bold')
        axes[1, 1].set_xlabel('Win Rate (%)')
        axes[1, 1].axvline(x=50, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Plot risk-adjusted comparison: Return/Volatility scatter plot
        for i, (regime, perf) in enumerate(regime_performance.items()):
            label = regime_labels[regimes.index(regime)]
            ret = perf['annualized_return'] * 100
            vol = perf['annualized_volatility'] * 100
            sharpe = perf['sharpe_ratio']
            
            # Scale marker size based on period length
            size = 100
            if 'regime_dates' in self.historical_results and regime in self.historical_results['regime_dates']:
                size = max(50, min(200, self.historical_results['regime_dates'][regime]['count'] / 2))
            
            # Color based on Sharpe ratio
            if sharpe >= 1.0:
                color = '#2ca02c'  # Good - Green
            elif sharpe >= 0.5:
                color = '#7fba2c'  # Moderate - Light Green
            elif sharpe >= 0:
                color = '#b8c62c'  # Marginal - Yellow-Green
            elif sharpe >= -0.5:
                color = '#e6912c'  # Poor - Orange
            else:
                color = '#d62728'  # Bad - Red
            
            axes[2, 0].scatter(vol, ret, s=size, color=color, alpha=0.7, edgecolors='black', label=label)
            axes[2, 0].text(vol+0.5, ret, label, fontsize=9)
            
        # Add reference lines
        max_vol = max(volatilities) * 1.1
        max_ret = max(abs(min(returns)), abs(max(returns))) * 1.1
        
        # Add Sharpe ratio reference lines
        sharpe_levels = [0.5, 1.0, 1.5, 2.0]
        for sharpe in sharpe_levels:
            vols = np.linspace(0, max_vol, 100)
            rets = sharpe * vols / 100 * 100  # Convert to percentage
            axes[2, 0].plot(vols, rets, 'k--', alpha=0.2)
            # Label the line at 3/4 of the way
            idx = int(len(vols) * 0.75)
            axes[2, 0].text(vols[idx], rets[idx], f"SR={sharpe}", fontsize=8, 
                          ha='left', va='bottom', alpha=0.7)
            
        axes[2, 0].set_title('Risk-Return Profile by Regime', fontweight='bold')
        axes[2, 0].set_xlabel('Volatility (%)')
        axes[2, 0].set_ylabel('Return (%)')
        axes[2, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[2, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
        axes[2, 0].grid(True, alpha=0.3)
        
        # Set axis limits with some padding
        axes[2, 0].set_xlim(0, max_vol)
        axes[2, 0].set_ylim(-max_ret, max_ret)
        
        # 6. Plot benchmark comparison or consistency metrics
        if 'benchmark_comparison' in self.historical_results and 'by_regime' in self.historical_results['benchmark_comparison']:
            # We have benchmark comparison data - plot alpha by regime
            benchmark_data = self.historical_results['benchmark_comparison']['by_regime']
            
            # Extract alpha values
            alphas = []
            alpha_labels = []
            alpha_significance = []
            
            for regime, data in benchmark_data.items():
                if 'relative_metrics' in data and 'alpha_annualized' in data['relative_metrics']:
                    alpha = data['relative_metrics']['alpha_annualized'] * 100  # Convert to percentage
                    alphas.append(alpha)
                    
                    # Get regime label
                    if regime in RegimeType.__members__:
                        label = RegimeType[regime].value.capitalize()
                    elif regime.isdigit() and int(regime) < len(RegimeType):
                        label = list(RegimeType)[int(regime)].value.capitalize()
                    else:
                        label = f"Regime {regime}"
                    
                    alpha_labels.append(label)
                    
                    # Get significance
                    sig = data['relative_metrics'].get('alpha_significant_05', False)
                    alpha_significance.append(sig)
            
            if alphas:
                # Sort by alpha value
                alpha_data = sorted(zip(alpha_labels, alphas, alpha_significance), key=lambda x: x[1])
                sorted_labels = [r for r, _, _ in alpha_data]
                sorted_alphas = [v for _, v, _ in alpha_data]
                sorted_sig = [s for _, _, s in alpha_data]
                
                bars = axes[2, 1].barh(sorted_labels, sorted_alphas)
                
                # Color based on alpha value and add significance markers
                for i, (bar, sig) in enumerate(zip(bars, sorted_sig)):
                    if sorted_alphas[i] > 0:
                        bar.set_color('#2ca02c')  # Positive - Green
                    else:
                        bar.set_color('#d62728')  # Negative - Red
                    
                    # Add significance marker
                    if sig:
                        axes[2, 1].text(sorted_alphas[i], i, '*', fontsize=14, 
                                      va='center', ha='left' if sorted_alphas[i] >= 0 else 'right',
                                      color='black')
                
                axes[2, 1].set_title('Annualized Alpha by Regime', fontweight='bold')
                axes[2, 1].set_xlabel('Alpha (%)')
                axes[2, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
                axes[2, 1].grid(True, alpha=0.3)
                
        elif 'regime_metrics' in self.historical_results:
            # No benchmark data - plot consistency metrics
            regime_metrics = self.historical_results['regime_metrics']
            metrics = list(regime_metrics.keys())
            consistency_scores = [regime_metrics[m]['consistency_score'] for m in metrics]
            
            # Create more descriptive labels
            metric_labels = {
                'annualized_return': 'Return Consistency',
                'sharpe_ratio': 'Sharpe Consistency',
                'max_drawdown': 'Drawdown Consistency'
            }
            
            plot_labels = [metric_labels.get(m, m) for m in metrics]
            
            # Sort by consistency score
            consistency_data = sorted(zip(plot_labels, consistency_scores), key=lambda x: x[1])
            sorted_labels = [l for l, _ in consistency_data]
            sorted_scores = [s for _, s in consistency_data]
            
            bars = axes[2, 1].barh(sorted_labels, sorted_scores)
            
            # Color based on consistency score
            for i, bar in enumerate(bars):
                # Gradient based on consistency
                if sorted_scores[i] >= 0.8:
                    bar.set_color('#2ca02c')  # Very consistent - Green
                elif sorted_scores[i] >= 0.6:
                    bar.set_color('#7fba2c')  # Consistent - Light Green
                elif sorted_scores[i] >= 0.4:
                    bar.set_color('#b8c62c')  # Moderate - Yellow-Green
                elif sorted_scores[i] >= 0.2:
                    bar.set_color('#e6912c')  # Inconsistent - Orange
                else:
                    bar.set_color('#d62728')  # Very inconsistent - Red
            
            axes[2, 1].set_title('Strategy Consistency Across Regimes', fontweight='bold')
            axes[2, 1].set_xlabel('Consistency Score (higher is better)')
            axes[2, 1].grid(True, alpha=0.3)
            
            # Add overall consistency score if available
            if 'overall_consistency' in self.historical_results:
                overall = self.historical_results['overall_consistency']
                axes[2, 1].text(0.5, -0.15, f"Overall Regime Consistency: {overall:.2f}/1.0", 
                              ha='center', va='center', fontsize=12, fontweight='bold',
                              transform=axes[2, 1].transAxes)
        
        # Add overall title
        fig.suptitle('Historical Market Regime Stress Test Results', fontsize=16, fontweight='bold', y=0.98)
        
        # Add legend for significance markers
        if 'statistical_significance' in self.historical_results:
            fig.text(0.5, 0.01, '* Statistical significance at 1% level, + Statistical significance at 5% level',
                   ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        return fig
    
    def _plot_monte_carlo_stress_test(self, figsize=(12, 10)) -> plt.Figure:
        """
        Plot Monte Carlo stress test results.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.monte_carlo_results is None:
            logger.warning("No Monte Carlo stress test results to plot")
            return None
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Get metrics statistics
        metrics_stats = self.monte_carlo_results['metrics_stats']
        
        # Plot total return distribution
        if 'total_return' in metrics_stats:
            total_return_stats = metrics_stats['total_return']
            
            # Create synthetic data for histogram
            mean = total_return_stats['mean']
            std = total_return_stats['std']
            
            # Generate data points from normal distribution
            x = np.linspace(mean - 3*std, mean + 3*std, 100)
            y = norm.pdf(x, mean, std)
            
            axes[0, 0].plot(x, y)
            axes[0, 0].fill_between(x, y, 0, alpha=0.3)
            axes[0, 0].axvline(x=total_return_stats['mean'], color='r', linestyle='--', label='Mean')
            axes[0, 0].axvline(x=total_return_stats['q5'], color='g', linestyle='--', label='5th Percentile')
            axes[0, 0].axvline(x=total_return_stats['q95'], color='g', linestyle='--', label='95th Percentile')
            axes[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
            axes[0, 0].set_title('Total Return Distribution')
            axes[0, 0].set_xlabel('Total Return')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
        # Plot Sharpe ratio distribution
        if 'sharpe_ratio' in metrics_stats:
            sharpe_stats = metrics_stats['sharpe_ratio']
            
            # Create synthetic data for histogram
            mean = sharpe_stats['mean']
            std = sharpe_stats['std']
            
            # Generate data points from normal distribution
            x = np.linspace(mean - 3*std, mean + 3*std, 100)
            y = norm.pdf(x, mean, std)
            
            axes[0, 1].plot(x, y)
            axes[0, 1].fill_between(x, y, 0, alpha=0.3)
            axes[0, 1].axvline(x=sharpe_stats['mean'], color='r', linestyle='--', label='Mean')
            axes[0, 1].axvline(x=sharpe_stats['q5'], color='g', linestyle='--', label='5th Percentile')
            axes[0, 1].axvline(x=sharpe_stats['q95'], color='g', linestyle='--', label='95th Percentile')
            axes[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
            axes[0, 1].set_title('Sharpe Ratio Distribution')
            axes[0, 1].set_xlabel('Sharpe Ratio')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
        # Plot max drawdown distribution
        if 'max_drawdown' in metrics_stats:
            drawdown_stats = metrics_stats['max_drawdown']
            
            # Create synthetic data for histogram
            mean = drawdown_stats['mean']
            std = drawdown_stats['std']
            
            # Generate data points from normal distribution
            x = np.linspace(mean - 3*std, mean + 3*std, 100)
            y = norm.pdf(x, mean, std)
            
            axes[1, 0].plot(x, y)
            axes[1, 0].fill_between(x, y, 0, alpha=0.3)
            axes[1, 0].axvline(x=drawdown_stats['mean'], color='r', linestyle='--', label='Mean')
            axes[1, 0].axvline(x=drawdown_stats['q5'], color='g', linestyle='--', label='5th Percentile')
            axes[1, 0].axvline(x=drawdown_stats['q95'], color='g', linestyle='--', label='95th Percentile')
            axes[1, 0].set_title('Maximum Drawdown Distribution')
            axes[1, 0].set_xlabel('Maximum Drawdown')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
        # Plot VaR and CVaR
        var = self.monte_carlo_results['var']
        cvar = self.monte_carlo_results['cvar']
        confidence_level = self.monte_carlo_results['confidence_level']
        
        # Create bar chart for VaR and CVaR
        axes[1, 1].bar(['VaR', 'CVaR'], [var, cvar])
        axes[1, 1].set_title(f'Value at Risk (VaR) and Conditional VaR (CVaR)\nConfidence Level: {confidence_level:.0%}')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def _plot_tail_risk_stress_test(self, figsize=(12, 6)) -> plt.Figure:
        """
        Plot tail risk stress test results.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.tail_risk_results is None:
            logger.warning("No tail risk stress test results to plot")
            return None
            
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Get VaR and CVaR metrics
        var_metrics = self.tail_risk_results['var_metrics']
        cvar_metrics = self.tail_risk_results['cvar_metrics']
        confidence_levels = [float(level) for level in var_metrics.keys()]
        
        # Sort by confidence level
        sorted_indices = np.argsort(confidence_levels)
        sorted_levels = [confidence_levels[i] for i in sorted_indices]
        sorted_vars = [var_metrics[f"{level:.2f}"] for level in sorted_levels]
        sorted_cvars = [cvar_metrics[f"{level:.2f}"] for level in sorted_levels]
        
        # Plot VaR by confidence level
        axes[0].plot(sorted_levels, sorted_vars, 'o-', label='VaR')
        axes[0].set_title('Value at Risk (VaR) by Confidence Level')
        axes[0].set_xlabel('Confidence Level')
        axes[0].set_ylabel('VaR')
        axes[0].grid(True, alpha=0.3)
        
        # Plot CVaR by confidence level
        axes[1].plot(sorted_levels, sorted_cvars, 'o-', label='CVaR')
        axes[1].set_title('Conditional VaR (CVaR) by Confidence Level')
        axes[1].set_xlabel('Confidence Level')
        axes[1].set_ylabel('CVaR')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def generate_stress_test_report(self, 
                                  output_dir: Optional[str] = None,
                                  include_plots: bool = True) -> Dict:
        """
        Generate a comprehensive stress test report.
        
        Args:
            output_dir: Directory to save report files
            include_plots: Whether to include plots in the report
            
        Returns:
            Dictionary with report data
        """
        # Ensure we have at least some stress test results
        if not hasattr(self, 'stress_test_results') or not self.stress_test_results:
            logger.warning("No stress test results available for report generation")
            return {}
            
        # Prepare report data
        report = {
            'test_types': list(self.stress_test_results.keys()),
            'results_summary': {},
            'worst_regime': None,
            'best_regime': None,
            'overall_robustness': None
        }
        
        # Add historical test summary if available
        if 'historical' in self.stress_test_results:
            historical = self.stress_test_results['historical']
            
            # Add overall performance
            report['results_summary']['overall_performance'] = historical['overall_performance']
            
            # Find best and worst regimes
            if 'regime_performance' in historical:
                regime_perf = historical['regime_performance']
                
                # Sort regimes by Sharpe ratio
                sorted_regimes = sorted(regime_perf.items(), 
                                     key=lambda x: x[1]['sharpe_ratio'],
                                     reverse=True)
                
                if sorted_regimes:
                    report['best_regime'] = {
                        'regime': sorted_regimes[0][0],
                        'performance': sorted_regimes[0][1]
                    }
                    
                    report['worst_regime'] = {
                        'regime': sorted_regimes[-1][0],
                        'performance': sorted_regimes[-1][1]
                    }
                    
                # Calculate overall robustness score
                sharpe_ratios = [perf['sharpe_ratio'] for perf in regime_perf.values()]
                if sharpe_ratios:
                    min_sharpe = min(sharpe_ratios)
                    max_sharpe = max(sharpe_ratios)
                    mean_sharpe = np.mean(sharpe_ratios)
                    
                    # Robustness score based on Sharpe ratio consistency
                    sharpe_range = max_sharpe - min_sharpe
                    mean_abs_sharpe = np.mean([abs(s) for s in sharpe_ratios])
                    
                    if mean_abs_sharpe > 0:
                        robustness = 1.0 - (sharpe_range / (2 * mean_abs_sharpe))
                        report['overall_robustness'] = max(0.0, min(1.0, robustness))
        
        # Add Monte Carlo test summary if available
        if 'monte_carlo' in self.stress_test_results:
            monte_carlo = self.stress_test_results['monte_carlo']
            
            # Add VaR and CVaR
            report['results_summary']['var'] = monte_carlo['var']
            report['results_summary']['cvar'] = monte_carlo['cvar']
            report['results_summary']['confidence_level'] = monte_carlo['confidence_level']
            
            # Add metrics mean values
            for metric, stats in monte_carlo['metrics_stats'].items():
                report['results_summary'][f'mean_{metric}'] = stats['mean']
                
        # Add tail risk test summary if available
        if 'tail_risk' in self.stress_test_results:
            tail_risk = self.stress_test_results['tail_risk']
            
            # Add worst drawdown
            report['results_summary']['worst_drawdown'] = tail_risk['worst_drawdown']
            
            # Add highest confidence VaR and CVaR
            confidence_levels = tail_risk['confidence_levels']
            highest_level = f"{max(confidence_levels):.2f}"
            
            report['results_summary']['high_confidence_var'] = tail_risk['var_metrics'][highest_level]
            report['results_summary']['high_confidence_cvar'] = tail_risk['cvar_metrics'][highest_level]
            
        # Generate and save report if output directory is specified
        if output_dir is not None:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save report as JSON
            report_path = os.path.join(output_dir, 'stress_test_report.json')
            
            # Prepare JSON-serializable report
            json_report = {
                'test_types': report['test_types'],
                'results_summary': {
                    k: float(v) if isinstance(v, (np.floating, float)) else v 
                    for k, v in report['results_summary'].items()
                },
                'overall_robustness': float(report['overall_robustness']) if report['overall_robustness'] is not None else None
            }
            
            # Add best and worst regimes
            if report['best_regime'] is not None:
                json_report['best_regime'] = {
                    'regime': report['best_regime']['regime'],
                    'performance': {
                        k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in report['best_regime']['performance'].items()
                    }
                }
                
            if report['worst_regime'] is not None:
                json_report['worst_regime'] = {
                    'regime': report['worst_regime']['regime'],
                    'performance': {
                        k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in report['worst_regime']['performance'].items()
                    }
                }
                
            # Save JSON report
            with open(report_path, 'w') as f:
                json.dump(json_report, f, indent=2)
                
            report['report_path'] = report_path
            
            # Generate and save plots if requested
            if include_plots:
                plots = {}
                
                # Historical stress test plot
                if 'historical' in self.stress_test_results:
                    fig = self._plot_historical_stress_test()
                    if fig is not None:
                        plot_path = os.path.join(output_dir, 'historical_stress_test.png')
                        fig.savefig(plot_path)
                        plt.close(fig)
                        
                        plots['historical'] = plot_path
                        
                # Monte Carlo stress test plot
                if 'monte_carlo' in self.stress_test_results:
                    fig = self._plot_monte_carlo_stress_test()
                    if fig is not None:
                        plot_path = os.path.join(output_dir, 'monte_carlo_stress_test.png')
                        fig.savefig(plot_path)
                        plt.close(fig)
                        
                        plots['monte_carlo'] = plot_path
                        
                # Tail risk stress test plot
                if 'tail_risk' in self.stress_test_results:
                    fig = self._plot_tail_risk_stress_test()
                    if fig is not None:
                        plot_path = os.path.join(output_dir, 'tail_risk_stress_test.png')
                        fig.savefig(plot_path)
                        plt.close(fig)
                        
                        plots['tail_risk'] = plot_path
                
                report['plots'] = plots
            
            logger.info(f"Saved stress test report to {report_path}")
            
        return report


def run_regime_stress_test(strategy_func: callable, 
                      market_data: pd.DataFrame, 
                      benchmark_data: pd.DataFrame = None,
                      method: str = 'hmm', 
                      n_regimes: int = 4,
                      output_dir: str = None,
                      plot_results: bool = True) -> Dict:
    """
    Convenience function to run a comprehensive regime-based stress test.
    
    Args:
        strategy_func: Function that takes market data and returns strategy returns
        market_data: DataFrame with market price data
        benchmark_data: Optional benchmark data for relative performance
        method: Regime detection method ('hmm', 'kmeans', 'threshold', etc.)
        n_regimes: Number of regimes to detect
        output_dir: Directory to save results and plots
        plot_results: Whether to generate and save plots
        
    Returns:
        Dictionary with comprehensive stress test results
    """
    # Create regime detector
    detection_method = None
    for enum_item in DetectionMethod:
        if enum_item.value == method.lower():
            detection_method = enum_item
            break
    
    if detection_method is None:
        logger.warning(f"Unknown detection method: {method}. Using HMM.")
        detection_method = DetectionMethod.HMM
        
    detector = MarketRegimeDetector(method=detection_method, n_regimes=n_regimes)
    
    # Load market data and detect regimes
    detector.load_market_data(market_data)
    regimes = detector.detect_regimes()
    
    # Create stress tester
    stress_tester = StressTester(
        strategy_func=strategy_func,
        market_data=market_data,
        regime_detector=detector
    )
    
    # Run historical stress test with benchmark comparison if available
    if benchmark_data is not None:
        results = stress_tester.run_historical_stress_test(benchmark_data=benchmark_data)
    else:
        results = stress_tester.run_historical_stress_test()
    
    # Generate report if output directory specified
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save results as JSON
        result_path = os.path.join(output_dir, 'regime_stress_test_results.json')
        
        # Prepare serializable results
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = _make_serializable(value)
            else:
                serializable_results[key] = value
                
        with open(result_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Generate and save plot if requested
        if plot_results:
            fig = stress_tester.plot_stress_test_results(StressTestType.HISTORICAL)
            if fig:
                plot_path = os.path.join(output_dir, 'regime_stress_test_plot.png')
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    return results

def _make_serializable(data):
    """Helper function to make data JSON serializable."""
    if isinstance(data, dict):
        return {k: _make_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_make_serializable(v) for v in data]
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return _make_serializable(data.tolist())
    elif isinstance(data, pd.DataFrame):
        return _make_serializable(data.to_dict())
    elif isinstance(data, pd.Series):
        return _make_serializable(data.to_dict())
    else:
        return data

def generate_regime_report(market_data: pd.DataFrame,
                          method: str = 'hmm',
                          n_regimes: int = 4,
                          lookback: int = 252,
                          output_dir: str = None) -> Dict:
    """
    Generate a report on market regimes without running a strategy.
    
    This function is useful for understanding market regimes without
    testing a specific strategy.
    
    Args:
        market_data: DataFrame with market price data
        method: Regime detection method ('hmm', 'kmeans', 'threshold', etc.)
        n_regimes: Number of regimes to detect
        lookback: Lookback window for feature calculation
        output_dir: Directory to save results and plots
        
    Returns:
        Dictionary with regime analysis results
    """
    # Create regime detector
    detection_method = None
    for enum_item in DetectionMethod:
        if enum_item.value == method.lower():
            detection_method = enum_item
            break
    
    if detection_method is None:
        logger.warning(f"Unknown detection method: {method}. Using HMM.")
        detection_method = DetectionMethod.HMM
        
    detector = MarketRegimeDetector(
        method=detection_method, 
        n_regimes=n_regimes,
        lookback_window=lookback
    )
    
    # Load market data and detect regimes
    detector.load_market_data(market_data)
    detector.detect_regimes()
    
    # Generate report
    report = detector.generate_regime_report(output_dir)
    
    return report