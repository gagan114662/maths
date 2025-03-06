"""
Volatility Assessment Agent for identifying and quantifying market noise and volatility.
"""
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, acf
from .base_agent import BaseAgent

class VolatilityAssessmentAgent(BaseAgent):
    def __init__(self, name: str = "VolatilityAgent", weight: float = 1.0):
        """
        Initialize the Volatility Assessment Agent.
        
        Args:
            name: Name of the agent
            weight: Agent's weight in the system
        """
        super().__init__(name, weight)
        self.lookback_periods = {
            'short': 20,   # 20 days
            'medium': 60,  # 60 days
            'long': 252    # 252 trading days
        }
        
    def _get_required_columns(self) -> List[str]:
        """Get list of required columns for volatility analysis."""
        return ['date', 'symbol', 'close', 'high', 'low', 'volume']

    def process(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process data to assess volatility and market noise.
        
        Args:
            data: Input data for volatility analysis
            
        Returns:
            Dictionary containing volatility metrics and analysis
        """
        if not self.validate_data(data):
            return {}
            
        results = {}
        
        # Group by symbol to process each security independently
        for symbol, group in data.groupby('symbol'):
            results[symbol] = {
                'volatility_metrics': self._calculate_volatility_metrics(group),
                'noise_metrics': self._calculate_noise_metrics(group),
                'stationarity': self._test_stationarity(group),
                'autocorrelation': self._calculate_autocorrelation(group),
                'regime_analysis': self._analyze_volatility_regime(group)
            }
            
        # Update agent's state
        self.state.update({
            'last_update': pd.Timestamp.now().isoformat(),
            'processed_symbols': list(results.keys()),
            'average_metrics': self._calculate_average_metrics(results)
        })
        
        # Log significant findings
        self._log_significant_findings(results)
        
        return results

    def _calculate_volatility_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate various volatility metrics."""
        metrics = {}
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        
        # Historical volatility for different periods
        for period_name, period in self.lookback_periods.items():
            vol = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
            metrics[f'{period_name}_term_volatility'] = vol.iloc[-1]
            
        # Parkinson volatility estimator (using high-low prices)
        hl_ratio = np.log(data['high'] / data['low'])
        metrics['parkinson_volatility'] = np.sqrt(1 / (4 * np.log(2)) * hl_ratio.pow(2).mean() * 252)
        
        # Volatility of volatility
        vol_of_vol = vol.rolling(window=20).std()
        metrics['volatility_of_volatility'] = vol_of_vol.iloc[-1]
        
        return metrics

    def _calculate_noise_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market noise metrics."""
        metrics = {}
        
        returns = data['close'].pct_change().dropna()
        
        # Noise ratio (fraction of price changes not attributable to trend)
        price_diff = data['close'].diff().abs().sum()
        total_variation = data['high'].sub(data['low']).abs().sum()
        metrics['noise_ratio'] = 1 - (price_diff / total_variation)
        
        # Distribution analysis
        metrics.update({
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'jarque_bera_stat': jarque_bera(returns)[0]
        })
        
        return metrics

    def _test_stationarity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test for stationarity using Augmented Dickey-Fuller test."""
        returns = data['close'].pct_change().dropna()
        adf_result = adfuller(returns)
        
        return {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'is_stationary': adf_result[1] < 0.05,
            'critical_values': adf_result[4]
        }

    def _calculate_autocorrelation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate autocorrelation at different lags."""
        returns = data['close'].pct_change().dropna()
        
        # Calculate ACF for different lags
        lags = [1, 5, 10, 22]  # 1 day, 1 week, 2 weeks, 1 month
        acf_values = acf(returns, nlags=max(lags))
        
        return {
            f'lag_{lag}': acf_values[lag] for lag in lags
        }

    def _analyze_volatility_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current volatility regime."""
        returns = data['close'].pct_change().dropna()
        current_vol = returns.rolling(window=20).std().iloc[-1]
        hist_vol = returns.std()
        
        # Determine regime based on current volatility relative to historical
        z_score = (current_vol - hist_vol.mean()) / hist_vol.std()
        
        regime = 'normal'
        if z_score > 2:
            regime = 'high_volatility'
        elif z_score < -2:
            regime = 'low_volatility'
            
        return {
            'current_regime': regime,
            'volatility_z_score': z_score,
            'regime_confidence': 1 - stats.norm.sf(abs(z_score))  # Confidence based on z-score
        }

    def _calculate_average_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate average metrics across all symbols."""
        avg_metrics = {}
        
        for symbol in results:
            vol_metrics = results[symbol]['volatility_metrics']
            for metric, value in vol_metrics.items():
                if metric not in avg_metrics:
                    avg_metrics[metric] = []
                avg_metrics[metric].append(value)
                
        return {k: np.mean(v) for k, v in avg_metrics.items()}

    def _log_significant_findings(self, results: Dict[str, Any]) -> None:
        """Log significant volatility findings."""
        for symbol, result in results.items():
            # Log high volatility regimes
            if result['regime_analysis']['current_regime'] == 'high_volatility':
                self.logger.warning(f"High volatility detected for {symbol}")
                
            # Log non-stationary price series
            if not result['stationarity']['is_stationary']:
                self.logger.info(f"Non-stationary price series detected for {symbol}")

    def get_confidence_score(self, data: pd.DataFrame) -> float:
        """
        Calculate confidence score based on volatility metrics quality.
        
        Args:
            data: Input data for confidence calculation
            
        Returns:
            Confidence score between 0 and 1
        """
        if not self.validate_data(data):
            return 0.0
            
        confidence_factors = []
        
        # Data quality factor
        missing_data = data[self._get_required_columns()].isnull().sum().sum()
        data_quality = 1 - (missing_data / (len(data) * len(self._get_required_columns())))
        confidence_factors.append(data_quality)
        
        # Time series length factor
        min_required_points = max(self.lookback_periods.values())
        length_factor = min(1.0, len(data) / min_required_points)
        confidence_factors.append(length_factor)
        
        # Volatility stability factor
        returns = data['close'].pct_change().dropna()
        vol_stability = 1 - min(1, returns.std().std())  # Higher stability = higher confidence
        confidence_factors.append(vol_stability)
        
        return np.mean(confidence_factors)

    def update(self, feedback: Dict[str, Any]) -> None:
        """
        Update agent's state based on feedback.
        
        Args:
            feedback: Feedback data for updating the agent
        """
        if 'weight_adjustment' in feedback:
            self.adjust_weight(feedback['weight_adjustment'])
            
        if 'lookback_periods' in feedback:
            self.lookback_periods.update(feedback['lookback_periods'])
            
        self.state.update({
            'last_feedback': feedback,
            'last_feedback_time': pd.Timestamp.now().isoformat()
        })