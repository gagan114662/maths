#!/usr/bin/env python
"""
Ensemble Learning Module

This module implements ensemble learning techniques for combining multiple trading
strategies, including weighted averaging, stacking, and dynamic strategy selection.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from typing import Dict, List, Tuple, Callable, Any, Optional
import logging
from datetime import datetime

class EnsembleLearner:
    """Implements ensemble learning for strategy combination."""
    
    def __init__(self,
                base_strategies: Dict[str, Callable],
                ensemble_method: str = 'weighted',
                meta_model: str = 'gbm',
                lookback_window: int = 252,
                rebalance_frequency: int = 21):
        """
        Initialize the ensemble learner.

        Args:
            base_strategies: Dictionary of strategy name -> strategy function
            ensemble_method: Method for combining strategies ('weighted', 'stacking', 'dynamic')
            meta_model: Model for strategy combination ('gbm', 'rf', 'linear')
            lookback_window: Window for training meta-model
            rebalance_frequency: Frequency to update strategy weights
        """
        self.base_strategies = base_strategies
        self.ensemble_method = ensemble_method
        self.meta_model = self._initialize_meta_model(meta_model)
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        
        # Runtime state
        self.strategy_weights = None
        self.strategy_performance = {}
        self.meta_features = {}
        self.current_regime = None
        
        self.logger = logging.getLogger(__name__)
        
    def fit(self,
           market_data: pd.DataFrame,
           strategy_predictions: Dict[str, pd.Series],
           target_returns: pd.Series):
        """
        Fit the ensemble model.
        
        Args:
            market_data: Market data for feature calculation
            strategy_predictions: Dictionary of strategy signals/predictions
            target_returns: Actual returns to train against
        """
        if self.ensemble_method == 'weighted':
            self._fit_weighted_ensemble(strategy_predictions, target_returns)
        elif self.ensemble_method == 'stacking':
            self._fit_stacking_ensemble(
                market_data, strategy_predictions, target_returns
            )
        else:  # dynamic
            self._fit_dynamic_ensemble(
                market_data, strategy_predictions, target_returns
            )
            
    def predict(self,
               market_data: pd.DataFrame,
               strategy_predictions: Dict[str, pd.Series]) -> pd.Series:
        """
        Generate ensemble predictions.
        
        Args:
            market_data: Current market data
            strategy_predictions: Current strategy predictions
            
        Returns:
            Combined strategy prediction
        """
        if self.ensemble_method == 'weighted':
            return self._predict_weighted(strategy_predictions)
        elif self.ensemble_method == 'stacking':
            return self._predict_stacking(market_data, strategy_predictions)
        else:  # dynamic
            return self._predict_dynamic(market_data, strategy_predictions)
            
    def _initialize_meta_model(self, model_type: str) -> Any:
        """Initialize meta-model for strategy combination."""
        if model_type == 'gbm':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            )
        elif model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=3
            )
        else:  # linear
            return LinearRegression()
            
    def _fit_weighted_ensemble(self,
                            strategy_predictions: Dict[str, pd.Series],
                            target_returns: pd.Series):
        """Fit weighted ensemble using strategy performance."""
        strategy_returns = {}
        
        # Calculate returns for each strategy
        for name, predictions in strategy_predictions.items():
            strategy_returns[name] = predictions * target_returns
            
        # Calculate Sharpe ratios
        strategy_sharpes = {}
        for name, returns in strategy_returns.items():
            sharpe = self._calculate_sharpe_ratio(returns)
            strategy_sharpes[name] = max(sharpe, 0)  # Non-negative weights
            
        # Normalize weights
        total_sharpe = sum(strategy_sharpes.values())
        if total_sharpe > 0:
            self.strategy_weights = {
                name: sharpe/total_sharpe 
                for name, sharpe in strategy_sharpes.items()
            }
        else:
            # Equal weights if all strategies perform poorly
            weight = 1.0 / len(strategy_predictions)
            self.strategy_weights = {
                name: weight for name in strategy_predictions.keys()
            }
            
    def _fit_stacking_ensemble(self,
                            market_data: pd.DataFrame,
                            strategy_predictions: Dict[str, pd.Series],
                            target_returns: pd.Series):
        """Fit stacking ensemble using meta-features."""
        # Prepare features
        features = self._prepare_meta_features(
            market_data, strategy_predictions
        )
        
        # Prepare target
        target = np.sign(target_returns)
        
        # Fit meta-model
        self.meta_model.fit(features, target)
        
        # Store feature importance
        if hasattr(self.meta_model, 'feature_importances_'):
            self.meta_features['importance'] = pd.Series(
                self.meta_model.feature_importances_,
                index=features.columns
            )
            
    def _fit_dynamic_ensemble(self,
                           market_data: pd.DataFrame,
                           strategy_predictions: Dict[str, pd.Series],
                           target_returns: pd.Series):
        """Fit dynamic ensemble using market regimes."""
        # Detect market regimes
        regimes = self._detect_market_regimes(market_data)
        
        # Fit regime-specific models
        self.regime_models = {}
        for regime in regimes.unique():
            regime_mask = regimes == regime
            
            # Prepare regime-specific data
            regime_predictions = {
                name: preds[regime_mask]
                for name, preds in strategy_predictions.items()
            }
            regime_returns = target_returns[regime_mask]
            
            # Fit regime-specific ensemble
            regime_model = EnsembleLearner(
                self.base_strategies,
                ensemble_method='stacking',
                meta_model='gbm'
            )
            regime_model.fit(
                market_data[regime_mask],
                regime_predictions,
                regime_returns
            )
            
            self.regime_models[regime] = regime_model
            
    def _predict_weighted(self,
                       strategy_predictions: Dict[str, pd.Series]) -> pd.Series:
        """Generate weighted ensemble predictions."""
        weighted_predictions = pd.Series(0, index=strategy_predictions[next(iter(strategy_predictions))].index)
        
        for name, predictions in strategy_predictions.items():
            weight = self.strategy_weights.get(name, 1.0/len(strategy_predictions))
            weighted_predictions += predictions * weight
            
        return weighted_predictions
    
    def _predict_stacking(self,
                        market_data: pd.DataFrame,
                        strategy_predictions: Dict[str, pd.Series]) -> pd.Series:
        """Generate stacking ensemble predictions."""
        # Prepare features
        features = self._prepare_meta_features(
            market_data, strategy_predictions
        )
        
        # Generate predictions
        predictions = self.meta_model.predict(features)
        
        return pd.Series(predictions, index=market_data.index)
    
    def _predict_dynamic(self,
                       market_data: pd.DataFrame,
                       strategy_predictions: Dict[str, pd.Series]) -> pd.Series:
        """Generate dynamic ensemble predictions."""
        # Detect current regime
        current_regime = self._detect_current_regime(market_data)
        
        # Use regime-specific model
        if current_regime in self.regime_models:
            return self.regime_models[current_regime].predict(
                market_data,
                strategy_predictions
            )
        else:
            # Fallback to weighted ensemble
            return self._predict_weighted(strategy_predictions)
    
    def _prepare_meta_features(self,
                            market_data: pd.DataFrame,
                            strategy_predictions: Dict[str, pd.Series]) -> pd.DataFrame:
        """Prepare features for meta-model."""
        features = pd.DataFrame()
        
        # Strategy predictions
        for name, predictions in strategy_predictions.items():
            features[f'strategy_{name}'] = predictions
            
        # Market features
        features['volatility'] = market_data['returns'].rolling(21).std()
        features['momentum'] = market_data['returns'].rolling(63).mean()
        features['trend'] = self._calculate_trend_feature(market_data)
        
        # Strategy agreement
        features['strategy_agreement'] = self._calculate_strategy_agreement(
            strategy_predictions
        )
        
        return features.fillna(0)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
            
        return np.sqrt(252) * returns.mean() / returns.std()
    
    def _detect_market_regimes(self, market_data: pd.DataFrame) -> pd.Series:
        """Detect market regimes using volatility and trend."""
        volatility = market_data['returns'].rolling(21).std() * np.sqrt(252)
        trend = self._calculate_trend_feature(market_data)
        
        regimes = pd.Series(index=market_data.index, dtype=str)
        
        # Classify regimes
        regimes[(volatility <= 0.15) & (trend > 0)] = 'low_vol_uptrend'
        regimes[(volatility <= 0.15) & (trend < 0)] = 'low_vol_downtrend'
        regimes[(volatility > 0.15) & (volatility <= 0.25)] = 'medium_vol'
        regimes[volatility > 0.25] = 'high_vol'
        
        return regimes.fillna('medium_vol')
    
    def _detect_current_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime."""
        if len(market_data) < 21:
            return 'medium_vol'
            
        current_vol = (
            market_data['returns'].tail(21).std() * np.sqrt(252)
        )
        current_trend = self._calculate_trend_feature(
            market_data.tail(63)
        ).iloc[-1]
        
        if current_vol <= 0.15:
            return 'low_vol_uptrend' if current_trend > 0 else 'low_vol_downtrend'
        elif current_vol <= 0.25:
            return 'medium_vol'
        else:
            return 'high_vol'
    
    def _calculate_trend_feature(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate trend feature using price ratios."""
        prices = market_data['close']
        ma20 = prices.rolling(20).mean()
        ma50 = prices.rolling(50).mean()
        
        return (ma20 / ma50 - 1)
    
    def _calculate_strategy_agreement(self,
                                  strategy_predictions: Dict[str, pd.Series]) -> pd.Series:
        """Calculate agreement level between strategies."""
        predictions = pd.DataFrame(strategy_predictions)
        return predictions.apply(lambda x: x.std(), axis=1)
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights."""
        return self.strategy_weights.copy() if self.strategy_weights else None
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if available."""
        return self.meta_features.get('importance')
    
    def plot_strategy_performance(self, save_path: Optional[str] = None):
        """Plot strategy performance comparison."""
        import matplotlib.pyplot as plt
        
        if not self.strategy_performance:
            return
            
        plt.figure(figsize=(12, 6))
        
        for name, perf in self.strategy_performance.items():
            plt.plot(perf.cumsum(), label=name)
            
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Strategy Performance Comparison')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        plt.close()