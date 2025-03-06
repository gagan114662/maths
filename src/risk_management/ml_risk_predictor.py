#!/usr/bin/env python
"""
Machine Learning Risk Predictor Module

This module implements machine learning models to predict various risk metrics
and potential market stress scenarios.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
from datetime import datetime, timedelta

class MLRiskPredictor:
    """
    Predicts various risk metrics using machine learning models.
    """
    
    def __init__(self, 
                lookback_window: int = 252,
                forecast_horizon: int = 21,
                feature_windows: List[int] = [5, 21, 63, 252]):
        """
        Initialize the ML risk predictor.

        Args:
            lookback_window: Historical window for feature calculation
            forecast_horizon: Number of days to forecast
            feature_windows: List of windows for feature calculation
        """
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.feature_windows = feature_windows
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.logger = logging.getLogger(__name__)
        
    def prepare_features(self, returns: pd.Series) -> pd.DataFrame:
        """
        Prepare features for risk prediction.
        
        Args:
            returns: Series of asset returns
            
        Returns:
            DataFrame of features
        """
        features = pd.DataFrame(index=returns.index)
        
        for window in self.feature_windows:
            # Rolling statistics
            rolling = returns.rolling(window=window)
            
            features[f'volatility_{window}d'] = rolling.std() * np.sqrt(252)
            features[f'skewness_{window}d'] = rolling.skew()
            features[f'kurtosis_{window}d'] = rolling.kurt()
            
            # Return features
            features[f'cum_return_{window}d'] = (1 + returns).rolling(window).prod() - 1
            features[f'max_drawdown_{window}d'] = (
                (1 + returns).rolling(window).apply(
                    lambda x: (1 + x).cumprod().max() - (1 + x).cumprod().min()
                )
            )
            
            # Directional features
            features[f'up_days_{window}d'] = (
                returns.rolling(window).apply(lambda x: (x > 0).sum() / len(x))
            )
            
            # Tail features
            features[f'var_95_{window}d'] = (
                returns.rolling(window).quantile(0.05)
            )
            features[f'cvar_95_{window}d'] = (
                returns.rolling(window).apply(
                    lambda x: x[x <= np.percentile(x, 5)].mean()
                )
            )
            
        # Add technical features
        features['rsi_14'] = self._calculate_rsi(returns, 14)
        features['volatility_trend'] = (
            features['volatility_21d'] / features['volatility_63d']
        )
        
        # Add sequential features
        features['day_of_week'] = returns.index.dayofweek
        features['month'] = returns.index.month
        
        return features.dropna()
    
    def prepare_targets(self, returns: pd.Series) -> pd.DataFrame:
        """
        Prepare target variables for risk prediction.
        
        Args:
            returns: Series of asset returns
            
        Returns:
            DataFrame of target variables
        """
        targets = pd.DataFrame(index=returns.index)
        
        # Forward-looking volatility
        future_rets = returns.shift(-self.forecast_horizon)
        targets['future_volatility'] = (
            pd.Series(index=returns.index, dtype=float)
        )
        
        for i in range(len(returns)-self.forecast_horizon):
            forward_vol = returns.iloc[i:i+self.forecast_horizon].std() * np.sqrt(252)
            targets['future_volatility'].iloc[i] = forward_vol
            
        # Forward-looking drawdown
        targets['future_drawdown'] = (
            pd.Series(index=returns.index, dtype=float)
        )
        
        for i in range(len(returns)-self.forecast_horizon):
            forward_returns = returns.iloc[i:i+self.forecast_horizon]
            cum_returns = (1 + forward_returns).cumprod()
            drawdown = (cum_returns.max() - cum_returns.min()) / cum_returns.max()
            targets['future_drawdown'].iloc[i] = drawdown
            
        # Forward-looking tail risk
        targets['future_var_95'] = (
            pd.Series(index=returns.index, dtype=float)
        )
        
        for i in range(len(returns)-self.forecast_horizon):
            forward_var = np.percentile(
                returns.iloc[i:i+self.forecast_horizon], 5
            )
            targets['future_var_95'].iloc[i] = forward_var
            
        return targets.dropna()
    
    def train_models(self, features: pd.DataFrame, targets: pd.DataFrame):
        """
        Train prediction models for each risk metric.
        
        Args:
            features: Feature DataFrame
            targets: Target DataFrame
        """
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for target_col in targets.columns:
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            self.scalers[target_col] = scaler
            
            # Train LightGBM model
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42
            )
            
            # Train with cross-validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(scaled_features):
                X_train = scaled_features[train_idx]
                y_train = targets[target_col].iloc[train_idx]
                X_val = scaled_features[val_idx]
                y_val = targets[target_col].iloc[val_idx]
                
                lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='rmse',
                    early_stopping_rounds=20,
                    verbose=False
                )
                
                cv_scores.append(
                    r2_score(y_val, lgb_model.predict(X_val))
                )
            
            self.logger.info(f"CV R² scores for {target_col}: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
            # Final fit on all data
            lgb_model.fit(scaled_features, targets[target_col])
            self.models[target_col] = lgb_model
            
            # Store feature importance
            self.feature_importance[target_col] = pd.Series(
                lgb_model.feature_importances_,
                index=features.columns
            ).sort_values(ascending=False)
    
    def predict_risk_metrics(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict risk metrics using trained models.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            DataFrame of predicted risk metrics
        """
        predictions = pd.DataFrame(index=features.index)
        
        for target, model in self.models.items():
            # Scale features
            scaled_features = self.scalers[target].transform(features)
            
            # Make predictions
            predictions[target] = model.predict(scaled_features)
            
        return predictions
    
    def get_feature_importance(self, target: str) -> pd.Series:
        """
        Get feature importance for a specific target.
        
        Args:
            target: Name of target variable
            
        Returns:
            Series of feature importance scores
        """
        if target not in self.feature_importance:
            raise ValueError(f"No feature importance available for {target}")
            
        return self.feature_importance[target]
    
    def _calculate_rsi(self, returns: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = returns
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def save_models(self, path: str):
        """Save trained models and scalers."""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'params': {
                'lookback_window': self.lookback_window,
                'forecast_horizon': self.forecast_horizon,
                'feature_windows': self.feature_windows
            }
        }
        joblib.dump(model_data, path)
        
    def load_models(self, path: str):
        """Load trained models and scalers."""
        model_data = joblib.load(path)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.lookback_window = model_data['params']['lookback_window']
        self.forecast_horizon = model_data['params']['forecast_horizon']
        self.feature_windows = model_data['params']['feature_windows']