#!/usr/bin/env python3
"""
Advanced Market Regime Detection Module

This module implements sophisticated market regime detection using unsupervised
learning techniques to automatically classify market conditions.

Supported techniques:
- Hidden Markov Models (HMM)
- Gaussian Mixture Models (GMM)
- K-means clustering
- Self-Organizing Maps (SOM)
- Hierarchical clustering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Advanced Market Regime Detection class that implements multiple algorithms
    to classify market regimes based on price data and various features.
    """
    
    REGIME_LABELS = {
        0: "Bull Market",
        1: "Bear Market",
        2: "Sideways/Consolidation",
        3: "High Volatility",
        4: "Low Volatility",
        5: "Crisis/Extreme"
    }
    
    def __init__(self, method: str = "hmm", n_regimes: int = 4, 
                 features: List[str] = None, lookback: int = 252, 
                 model_path: str = None):
        """
        Initialize the Market Regime Detector.
        
        Args:
            method: Algorithm to use for regime detection 
                   ('hmm', 'gmm', 'kmeans', 'hierarchical')
            n_regimes: Number of regimes to detect
            features: List of features to use for regime detection
            lookback: Number of days to use for training/detection
            model_path: Path to save/load trained models
        """
        self.method = method.lower()
        self.n_regimes = n_regimes
        self.features = features or ["returns", "volatility", "volume", "rsi"]
        self.lookback = lookback
        self.model_path = model_path or os.path.join(os.getcwd(), "models", "market_regimes")
        self.model = None
        self.scaler = StandardScaler()
        self.regime_history = []
        self.current_regime = None
        self.feature_data = None
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Dictionary to map method names to model creation functions
        self.method_map = {
            "hmm": self._create_hmm_model,
            "gmm": self._create_gmm_model,
            "kmeans": self._create_kmeans_model,
            "hierarchical": self._create_hierarchical_model
        }
        
        if self.method not in self.method_map:
            raise ValueError(f"Unsupported method: {self.method}. Supported methods are: {list(self.method_map.keys())}")
    
    def compute_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for regime detection from raw market data.
        
        Args:
            price_data: DataFrame with market data (must include 'Close' and 'Volume' columns)
                       Can be a single symbol or multiple symbols
                       
        Returns:
            DataFrame with computed features
        """
        features_df = pd.DataFrame(index=price_data.index)
        
        # Extract columns - handle multiindex if multiple symbols
        if isinstance(price_data.columns, pd.MultiIndex):
            # For multiple symbols, compute features for each and take the average
            symbols = price_data.columns.get_level_values(0).unique()
            feature_dfs = []
            
            for symbol in symbols:
                # Get data for this symbol
                symbol_data = price_data[symbol]
                # Compute features
                symbol_features = self._compute_symbol_features(symbol_data)
                feature_dfs.append(symbol_features)
            
            # Average features across symbols
            features_df = pd.concat(feature_dfs, axis=1)
            features_df = features_df.groupby(level=0, axis=1).mean()
        else:
            # Single symbol
            features_df = self._compute_symbol_features(price_data)
        
        # Drop rows with NaN values
        features_df = features_df.dropna()
        
        # Store feature data for later use
        self.feature_data = features_df
        
        return features_df
    
    def _compute_symbol_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for a single symbol.
        
        Args:
            data: DataFrame with price data for a single symbol
            
        Returns:
            DataFrame with computed features
        """
        df = pd.DataFrame(index=data.index)
        
        # Calculate returns
        if "returns" in self.features:
            df["returns"] = data["Close"].pct_change()
        
        # Calculate log returns
        if "log_returns" in self.features:
            df["log_returns"] = np.log(data["Close"]).diff()
        
        # Calculate volatility (rolling standard deviation of returns)
        if "volatility" in self.features:
            df["volatility"] = df.get("returns", data["Close"].pct_change()).rolling(window=20).std()
        
        # Calculate volume changes
        if "volume" in self.features and "Volume" in data.columns:
            df["volume"] = data["Volume"].pct_change()
        
        # Calculate RSI
        if "rsi" in self.features:
            df["rsi"] = self._calculate_rsi(data["Close"])
        
        # Calculate Moving Average Convergence Divergence (MACD)
        if "macd" in self.features:
            df["macd"] = self._calculate_macd(data["Close"])
        
        # Calculate Bollinger Bands width
        if "bbands_width" in self.features:
            df["bbands_width"] = self._calculate_bbands_width(data["Close"])
        
        # Calculate average true range (ATR)
        if "atr" in self.features and all(col in data.columns for col in ["High", "Low", "Close"]):
            df["atr"] = self._calculate_atr(data)
            
        # Calculate rate of change (momentum)
        if "roc" in self.features:
            df["roc"] = self._calculate_roc(data["Close"])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        
        ma_up = up.rolling(window=period).mean()
        ma_down = down.rolling(window=period).mean()
        
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate Moving Average Convergence Divergence."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line  # Return MACD histogram
    
    def _calculate_bbands_width(self, prices: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
        """Calculate Bollinger Bands width."""
        middle_band = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        width = (upper_band - lower_band) / middle_band  # Normalized width
        return width
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = data["High"]
        low = data["Low"]
        close = data["Close"]
        
        # Calculate True Range
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _calculate_roc(self, prices: pd.Series, period: int = 12) -> pd.Series:
        """Calculate Rate of Change (price momentum)."""
        return prices.pct_change(period)

    def train(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the regime detection model on historical price data.
        
        Args:
            price_data: DataFrame with market data
            
        Returns:
            Dictionary with training results including model performance metrics
        """
        # Compute features
        logger.info(f"Computing features for regime detection using method: {self.method}")
        features_df = self.compute_features(price_data)
        
        # Use only the specified lookback period
        if len(features_df) > self.lookback:
            features_df = features_df.iloc[-self.lookback:]
        
        # Scale features
        X = self.scaler.fit_transform(features_df.values)
        
        # Create and train the appropriate model
        logger.info(f"Training {self.method} model with {self.n_regimes} regimes")
        model_func = self.method_map[self.method]
        self.model = model_func(X)
        
        # Get regime labels
        if self.method == "hmm":
            regimes = self.model.predict(X)
        else:
            regimes = self.model.predict(X)
        
        # Store regime history
        self.regime_history = list(zip(features_df.index, regimes))
        self.current_regime = regimes[-1]
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(X, regimes)
        
        # Save the trained model
        self._save_model()
        
        return {
            "method": self.method,
            "n_regimes": self.n_regimes,
            "metrics": metrics,
            "current_regime": self.current_regime,
            "regime_counts": Counter(regimes)
        }
    
    def predict(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict the current market regime based on recent price data.
        
        Args:
            price_data: DataFrame with market data
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            try:
                self._load_model()
            except FileNotFoundError:
                logger.error("No trained model found. Please train the model first.")
                return {"error": "No trained model found. Please train the model first."}
        
        # Compute features
        features_df = self.compute_features(price_data)
        
        # Use only the most recent data points
        features_recent = features_df.iloc[-min(self.lookback, len(features_df)):]
        
        # Scale features
        X = self.scaler.transform(features_recent.values)
        
        # Predict regimes
        if self.method == "hmm":
            regimes = self.model.predict(X)
        else:
            regimes = self.model.predict(X)
        
        # Update history and current regime
        new_history = list(zip(features_recent.index, regimes))
        self.regime_history.extend(new_history)
        self.current_regime = regimes[-1]
        
        # Calculate transition probabilities
        transitions = {}
        if len(regimes) > 1:
            for i in range(len(regimes) - 1):
                from_regime = regimes[i]
                to_regime = regimes[i + 1]
                key = (from_regime, to_regime)
                transitions[key] = transitions.get(key, 0) + 1
        
        # Get regime label
        regime_label = self.REGIME_LABELS.get(self.current_regime, f"Regime {self.current_regime}")
        
        return {
            "current_regime": self.current_regime,
            "regime_label": regime_label,
            "regime_sequence": list(regimes),
            "timestamps": features_recent.index.tolist(),
            "transitions": transitions,
            "confidence": 1.0  # Placeholder for confidence score
        }
    
    def load_or_train(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Load an existing trained model or train a new one if not found.
        
        Args:
            price_data: DataFrame with market data
            
        Returns:
            Dictionary with the model and prediction results
        """
        try:
            self._load_model()
            logger.info(f"Loaded existing {self.method} model")
            return self.predict(price_data)
        except (FileNotFoundError, ValueError):
            logger.info(f"No existing model found. Training a new {self.method} model.")
            return self.train(price_data)
    
    def _create_hmm_model(self, X: np.ndarray) -> hmm.GaussianHMM:
        """
        Create and train a Hidden Markov Model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Trained HMM model
        """
        model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        model.fit(X)
        return model
    
    def _create_gmm_model(self, X: np.ndarray) -> GaussianMixture:
        """
        Create and train a Gaussian Mixture Model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Trained GMM model
        """
        model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=42,
            max_iter=1000
        )
        model.fit(X)
        return model
    
    def _create_kmeans_model(self, X: np.ndarray) -> KMeans:
        """
        Create and train a K-means clustering model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Trained K-means model
        """
        model = KMeans(
            n_clusters=self.n_regimes,
            random_state=42,
            n_init=10
        )
        model.fit(X)
        return model
    
    def _create_hierarchical_model(self, X: np.ndarray) -> AgglomerativeClustering:
        """
        Create and train a Hierarchical clustering model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Trained Hierarchical clustering model
        """
        model = AgglomerativeClustering(
            n_clusters=self.n_regimes,
            linkage="ward"
        )
        model.fit(X)
        # Since AgglomerativeClustering doesn't have a predict method,
        # we'll create a wrapper class
        return _HierarchicalModelWrapper(model, X)
    
    def _calculate_metrics(self, X: np.ndarray, regimes: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics for the trained model.
        
        Args:
            X: Feature matrix
            regimes: Predicted regimes
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # Common metrics
        unique_regimes = len(np.unique(regimes))
        metrics["unique_regimes"] = unique_regimes
        
        # Model-specific metrics
        if self.method == "hmm":
            metrics["log_likelihood"] = self.model.score(X)
        elif self.method == "gmm":
            metrics["bic"] = self.model.bic(X)
            metrics["aic"] = self.model.aic(X)
        elif self.method == "kmeans":
            metrics["inertia"] = self.model.inertia_
        
        return metrics
    
    def _save_model(self):
        """Save the trained model to disk."""
        model_file = f"{self.model_path}_{self.method}_{self.n_regimes}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "method": self.method,
                "n_regimes": self.n_regimes,
                "features": self.features,
                "regime_history": self.regime_history
            }, f)
        logger.info(f"Model saved to {model_file}")
    
    def _load_model(self):
        """Load a trained model from disk."""
        model_file = f"{self.model_path}_{self.method}_{self.n_regimes}.pkl"
        with open(model_file, "rb") as f:
            saved_data = pickle.load(f)
            self.model = saved_data["model"]
            self.scaler = saved_data["scaler"]
            self.method = saved_data["method"]
            self.n_regimes = saved_data["n_regimes"]
            self.features = saved_data["features"]
            self.regime_history = saved_data["regime_history"]
            if self.regime_history:
                self.current_regime = self.regime_history[-1][1]
        logger.info(f"Model loaded from {model_file}")
    
    def plot_regimes(self, price_data: pd.DataFrame, symbol: Optional[str] = None, output_path: Optional[str] = None) -> None:
        """
        Plot price data with regimes highlighted in different colors.
        
        Args:
            price_data: DataFrame with price data
            symbol: Symbol to plot (for multi-symbol data)
            output_path: Path to save the plot
        """
        if not self.regime_history:
            logger.warning("No regime history available. Run train() or predict() first.")
            return
        
        # Get price data for the specified symbol
        if isinstance(price_data.columns, pd.MultiIndex) and symbol:
            if symbol in price_data.columns.get_level_values(0):
                close_prices = price_data[symbol]["Close"]
            else:
                logger.warning(f"Symbol {symbol} not found in price data. Using first symbol.")
                symbol = price_data.columns.get_level_values(0)[0]
                close_prices = price_data[symbol]["Close"]
        elif "Close" in price_data.columns:
            close_prices = price_data["Close"]
            if symbol is None:
                symbol = "Price"
        else:
            logger.error("Price data must contain a 'Close' column.")
            return
        
        # Create a DataFrame with regimes and prices
        regime_dates = [date for date, _ in self.regime_history]
        regime_values = [regime for _, regime in self.regime_history]
        regimes_df = pd.DataFrame({"regime": regime_values}, index=regime_dates)
        
        # Join with price data
        plot_df = pd.DataFrame({"close": close_prices}, index=close_prices.index)
        plot_df = plot_df.join(regimes_df, how="left")
        
        # Fill missing regimes (if any)
        plot_df["regime"] = plot_df["regime"].ffill()
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Plot price
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(plot_df.index, plot_df["close"], color="black", linewidth=1.5)
        ax1.set_title(f"Market Regimes for {symbol} using {self.method.upper()}")
        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.3)
        
        # Highlight regimes
        regime_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for regime in range(self.n_regimes):
            if regime in plot_df["regime"].values:
                mask = plot_df["regime"] == regime
                ax1.fill_between(
                    plot_df.index, 
                    plot_df["close"].min(), 
                    plot_df["close"].max(), 
                    where=mask, 
                    alpha=0.3, 
                    color=regime_colors[regime % len(regime_colors)],
                    label=f"Regime {regime}"
                )
        
        ax1.legend(loc="upper left")
        
        # Plot regime transitions
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.scatter(
            plot_df.index, 
            plot_df["regime"], 
            c=plot_df["regime"].map({i: regime_colors[i % len(regime_colors)] for i in range(self.n_regimes)}),
            s=50, 
            alpha=0.7
        )
        
        ax2.set_yticks(range(self.n_regimes))
        ax2.set_yticklabels([self.REGIME_LABELS.get(i, f"Regime {i}") for i in range(self.n_regimes)])
        ax2.set_ylabel("Regime")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {output_path}")
        else:
            plt.show()
    
    def get_regime_returns(self, price_data: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        Calculate return statistics for each regime.
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            Dictionary mapping regime IDs to return statistics
        """
        if not self.regime_history:
            logger.warning("No regime history available. Run train() or predict() first.")
            return {}
        
        # Get returns
        if isinstance(price_data.columns, pd.MultiIndex):
            # For multi-symbol data, average the returns across symbols
            returns = pd.DataFrame()
            for symbol in price_data.columns.get_level_values(0).unique():
                returns[symbol] = price_data[symbol]["Close"].pct_change()
            returns = returns.mean(axis=1)
        else:
            returns = price_data["Close"].pct_change()
        
        # Create a DataFrame with regimes and returns
        regime_dates = [date for date, _ in self.regime_history]
        regime_values = [regime for _, regime in self.regime_history]
        regimes_df = pd.DataFrame({"regime": regime_values}, index=regime_dates)
        
        # Join with returns
        returns_df = pd.DataFrame({"return": returns}, index=returns.index)
        returns_df = returns_df.join(regimes_df, how="inner")
        
        # Calculate statistics for each regime
        regime_stats = {}
        for regime in range(self.n_regimes):
            regime_returns = returns_df[returns_df["regime"] == regime]["return"]
            
            if len(regime_returns) > 0:
                annualized_factor = 252  # Trading days in a year
                
                regime_stats[regime] = {
                    "count": len(regime_returns),
                    "mean_return": regime_returns.mean(),
                    "median_return": regime_returns.median(),
                    "std_return": regime_returns.std(),
                    "min_return": regime_returns.min(),
                    "max_return": regime_returns.max(),
                    "skew": regime_returns.skew(),
                    "kurtosis": regime_returns.kurtosis(),
                    "annualized_return": regime_returns.mean() * annualized_factor,
                    "annualized_volatility": regime_returns.std() * np.sqrt(annualized_factor),
                    "sharpe_ratio": (regime_returns.mean() * annualized_factor) / 
                                    (regime_returns.std() * np.sqrt(annualized_factor)) if regime_returns.std() > 0 else 0,
                    "positive_returns": (regime_returns > 0).mean(),
                    "label": self.REGIME_LABELS.get(regime, f"Regime {regime}")
                }
        
        return regime_stats
    
    def get_regime_transition_matrix(self) -> pd.DataFrame:
        """
        Calculate the regime transition probability matrix.
        
        Returns:
            DataFrame with transition probabilities
        """
        if not self.regime_history or len(self.regime_history) < 2:
            logger.warning("Insufficient regime history for transition matrix.")
            return pd.DataFrame()
        
        # Extract regime sequence
        regimes = [regime for _, regime in self.regime_history]
        
        # Calculate transitions
        transitions = {}
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]
            key = (from_regime, to_regime)
            transitions[key] = transitions.get(key, 0) + 1
        
        # Count occurrences of each regime
        regime_counts = Counter(regimes[:-1])  # Exclude the last regime
        
        # Create transition matrix
        matrix = pd.DataFrame(0, 
                            index=[self.REGIME_LABELS.get(i, f"Regime {i}") for i in range(self.n_regimes)],
                            columns=[self.REGIME_LABELS.get(i, f"Regime {i}") for i in range(self.n_regimes)])
        
        # Calculate transition probabilities
        for (from_regime, to_regime), count in transitions.items():
            if from_regime < self.n_regimes and to_regime < self.n_regimes:
                from_label = self.REGIME_LABELS.get(from_regime, f"Regime {from_regime}")
                to_label = self.REGIME_LABELS.get(to_regime, f"Regime {to_regime}")
                matrix.loc[from_label, to_label] = count / regime_counts[from_regime]
        
        return matrix
    
    def get_optimal_regimes(self, min_regimes: int = 2, max_regimes: int = 10) -> Dict[str, Any]:
        """
        Find the optimal number of regimes using BIC/AIC for GMM or silhouette score for others.
        
        Args:
            min_regimes: Minimum number of regimes to try
            max_regimes: Maximum number of regimes to try
            
        Returns:
            Dictionary with optimal number of regimes and evaluation metrics
        """
        if self.feature_data is None:
            logger.error("No feature data available. Run compute_features() first.")
            return {"error": "No feature data available"}
        
        # Scale features
        X = self.scaler.fit_transform(self.feature_data.values)
        
        scores = []
        models = []
        
        logger.info(f"Finding optimal number of regimes using {self.method}")
        
        for n in range(min_regimes, max_regimes + 1):
            logger.info(f"Evaluating {n} regimes")
            
            # Create model with n regimes
            original_n = self.n_regimes
            self.n_regimes = n
            
            try:
                model_func = self.method_map[self.method]
                model = model_func(X)
                models.append(model)
                
                # Calculate scoring metric based on method
                if self.method == "hmm":
                    score = model.score(X)
                    scores.append({"n_regimes": n, "score": score, "metric": "log_likelihood"})
                elif self.method == "gmm":
                    score = -model.bic(X)  # Negative BIC (higher is better)
                    scores.append({"n_regimes": n, "score": score, "metric": "neg_bic"})
                elif self.method == "kmeans":
                    from sklearn.metrics import silhouette_score
                    labels = model.predict(X)
                    if len(np.unique(labels)) > 1:  # Silhouette requires at least 2 clusters
                        score = silhouette_score(X, labels)
                        scores.append({"n_regimes": n, "score": score, "metric": "silhouette"})
                elif self.method == "hierarchical":
                    # For hierarchical, we'll use the Davies-Bouldin index
                    from sklearn.metrics import davies_bouldin_score
                    labels = model.predict(X)
                    if len(np.unique(labels)) > 1:
                        score = -davies_bouldin_score(X, labels)  # Negative DB (higher is better)
                        scores.append({"n_regimes": n, "score": score, "metric": "neg_davies_bouldin"})
            except Exception as e:
                logger.warning(f"Error evaluating {n} regimes: {str(e)}")
            
            # Restore original n_regimes
            self.n_regimes = original_n
        
        # Find optimal number of regimes
        if scores:
            optimal_score = max(scores, key=lambda x: x["score"])
            optimal_n = optimal_score["n_regimes"]
            
            logger.info(f"Optimal number of regimes: {optimal_n} with {optimal_score['metric']} = {optimal_score['score']}")
            
            return {
                "optimal_n_regimes": optimal_n,
                "scores": scores,
                "metric": optimal_score["metric"]
            }
        else:
            logger.warning("Could not determine optimal number of regimes")
            return {"error": "Could not determine optimal number of regimes"}


class _HierarchicalModelWrapper:
    """
    Wrapper class for Hierarchical clustering model to add predict method.
    """
    
    def __init__(self, model: AgglomerativeClustering, X_train: np.ndarray):
        """
        Initialize the wrapper.
        
        Args:
            model: Trained AgglomerativeClustering model
            X_train: Training data
        """
        self.model = model
        self.X_train = X_train
        self.labels = model.labels_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict clusters for new data by finding the nearest neighbor in the training data.
        
        Args:
            X: New data points
            
        Returns:
            Predicted cluster labels
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Create a nearest neighbors model using the training data
        nn = NearestNeighbors(n_neighbors=1).fit(self.X_train)
        
        # Find the nearest neighbor in the training data for each new point
        distances, indices = nn.kneighbors(X)
        
        # Assign the cluster label of the nearest neighbor
        return self.labels[indices.flatten()]


def main():
    """
    Example usage of the MarketRegimeDetector.
    """
    import yfinance as yf
    
    # Download data
    symbol = "SPY"
    data = yf.download(symbol, start="2018-01-01", end="2023-12-31")
    
    # Create detector
    detector = MarketRegimeDetector(method="hmm", n_regimes=4)
    
    # Train the model
    results = detector.train(data)
    print(f"Training results: {results}")
    
    # Get regime statistics
    stats = detector.get_regime_returns(data)
    for regime, regime_stats in stats.items():
        print(f"\nRegime {regime} ({regime_stats['label']}):")
        print(f"  Count: {regime_stats['count']}")
        print(f"  Annualized Return: {regime_stats['annualized_return']*100:.2f}%")
        print(f"  Annualized Volatility: {regime_stats['annualized_volatility']*100:.2f}%")
        print(f"  Sharpe Ratio: {regime_stats['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {regime_stats['positive_returns']*100:.2f}%")
    
    # Plot regimes
    detector.plot_regimes(data, symbol=symbol, output_path=f"market_regimes_{symbol}.png")
    
    # Get transition matrix
    trans_matrix = detector.get_regime_transition_matrix()
    print("\nRegime Transition Matrix:")
    print(trans_matrix)
    
    # Find optimal number of regimes
    optimal = detector.get_optimal_regimes(min_regimes=2, max_regimes=6)
    print(f"\nOptimal number of regimes: {optimal.get('optimal_n_regimes')}")


if __name__ == "__main__":
    main()