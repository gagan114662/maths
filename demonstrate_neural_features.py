#!/usr/bin/env python3
"""
Neural Feature Discovery Demonstration

This script demonstrates the usage of the neural feature discovery module
on financial data to extract meaningful features for trading strategies.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import yfinance as yf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/neural_features_demo.log')
    ]
)
logger = logging.getLogger(__name__)

# Import the neural feature discovery module
from src.neural_feature_discovery import NeuralFeatureDiscovery


def download_data(tickers, start_date, end_date):
    """
    Download historical market data for the specified tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for data download
        end_date: End date for data download
        
    Returns:
        DataFrame with market data
    """
    logger.info(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    # Download data
    data_dict = {}
    for ticker in tickers:
        logger.info(f"Downloading data for {ticker}")
        try:
            ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not ticker_data.empty:
                data_dict[ticker] = ticker_data
            else:
                logger.warning(f"No data available for {ticker}")
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {str(e)}")
    
    return data_dict


def prepare_features(data_dict):
    """
    Prepare features from market data.
    
    Args:
        data_dict: Dictionary of DataFrames with market data
        
    Returns:
        DataFrame with features
    """
    logger.info("Preparing features from market data")
    
    # Create an empty list to store DataFrames for each ticker
    dfs = []
    
    for ticker, data in data_dict.items():
        # Create a copy of the data
        df = data.copy()
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_10d'] = df['Close'].pct_change(10)
        df['returns_20d'] = df['Close'].pct_change(20)
        
        # Calculate volatility
        df['volatility_10d'] = df['returns'].rolling(10).std()
        df['volatility_20d'] = df['returns'].rolling(20).std()
        
        # Calculate moving averages
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['sma_200'] = df['Close'].rolling(200).mean()
        
        # Calculate moving average ratios
        df['sma_ratio_5_20'] = df['sma_5'] / df['sma_20']
        df['sma_ratio_10_50'] = df['sma_10'] / df['sma_50']
        df['sma_ratio_50_200'] = df['sma_50'] / df['sma_200']
        
        # Calculate momentum
        df['momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Calculate volume features
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
        
        # Calculate RSI (14-day)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        df['macd'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Add ticker as a column
        df['ticker'] = ticker
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select features for analysis
        features = ['ticker', 'returns', 'returns_5d', 'returns_10d', 'returns_20d',
                    'volatility_10d', 'volatility_20d', 'sma_ratio_5_20', 'sma_ratio_10_50',
                    'sma_ratio_50_200', 'momentum_5d', 'momentum_10d', 'momentum_20d',
                    'volume_ratio', 'volume_trend', 'rsi_14', 'macd', 'macd_signal', 'macd_hist']
        
        # Append to list of DataFrames
        dfs.append(df[features])
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    logger.info(f"Prepared {len(combined_df)} samples with {len(features)} features")
    
    return combined_df


def prepare_target(data_dict, forecast_horizon=5):
    """
    Prepare target variable (future returns) for each ticker.
    
    Args:
        data_dict: Dictionary of DataFrames with market data
        forecast_horizon: Number of days to forecast ahead
        
    Returns:
        DataFrame with target variable
    """
    logger.info(f"Preparing target variable with forecast horizon of {forecast_horizon} days")
    
    # Create an empty list to store DataFrames for each ticker
    dfs = []
    
    for ticker, data in data_dict.items():
        # Create a copy of the data
        df = data.copy()
        
        # Calculate future returns
        df[f'future_returns_{forecast_horizon}d'] = df['Close'].pct_change(forecast_horizon).shift(-forecast_horizon)
        
        # Add ticker as a column
        df['ticker'] = ticker
        
        # Select only the necessary columns
        target_df = df[['ticker', f'future_returns_{forecast_horizon}d']]
        
        # Drop rows with NaN values
        target_df = target_df.dropna()
        
        # Append to list of DataFrames
        dfs.append(target_df)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    logger.info(f"Prepared {len(combined_df)} samples with target variable")
    
    return combined_df


def run_demo():
    """
    Run the neural feature discovery demonstration.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Neural Feature Discovery Demonstration")
    parser.add_argument("--tickers", default="SPY,QQQ,IWM,DIA,XLF,XLK,XLE,XLV,XLI,XLP", help="Comma-separated list of ticker symbols")
    parser.add_argument("--start-date", default="2015-01-01", help="Start date for data download (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="End date for data download (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in trading days")
    parser.add_argument("--window", type=int, default=60, help="Window size for temporal features")
    parser.add_argument("--features", type=int, default=10, help="Number of features to discover")
    parser.add_argument("--extractor", default="temporal", choices=["autoencoder", "temporal", "attention"], help="Feature extractor type")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--output", default="output/neural_features", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Set end date if not provided
    if args.end_date is None:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Parse ticker symbols
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Download data
        data_dict = download_data(tickers, args.start_date, args.end_date)
        
        if not data_dict:
            logger.error("No data downloaded. Exiting.")
            return 1
        
        # Prepare features and target
        features_df = prepare_features(data_dict)
        target_df = prepare_target(data_dict, args.horizon)
        
        # Merge features and target
        merged_df = pd.merge(features_df, target_df, on='ticker')
        
        # Save processed data
        merged_df.to_csv(os.path.join(args.output, "processed_data.csv"), index=False)
        logger.info(f"Saved processed data to {os.path.join(args.output, 'processed_data.csv')}")
        
        # Convert ticker to one-hot encoding
        ticker_dummies = pd.get_dummies(merged_df['ticker'], prefix='ticker')
        merged_df = pd.concat([merged_df.drop('ticker', axis=1), ticker_dummies], axis=1)
        
        # Prepare data for feature discovery
        X = merged_df.drop(f'future_returns_{args.horizon}d', axis=1)
        y = merged_df[f'future_returns_{args.horizon}d']
        
        # Initialize feature discovery
        feature_discovery = NeuralFeatureDiscovery(
            extractor_type=args.extractor,
            window_size=args.window,
            latent_dim=args.features,
            batch_size=64,
            learning_rate=0.001,
            num_epochs=args.epochs
        )
        
        # Run feature discovery
        logger.info(f"Running feature discovery with {args.extractor} extractor")
        results = feature_discovery.fit(X, y)
        
        # Save feature discovery system
        feature_discovery.save(os.path.join(args.output, "feature_discovery"))
        logger.info(f"Saved feature discovery system to {os.path.join(args.output, 'feature_discovery')}")
        
        # Get top features
        top_features = feature_discovery.get_top_features(10)
        logger.info("Top 10 discovered features:")
        for _, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('Top Discovered Features')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "top_features.png"))
        logger.info(f"Saved top features plot to {os.path.join(args.output, 'top_features.png')}")
        
        # Plot training history
        plt.figure(figsize=(12, 6))
        plt.plot(results['history']['train_loss'], label='Train Loss')
        plt.plot(results['history']['val_loss'], label='Validation Loss')
        plt.title('Feature Extractor Training')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "training_history.png"))
        logger.info(f"Saved training history plot to {os.path.join(args.output, 'training_history.png')}")
        
        # Evaluate feature importance
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Get original and discovered features
        X_original = X
        X_discovered = feature_discovery.discovered_features
        
        # Split data into train and test sets
        X_orig_train, X_orig_test, y_train, y_test = train_test_split(X_original, y, test_size=0.2, random_state=42)
        X_disc_train, X_disc_test = train_test_split(X_discovered, test_size=0.2, random_state=42)
        
        # Train models on original and discovered features
        model_orig = RandomForestRegressor(n_estimators=100, random_state=42)
        model_orig.fit(X_orig_train, y_train)
        
        model_disc = RandomForestRegressor(n_estimators=100, random_state=42)
        model_disc.fit(X_disc_train, y_train)
        
        # Evaluate models
        y_pred_orig = model_orig.predict(X_orig_test)
        y_pred_disc = model_disc.predict(X_disc_test)
        
        rmse_orig = np.sqrt(mean_squared_error(y_test, y_pred_orig))
        rmse_disc = np.sqrt(mean_squared_error(y_test, y_pred_disc))
        
        r2_orig = r2_score(y_test, y_pred_orig)
        r2_disc = r2_score(y_test, y_pred_disc)
        
        logger.info(f"Original features RMSE: {rmse_orig:.6f}, R²: {r2_orig:.6f}")
        logger.info(f"Discovered features RMSE: {rmse_disc:.6f}, R²: {r2_disc:.6f}")
        
        # Plot feature clusters
        plt.figure(figsize=(12, 10))
        feature_discovery.plot_feature_clusters(n_clusters=4)
        plt.savefig(os.path.join(args.output, "feature_clusters.png"))
        logger.info(f"Saved feature clusters plot to {os.path.join(args.output, 'feature_clusters.png')}")
        
        # Write summary report
        with open(os.path.join(args.output, "summary_report.txt"), "w") as f:
            f.write("NEURAL FEATURE DISCOVERY DEMONSTRATION\n")
            f.write("======================================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  Tickers: {', '.join(tickers)}\n")
            f.write(f"  Date range: {args.start_date} to {args.end_date}\n")
            f.write(f"  Forecast horizon: {args.horizon} days\n")
            f.write(f"  Feature extractor: {args.extractor}\n")
            f.write(f"  Window size: {args.window}\n")
            f.write(f"  Number of features: {args.features}\n")
            f.write(f"  Training epochs: {args.epochs}\n\n")
            
            f.write("Results:\n")
            f.write(f"  Original features RMSE: {rmse_orig:.6f}, R²: {r2_orig:.6f}\n")
            f.write(f"  Discovered features RMSE: {rmse_disc:.6f}, R²: {r2_disc:.6f}\n\n")
            
            f.write("Top 10 discovered features:\n")
            for _, row in top_features.iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
                
            f.write("\nNEURAL FEATURE DISCOVERY MODULE IMPLEMENTATION COMPLETED SUCCESSFULLY\n")
        
        logger.info(f"Saved summary report to {os.path.join(args.output, 'summary_report.txt')}")
        
        print("\nNEURAL FEATURE DISCOVERY MODULE IMPLEMENTATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Discovered {args.features} features from the data")
        print(f"Results saved to {args.output}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in neural feature discovery demo: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(run_demo())