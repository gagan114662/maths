#!/usr/bin/env python
"""
Demonstration script for alternative data integration.

This script demonstrates how to use the AlternativeDataProcessor and
how to integrate it with the MarketDataProcessor to enhance trading signals.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Import processors
from src.data_processors.market_data_processor import MarketDataProcessor
from src.data_processors.alternative_data_processor import AlternativeDataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/alternative_data_demo.log')
    ]
)
logger = logging.getLogger(__name__)

def visualize_alternative_data_impact(df, symbol):
    """Visualize the impact of alternative data on market data."""
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Price and sentiment overlay
    plt.subplot(3, 1, 1)
    plt.title(f"{symbol} Price and News Sentiment")
    plt.plot(df.index, df['close'], 'b-', label='Close Price')
    plt.grid(True, alpha=0.3)
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    
    # Add sentiment as secondary axis
    ax2 = plt.twinx()
    if 'news_sentiment' in df.columns:
        ax2.plot(df.index, df['news_sentiment'], 'g-', label='News Sentiment')
        ax2.set_ylabel('Sentiment Score')
        ax2.legend(loc='upper right')
    
    # Plot 2: Social media activity
    plt.subplot(3, 1, 2)
    plt.title(f"{symbol} Social Media Activity")
    
    # Plot available social media metrics
    social_cols = [col for col in df.columns if 'social_' in col and 'volume' in col or 'count' in col]
    for col in social_cols[:3]:  # Limit to first 3 for clarity
        plt.plot(df.index, df[col], label=col.replace('social_', '').replace('_', ' ').title())
    
    plt.grid(True, alpha=0.3)
    plt.ylabel('Activity Volume')
    plt.legend()
    
    # Plot 3: Event markers
    plt.subplot(3, 1, 3)
    plt.title(f"{symbol} Corporate Events and Returns")
    plt.plot(df.index, df['returns'].rolling(5).mean(), 'b-', label='5-Day Rolling Returns')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.grid(True, alpha=0.3)
    plt.ylabel('Returns')
    
    # Mark corporate events if available
    event_cols = [col for col in df.columns if 'event_' in col]
    for col in event_cols:
        event_dates = df[df[col] > 0].index
        plt.scatter(event_dates, np.zeros_like(event_dates, dtype=float), 
                   marker='^', s=100, label=col.replace('event_', '').title())
    
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"visualizations/{symbol}_alternative_data.png")
    plt.close()
    
    logger.info(f"Saved visualization for {symbol}")

def visualize_feature_importance(importance_dict, symbol, top_n=20):
    """Visualize feature importance for a symbol."""
    if not importance_dict or symbol not in importance_dict:
        logger.warning(f"No importance data available for {symbol}")
        return
    
    importances = importance_dict[symbol]
    if not importances:
        logger.warning(f"Empty importance data for {symbol}")
        return
    
    # Get top N features
    sorted_features = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True)[:top_n])
    
    plt.figure(figsize=(12, 8))
    plt.barh(list(sorted_features.keys()), list(sorted_features.values()))
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importance for {symbol}')
    plt.gca().invert_yaxis()  # Display with highest importance at the top
    plt.tight_layout()
    plt.savefig(f"visualizations/{symbol}_feature_importance.png")
    plt.close()
    
    logger.info(f"Saved feature importance visualization for {symbol}")

def run_demo():
    """Run the alternative data integration demonstration."""
    logger.info("Starting alternative data integration demonstration")
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Initialize the market data processor
    processor = MarketDataProcessor()
    
    # Define symbols to analyze
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Get market data with alternative data integration
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    logger.info(f"Getting market data with alternative data for {len(symbols)} symbols")
    data_dict = processor.get_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        clean=True,
        include_alternative_data=True
    )
    
    # Analyze and visualize the data
    for symbol in symbols:
        if symbol in data_dict and not data_dict[symbol].empty:
            logger.info(f"Processing {symbol}")
            df = data_dict[symbol]
            
            # Print data statistics
            alt_data_cols = [col for col in df.columns if any(x in col for x in ['news_', 'social_', 'event_', 'macro_'])]
            logger.info(f"{symbol} has {len(alt_data_cols)} alternative data columns")
            
            # Save to CSV for inspection
            df.to_csv(f"visualizations/{symbol}_with_alt_data.csv")
            
            # Visualize
            visualize_alternative_data_impact(df, symbol)
    
    # Analyze feature importance
    logger.info("Analyzing feature importance")
    importance_dict = processor.analyze_alternative_data_importance(symbols)
    
    # Visualize feature importance
    for symbol in symbols:
        visualize_feature_importance(importance_dict, symbol)
    
    logger.info("Alternative data demonstration completed")

if __name__ == "__main__":
    run_demo()