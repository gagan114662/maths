"""
Alternative Data Processor for enhanced trading signals.

This module provides functionality to integrate and process alternative data sources
such as news sentiment, social media, satellite imagery, and other non-traditional
data sources to generate enhanced trading signals.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Any, Optional, Union, Tuple

# Setup logging
logger = logging.getLogger(__name__)

class AlternativeDataProcessor:
    """
    Processor for alternative data sources to enhance trading signals.
    
    This class provides functionality to download, process, and integrate
    alternative data sources with traditional market data for enhanced
    signal generation.
    """
    
    def __init__(self, cache_dir: str = "data/alternative_data_cache", config_path: Optional[str] = None):
        """
        Initialize the AlternativeDataProcessor.
        
        Args:
            cache_dir (str): Directory to cache alternative data
            config_path (str, optional): Path to configuration file for API keys and settings
        """
        self.cache_dir = cache_dir
        self.config = self._load_config(config_path)
        self._setup_cache_dir()
        logger.info(f"AlternativeDataProcessor initialized with cache at {cache_dir}")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path (str, optional): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        default_config = {
            "news_api_key": os.environ.get("NEWS_API_KEY", ""),
            "twitter_api_key": os.environ.get("TWITTER_API_KEY", ""),
            "twitter_api_secret": os.environ.get("TWITTER_API_SECRET", ""),
            "reddit_client_id": os.environ.get("REDDIT_CLIENT_ID", ""),
            "reddit_client_secret": os.environ.get("REDDIT_CLIENT_SECRET", ""),
            "satellite_api_key": os.environ.get("SATELLITE_API_KEY", ""),
            "macro_economic_api_key": os.environ.get("MACRO_ECONOMIC_API_KEY", ""),
            "update_frequency": {
                "news": 24,  # hours
                "social_media": 12,  # hours
                "satellite": 168,  # hours (weekly)
                "macro_economic": 24,  # hours
                "corporate_events": 24,  # hours
            },
            "sentiment_analysis": {
                "model": "vader",  # Options: vader, textblob, transformers
                "transformers_model": "finbert"  # Used if model is transformers
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Update default config with user settings
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
        
        return default_config
    
    def _setup_cache_dir(self) -> None:
        """Create cache directory structure if it doesn't exist."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Create subdirectories for each data type
        data_types = ["news", "social_media", "satellite", "macro_economic", "corporate_events"]
        for data_type in data_types:
            sub_dir = os.path.join(self.cache_dir, data_type)
            os.makedirs(sub_dir, exist_ok=True)
            
        logger.info(f"Cache directory structure created at {self.cache_dir}")
    
    def get_news_sentiment(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """
        Get news sentiment data for a specific symbol.
        
        Args:
            symbol (str): Stock symbol to retrieve news for
            days (int): Number of days of historical news to retrieve
            
        Returns:
            pd.DataFrame: DataFrame with news sentiment data
        """
        cache_file = os.path.join(self.cache_dir, "news", f"{symbol}_news.csv")
        
        # Check if cached data exists and is recent enough
        if os.path.exists(cache_file):
            cached_data = pd.read_csv(cache_file)
            if not cached_data.empty:
                cached_data['date'] = pd.to_datetime(cached_data['date'])
                most_recent = cached_data['date'].max()
                if most_recent.date() >= (datetime.now() - timedelta(hours=self.config['update_frequency']['news'])).date():
                    logger.info(f"Using cached news sentiment data for {symbol}")
                    return cached_data
        
        # If no recent cache exists, fetch new data
        logger.info(f"Fetching news sentiment data for {symbol}")
        try:
            api_key = self.config.get("news_api_key")
            if not api_key:
                logger.warning("News API key not found in configuration")
                return pd.DataFrame()
                
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch news data from API
            url = f"https://newsapi.org/v2/everything"
            params = {
                "q": f"{symbol} stock",
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": api_key
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                logger.error(f"Error fetching news data: {response.status_code} - {response.text}")
                return pd.DataFrame()
                
            news_data = response.json()
            
            # Process and analyze sentiment
            sentiment_data = self._analyze_news_sentiment(news_data, symbol)
            
            # Cache the data
            sentiment_data.to_csv(cache_file, index=False)
            
            return sentiment_data
        
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return pd.DataFrame()
    
    def _analyze_news_sentiment(self, news_data: Dict[str, Any], symbol: str) -> pd.DataFrame:
        """
        Analyze sentiment for news articles.
        
        Args:
            news_data (Dict[str, Any]): News data from API
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: DataFrame with sentiment analysis
        """
        if "articles" not in news_data or not news_data["articles"]:
            return pd.DataFrame()
            
        articles = news_data["articles"]
        
        # Select sentiment model based on configuration
        sentiment_model = self.config["sentiment_analysis"]["model"]
        
        if sentiment_model == "vader":
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            try:
                sid = SentimentIntensityAnalyzer()
            except:
                # NLTK resources might need to be downloaded
                import nltk
                nltk.download('vader_lexicon', quiet=True)
                sid = SentimentIntensityAnalyzer()
                
            sentiment_data = []
            
            for article in articles:
                if not article["title"] or not article["description"]:
                    continue
                    
                # Combine title and description for analysis
                text = f"{article['title']} {article['description']}"
                
                # Get sentiment scores
                sentiment = sid.polarity_scores(text)
                
                sentiment_data.append({
                    "date": article["publishedAt"][:10],
                    "title": article["title"],
                    "source": article["source"]["name"],
                    "compound_score": sentiment["compound"],
                    "positive_score": sentiment["pos"],
                    "negative_score": sentiment["neg"],
                    "neutral_score": sentiment["neu"],
                    "url": article["url"]
                })
                
        elif sentiment_model == "textblob":
            from textblob import TextBlob
            
            sentiment_data = []
            
            for article in articles:
                if not article["title"] or not article["description"]:
                    continue
                    
                # Combine title and description for analysis
                text = f"{article['title']} {article['description']}"
                
                # Get sentiment 
                blob = TextBlob(text)
                sentiment = blob.sentiment
                
                sentiment_data.append({
                    "date": article["publishedAt"][:10],
                    "title": article["title"],
                    "source": article["source"]["name"],
                    "polarity": sentiment.polarity,
                    "subjectivity": sentiment.subjectivity,
                    "url": article["url"]
                })
                
        elif sentiment_model == "transformers":
            # Using a pre-trained transformer model like FinBERT
            # Note: This requires transformers library
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch
                
                model_name = self.config["sentiment_analysis"]["transformers_model"]
                if model_name == "finbert":
                    model_path = "ProsusAI/finbert"
                else:
                    model_path = model_name
                    
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                
                sentiment_data = []
                
                for article in articles:
                    if not article["title"] or not article["description"]:
                        continue
                        
                    # Combine title and description for analysis
                    text = f"{article['title']} {article['description']}"
                    
                    # Tokenize and get sentiment
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Get sentiment scores
                    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                    sentiment_data.append({
                        "date": article["publishedAt"][:10],
                        "title": article["title"],
                        "source": article["source"]["name"],
                        "positive_score": scores[0][0].item(),
                        "negative_score": scores[0][1].item(),
                        "neutral_score": scores[0][2].item() if scores.shape[1] > 2 else 0,
                        "url": article["url"]
                    })
            except ImportError:
                logger.error("Transformers library not available. Falling back to VADER sentiment analysis.")
                # Fallback to VADER
                return self._analyze_news_sentiment(news_data, symbol)
        else:
            logger.warning(f"Unknown sentiment model: {sentiment_model}. Falling back to VADER.")
            # Recursively call with vader model
            old_model = self.config["sentiment_analysis"]["model"]
            self.config["sentiment_analysis"]["model"] = "vader"
            result = self._analyze_news_sentiment(news_data, symbol)
            self.config["sentiment_analysis"]["model"] = old_model
            return result
            
        # Convert to DataFrame
        df = pd.DataFrame(sentiment_data)
        
        if df.empty:
            return df
            
        # Add symbol column
        df["symbol"] = symbol
        
        # Convert date strings to datetime
        df["date"] = pd.to_datetime(df["date"])
        
        # Aggregate by date
        daily_sentiment = df.groupby(["symbol", df["date"].dt.date]).agg({
            "compound_score": "mean" if "compound_score" in df.columns else None,
            "positive_score": "mean",
            "negative_score": "mean",
            "neutral_score": "mean" if "neutral_score" in df.columns else None,
            "polarity": "mean" if "polarity" in df.columns else None,
            "subjectivity": "mean" if "subjectivity" in df.columns else None,
            "title": "count"
        }).reset_index()
        
        # Rename columns
        daily_sentiment.rename(columns={"title": "article_count"}, inplace=True)
        
        return daily_sentiment
    
    def get_social_media_sentiment(self, symbol: str, days: int = 7, platform: str = "all") -> pd.DataFrame:
        """
        Get social media sentiment data for a specific symbol.
        
        Args:
            symbol (str): Stock symbol to retrieve sentiment for
            days (int): Number of days of historical data to retrieve
            platform (str): Platform to retrieve data from ("twitter", "reddit", "all")
            
        Returns:
            pd.DataFrame: DataFrame with social media sentiment data
        """
        platforms = ["twitter", "reddit"] if platform == "all" else [platform]
        combined_data = []
        
        for p in platforms:
            cache_file = os.path.join(self.cache_dir, "social_media", f"{symbol}_{p}.csv")
            
            # Check if cached data exists and is recent enough
            if os.path.exists(cache_file):
                cached_data = pd.read_csv(cache_file)
                if not cached_data.empty:
                    cached_data['date'] = pd.to_datetime(cached_data['date'])
                    most_recent = cached_data['date'].max()
                    if most_recent.date() >= (datetime.now() - timedelta(hours=self.config['update_frequency']['social_media'])).date():
                        logger.info(f"Using cached {p} sentiment data for {symbol}")
                        combined_data.append(cached_data)
                        continue
            
            # Fetch data for specific platform
            if p == "twitter":
                platform_data = self._get_twitter_sentiment(symbol, days)
            elif p == "reddit":
                platform_data = self._get_reddit_sentiment(symbol, days)
            else:
                logger.warning(f"Unsupported platform: {p}")
                continue
                
            if not platform_data.empty:
                # Add platform column and cache
                platform_data["platform"] = p
                platform_data.to_csv(cache_file, index=False)
                combined_data.append(platform_data)
        
        if not combined_data:
            return pd.DataFrame()
            
        # Combine data from all platforms
        result = pd.concat(combined_data, ignore_index=True)
        
        return result
    
    def _get_twitter_sentiment(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Get Twitter sentiment data.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days
            
        Returns:
            pd.DataFrame: Twitter sentiment data
        """
        logger.info(f"Fetching Twitter sentiment data for {symbol}")
        
        # This is a placeholder for actual Twitter API integration
        # In a real implementation, you would use the Twitter API or a service like TwitterAPI
        # For now, we'll generate some random data for demonstration
        
        # Check if API keys are available
        api_key = self.config.get("twitter_api_key")
        api_secret = self.config.get("twitter_api_secret")
        
        if not api_key or not api_secret:
            logger.warning("Twitter API credentials not found in configuration")
            # Return dummy data for demonstration
            return self._generate_dummy_social_data(symbol, days, "twitter")
        
        # TODO: Implement actual Twitter API integration
        # For now, return dummy data
        return self._generate_dummy_social_data(symbol, days, "twitter")
    
    def _get_reddit_sentiment(self, symbol: str, days: int) -> pd.DataFrame:
        """
        Get Reddit sentiment data.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days
            
        Returns:
            pd.DataFrame: Reddit sentiment data
        """
        logger.info(f"Fetching Reddit sentiment data for {symbol}")
        
        # This is a placeholder for actual Reddit API integration
        # In a real implementation, you would use the Reddit API or PRAW
        # For now, we'll generate some random data for demonstration
        
        # Check if API keys are available
        client_id = self.config.get("reddit_client_id")
        client_secret = self.config.get("reddit_client_secret")
        
        if not client_id or not client_secret:
            logger.warning("Reddit API credentials not found in configuration")
            # Return dummy data for demonstration
            return self._generate_dummy_social_data(symbol, days, "reddit")
        
        # TODO: Implement actual Reddit API integration
        # For now, return dummy data
        return self._generate_dummy_social_data(symbol, days, "reddit")
    
    def _generate_dummy_social_data(self, symbol: str, days: int, platform: str) -> pd.DataFrame:
        """
        Generate dummy social media data for demonstration purposes.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days
            platform (str): Social media platform
            
        Returns:
            pd.DataFrame: Dummy social media data
        """
        # Generate date range
        end_date = datetime.now().date()
        date_range = [end_date - timedelta(days=i) for i in range(days)]
        
        # Generate dummy sentiment scores
        np.random.seed(hash(symbol + platform) % 10000)  # Seed for reproducibility
        
        data = []
        for date in date_range:
            # Base sentiment slightly positive
            base_sentiment = 0.1
            # Add some random variation
            daily_sentiment = base_sentiment + np.random.normal(0, 0.3)
            # Clamp between -1 and 1
            daily_sentiment = max(min(daily_sentiment, 1.0), -1.0)
            
            # Generate post volume with weekly pattern (higher on weekdays)
            base_volume = 50 if date.weekday() < 5 else 25
            post_volume = int(base_volume * (1 + np.random.normal(0, 0.3)))
            
            data.append({
                "date": date,
                "symbol": symbol,
                "sentiment_score": daily_sentiment,
                "post_volume": post_volume,
                "positive_ratio": (daily_sentiment + 1) / 2,  # Convert -1,1 to 0,1
                "negative_ratio": 1 - (daily_sentiment + 1) / 2
            })
            
        return pd.DataFrame(data)
    
    def get_satellite_imagery_data(self, symbol: str, location_type: str, metric: str) -> pd.DataFrame:
        """
        Get satellite imagery-derived data.
        
        Args:
            symbol (str): Stock symbol
            location_type (str): Type of location to analyze ("retail", "manufacturing", "shipping", "oil")
            metric (str): Metric to retrieve ("foot_traffic", "car_count", "container_count", "build_progress")
            
        Returns:
            pd.DataFrame: Satellite imagery-derived data
        """
        cache_file = os.path.join(self.cache_dir, "satellite", f"{symbol}_{location_type}_{metric}.csv")
        
        # Check if cached data exists and is recent enough
        if os.path.exists(cache_file):
            cached_data = pd.read_csv(cache_file)
            if not cached_data.empty:
                cached_data['date'] = pd.to_datetime(cached_data['date'])
                most_recent = cached_data['date'].max()
                if most_recent.date() >= (datetime.now() - timedelta(hours=self.config['update_frequency']['satellite'])).date():
                    logger.info(f"Using cached satellite imagery data for {symbol} ({location_type}, {metric})")
                    return cached_data
        
        logger.info(f"Fetching satellite imagery data for {symbol} ({location_type}, {metric})")
        
        # This is a placeholder for actual satellite imagery API integration
        # In a real implementation, you would use an API like Planet, Sentinel Hub, etc.
        
        # Check if API key is available
        api_key = self.config.get("satellite_api_key")
        
        if not api_key:
            logger.warning("Satellite imagery API key not found in configuration")
            # Return dummy data for demonstration
            return self._generate_dummy_satellite_data(symbol, location_type, metric)
        
        # TODO: Implement actual satellite imagery API integration
        # For now, return dummy data
        dummy_data = self._generate_dummy_satellite_data(symbol, location_type, metric)
        
        # Cache the data
        dummy_data.to_csv(cache_file, index=False)
        
        return dummy_data
    
    def _generate_dummy_satellite_data(self, symbol: str, location_type: str, metric: str) -> pd.DataFrame:
        """
        Generate dummy satellite imagery data for demonstration purposes.
        
        Args:
            symbol (str): Stock symbol
            location_type (str): Type of location
            metric (str): Metric to generate
            
        Returns:
            pd.DataFrame: Dummy satellite data
        """
        # Generate weekly data for the last 6 months
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=180)
        
        # Generate weekly dates
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            date_range.append(current_date)
            current_date += timedelta(days=7)
        
        # Seed for reproducibility
        np.random.seed(hash(symbol + location_type + metric) % 10000)
        
        # Generate base trend with seasonal component
        n_points = len(date_range)
        time = np.linspace(0, 1, n_points)
        
        # Base trend (linear or slightly exponential)
        if np.random.random() < 0.7:  # 70% chance of upward trend
            trend = 100 + 20 * time
        else:
            trend = 100 - 10 * time
            
        # Add seasonal component
        season = 10 * np.sin(2 * np.pi * 4 * time)  # 4 cycles over the period
        
        # Combine and add noise
        values = trend + season + np.random.normal(0, 5, n_points)
        
        # Ensure values are positive
        values = np.maximum(values, 0)
        
        # Create DataFrame
        data = []
        for i, date in enumerate(date_range):
            data.append({
                "date": date,
                "symbol": symbol,
                "location_type": location_type,
                "metric": metric,
                "value": values[i],
                "change_pct": 0 if i == 0 else ((values[i] - values[i-1]) / values[i-1]) * 100
            })
            
        return pd.DataFrame(data)
    
    def get_macro_economic_data(self, indicator: str, country: str = "US") -> pd.DataFrame:
        """
        Get macroeconomic indicator data.
        
        Args:
            indicator (str): Economic indicator to retrieve
                             ("gdp", "unemployment", "inflation", "interest_rate", "consumer_sentiment")
            country (str): Country code
            
        Returns:
            pd.DataFrame: Macroeconomic data
        """
        cache_file = os.path.join(self.cache_dir, "macro_economic", f"{indicator}_{country}.csv")
        
        # Check if cached data exists and is recent enough
        if os.path.exists(cache_file):
            cached_data = pd.read_csv(cache_file)
            if not cached_data.empty:
                cached_data['date'] = pd.to_datetime(cached_data['date'])
                most_recent = cached_data['date'].max()
                if most_recent.date() >= (datetime.now() - timedelta(hours=self.config['update_frequency']['macro_economic'])).date():
                    logger.info(f"Using cached macroeconomic data for {indicator} ({country})")
                    return cached_data
        
        logger.info(f"Fetching macroeconomic data for {indicator} ({country})")
        
        # This is a placeholder for actual economic data API integration
        # In a real implementation, you would use an API like Alpha Vantage, FRED, World Bank, etc.
        
        # Check if API key is available
        api_key = self.config.get("macro_economic_api_key")
        
        if not api_key:
            logger.warning("Macroeconomic API key not found in configuration")
            # Return dummy data for demonstration
            return self._generate_dummy_macro_data(indicator, country)
        
        # TODO: Implement actual macroeconomic API integration
        # For now, return dummy data
        dummy_data = self._generate_dummy_macro_data(indicator, country)
        
        # Cache the data
        dummy_data.to_csv(cache_file, index=False)
        
        return dummy_data
    
    def _generate_dummy_macro_data(self, indicator: str, country: str) -> pd.DataFrame:
        """
        Generate dummy macroeconomic data for demonstration purposes.
        
        Args:
            indicator (str): Economic indicator
            country (str): Country code
            
        Returns:
            pd.DataFrame: Dummy macroeconomic data
        """
        # Determine frequency based on indicator
        if indicator in ["gdp"]:
            freq = "Q"  # Quarterly
            periods = 20  # 5 years
        elif indicator in ["unemployment", "inflation", "consumer_sentiment"]:
            freq = "M"  # Monthly
            periods = 36  # 3 years
        elif indicator in ["interest_rate"]:
            freq = "D"  # Daily (but with fewer changes)
            periods = 180  # ~6 months
        else:
            freq = "M"  # Default to monthly
            periods = 36
            
        # Generate date range
        if freq == "Q":
            end_date = pd.Timestamp(datetime.now().date()).to_period("Q").to_timestamp()
            date_range = pd.date_range(end=end_date, periods=periods, freq=freq)
        elif freq == "M":
            end_date = pd.Timestamp(datetime.now().date()).to_period("M").to_timestamp()
            date_range = pd.date_range(end=end_date, periods=periods, freq=freq)
        else:
            end_date = datetime.now().date()
            date_range = pd.date_range(end=end_date, periods=periods, freq=freq)
            
        # Seed for reproducibility
        np.random.seed(hash(indicator + country) % 10000)
        
        # Generate data based on indicator type
        data = []
        
        # Initial value depends on indicator
        if indicator == "gdp":
            initial_value = 21000  # Billions of USD
            change_range = (-0.03, 0.05)  # Quarterly GDP growth range
        elif indicator == "unemployment":
            initial_value = 4.0  # Percent
            change_range = (-0.5, 0.7)  # Monthly unemployment change range
        elif indicator == "inflation":
            initial_value = 2.5  # Percent
            change_range = (-0.3, 0.5)  # Monthly inflation change range
        elif indicator == "interest_rate":
            initial_value = 3.0  # Percent
            change_range = (-0.25, 0.25)  # Interest rate step change
        elif indicator == "consumer_sentiment":
            initial_value = 95  # Index
            change_range = (-5, 5)  # Monthly sentiment change range
        else:
            initial_value = 100  # Generic index
            change_range = (-2, 2)  # Generic change range
            
        # Generate time series with appropriate characteristics
        value = initial_value
        for date in date_range:
            # Determine change based on indicator
            if indicator == "interest_rate":
                # Interest rates change less frequently and in steps
                if np.random.random() < 0.2:  # 20% chance of change
                    change = np.random.choice([-0.25, 0, 0.25])
                else:
                    change = 0
            else:
                # Other indicators have more continuous changes
                change = np.random.uniform(change_range[0], change_range[1])
                
            # Update value
            value += change
            
            # Ensure value makes sense (non-negative, etc.)
            if indicator in ["unemployment", "inflation", "interest_rate"]:
                value = max(value, 0)  # Rates can't be negative
            elif indicator == "consumer_sentiment":
                value = max(min(value, 150), 0)  # Limit sentiment range
                
            data.append({
                "date": date.date(),
                "country": country,
                "indicator": indicator,
                "value": round(value, 2),
                "change": round(change, 2)
            })
            
        return pd.DataFrame(data)
    
    def get_corporate_events(self, symbol: str, event_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get corporate events data for a specific symbol.
        
        Args:
            symbol (str): Stock symbol
            event_types (List[str], optional): Types of events to include
                                              ("earnings", "dividends", "splits", "insider_trading")
            
        Returns:
            pd.DataFrame: Corporate events data
        """
        if event_types is None:
            event_types = ["earnings", "dividends", "splits", "insider_trading"]
            
        cache_file = os.path.join(self.cache_dir, "corporate_events", f"{symbol}_events.csv")
        
        # Check if cached data exists and is recent enough
        if os.path.exists(cache_file):
            cached_data = pd.read_csv(cache_file)
            if not cached_data.empty:
                cached_data['date'] = pd.to_datetime(cached_data['date'])
                most_recent = cached_data['date'].max()
                if most_recent.date() >= (datetime.now() - timedelta(hours=self.config['update_frequency']['corporate_events'])).date():
                    logger.info(f"Using cached corporate events data for {symbol}")
                    # Filter by requested event types
                    return cached_data[cached_data['event_type'].isin(event_types)]
        
        logger.info(f"Fetching corporate events data for {symbol}")
        
        # This is a placeholder for actual corporate events API integration
        # In a real implementation, you would use an API like Alpha Vantage, IEX Cloud, etc.
        
        # Generate dummy data for demonstration
        dummy_data = self._generate_dummy_corporate_events(symbol, event_types)
        
        # Cache the data
        dummy_data.to_csv(cache_file, index=False)
        
        return dummy_data
    
    def _generate_dummy_corporate_events(self, symbol: str, event_types: List[str]) -> pd.DataFrame:
        """
        Generate dummy corporate events data for demonstration purposes.
        
        Args:
            symbol (str): Stock symbol
            event_types (List[str]): Types of events to include
            
        Returns:
            pd.DataFrame: Dummy corporate events data
        """
        # Seed for reproducibility
        np.random.seed(hash(symbol) % 10000)
        
        # Generate events for the next year
        end_date = datetime.now().date() + timedelta(days=365)
        start_date = datetime.now().date() - timedelta(days=365)
        
        data = []
        
        # Generate earnings events (quarterly)
        if "earnings" in event_types:
            # Start with the most recent quarter end
            current_quarter_end = pd.Timestamp(start_date).to_period("Q").to_timestamp()
            
            while current_quarter_end.date() <= end_date:
                # Earnings typically reported 2-4 weeks after quarter end
                earnings_date = current_quarter_end + timedelta(days=np.random.randint(14, 28))
                
                # Only include if within our date range
                if start_date <= earnings_date.date() <= end_date:
                    # Generate some realistic EPS values
                    expected_eps = round(np.random.uniform(0.5, 2.5), 2)
                    
                    # Actual EPS beats or misses estimates
                    beat_miss = np.random.uniform(-0.3, 0.4)
                    actual_eps = round(expected_eps + beat_miss, 2)
                    
                    data.append({
                        "date": earnings_date.date(),
                        "symbol": symbol,
                        "event_type": "earnings",
                        "details": {
                            "expected_eps": expected_eps,
                            "actual_eps": actual_eps,
                            "beat_miss": round(actual_eps - expected_eps, 2),
                            "beat_miss_pct": round(((actual_eps - expected_eps) / expected_eps) * 100, 2)
                        }
                    })
                
                # Move to next quarter
                current_quarter_end = current_quarter_end + pd.DateOffset(months=3)
        
        # Generate dividend events (quarterly or monthly)
        if "dividends" in event_types:
            # Determine dividend frequency
            is_monthly_dividend = np.random.random() < 0.2  # 20% chance of monthly dividends
            
            # Determine dividend amount
            base_dividend = round(np.random.uniform(0.1, 1.0), 2)
            
            # Start with the beginning of our range
            current_date = start_date
            
            while current_date <= end_date:
                # Add some variation to dividend dates
                if is_monthly_dividend:
                    # Monthly dividends
                    days_to_add = np.random.randint(28, 31)
                else:
                    # Quarterly dividends
                    days_to_add = np.random.randint(89, 92)
                    
                current_date = current_date + timedelta(days=days_to_add)
                
                # Only include if within our date range
                if current_date <= end_date:
                    # Add a small random change to dividend amount (occasionally)
                    if np.random.random() < 0.2:  # 20% chance of dividend change
                        base_dividend = round(base_dividend * (1 + np.random.uniform(-0.05, 0.1)), 2)
                        
                    data.append({
                        "date": current_date,
                        "symbol": symbol,
                        "event_type": "dividends",
                        "details": {
                            "amount": base_dividend,
                            "ex_date": (current_date - timedelta(days=np.random.randint(10, 15))).isoformat(),
                            "record_date": (current_date - timedelta(days=np.random.randint(5, 8))).isoformat(),
                            "payment_date": current_date.isoformat()
                        }
                    })
        
        # Generate stock splits (rare events)
        if "splits" in event_types and np.random.random() < 0.3:  # 30% chance of having a split
            # Generate a split date
            split_date = start_date + timedelta(days=np.random.randint(30, 330))
            
            # Generate split ratio
            split_ratios = [(2, 1), (3, 1), (3, 2), (4, 1), (5, 1), (10, 1)]
            ratio = split_ratios[np.random.randint(0, len(split_ratios))]
            
            data.append({
                "date": split_date,
                "symbol": symbol,
                "event_type": "splits",
                "details": {
                    "ratio": f"{ratio[0]}:{ratio[1]}",
                    "announcement_date": (split_date - timedelta(days=np.random.randint(14, 30))).isoformat()
                }
            })
        
        # Generate insider trading events (sporadic)
        if "insider_trading" in event_types:
            # Generate a few insider trading events
            num_events = np.random.randint(3, 10)
            
            for _ in range(num_events):
                # Random date within our range
                event_date = start_date + timedelta(days=np.random.randint(1, 365*2))
                
                # Only include if within our date range
                if event_date <= end_date:
                    # Transaction type
                    transaction_type = np.random.choice(["buy", "sell"], p=[0.3, 0.7])  # More sells than buys
                    
                    # Transaction size
                    shares = np.random.randint(1000, 50000)
                    price = round(np.random.uniform(20, 200), 2)
                    value = shares * price
                    
                    # Insider name (just for demo)
                    insider_titles = ["CEO", "CFO", "CTO", "Director", "VP", "SVP"]
                    insider = f"{np.random.choice(['John', 'Jane', 'Robert', 'Mary', 'David', 'Lisa'])} {np.random.choice(['Smith', 'Jones', 'Williams', 'Brown', 'Miller', 'Davis'])} ({np.random.choice(insider_titles)})"
                    
                    data.append({
                        "date": event_date,
                        "symbol": symbol,
                        "event_type": "insider_trading",
                        "details": {
                            "insider": insider,
                            "transaction_type": transaction_type,
                            "shares": shares,
                            "price": price,
                            "value": value,
                            "filing_date": (event_date + timedelta(days=np.random.randint(1, 3))).isoformat()
                        }
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
            
        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])
        
        # Convert details dictionary to string
        df["details"] = df["details"].apply(json.dumps)
        
        # Sort by date
        df.sort_values(by="date", inplace=True)
        
        return df
    
    def integrate_alternative_data(self, market_data: pd.DataFrame, symbol: str, 
                                  include_news: bool = True, 
                                  include_social: bool = True, 
                                  include_satellite: bool = False,
                                  include_macro: bool = True,
                                  include_events: bool = True) -> pd.DataFrame:
        """
        Integrate alternative data sources with traditional market data.
        
        Args:
            market_data (pd.DataFrame): Traditional market price data
            symbol (str): Stock symbol
            include_news (bool): Whether to include news sentiment
            include_social (bool): Whether to include social media sentiment
            include_satellite (bool): Whether to include satellite imagery data
            include_macro (bool): Whether to include macroeconomic data
            include_events (bool): Whether to include corporate events
            
        Returns:
            pd.DataFrame: Integrated data with alternative data features
        """
        if market_data.empty:
            return market_data
            
        # Ensure date column is datetime
        if "date" in market_data.columns:
            date_col = "date"
        else:
            # Try to find date column
            date_cols = [col for col in market_data.columns if "date" in col.lower()]
            if date_cols:
                date_col = date_cols[0]
            else:
                # If no date column found, use index
                market_data = market_data.reset_index()
                date_col = "index"
                
        market_data[date_col] = pd.to_datetime(market_data[date_col])
        
        # Get date range from market data
        start_date = market_data[date_col].min()
        end_date = market_data[date_col].max()
        days = (end_date - start_date).days + 1
        
        # Add news sentiment features
        if include_news:
            news_data = self.get_news_sentiment(symbol, days=days)
            if not news_data.empty:
                # Rename columns to avoid conflicts
                news_data = news_data.rename(columns={
                    "compound_score": "news_sentiment",
                    "positive_score": "news_positive",
                    "negative_score": "news_negative",
                    "neutral_score": "news_neutral",
                    "polarity": "news_polarity",
                    "subjectivity": "news_subjectivity",
                    "article_count": "news_article_count"
                })
                
                # Merge with market data
                market_data = pd.merge_asof(
                    market_data.sort_values(date_col),
                    news_data[["date", "news_sentiment", "news_positive", "news_negative", 
                               "news_neutral", "news_article_count"]].sort_values("date"),
                    left_on=date_col,
                    right_on="date",
                    direction="backward"
                )
                
                # Fill missing values with 0 (no news)
                news_cols = [col for col in market_data.columns if col.startswith("news_")]
                market_data[news_cols] = market_data[news_cols].fillna(0)
        
        # Add social media sentiment features
        if include_social:
            social_data = self.get_social_media_sentiment(symbol, days=days)
            if not social_data.empty:
                # Aggregate by date and platform
                social_agg = social_data.groupby(["date", "platform"]).agg({
                    "sentiment_score": "mean",
                    "post_volume": "sum",
                    "positive_ratio": "mean",
                    "negative_ratio": "mean"
                }).reset_index()
                
                # Pivot to get columns by platform
                social_pivot = social_agg.pivot(
                    index="date",
                    columns="platform",
                    values=["sentiment_score", "post_volume", "positive_ratio", "negative_ratio"]
                )
                
                # Flatten column names
                social_pivot.columns = [f"social_{platform}_{col}" 
                                      for col, platform in social_pivot.columns]
                social_pivot = social_pivot.reset_index()
                
                # Merge with market data
                market_data = pd.merge_asof(
                    market_data.sort_values(date_col),
                    social_pivot.sort_values("date"),
                    left_on=date_col,
                    right_on="date",
                    direction="backward"
                )
                
                # Fill missing values
                social_cols = [col for col in market_data.columns if col.startswith("social_")]
                market_data[social_cols] = market_data[social_cols].fillna(0)
        
        # Add satellite imagery data features
        if include_satellite:
            location_types = ["retail", "manufacturing", "shipping"]
            metrics = ["foot_traffic", "car_count", "container_count"]
            
            for loc_type in location_types:
                for metric in metrics:
                    # Only get relevant metrics for each location type
                    if (loc_type == "retail" and metric in ["foot_traffic", "car_count"]) or \
                       (loc_type == "manufacturing" and metric in ["car_count"]) or \
                       (loc_type == "shipping" and metric in ["container_count"]):
                        
                        sat_data = self.get_satellite_imagery_data(symbol, loc_type, metric)
                        if not sat_data.empty:
                            # Rename columns
                            col_prefix = f"sat_{loc_type}_{metric}"
                            sat_data = sat_data.rename(columns={
                                "value": f"{col_prefix}_value",
                                "change_pct": f"{col_prefix}_change"
                            })
                            
                            # Merge with market data
                            market_data = pd.merge_asof(
                                market_data.sort_values(date_col),
                                sat_data[["date", f"{col_prefix}_value", f"{col_prefix}_change"]].sort_values("date"),
                                left_on=date_col,
                                right_on="date",
                                direction="backward"
                            )
                            
                            # Forward fill missing values (satellite data is less frequent)
                            sat_cols = [col for col in market_data.columns if col.startswith(col_prefix)]
                            market_data[sat_cols] = market_data[sat_cols].fillna(method="ffill")
        
        # Add macroeconomic data features
        if include_macro:
            # Key economic indicators
            indicators = ["gdp", "unemployment", "inflation", "interest_rate", "consumer_sentiment"]
            
            for indicator in indicators:
                macro_data = self.get_macro_economic_data(indicator)
                if not macro_data.empty:
                    # Rename columns
                    col_prefix = f"macro_{indicator}"
                    macro_data = macro_data.rename(columns={
                        "value": f"{col_prefix}_value",
                        "change": f"{col_prefix}_change"
                    })
                    
                    # Merge with market data
                    market_data = pd.merge_asof(
                        market_data.sort_values(date_col),
                        macro_data[["date", f"{col_prefix}_value", f"{col_prefix}_change"]].sort_values("date"),
                        left_on=date_col,
                        right_on="date",
                        direction="backward"
                    )
                    
                    # Forward fill missing values (macro data is less frequent)
                    macro_cols = [col for col in market_data.columns if col.startswith(col_prefix)]
                    market_data[macro_cols] = market_data[macro_cols].fillna(method="ffill")
        
        # Add corporate events features
        if include_events:
            events_data = self.get_corporate_events(symbol)
            if not events_data.empty:
                # Create dummy variables for event types
                event_dummies = pd.get_dummies(events_data, columns=["event_type"], prefix="event")
                
                # Group by date to handle multiple events on the same day
                event_features = event_dummies.groupby("date").sum().reset_index()
                
                # Merge with market data
                market_data = pd.merge_asof(
                    market_data.sort_values(date_col),
                    event_features.sort_values("date"),
                    left_on=date_col,
                    right_on="date",
                    direction="forward",  # Use 'forward' to capture upcoming events
                    tolerance=pd.Timedelta(days=7)  # Consider events within 7 days
                )
                
                # Replace NaN with 0 for event columns
                event_cols = [col for col in market_data.columns if col.startswith("event_")]
                market_data[event_cols] = market_data[event_cols].fillna(0)
                
                # Add days to next earnings feature
                earnings_dates = events_data[events_data["event_type"] == "earnings"]["date"]
                if not earnings_dates.empty:
                    # Convert to numpy array for faster processing
                    earnings_dates_array = earnings_dates.values
                    
                    # Calculate days to next earnings for each date in market data
                    days_to_earnings = []
                    for date in market_data[date_col]:
                        # Find future earnings dates
                        future_earnings = earnings_dates_array[earnings_dates_array > date]
                        if len(future_earnings) > 0:
                            # Get the nearest future earnings date
                            nearest = min(future_earnings)
                            days_diff = (nearest - date).astype('timedelta64[D]') / np.timedelta64(1, 'D')
                            days_to_earnings.append(days_diff)
                        else:
                            days_to_earnings.append(None)
                            
                    market_data["days_to_earnings"] = days_to_earnings
                    
                    # Fill missing values with a large number
                    market_data["days_to_earnings"] = market_data["days_to_earnings"].fillna(365)
        
        # Drop duplicate date column if it was added during merges
        if "date" in market_data.columns and date_col != "date":
            market_data = market_data.drop(columns=["date"])
            
        return market_data
    
    def generate_alternative_data_features(self, market_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Generate features from alternative data sources for machine learning models.
        
        Args:
            market_data (pd.DataFrame): Market data with alternative data
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Feature DataFrame ready for ML models
        """
        # First integrate alternative data if not already done
        if not any(col.startswith(("news_", "social_", "sat_", "macro_", "event_")) for col in market_data.columns):
            market_data = self.integrate_alternative_data(market_data, symbol)
            
        # Create feature DataFrame
        features = market_data.copy()
        
        # Ensure we have a date column
        date_cols = [col for col in features.columns if "date" in col.lower()]
        if date_cols:
            date_col = date_cols[0]
        else:
            features["date"] = pd.to_datetime(features.index)
            date_col = "date"
            
        # Add time-based features
        features["day_of_week"] = features[date_col].dt.dayofweek
        features["month"] = features[date_col].dt.month
        features["quarter"] = features[date_col].dt.quarter
        features["is_month_end"] = features[date_col].dt.is_month_end.astype(int)
        features["is_quarter_end"] = features[date_col].dt.is_quarter_end.astype(int)
        
        # Sentiment smoothing features
        sentiment_cols = [col for col in features.columns if "sentiment" in col.lower()]
        for col in sentiment_cols:
            # Add rolling averages
            features[f"{col}_5d_avg"] = features[col].rolling(window=5).mean()
            features[f"{col}_10d_avg"] = features[col].rolling(window=10).mean()
            features[f"{col}_20d_avg"] = features[col].rolling(window=20).mean()
            
            # Add trends (direction of sentiment)
            features[f"{col}_trend"] = features[col].diff().rolling(window=5).mean()
            
        # Social media volume features
        volume_cols = [col for col in features.columns if "volume" in col.lower() or "count" in col.lower()]
        for col in volume_cols:
            # Add rolling averages
            features[f"{col}_5d_avg"] = features[col].rolling(window=5).mean()
            
            # Add relative volume (compared to recent average)
            features[f"{col}_rel_vol"] = features[col] / features[col].rolling(window=10).mean()
            
        # Event proximity features
        if "days_to_earnings" in features.columns:
            # Create bins for earnings proximity
            features["earnings_proximity"] = pd.cut(
                features["days_to_earnings"],
                bins=[0, 7, 14, 30, float("inf")],
                labels=["very_close", "close", "near", "far"]
            )
            
            # Convert to dummy variables
            earnings_dummies = pd.get_dummies(features["earnings_proximity"], prefix="earnings_prox")
            features = pd.concat([features, earnings_dummies], axis=1)
            
        # Macro trend features
        macro_cols = [col for col in features.columns if col.startswith("macro_")]
        for col in macro_cols:
            if "value" in col:
                # Add trend direction
                features[f"{col}_trend"] = np.sign(features[col].diff(5))
                
                # Add normalized value (z-score)
                features[f"{col}_zscore"] = (features[col] - features[col].rolling(window=180).mean()) / features[col].rolling(window=180).std()
        
        # Fill NaN values
        numeric_cols = features.select_dtypes(include=["float64", "int64"]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0)
        
        # Drop original date column if requested (useful for ML models)
        # features = features.drop(columns=[date_col])
        
        return features
    
    def get_alternative_data_importance(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """
        Calculate the importance of alternative data features for predicting the target.
        
        Args:
            features (pd.DataFrame): Feature DataFrame
            target (pd.Series): Target variable (e.g., future returns)
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Get numeric features only
            numeric_features = features.select_dtypes(include=["float64", "int64"])
            
            # Remove any remaining NaN values
            numeric_features = numeric_features.fillna(0)
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(numeric_features)
            
            # Train a random forest to get feature importances
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(scaled_features, target)
            
            # Get feature importances
            importances = rf.feature_importances_
            
            # Map importances to feature names
            importance_dict = dict(zip(numeric_features.columns, importances))
            
            # Sort by importance
            sorted_importances = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
            
            return sorted_importances
            
        except ImportError:
            logger.warning("scikit-learn not available. Feature importance calculation skipped.")
            return {}
            
        except Exception as e:
            logger.error(f"Error calculating feature importances: {e}")
            return {}