#!/usr/bin/env python
"""
QuantConnect Strategy Adapter Module with Market Regime Detection

This module provides functionality to convert Mathematricks strategies to QuantConnect format
with advanced market regime detection capabilities. It adapts strategies to be regime-aware,
allowing them to optimize their parameters and behavior based on different market conditions.

Key features:
1. Market regime detection using unsupervised learning
2. Regime-specific parameter optimization
3. Adaptive position sizing based on current market regime
4. Dynamic risk management that responds to market conditions
5. Transition handling between different market regimes
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class QuantConnectAdapter:
    """
    Base adapter class for converting Mathematricks strategies to QuantConnect format.
    
    This class provides methods to:
    1. Parse strategy JSON files
    2. Generate QuantConnect Python files
    3. Upload to QuantConnect platform
    """
    
    def __init__(self, strategy_path=None, output_path=None):
        """
        Initialize the adapter.
        
        Args:
            strategy_path (str): Path to strategy JSON or Python file
            output_path (str): Directory to save generated QuantConnect algorithm
        """
        self.strategy_path = strategy_path
        self.output_path = output_path or os.path.join(os.getcwd(), 'qc_algorithms')
        self.strategy_data = None
        self.template = self._load_template()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
    
    def _load_template(self):
        """Load the QuantConnect algorithm template."""
        return """
# QuantConnect Algorithm Template
# Generated from the Mathematricks strategy system

from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Indicators")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Indicators import *
from QuantConnect.Data.Market import TradeBar, QuoteBar
from datetime import datetime, timedelta

class {class_name}(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate({start_year}, {start_month}, {start_day})  # Set Start Date
        self.SetEndDate({end_year}, {end_month}, {end_day})          # Set End Date
        self.SetCash({initial_cash})                                 # Set Strategy Cash

        # Strategy parameters
{params}

        # Add universe selection
{universe}

        # Initialize indicators
{indicators}

        # Set warmup period
        self.SetWarmUp({warmup_period})
        
        # Initialize tracking variables
{tracking_vars}

    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm.'''
        # Check if we're still in the warmup period
        if self.IsWarmingUp:
            return
            
{on_data}

{helper_methods}
"""
    
    def load_strategy(self):
        """Load strategy data from file."""
        if not self.strategy_path:
            raise ValueError("Strategy path not specified")
            
        if self.strategy_path.endswith('.json'):
            with open(self.strategy_path, 'r') as f:
                self.strategy_data = json.load(f)
            logger.info(f"Loaded strategy from JSON: {self.strategy_path}")
            
        else:
            raise ValueError("Unsupported strategy file format. Use JSON file.")
        
        return self.strategy_data
    
    def generate_algorithm(self, start_date=None, end_date=None, cash=100000):
        """
        Generate a QuantConnect algorithm from the loaded strategy.
        
        Args:
            start_date (str): Backtest start date (YYYY-MM-DD)
            end_date (str): Backtest end date (YYYY-MM-DD)
            cash (int): Initial capital
            
        Returns:
            str: Generated algorithm code
        """
        if not self.strategy_data:
            self.load_strategy()
            
        strategy_name = self.strategy_data['strategy']['Strategy Name']
        class_name = ''.join(x for x in strategy_name if x.isalnum()) + "Algorithm"
        
        # Set dates for backtesting
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        start_year, start_month, start_day = start_date.split('-')
        end_year, end_month, end_day = end_date.split('-')
        
        # Generate parameters section
        params = self._generate_parameters()
        
        # Generate universe section (default to S&P 500)
        universe = self._generate_universe()
        
        # Generate indicators section
        indicators = self._generate_indicators()
        
        # Generate tracking variables
        tracking_vars = self._generate_tracking_vars()
        
        # Generate OnData method
        on_data = self._generate_on_data()
        
        # Generate helper methods
        helper_methods = self._generate_helper_methods()
        
        # Fill template
        algorithm_code = self.template.format(
            class_name=class_name,
            start_year=start_year,
            start_month=start_month,
            start_day=start_day,
            end_year=end_year,
            end_month=end_month,
            end_day=end_day,
            initial_cash=cash,
            params=params,
            universe=universe,
            indicators=indicators,
            warmup_period=50,  # Default warmup period
            tracking_vars=tracking_vars,
            on_data=on_data,
            helper_methods=helper_methods
        )
        
        # Save to file
        output_file = os.path.join(self.output_path, f"{class_name}.py")
        with open(output_file, 'w') as f:
            f.write(algorithm_code)
            
        logger.info(f"Generated QuantConnect algorithm: {output_file}")
        return algorithm_code
    
    def _generate_parameters(self):
        """Generate strategy parameters."""
        params = []
        
        # Default parameters based on strategy type
        params.append("        # Technical indicator parameters")
        params.append("        self.ema_short_period = 8")
        params.append("        self.ema_medium_period = 21")
        params.append("        self.ema_long_period = 50")
        params.append("        self.rsi_period = 14")
        params.append("        self.atr_period = 14")
        params.append("        self.volume_ma_period = 20")
        
        # Position sizing and risk management parameters
        params.append("\n        # Position sizing parameters")
        params.append("        self.max_position_size = 0.05  # Maximum position size as percentage of portfolio")
        
        # Add risk management parameters
        params.append("\n        # Risk management parameters")
        
        # Extract stop loss from strategy if available
        risk_mgmt = self.strategy_data['strategy'].get('Risk Management', {})
        stop_loss_text = risk_mgmt.get('Stop Loss', '')
        if 'ATR' in stop_loss_text:
            params.append("        self.stop_loss_atr_multiple = 2.5")
        
        params.append("        self.trailing_stop_percentage = 0.02  # 2% trailing stop")
        
        return "\n".join(params)
    
    def _generate_universe(self):
        """Generate universe selection code."""
        universe_lines = []
        universe_setting = self.strategy_data['strategy'].get('Universe', 'S&P 500 constituents')
        
        if 'S&P 500' in universe_setting:
            universe_lines.append("        # Use S&P 500 constituents")
            universe_lines.append("        self.symbol = self.AddEquity('SPY', Resolution.Daily).Symbol")
            universe_lines.append("        # Alternatively, use S&P 500 constituents:")
            universe_lines.append("        # self.UniverseSettings.Resolution = Resolution.Daily")
            universe_lines.append("        # self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)")
        else:
            # Default to SPY if universe not specified
            universe_lines.append("        # Default symbol")
            universe_lines.append("        self.symbol = self.AddEquity('SPY', Resolution.Daily).Symbol")
        
        return "\n".join(universe_lines)
    
    def _generate_indicators(self):
        """Generate indicator initialization code."""
        indicators = []
        
        # Add common indicators based on strategy type
        indicators.append("        # Moving averages")
        indicators.append("        self.ema_short = ExponentialMovingAverage(self.ema_short_period)")
        indicators.append("        self.ema_medium = ExponentialMovingAverage(self.ema_medium_period)")
        indicators.append("        self.ema_long = ExponentialMovingAverage(self.ema_long_period)")
        indicators.append("        self.RegisterIndicator(self.symbol, self.ema_short, Resolution.Daily)")
        indicators.append("        self.RegisterIndicator(self.symbol, self.ema_medium, Resolution.Daily)")
        indicators.append("        self.RegisterIndicator(self.symbol, self.ema_long, Resolution.Daily)")
        
        indicators.append("\n        # RSI indicator")
        indicators.append("        self.rsi = RelativeStrengthIndex(self.rsi_period)")
        indicators.append("        self.RegisterIndicator(self.symbol, self.rsi, Resolution.Daily)")
        
        indicators.append("\n        # ATR indicator for volatility measurement")
        indicators.append("        self.atr = AverageTrueRange(self.atr_period)")
        indicators.append("        self.RegisterIndicator(self.symbol, self.atr, Resolution.Daily)")
        
        indicators.append("\n        # Volume moving average")
        indicators.append("        self.volume_ma = SimpleMovingAverage(self.volume_ma_period)")
        indicators.append("        # Custom volume indicator registration")
        indicators.append("        self.RegisterIndicator(self.symbol, self.volume_ma, Resolution.Daily,")
        indicators.append("            lambda x: x.Volume)")
        
        return "\n".join(indicators)
    
    def _generate_tracking_vars(self):
        """Generate tracking variables code."""
        tracking_vars = []
        
        tracking_vars.append("        # Position tracking")
        tracking_vars.append("        self.invested = False")
        tracking_vars.append("        self.entry_price = 0")
        tracking_vars.append("        self.highest_price = 0")
        tracking_vars.append("        self.stop_price = 0")
        
        return "\n".join(tracking_vars)
    
    def _generate_on_data(self):
        """Generate OnData method implementation."""
        # Extract strategy rules
        entry_rules = self.strategy_data['strategy'].get('Entry Rules', [])
        exit_rules = self.strategy_data['strategy'].get('Exit Rules', [])
        
        on_data_lines = []
        on_data_lines.append("        # Get current data for our symbol")
        on_data_lines.append("        if not self.Securities.ContainsKey(self.symbol):")
        on_data_lines.append("            return")
        on_data_lines.append("")
        on_data_lines.append("        if not data.ContainsKey(self.symbol):")
        on_data_lines.append("            return")
        on_data_lines.append("")
        on_data_lines.append("        # Get current price and update tracking variables")
        on_data_lines.append("        current_price = self.Securities[self.symbol].Price")
        on_data_lines.append("")
        on_data_lines.append("        # Update highest price if we're invested (for trailing stop)")
        on_data_lines.append("        if self.invested and current_price > self.highest_price:")
        on_data_lines.append("            self.highest_price = current_price")
        on_data_lines.append("            # Update trailing stop")
        on_data_lines.append("            self.stop_price = self.highest_price * (1 - self.trailing_stop_percentage)")
        on_data_lines.append("")
        
        # Check if we need to exit based on trailing stop
        on_data_lines.append("        # Check for exit based on trailing stop")
        on_data_lines.append("        if self.invested and current_price < self.stop_price:")
        on_data_lines.append("            self.Liquidate(self.symbol)")
        on_data_lines.append("            self.Debug(f\"Exit: Trailing stop triggered at {current_price}\")")
        on_data_lines.append("            self.invested = False")
        on_data_lines.append("            return")
        on_data_lines.append("")
        
        # Check entry conditions
        on_data_lines.append("        # Entry conditions")
        on_data_lines.append("        if not self.invested:")
        on_data_lines.append("            # Check if indicators are ready")
        on_data_lines.append("            if not (self.ema_short.IsReady and self.ema_medium.IsReady and")
        on_data_lines.append("                    self.ema_long.IsReady and self.rsi.IsReady and self.atr.IsReady):")
        on_data_lines.append("                return")
        on_data_lines.append("")
        on_data_lines.append("            # Check entry conditions")
        on_data_lines.append("            uptrend_structure = (")
        on_data_lines.append("                current_price > self.ema_short.Current.Value > ")
        on_data_lines.append("                self.ema_medium.Current.Value > self.ema_long.Current.Value")
        on_data_lines.append("            )")
        on_data_lines.append("")
        on_data_lines.append("            rsi_condition = (")
        on_data_lines.append("                self.rsi.Current.Value > 50 and")
        on_data_lines.append("                self.rsi.Current.Value < 65 and")
        on_data_lines.append("                self.rsi.Current.Value > self.rsi.Current.Value")  # Simplification for rising RSI
        on_data_lines.append("            )")
        on_data_lines.append("")
        on_data_lines.append("            volume_confirmation = (")
        on_data_lines.append("                data[self.symbol].Volume > self.volume_ma.Current.Value")
        on_data_lines.append("            )")
        on_data_lines.append("")
        on_data_lines.append("            # Buy signal")
        on_data_lines.append("            if uptrend_structure and rsi_condition and volume_confirmation:")
        on_data_lines.append("                # Calculate position size based on ATR")
        on_data_lines.append("                portfolio_value = self.Portfolio.TotalPortfolioValue")
        on_data_lines.append("                risk_per_share = self.atr.Current.Value * self.stop_loss_atr_multiple")
        on_data_lines.append("                self.entry_price = current_price")
        on_data_lines.append("                self.highest_price = current_price")
        on_data_lines.append("                self.stop_price = current_price - risk_per_share")
        on_data_lines.append("")
        on_data_lines.append("                # Calculate position size (limit to max position size)")
        on_data_lines.append("                risk_amount = portfolio_value * 0.01  # Risk 1% per trade")
        on_data_lines.append("                shares_to_buy = int(risk_amount / risk_per_share)")
        on_data_lines.append("                max_shares = int(portfolio_value * self.max_position_size / current_price)")
        on_data_lines.append("                shares_to_buy = min(shares_to_buy, max_shares)")
        on_data_lines.append("")
        on_data_lines.append("                if shares_to_buy > 0:")
        on_data_lines.append("                    self.MarketOrder(self.symbol, shares_to_buy)")
        on_data_lines.append("                    self.Debug(f\"Buy: {shares_to_buy} shares at {current_price}\")")
        on_data_lines.append("                    self.invested = True")
        on_data_lines.append("")
        
        # Check exit conditions
        on_data_lines.append("        # Exit conditions")
        on_data_lines.append("        elif self.invested:")
        on_data_lines.append("            rsi_overbought_declining = (")
        on_data_lines.append("                self.rsi.Current.Value > 65 and")
        on_data_lines.append("                self.rsi.Current.Value < self.rsi.Current.Value")  # Simplification for declining RSI
        on_data_lines.append("            )")
        on_data_lines.append("")
        on_data_lines.append("            price_below_short_ema = current_price < self.ema_short.Current.Value")
        on_data_lines.append("")
        on_data_lines.append("            # Check if short EMA crosses below medium EMA")
        on_data_lines.append("            short_ema_crossover = (")
        on_data_lines.append("                self.ema_short.Current.Value < self.ema_medium.Current.Value")
        on_data_lines.append("            )")
        on_data_lines.append("")
        on_data_lines.append("            # Sell signal")
        on_data_lines.append("            if rsi_overbought_declining or price_below_short_ema or short_ema_crossover:")
        on_data_lines.append("                self.Liquidate(self.symbol)")
        on_data_lines.append("                self.Debug(f\"Exit: Signal triggered at {current_price}\")")
        on_data_lines.append("                self.invested = False")
        
        return "\n".join(on_data_lines)
    
    def _generate_helper_methods(self):
        """Generate helper methods for the algorithm."""
        helper_methods = []
        
        # Add CoarseSelectionFunction for universe selection
        helper_methods.append("    def CoarseSelectionFunction(self, coarse):")
        helper_methods.append("        '''")
        helper_methods.append("        Coarse selection function for universe selection.")
        helper_methods.append("        ")
        helper_methods.append("        This function filters the universe based on price, volume, and dollar volume.")
        helper_methods.append("        '''")
        helper_methods.append("        if self.Time.day != 1:")
        helper_methods.append("            return Universe.Unchanged")
        helper_methods.append("            ")
        helper_methods.append("        # Sort by dollar volume and take top stocks")
        helper_methods.append("        selected = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)")
        helper_methods.append("        selected = [x.Symbol for x in selected if x.Price > 5 and x.DollarVolume > 10000000]")
        helper_methods.append("        return selected[:100]  # Take top 100 by dollar volume")
        helper_methods.append("")
        
        # Add FineSelectionFunction for universe selection
        helper_methods.append("    def FineSelectionFunction(self, fine):")
        helper_methods.append("        '''")
        helper_methods.append("        Fine selection function for universe selection.")
        helper_methods.append("        ")
        helper_methods.append("        This function filters the universe based on fundamental factors.")
        helper_methods.append("        '''")
        helper_methods.append("        # Filter stocks in S&P 500")
        helper_methods.append("        fine = [f for f in fine if f.AssetClassification.MorningstarSectorCode == MorningstarSectorCode.Technology]")
        helper_methods.append("        ")
        helper_methods.append("        # Sort by PE ratio and take top stocks")
        helper_methods.append("        fine = sorted(fine, key=lambda f: f.ValuationRatios.PERatio if f.ValuationRatios.PERatio > 0 else float('inf'))")
        helper_methods.append("        ")
        helper_methods.append("        return [f.Symbol for f in fine[:10]]  # Take top 10 by PE ratio")
        
        return "\n".join(helper_methods)


class MarketRegimeDetectionAdapter(QuantConnectAdapter):
    """
    Enhanced adapter class that integrates market regime detection into QuantConnect algorithms.
    This allows strategies to adapt their parameters and behavior based on the current market regime.
    """
    
    def __init__(self, strategy_path=None, output_path=None, num_regimes=4, regime_detection_method="hmm"):
        """
        Initialize the regime-aware adapter.
        
        Args:
            strategy_path (str): Path to strategy JSON or Python file
            output_path (str): Directory to save generated QuantConnect algorithm
            num_regimes (int): Number of market regimes to detect
            regime_detection_method (str): Method for regime detection ('hmm', 'kmeans')
        """
        super().__init__(strategy_path, output_path)
        self.num_regimes = num_regimes
        self.regime_detection_method = regime_detection_method
        self.template = self._load_regime_aware_template()
    
    def _load_regime_aware_template(self):
        """Load the QuantConnect algorithm template with market regime detection."""
        return """
# QuantConnect Algorithm Template with Market Regime Detection
# Generated from the Mathematricks strategy system

from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Indicators")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Indicators import *
from QuantConnect.Data.Market import TradeBar, QuoteBar
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from enum import Enum

class MarketRegime(Enum):
    """Enum for different market regimes"""
    BullMarket = 0
    BearMarket = 1
    SidewaysMarket = 2
    HighVolatility = 3
    LowVolatility = 4
    Crisis = 5

class {class_name}(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate({start_year}, {start_month}, {start_day})  # Set Start Date
        self.SetEndDate({end_year}, {end_month}, {end_day})          # Set End Date
        self.SetCash({initial_cash})                                 # Set Strategy Cash

        # Strategy parameters
{params}

        # Market regime parameters
{regime_params}

        # Add universe selection
{universe}

        # Initialize indicators
{indicators}

        # Initialize market regime indicators
{regime_indicators}
        
        # Set warmup period
        self.SetWarmUp({warmup_period})
        
        # Initialize tracking variables
{tracking_vars}

        # Schedule market regime detection to run weekly
        self.Schedule.On(self.DateRules.WeekStart(), 
                        self.TimeRules.AfterMarketOpen("SPY"), 
                        self.DetectMarketRegime)

    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm.'''
        # Check if we're still in the warmup period
        if self.IsWarmingUp:
            return
            
{on_data}

    def DetectMarketRegime(self):
        '''
        Detect the current market regime based on recent market data.
        This method is scheduled to run weekly.
        '''
        # Skip if not enough data
        if not self.feature_lookback_period_ready:
            return
            
        # Get historical data for the main symbol
        history = self.History(self.symbol, self.feature_lookback_period, Resolution.Daily)
        if history.empty or len(history) < self.feature_lookback_period:
            return
            
        # Calculate features for regime detection
        features = self.CalculateRegimeFeatures(history)
        
        # Scale features
        scaled_features = self.regime_scaler.transform([features])
        
        # Detect regime using the method specified (simplified method-specific logic)
        if self.regime_detection_method == "kmeans":
            # K-means clustering
            regime_id = self.DetectRegimeWithKMeans(scaled_features)
        else:
            # Default to simple rules-based detection
            regime_id = self.DetectRegimeWithRules(features)
        
        # Update current regime if it has changed
        if regime_id != self.current_regime_id:
            old_regime = self.current_regime_name
            self.current_regime_id = regime_id
            self.current_regime_name = self.GetRegimeName(regime_id)
            self.Debug(f"Market regime changed from {{old_regime}} to {{self.current_regime_name}}")
            
            # Update strategy parameters based on the new regime
            self.UpdateStrategyParameters()
        
    def CalculateRegimeFeatures(self, history):
        '''Calculate features for regime detection.'''
        # Convert history to pandas DataFrame if needed
        if not isinstance(history, pd.DataFrame):
            hist_df = pd.DataFrame(history)
        else:
            hist_df = history
        
        # Calculate returns-based features
        returns = hist_df['close'].pct_change().dropna().values
        log_returns = np.log(hist_df['close']).diff().dropna().values
        
        # Calculate volatility
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        # Calculate trend features
        sma_20 = hist_df['close'].rolling(window=20).mean().values[-1] if len(hist_df) >= 20 else hist_df['close'].mean()
        sma_50 = hist_df['close'].rolling(window=50).mean().values[-1] if len(hist_df) >= 50 else hist_df['close'].mean()
        sma_200 = hist_df['close'].rolling(window=200).mean().values[-1] if len(hist_df) >= 200 else hist_df['close'].mean()
        
        price = hist_df['close'].values[-1]
        trend_20_50 = (sma_20 / sma_50) - 1 if sma_50 != 0 else 0
        trend_50_200 = (sma_50 / sma_200) - 1 if sma_200 != 0 else 0
        
        # Calculate momentum features
        momentum_20 = (price / hist_df['close'].values[-21]) - 1 if len(hist_df) >= 21 else 0
        momentum_60 = (price / hist_df['close'].values[-61]) - 1 if len(hist_df) >= 61 else 0
        
        # Calculate drawdown
        rolling_max = hist_df['close'].expanding().max()
        drawdown = (hist_df['close'] / rolling_max) - 1
        max_drawdown = drawdown.min()
        
        # Return array of features
        return [volatility, trend_20_50, trend_50_200, momentum_20, momentum_60, max_drawdown]
    
    def DetectRegimeWithKMeans(self, scaled_features):
        '''Detect market regime using K-means clustering.'''
        # Note: In a real implementation, we would have a pre-trained K-means model
        # For this template, we'll use a simplified approach
        
        # Use manually defined centroids for different regimes
        # Format: [volatility, trend_20_50, trend_50_200, momentum_20, momentum_60, max_drawdown]
        centroids = [
            [0.10, 0.02, 0.03, 0.05, 0.10, -0.05],  # Bull Market (low vol, positive trends)
            [0.25, -0.03, -0.05, -0.08, -0.15, -0.15],  # Bear Market (higher vol, negative trends)
            [0.12, 0.00, 0.00, 0.01, 0.00, -0.08],  # Sideways Market (moderate vol, neutral trends)
            [0.35, 0.01, -0.01, 0.02, 0.05, -0.20],  # High Volatility (very high vol, mixed trends)
            [0.08, 0.01, 0.01, 0.02, 0.05, -0.03],  # Low Volatility (very low vol, slight positive trend)
            [0.50, -0.05, -0.08, -0.15, -0.30, -0.30]   # Crisis (extreme vol, strong negative trends)
        ]
        
        # Find closest centroid
        min_dist = float('inf')
        closest_regime = 0
        
        for i, centroid in enumerate(centroids):
            if i >= self.num_regimes:
                break
                
            # Calculate Euclidean distance
            dist = np.sqrt(np.sum((scaled_features[0] - centroid) ** 2))
            
            if dist < min_dist:
                min_dist = dist
                closest_regime = i
        
        return closest_regime
    
    def DetectRegimeWithRules(self, features):
        '''Detect market regime using simple rules.'''
        volatility, trend_20_50, trend_50_200, momentum_20, momentum_60, max_drawdown = features
        
        # Bull Market: Low-moderate volatility, positive trends and momentum
        if trend_20_50 > 0.01 and trend_50_200 > 0 and momentum_20 > 0 and volatility < 0.20:
            return MarketRegime.BullMarket.value
            
        # Bear Market: Moderate-high volatility, negative trends and momentum
        elif trend_20_50 < -0.01 and trend_50_200 < 0 and momentum_20 < 0:
            return MarketRegime.BearMarket.value
            
        # High Volatility: Very high volatility regardless of trend
        elif volatility > 0.30:
            return MarketRegime.HighVolatility.value
            
        # Crisis: Extreme volatility and drawdown
        elif volatility > 0.40 and max_drawdown < -0.25:
            return MarketRegime.Crisis.value
            
        # Low Volatility: Very low volatility
        elif volatility < 0.10:
            return MarketRegime.LowVolatility.value
            
        # Sideways Market: Default case
        else:
            return MarketRegime.SidewaysMarket.value
    
    def GetRegimeName(self, regime_id):
        '''Get the name of a market regime based on its ID.'''
        try:
            return MarketRegime(regime_id).name
        except:
            return f"Regime {regime_id}"
    
    def UpdateStrategyParameters(self):
        '''Update strategy parameters based on current market regime.'''
        if self.current_regime_id == MarketRegime.BullMarket.value:
            # More aggressive in bull markets
            self.max_position_size = self.regime_params["bull_market"]["max_position_size"]
            self.stop_loss_atr_multiple = self.regime_params["bull_market"]["stop_loss_atr_multiple"]
            self.trailing_stop_percentage = self.regime_params["bull_market"]["trailing_stop_percentage"]
            self.Debug("Applied Bull Market parameters")
            
        elif self.current_regime_id == MarketRegime.BearMarket.value:
            # More conservative in bear markets
            self.max_position_size = self.regime_params["bear_market"]["max_position_size"]
            self.stop_loss_atr_multiple = self.regime_params["bear_market"]["stop_loss_atr_multiple"]
            self.trailing_stop_percentage = self.regime_params["bear_market"]["trailing_stop_percentage"]
            self.Debug("Applied Bear Market parameters")
            
        elif self.current_regime_id == MarketRegime.SidewaysMarket.value:
            # Optimized for range-bound markets
            self.max_position_size = self.regime_params["sideways_market"]["max_position_size"]
            self.stop_loss_atr_multiple = self.regime_params["sideways_market"]["stop_loss_atr_multiple"]
            self.trailing_stop_percentage = self.regime_params["sideways_market"]["trailing_stop_percentage"]
            self.Debug("Applied Sideways Market parameters")
            
        elif self.current_regime_id == MarketRegime.HighVolatility.value:
            # Adapted for high volatility
            self.max_position_size = self.regime_params["high_volatility"]["max_position_size"]
            self.stop_loss_atr_multiple = self.regime_params["high_volatility"]["stop_loss_atr_multiple"]
            self.trailing_stop_percentage = self.regime_params["high_volatility"]["trailing_stop_percentage"]
            self.Debug("Applied High Volatility parameters")
            
        elif self.current_regime_id == MarketRegime.LowVolatility.value:
            # Optimized for low volatility
            self.max_position_size = self.regime_params["low_volatility"]["max_position_size"]
            self.stop_loss_atr_multiple = self.regime_params["low_volatility"]["stop_loss_atr_multiple"]
            self.trailing_stop_percentage = self.regime_params["low_volatility"]["trailing_stop_percentage"]
            self.Debug("Applied Low Volatility parameters")
            
        elif self.current_regime_id == MarketRegime.Crisis.value:
            # Very conservative in crisis
            self.max_position_size = self.regime_params["crisis"]["max_position_size"]
            self.stop_loss_atr_multiple = self.regime_params["crisis"]["stop_loss_atr_multiple"]
            self.trailing_stop_percentage = self.regime_params["crisis"]["trailing_stop_percentage"]
            self.Debug("Applied Crisis parameters")

{on_data}

{helper_methods}
"""
    
    def _generate_regime_params(self):
        """Generate regime-specific parameters."""
        regime_params = []
        
        regime_params.append("        # Market regime detection parameters")
        regime_params.append("        self.regime_detection_method = \"" + self.regime_detection_method + "\"")
        regime_params.append("        self.num_regimes = " + str(self.num_regimes))
        regime_params.append("        self.feature_lookback_period = 252  # One year of data")
        regime_params.append("        self.feature_lookback_period_ready = False")
        regime_params.append("        self.regime_scaler = StandardScaler()")
        regime_params.append("        self.current_regime_id = MarketRegime.BullMarket.value  # Default starting regime")
        regime_params.append("        self.current_regime_name = MarketRegime(self.current_regime_id).name")
        
        # Define regime-specific parameters
        regime_params.append("\n        # Regime-specific parameters")
        regime_params.append("        self.regime_params = {")
        
        # Bull Market parameters
        regime_params.append("            \"bull_market\": {")
        regime_params.append("                \"max_position_size\": 0.08,  # More aggressive")
        regime_params.append("                \"stop_loss_atr_multiple\": 2.5,  # Wider stops")
        regime_params.append("                \"trailing_stop_percentage\": 0.03,  # Wider trailing stop")
        regime_params.append("            },")
        
        # Bear Market parameters
        regime_params.append("            \"bear_market\": {")
        regime_params.append("                \"max_position_size\": 0.03,  # More conservative")
        regime_params.append("                \"stop_loss_atr_multiple\": 1.8,  # Tighter stops")
        regime_params.append("                \"trailing_stop_percentage\": 0.015,  # Tighter trailing stop")
        regime_params.append("            },")
        
        # Sideways Market parameters
        regime_params.append("            \"sideways_market\": {")
        regime_params.append("                \"max_position_size\": 0.04,  # Moderate")
        regime_params.append("                \"stop_loss_atr_multiple\": 1.5,  # Tighter stops for range-bound")
        regime_params.append("                \"trailing_stop_percentage\": 0.02,  # Standard trailing stop")
        regime_params.append("            },")
        
        # High Volatility parameters
        regime_params.append("            \"high_volatility\": {")
        regime_params.append("                \"max_position_size\": 0.03,  # Conservative")
        regime_params.append("                \"stop_loss_atr_multiple\": 3.0,  # Wider stops for volatility")
        regime_params.append("                \"trailing_stop_percentage\": 0.025,  # Moderate trailing stop")
        regime_params.append("            },")
        
        # Low Volatility parameters
        regime_params.append("            \"low_volatility\": {")
        regime_params.append("                \"max_position_size\": 0.07,  # More aggressive")
        regime_params.append("                \"stop_loss_atr_multiple\": 1.2,  # Tighter stops")
        regime_params.append("                \"trailing_stop_percentage\": 0.015,  # Tighter trailing stop")
        regime_params.append("            },")
        
        # Crisis parameters
        regime_params.append("            \"crisis\": {")
        regime_params.append("                \"max_position_size\": 0.02,  # Very conservative")
        regime_params.append("                \"stop_loss_atr_multiple\": 4.0,  # Very wide stops for extreme volatility")
        regime_params.append("                \"trailing_stop_percentage\": 0.04,  # Wider trailing stop for volatility")
        regime_params.append("            }")
        regime_params.append("        }")
        
        return "\n".join(regime_params)
    
    def _generate_regime_indicators(self):
        """Generate market regime detection indicators."""
        indicators = []
        
        indicators.append("        # Initialize volatility indicators for regime detection")
        indicators.append("        self.volatility_window = 30")
        indicators.append("        self.relative_strength = RelativeStrengthIndex(14)")
        indicators.append("        self.standard_deviation = StandardDeviation(self.volatility_window)")
        indicators.append("        self.bollinger_bandwidth = BollingerBands(self.volatility_window, 2)")
        
        indicators.append("\n        # Register indicators for regime detection")
        indicators.append("        self.RegisterIndicator(self.symbol, self.relative_strength, Resolution.Daily)")
        indicators.append("        self.RegisterIndicator(self.symbol, self.standard_deviation, Resolution.Daily)")
        indicators.append("        self.RegisterIndicator(self.symbol, self.bollinger_bandwidth, Resolution.Daily)")
        
        indicators.append("\n        # Custom indicator tracking")
        indicators.append("        self.regime_feature_history = []  # Track features for regime detection")
        
        return "\n".join(indicators)
    
    def _generate_tracking_vars(self):
        """Generate tracking variables including regime-specific ones."""
        tracking_vars = super()._generate_tracking_vars()
        
        # Add regime tracking variables
        additional_vars = [
            "",
            "# Regime tracking",
            "self.last_regime_change_date = self.Time",
            "self.regime_history = []  # Track regime changes over time"
        ]
        
        return tracking_vars + "\n        ".join(additional_vars)
    
    def _generate_helper_methods(self):
        """Generate helper methods including regime-specific ones."""
        helper_methods = super()._generate_helper_methods()
        
        # Add regime helper methods
        additional_methods = [
            "",
            "def OnEndOfDay(self):",
            "    '''Track regime features at end of day.'''",
            "    # Update feature history when indicators are ready",
            "    if (self.standard_deviation.IsReady and self.relative_strength.IsReady and",
            "            self.bollinger_bandwidth.IsReady):",
            "        # Check if warmup period is complete",
            "        if not self.feature_lookback_period_ready and len(self.regime_feature_history) >= self.feature_lookback_period:",
            "            self.feature_lookback_period_ready = True",
            "            self.Debug(\"Feature lookback period ready for regime detection\")",
            "",
            "        # Get current price",
            "        current_price = self.Securities[self.symbol].Price",
            "",
            "        # Log regime status",
            "        self.Log(f\"Current Regime: {self.current_regime_name}, RSI: {self.relative_strength.Current.Value:.2f}, Volatility: {self.standard_deviation.Current.Value:.4f}\")",
            "",
            "        # Record regime state to charts",
            "        self.Plot(\"Regimes\", \"Current Regime ID\", self.current_regime_id)",
            "        self.Plot(\"Strategy Parameters\", \"Max Position Size\", self.max_position_size)",
            "        self.Plot(\"Strategy Parameters\", \"Stop Loss ATR Multiple\", self.stop_loss_atr_multiple)"
        ]
        
        return helper_methods + "\n    ".join(additional_methods)
    
    def generate_algorithm(self, start_date=None, end_date=None, cash=100000):
        """
        Generate a QuantConnect algorithm with market regime detection.
        
        Args:
            start_date (str): Backtest start date (YYYY-MM-DD)
            end_date (str): Backtest end date (YYYY-MM-DD)
            cash (int): Initial capital
            
        Returns:
            str: Generated algorithm code
        """
        if not self.strategy_data:
            self.load_strategy()
            
        strategy_name = self.strategy_data['strategy']['Strategy Name']
        class_name = ''.join(x for x in strategy_name if x.isalnum()) + "Algorithm"
        
        # Set dates for backtesting
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        start_year, start_month, start_day = start_date.split('-')
        end_year, end_month, end_day = end_date.split('-')
        
        # Generate parameters section
        params = self._generate_parameters()
        
        # Generate regime parameters
        regime_params = self._generate_regime_params()
        
        # Generate universe section (default to S&P 500)
        universe = self._generate_universe()
        
        # Generate indicators section
        indicators = self._generate_indicators()
        
        # Generate regime detection indicators
        regime_indicators = self._generate_regime_indicators()
        
        # Generate tracking variables
        tracking_vars = self._generate_tracking_vars()
        
        # Generate OnData method
        on_data = self._generate_on_data()
        
        # Generate helper methods
        helper_methods = self._generate_helper_methods()
        
        # Fill template
        algorithm_code = self.template.format(
            class_name=class_name,
            start_year=start_year,
            start_month=start_month,
            start_day=start_day,
            end_year=end_year,
            end_month=end_month,
            end_day=end_day,
            initial_cash=cash,
            params=params,
            regime_params=regime_params,
            universe=universe,
            indicators=indicators,
            regime_indicators=regime_indicators,
            warmup_period=252,  # Longer warmup for regime detection
            tracking_vars=tracking_vars,
            on_data=on_data,
            helper_methods=helper_methods
        )
        
        # Save to file
        output_file = os.path.join(self.output_path, f"RegimeAware{class_name}.py")
        with open(output_file, 'w') as f:
            f.write(algorithm_code)
            
        logger.info(f"Generated Regime-Aware QuantConnect algorithm: {output_file}")
        return algorithm_code


class QuantConnectStrategy:
    """Helper class to generate a specific strategy using the adapter."""
    
    @staticmethod
    def generate_momentum_rsi_volatility(strategy_json, output_path, start_date=None, end_date=None, cash=100000):
        """
        Generate a Momentum RSI Volatility strategy for QuantConnect.
        
        Args:
            strategy_json (str): Path to strategy JSON file
            output_path (str): Directory to save generated algorithm
            start_date (str): Backtest start date (YYYY-MM-DD)
            end_date (str): Backtest end date (YYYY-MM-DD)
            cash (int): Initial capital
        
        Returns:
            str: Path to generated algorithm file
        """
        adapter = QuantConnectAdapter(strategy_json, output_path)
        adapter.load_strategy()
        algorithm_code = adapter.generate_algorithm(start_date, end_date, cash)
        
        return os.path.join(output_path, f"{adapter.strategy_data['strategy']['Strategy Name'].replace(' ', '')}Algorithm.py")
    
    @staticmethod
    def generate_regime_aware_strategy(strategy_json, output_path, start_date=None, end_date=None, 
                                      cash=100000, num_regimes=4, regime_detection_method="hmm"):
        """
        Generate a Regime-Aware strategy for QuantConnect.
        
        Args:
            strategy_json (str): Path to strategy JSON file
            output_path (str): Directory to save generated algorithm
            start_date (str): Backtest start date (YYYY-MM-DD)
            end_date (str): Backtest end date (YYYY-MM-DD)
            cash (int): Initial capital
            num_regimes (int): Number of market regimes to detect
            regime_detection_method (str): Method for regime detection ('hmm', 'kmeans')
        
        Returns:
            str: Path to generated algorithm file
        """
        adapter = MarketRegimeDetectionAdapter(
            strategy_json, 
            output_path,
            num_regimes=num_regimes,
            regime_detection_method=regime_detection_method
        )
        adapter.load_strategy()
        algorithm_code = adapter.generate_algorithm(start_date, end_date, cash)
        
        return os.path.join(output_path, f"RegimeAware{adapter.strategy_data['strategy']['Strategy Name'].replace(' ', '')}Algorithm.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Mathematricks strategies to QuantConnect format with market regime detection")
    parser.add_argument("--strategy", required=True, help="Path to strategy JSON file")
    parser.add_argument("--output", default="qc_algorithms", help="Output directory for generated algorithm")
    parser.add_argument("--start", default=None, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=int, default=100000, help="Initial capital")
    parser.add_argument("--regimes", type=int, default=4, help="Number of market regimes to detect")
    parser.add_argument("--method", default="hmm", choices=["hmm", "kmeans"], help="Regime detection method")
    parser.add_argument("--regime-aware", action="store_true", help="Generate regime-aware strategy")
    
    args = parser.parse_args()
    
    # Generate algorithm
    if args.regime_aware:
        qc_file = QuantConnectStrategy.generate_regime_aware_strategy(
            args.strategy, 
            args.output, 
            args.start, 
            args.end, 
            args.cash,
            args.regimes,
            args.method
        )
        print(f"Generated Regime-Aware QuantConnect algorithm: {qc_file}")
    else:
        qc_file = QuantConnectStrategy.generate_momentum_rsi_volatility(
            args.strategy, 
            args.output, 
            args.start, 
            args.end, 
            args.cash
        )
        print(f"Generated QuantConnect algorithm: {qc_file}")