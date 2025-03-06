"""
Multi-asset strategy generation module.

This module provides tools to generate trading strategies that operate
across multiple asset classes and markets.
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt
import uuid
import re

from .correlations import CrossMarketCorrelationAnalyzer
from .opportunities import OpportunityDetector

logger = logging.getLogger(__name__)

class MultiAssetStrategyGenerator:
    """
    Generates multi-asset trading strategies for cross-market opportunities.
    
    This class creates trading strategies that operate across multiple 
    asset classes and markets, leveraging cross-market relationships
    and opportunities.
    """
    
    def __init__(self, 
                strategy_template_dir: str = None,
                output_dir: str = None,
                min_confidence: str = 'medium'):
        """
        Initialize the strategy generator.
        
        Args:
            strategy_template_dir: Directory containing strategy templates
            output_dir: Directory to write generated strategies
            min_confidence: Minimum confidence level for opportunities
        """
        self.strategy_template_dir = strategy_template_dir
        self.output_dir = output_dir
        self.min_confidence = min_confidence
        self.correlation_analyzer = None
        self.opportunity_detector = None
        self.templates = {
            'lead_lag': None,
            'pairs_trading': None,
            'cross_asset': None
        }
        
        # Initialize opportunity detector
        self.opportunity_detector = OpportunityDetector()
        
        # Load templates if directory provided
        if strategy_template_dir and os.path.exists(strategy_template_dir):
            self._load_templates()
        
    def set_correlation_analyzer(self, analyzer: CrossMarketCorrelationAnalyzer) -> None:
        """
        Set the correlation analyzer to use.
        
        Args:
            analyzer: CrossMarketCorrelationAnalyzer instance
        """
        self.correlation_analyzer = analyzer
        
        # Also set in opportunity detector
        if self.opportunity_detector:
            self.opportunity_detector.set_correlation_analyzer(analyzer)
    
    def _load_templates(self) -> None:
        """Load strategy templates from template directory."""
        if not self.strategy_template_dir:
            logger.warning("No template directory provided, using default templates")
            return
            
        for strategy_type in self.templates.keys():
            template_path = os.path.join(
                self.strategy_template_dir, 
                f"{strategy_type}_template.py"
            )
            
            if os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    self.templates[strategy_type] = f.read()
                logger.info(f"Loaded {strategy_type} template from {template_path}")
            else:
                logger.warning(f"No template found at {template_path}, using default")
                
    def _get_default_template(self, strategy_type: str) -> str:
        """
        Get default template for a strategy type.
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            Template string
        """
        if strategy_type == 'lead_lag':
            return """
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from AlgorithmImports import *

class {class_name}(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate({start_year}, {start_month}, {start_day})  # Set Start Date
        self.SetCash({initial_cash})  # Set Strategy Cash
        
        # Add securities
        self.lead_symbol = self.AddEquity("{lead_symbol}", Resolution.Daily).Symbol
        self.lag_symbol = self.AddEquity("{lag_symbol}", Resolution.Daily).Symbol
        
        # Set up lag window for tracking lead asset
        self.lag_window = {lag_window}
        self.history_window = max(200, self.lag_window * 2)  # History window needs to be at least 2x lag
        self.lead_history = RollingWindow[float](self.history_window)
        
        # Set up parameters
        self.lookback = {lookback}
        self.entry_threshold = {entry_threshold}
        self.exit_threshold = {exit_threshold}
        self.stop_loss = {stop_loss}
        self.correlation_sign = {correlation_sign}  # 1 for positive correlation, -1 for negative
        
        # Schedule rebalancing
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.AfterMarketOpen(), self.Rebalance)
        
    def OnData(self, data):
        # Update lead history
        if data.ContainsKey(self.lead_symbol) and data[self.lead_symbol] is not None:
            self.lead_history.Add(data[self.lead_symbol].Close)
            
    def Rebalance(self):
        # Skip if lead history is not full
        if not self.lead_history.IsReady:
            return
            
        # Calculate lead asset return over lag window
        if len(self.lead_history) < self.lag_window + 1:
            return
            
        lead_current = self.lead_history[0]
        lead_lagged = self.lead_history[self.lag_window]
        lead_return = (lead_current / lead_lagged) - 1
        
        # Get current allocation
        lag_position = self.Portfolio[self.lag_symbol].Quantity
        
        # Determine target position based on lead asset movement
        if self.correlation_sign * lead_return > self.entry_threshold and lag_position == 0:
            # Enter long position in lag asset
            self.SetHoldings(self.lag_symbol, 0.9)  # 90% allocation
            self.Log(f"Entering position in {{self.lag_symbol}} based on {{self.lead_symbol}} movement of {{lead_return:.2%}}")
            
        elif self.correlation_sign * lead_return < -self.exit_threshold and lag_position > 0:
            # Exit position in lag asset
            self.Liquidate(self.lag_symbol)
            self.Log(f"Exiting position in {{self.lag_symbol}} based on {{self.lead_symbol}} movement of {{lead_return:.2%}}")
            
        # Stop loss check
        if lag_position > 0:
            entry_price = self.Portfolio[self.lag_symbol].AveragePrice
            current_price = self.Securities[self.lag_symbol].Price
            position_return = (current_price / entry_price) - 1
            
            if position_return < -self.stop_loss:
                self.Liquidate(self.lag_symbol)
                self.Log(f"Stop loss triggered for {{self.lag_symbol}} at {{position_return:.2%}}")
"""
        elif strategy_type == 'pairs_trading':
            return """
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from AlgorithmImports import *

class {class_name}(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate({start_year}, {start_month}, {start_day})  # Set Start Date
        self.SetCash({initial_cash})  # Set Strategy Cash
        
        # Add securities
        self.symbol1 = self.AddEquity("{symbol1}", Resolution.Daily).Symbol
        self.symbol2 = self.AddEquity("{symbol2}", Resolution.Daily).Symbol
        
        # Set up parameters
        self.lookback = {lookback}
        self.hedge_ratio = {hedge_ratio}
        self.entry_threshold = {entry_threshold}
        self.exit_threshold = {exit_threshold}
        self.stop_loss = {stop_loss}
        self.max_holding_days = {max_holding_days}
        
        # Initialize state variables
        self.in_position = False
        self.position_direction = 0  # 1 for long S1/short S2, -1 for short S1/long S2
        self.entry_time = None
        self.entry_spread = 0
        
        # Historical data for z-score calculation
        self.spread_history = RollingWindow[float](self.lookback)
        self.history_loaded = False
        
        # Schedule rebalancing
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.AfterMarketOpen(), self.Rebalance)
        
        # Load historical data
        history = self.History(self.Securities.Keys, self.lookback)
        if len(history) >= self.lookback:
            history_by_symbol = history.GroupBy("Symbol")
            self.PreloadHistoricalData(history_by_symbol)
        
    def PreloadHistoricalData(self, history_by_symbol):
        # Preload historical spread data.
        if self.symbol1 not in history_by_symbol or self.symbol2 not in history_by_symbol:
            return
            
        symbol1_prices = []
        symbol2_prices = []
        
        # Extract historical prices
        for bar in history_by_symbol[self.symbol1]:
            symbol1_prices.append(bar.Close)
            
        for bar in history_by_symbol[self.symbol2]:
            symbol2_prices.append(bar.Close)
            
        # Calculate spreads and add to history window
        for i in range(min(len(symbol1_prices), len(symbol2_prices))):
            spread = symbol1_prices[i] - self.hedge_ratio * symbol2_prices[i]
            self.spread_history.Add(spread)
            
        self.history_loaded = self.spread_history.IsReady
        
    def OnData(self, data):
        # Skip if we don't have both symbols
        if not (data.ContainsKey(self.symbol1) and data.ContainsKey(self.symbol2)):
            return
            
        # Calculate current spread
        price1 = data[self.symbol1].Close
        price2 = data[self.symbol2].Close
        spread = price1 - self.hedge_ratio * price2
        
        # Add to history
        self.spread_history.Add(spread)
        
        # Skip further processing if history isn't loaded
        if not self.history_loaded and not self.spread_history.IsReady:
            return
            
        self.history_loaded = True
    
    def Rebalance(self):
        # Skip if historical data isn't ready
        if not self.history_loaded:
            return
            
        # Current spread
        price1 = self.Securities[self.symbol1].Price
        price2 = self.Securities[self.symbol2].Price
        current_spread = price1 - self.hedge_ratio * price2
        
        # Calculate z-score
        spread_values = [self.spread_history[i] for i in range(self.spread_history.Count)]
        spread_mean = np.mean(spread_values)
        spread_std = np.std(spread_values)
        
        if spread_std == 0:
            return
            
        z_score = (current_spread - spread_mean) / spread_std
        
        # Check if we should exit an existing position
        if self.in_position:
            # Check max holding period
            if (self.Time - self.entry_time).days >= self.max_holding_days:
                self.ExitPosition()
                self.Log(f"Exiting position due to max holding period reached")
                return
                
            # Check if spread reverted to mean
            if (self.position_direction == 1 and z_score <= self.exit_threshold) or \
               (self.position_direction == -1 and z_score >= -self.exit_threshold):
                self.ExitPosition()
                self.Log(f"Exiting position as spread reverted to mean, z-score: {{z_score:.2f}}")
                return
                
            # Check stop loss - spread moved further away
            current_distance = abs(current_spread - spread_mean)
            entry_distance = abs(self.entry_spread - spread_mean)
            
            if current_distance > entry_distance * (1 + self.stop_loss):
                self.ExitPosition()
                self.Log(f"Exiting position due to stop loss, spread moved further away")
                return
                
        # Check if we should enter a new position
        else:
            # Enter if spread is sufficiently far from mean
            if z_score >= self.entry_threshold:
                # Spread is high: short symbol1, long symbol2
                self.EnterPosition(-1)
                self.Log(f"Entering position: short {{self.symbol1}}, long {{self.symbol2}}, z-score: {{z_score:.2f}}")
                
            elif z_score <= -self.entry_threshold:
                # Spread is low: long symbol1, short symbol2
                self.EnterPosition(1)
                self.Log(f"Entering position: long {{self.symbol1}}, short {{self.symbol2}}, z-score: {{z_score:.2f}}")
    
    def EnterPosition(self, direction):
        # Enter a pairs trade position.
        # Calculate notional value for each leg
        price1 = self.Securities[self.symbol1].Price
        price2 = self.Securities[self.symbol2].Price
        
        # Record entry details
        self.in_position = True
        self.position_direction = direction
        self.entry_time = self.Time
        self.entry_spread = price1 - self.hedge_ratio * price2
        
        # Calculate position sizes assuming equal dollar amount on each side
        portfolio_value = self.Portfolio.TotalPortfolioValue
        trade_value = portfolio_value * 0.4  # 40% of portfolio per leg (80% total)
        
        # Calculate number of shares
        shares1 = trade_value / price1
        shares2 = (trade_value / price2) * self.hedge_ratio
        
        # Enter positions
        if direction == 1:
            self.SetHoldings(self.symbol1, 0.4)  # Long
            self.SetHoldings(self.symbol2, -0.4 * self.hedge_ratio)  # Short
        else:
            self.SetHoldings(self.symbol1, -0.4)  # Short
            self.SetHoldings(self.symbol2, 0.4 * self.hedge_ratio)  # Long
    
    def ExitPosition(self):
        # Exit current pairs trade position.
        if self.in_position:
            self.Liquidate(self.symbol1)
            self.Liquidate(self.symbol2)
            self.in_position = False
            self.position_direction = 0
"""
        elif strategy_type == 'cross_asset':
            return """
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from AlgorithmImports import *

class {class_name}(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate({start_year}, {start_month}, {start_day})  # Set Start Date
        self.SetCash({initial_cash})  # Set Strategy Cash
        
        # Add securities from different asset classes
        self.symbols = []
        {symbol_initialization}
        
        # Set up parameters
        self.lookback = {lookback}
        self.rebalance_days = {rebalance_days}
        self.position_weights = {position_weights}
        self.allocation_method = "{allocation_method}"  # 'equal', 'risk_parity', 'momentum'
        self.last_rebalance = datetime(1900, 1, 1)  # Initial value
        
        # Track correlations for monitoring
        self.correlation_window = {correlation_window}
        self.price_histories = {{}}
        
        # Initialize price histories for correlation tracking
        for symbol in self.symbols:
            self.price_histories[symbol] = RollingWindow[float](self.correlation_window)
        
        # Schedule rebalance
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.AfterMarketOpen(), self.Rebalance)
        
        # Load historical data for correlation analysis
        history = self.History(self.Securities.Keys, self.correlation_window)
        if not history.empty:
            history_by_symbol = history.GroupBy("Symbol")
            self.PreloadHistoricalData(history_by_symbol)
        
    def PreloadHistoricalData(self, history_by_symbol):
        # Preload historical price data.
        for symbol in self.symbols:
            if symbol not in history_by_symbol:
                continue
                
            for bar in history_by_symbol[symbol]:
                self.price_histories[symbol].Add(bar.Close)
    
    def OnData(self, data):
        # Update price histories
        for symbol in self.symbols:
            if data.ContainsKey(symbol) and data[symbol] is not None:
                self.price_histories[symbol].Add(data[symbol].Close)
    
    def Rebalance(self):
        # Check if enough time has passed since last rebalance
        if (self.Time - self.last_rebalance).days < self.rebalance_days:
            return
            
        # Update correlations
        self.UpdateCorrelations()
        
        # Calculate target weights
        target_weights = self.CalculateTargetWeights()
        
        # Set holdings based on target weights
        for symbol, weight in target_weights.items():
            self.SetHoldings(symbol, weight)
            
        self.last_rebalance = self.Time
        self.Log(f"Rebalanced portfolio with weights: {{target_weights}}")
    
    def UpdateCorrelations(self):
        # Update correlation matrix between assets.
        # Skip if we don't have enough data
        all_ready = all(self.price_histories[symbol].IsReady for symbol in self.symbols)
        if not all_ready:
            return
            
        # Create price dictionary
        prices = {{}}
        for symbol in self.symbols:
            prices[str(symbol)] = [self.price_histories[symbol][i] for i in range(self.price_histories[symbol].Count)]
            
        # Create DataFrame for correlation
        df = pd.DataFrame(prices)
        
        # Calculate correlations
        self.correlation_matrix = df.corr()
        
        # Log significant correlation changes
        # Implementation depends on how you want to track correlation changes
    
    def CalculateTargetWeights(self):
        # Calculate target weights for each asset.
        weights = {{}}
        
        if self.allocation_method == 'equal':
            # Equal weighting
            weight = 1.0 / len(self.symbols)
            for symbol in self.symbols:
                weights[symbol] = weight
                
        elif self.allocation_method == 'risk_parity':
            # Simple risk parity based on inverse volatility
            vols = {{}}
            
            # Calculate volatilities
            for symbol in self.symbols:
                if not self.price_histories[symbol].IsReady:
                    weights[symbol] = 0
                    continue
                    
                prices = [self.price_histories[symbol][i] for i in range(self.price_histories[symbol].Count)]
                returns = [prices[i] / prices[i+1] - 1 for i in range(len(prices) - 1)]
                vols[symbol] = np.std(returns) if len(returns) > 0 else 1
            
            # Inverse volatility weights
            total_inv_vol = sum(1/vol if vol > 0 else 0 for vol in vols.values())
            
            if total_inv_vol > 0:
                for symbol in self.symbols:
                    weights[symbol] = (1/vols[symbol]) / total_inv_vol if vols[symbol] > 0 else 0
            else:
                # Fallback to equal weighting
                weight = 1.0 / len(self.symbols)
                for symbol in self.symbols:
                    weights[symbol] = weight
                    
        elif self.allocation_method == 'momentum':
            # Momentum-based weighting
            returns = {{}}
            
            # Calculate returns over lookback period
            for symbol in self.symbols:
                if not self.price_histories[symbol].IsReady:
                    weights[symbol] = 0
                    continue
                    
                current_price = self.Securities[symbol].Price
                history_price = self.price_histories[symbol][min(self.lookback, self.price_histories[symbol].Count - 1)]
                
                returns[symbol] = current_price / history_price - 1
            
            # Allocate to positive momentum assets
            positive_symbols = [s for s, r in returns.items() if r > 0]
            
            if positive_symbols:
                weight = 1.0 / len(positive_symbols)
                for symbol in self.symbols:
                    weights[symbol] = weight if symbol in positive_symbols else 0
            else:
                # No positive momentum, stay in cash
                for symbol in self.symbols:
                    weights[symbol] = 0
                    
        else:
            # Fallback to equal weighting
            weight = 1.0 / len(self.symbols)
            for symbol in self.symbols:
                weights[symbol] = weight
                
        # Apply position weights multiplier
        for symbol in weights:
            weights[symbol] *= self.position_weights.get(str(symbol), 1.0)
            
        return weights
"""
        else:
            logger.error(f"No default template available for {strategy_type}")
            return ""
    
    def generate_lead_lag_strategy(self, opportunity: Dict) -> Dict:
        """
        Generate a lead-lag strategy.
        
        Args:
            opportunity: Lead-lag opportunity from OpportunityDetector
            
        Returns:
            Dictionary with strategy details and code
        """
        # Extract key information
        lead_symbol = opportunity.get('lead_asset')
        lag_symbol = opportunity.get('lag_asset')
        optimal_lag = opportunity.get('optimal_lag', 1)
        correlation = opportunity.get('correlation', 0)
        
        if not lead_symbol or not lag_symbol or not optimal_lag:
            logger.error("Missing required information for lead-lag strategy")
            return {}
            
        # Setup strategy parameters
        strategy_params = {
            'class_name': f"LeadLag_{lead_symbol}_{lag_symbol}_Strategy",
            'start_year': datetime.now().year - 3,
            'start_month': 1,
            'start_day': 1,
            'initial_cash': 100000,
            'lead_symbol': lead_symbol,
            'lag_symbol': lag_symbol,
            'lag_window': abs(optimal_lag),
            'lookback': 252,  # One year of trading days
            'entry_threshold': 0.02,  # 2% move in lead asset
            'exit_threshold': 0.01,  # 1% reverse move in lead asset
            'stop_loss': 0.05,  # 5% stop loss
            'correlation_sign': 1 if correlation >= 0 else -1
        }
        
        # Get strategy template
        template = self.templates.get('lead_lag')
        if not template:
            template = self._get_default_template('lead_lag')
            
        # Fill in template
        strategy_code = template.format(**strategy_params)
        
        # Generate full strategy info
        strategy_info = {
            'type': 'lead_lag',
            'name': strategy_params['class_name'],
            'description': f"Lead-Lag strategy exploiting the relationship between {lead_symbol} and {lag_symbol}",
            'parameters': strategy_params,
            'opportunity': opportunity,
            'code': strategy_code,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'id': str(uuid.uuid4())
        }
        
        return strategy_info
    
    def generate_pairs_trading_strategy(self, opportunity: Dict) -> Dict:
        """
        Generate a pairs trading strategy.
        
        Args:
            opportunity: Pairs trading opportunity from OpportunityDetector
            
        Returns:
            Dictionary with strategy details and code
        """
        # Extract key information
        symbol1 = opportunity.get('symbol1')
        symbol2 = opportunity.get('symbol2')
        hedge_ratio = opportunity.get('hedge_ratio', 1.0)
        half_life = opportunity.get('half_life')
        
        if not symbol1 or not symbol2 or not half_life:
            logger.error("Missing required information for pairs trading strategy")
            return {}
            
        # Setup strategy parameters
        strategy_params = {
            'class_name': f"PairsTrading_{symbol1}_{symbol2}_Strategy",
            'start_year': datetime.now().year - 3,
            'start_month': 1,
            'start_day': 1,
            'initial_cash': 100000,
            'symbol1': symbol1,
            'symbol2': symbol2,
            'hedge_ratio': hedge_ratio,
            'lookback': max(60, int(half_life * 4)),  # 4x half-life for statistical significance
            'entry_threshold': 2.0,  # Enter at 2 standard deviations
            'exit_threshold': 0.5,  # Exit at 0.5 standard deviations
            'stop_loss': 0.5,  # 50% stop loss on spread divergence
            'max_holding_days': int(half_life * 3)  # 3x half-life as max holding period
        }
        
        # Get strategy template
        template = self.templates.get('pairs_trading')
        if not template:
            template = self._get_default_template('pairs_trading')
            
        # Fill in template
        strategy_code = template.format(**strategy_params)
        
        # Generate full strategy info
        strategy_info = {
            'type': 'pairs_trading',
            'name': strategy_params['class_name'],
            'description': f"Pairs Trading strategy exploiting cointegration between {symbol1} and {symbol2}",
            'parameters': strategy_params,
            'opportunity': opportunity,
            'code': strategy_code,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'id': str(uuid.uuid4())
        }
        
        return strategy_info
    
    def generate_cross_asset_strategy(self, opportunities: List[Dict], name: str = None) -> Dict:
        """
        Generate a cross-asset strategy using multiple opportunities.
        
        Args:
            opportunities: List of opportunities from OpportunityDetector
            name: Name for the strategy
            
        Returns:
            Dictionary with strategy details and code
        """
        if not opportunities:
            logger.error("No opportunities provided for cross-asset strategy")
            return {}
            
        # Collect unique symbols across all opportunities
        symbols = set()
        for opp in opportunities:
            if 'symbol1' in opp and 'symbol2' in opp:
                symbols.add(opp.get('symbol1'))
                symbols.add(opp.get('symbol2'))
            elif 'lead_asset' in opp and 'lag_asset' in opp:
                symbols.add(opp.get('lead_asset'))
                symbols.add(opp.get('lag_asset'))
            elif 'symbol' in opp:
                symbols.add(opp.get('symbol'))
        
        if not symbols:
            logger.error("No valid symbols found in opportunities")
            return {}
            
        # Generate symbols list
        symbols = list(symbols)
        
        # Generate symbol initialization code
        symbol_initialization = ""
        for i, symbol in enumerate(symbols):
            symbol_initialization += f"        self.symbols.append(self.AddEquity('{symbol}', Resolution.Daily).Symbol)\n"
            
        # Create position weights dictionary
        # Start with equal weights and adjust based on opportunities
        position_weights = {str(symbol): 1.0 for symbol in symbols}
        
        # Setup strategy parameters
        strategy_params = {
            'class_name': name or f"CrossAsset_Strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_year': datetime.now().year - 3,
            'start_month': 1,
            'start_day': 1,
            'initial_cash': 100000,
            'symbol_initialization': symbol_initialization,
            'symbols': symbols,
            'lookback': 63,  # ~3 months of trading days
            'rebalance_days': 21,  # Monthly rebalancing
            'position_weights': position_weights,
            'allocation_method': 'risk_parity',  # Use risk parity by default
            'correlation_window': 252  # One year of trading days
        }
        
        # Get strategy template
        template = self.templates.get('cross_asset')
        if not template:
            template = self._get_default_template('cross_asset')
            
        # Fill in template
        strategy_code = template.format(**strategy_params)
        
        # Generate full strategy info
        strategy_info = {
            'type': 'cross_asset',
            'name': strategy_params['class_name'],
            'description': f"Cross-Asset strategy using {len(symbols)} assets from different markets",
            'parameters': strategy_params,
            'opportunities': opportunities,
            'code': strategy_code,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'id': str(uuid.uuid4())
        }
        
        return strategy_info
    
    def generate_all_strategies(self, opportunities: Dict[str, List[Dict]] = None) -> Dict[str, List[Dict]]:
        """
        Generate strategies for all detected opportunities.
        
        Args:
            opportunities: Dictionary of opportunities (from OpportunityDetector)
            
        Returns:
            Dictionary mapping strategy types to lists of strategy info
        """
        if not self.correlation_analyzer:
            logger.error("Correlation analyzer not set")
            return {}
            
        # Make sure opportunity detector is set up
        if not self.opportunity_detector:
            self.opportunity_detector = OpportunityDetector()
            self.opportunity_detector.set_correlation_analyzer(self.correlation_analyzer)
            
        # Get opportunities if not provided
        if not opportunities:
            opportunities = self.opportunity_detector.find_all_opportunities(min_confidence=self.min_confidence)
            
        if not opportunities:
            logger.error("No opportunities found")
            return {}
            
        # Generate strategies for each opportunity type
        strategies = {}
        
        # Generate lead-lag strategies
        if 'lead_lag' in opportunities:
            lead_lag_strategies = []
            for opp in opportunities['lead_lag']:
                strategy = self.generate_lead_lag_strategy(opp)
                if strategy:
                    lead_lag_strategies.append(strategy)
                    
            if lead_lag_strategies:
                strategies['lead_lag'] = lead_lag_strategies
                
        # Generate pairs trading strategies
        if 'pairs_trading' in opportunities:
            pairs_strategies = []
            for opp in opportunities['pairs_trading']:
                strategy = self.generate_pairs_trading_strategy(opp)
                if strategy:
                    pairs_strategies.append(strategy)
                    
            if pairs_strategies:
                strategies['pairs_trading'] = pairs_strategies
                
        # Generate cross-asset strategies
        # For cross-asset, we might group several opportunities into a single strategy
        if 'cross_asset_correlation' in opportunities:
            cross_asset_opp = opportunities['cross_asset_correlation']
            
            if cross_asset_opp:
                # Generate one cross-asset strategy using multiple opportunities
                strategy = self.generate_cross_asset_strategy(cross_asset_opp)
                if strategy:
                    strategies['cross_asset'] = [strategy]
        
        return strategies
    
    def save_strategy(self, strategy: Dict, output_dir: str = None) -> str:
        """
        Save a generated strategy to disk.
        
        Args:
            strategy: Strategy information dictionary
            output_dir: Directory to save strategy (uses self.output_dir if None)
            
        Returns:
            Path to saved strategy file
        """
        output_dir = output_dir or self.output_dir
        
        if not output_dir:
            logger.error("No output directory specified")
            return ""
            
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create filename
        strategy_name = re.sub(r'[^a-zA-Z0-9_]', '_', strategy.get('name', 'strategy'))
        code_filename = f"{strategy_name}.py"
        info_filename = f"{strategy_name}_info.json"
        
        # Save strategy code
        code_path = os.path.join(output_dir, code_filename)
        with open(code_path, 'w') as f:
            f.write(strategy.get('code', ''))
            
        # Save strategy info (exclude code to avoid duplication)
        info = strategy.copy()
        info.pop('code', None)  # Remove code from info file
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_numpy_types(obj.tolist())
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)
                
        info = convert_numpy_types(info)
        
        info_path = os.path.join(output_dir, info_filename)
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        logger.info(f"Saved strategy to {code_path} and {info_path}")
        
        return code_path
    
    def save_all_strategies(self, strategies: Dict[str, List[Dict]], output_dir: str = None) -> List[str]:
        """
        Save all generated strategies to disk.
        
        Args:
            strategies: Dictionary mapping strategy types to lists of strategies
            output_dir: Directory to save strategies (uses self.output_dir if None)
            
        Returns:
            List of paths to saved strategy files
        """
        output_dir = output_dir or self.output_dir
        
        if not output_dir:
            logger.error("No output directory specified")
            return []
            
        saved_paths = []
        
        for strategy_type, strategy_list in strategies.items():
            # Create type-specific subdirectory
            type_dir = os.path.join(output_dir, strategy_type)
            if not os.path.exists(type_dir):
                os.makedirs(type_dir)
                
            for strategy in strategy_list:
                path = self.save_strategy(strategy, type_dir)
                if path:
                    saved_paths.append(path)
                    
        return saved_paths
    
    def generate_and_save_strategies(self, output_dir: str = None) -> Dict:
        """
        Generate and save all strategies in one operation.
        
        Args:
            output_dir: Directory to save strategies (uses self.output_dir if None)
            
        Returns:
            Summary of generated strategies
        """
        output_dir = output_dir or self.output_dir
        
        if not output_dir:
            logger.error("No output directory specified")
            return {}
            
        # Make sure correlation analyzer is set
        if not self.correlation_analyzer:
            logger.error("Correlation analyzer not set")
            return {}
            
        # Generate strategies
        strategies = self.generate_all_strategies()
        
        if not strategies:
            logger.warning("No strategies generated")
            return {"status": "warning", "message": "No strategies generated"}
            
        # Save strategies
        saved_paths = self.save_all_strategies(strategies, output_dir)
        
        # Generate summary
        summary = {
            "status": "success",
            "total_strategies": sum(len(s) for s in strategies.values()),
            "strategies_by_type": {t: len(s) for t, s in strategies.items()},
            "saved_paths": saved_paths,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary
    
    def visualize_strategy_network(self, strategies: Dict[str, List[Dict]], figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Visualize strategies and their relationships as a network.
        
        Args:
            strategies: Dictionary mapping strategy types to lists of strategies
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        try:
            import networkx as nx
        except ImportError:
            logger.error("networkx package required for network visualization")
            return None
            
        if not strategies:
            logger.error("No strategies to visualize")
            return None
            
        # Create graph
        G = nx.Graph()
        
        # Add nodes for each asset
        assets = set()
        
        for strategy_type, strategy_list in strategies.items():
            for strategy in strategy_list:
                # Extract assets from parameters
                if strategy_type == 'lead_lag':
                    assets.add(strategy['parameters']['lead_symbol'])
                    assets.add(strategy['parameters']['lag_symbol'])
                elif strategy_type == 'pairs_trading':
                    assets.add(strategy['parameters']['symbol1'])
                    assets.add(strategy['parameters']['symbol2'])
                elif strategy_type == 'cross_asset':
                    assets.update(strategy['parameters']['symbols'])
                    
        # Add asset nodes
        for asset in assets:
            G.add_node(asset, type='asset')
            
        # Add strategy nodes
        for strategy_type, strategy_list in strategies.items():
            for strategy in strategy_list:
                strategy_id = strategy.get('id', f"{strategy['name']}")
                G.add_node(strategy_id, type='strategy', strategy_type=strategy_type)
                
                # Add edges between strategy and assets
                if strategy_type == 'lead_lag':
                    G.add_edge(strategy_id, strategy['parameters']['lead_symbol'], relationship='uses')
                    G.add_edge(strategy_id, strategy['parameters']['lag_symbol'], relationship='trades')
                elif strategy_type == 'pairs_trading':
                    G.add_edge(strategy_id, strategy['parameters']['symbol1'], relationship='trades')
                    G.add_edge(strategy_id, strategy['parameters']['symbol2'], relationship='trades')
                elif strategy_type == 'cross_asset':
                    for symbol in strategy['parameters']['symbols']:
                        G.add_edge(strategy_id, symbol, relationship='trades')
                        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define node colors
        color_map = []
        for node in G.nodes():
            if G.nodes[node]['type'] == 'asset':
                color_map.append('skyblue')
            else:
                if G.nodes[node].get('strategy_type') == 'lead_lag':
                    color_map.append('lightgreen')
                elif G.nodes[node].get('strategy_type') == 'pairs_trading':
                    color_map.append('coral')
                else:
                    color_map.append('purple')
                    
        # Define node sizes
        node_sizes = []
        for node in G.nodes():
            if G.nodes[node]['type'] == 'asset':
                node_sizes.append(300)
            else:
                node_sizes.append(500)
                
        # Define layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        # Add legend
        # This requires creating proxy artists
        import matplotlib.patches as mpatches
        
        asset_patch = mpatches.Patch(color='skyblue', label='Asset')
        lead_lag_patch = mpatches.Patch(color='lightgreen', label='Lead-Lag Strategy')
        pairs_patch = mpatches.Patch(color='coral', label='Pairs Trading Strategy')
        cross_asset_patch = mpatches.Patch(color='purple', label='Cross-Asset Strategy')
        
        plt.legend(handles=[asset_patch, lead_lag_patch, pairs_patch, cross_asset_patch])
        
        # Set title and remove axis
        plt.title('Strategy and Asset Network')
        plt.axis('off')
        
        return fig