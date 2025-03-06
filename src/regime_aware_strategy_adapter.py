#!/usr/bin/env python3
"""
Regime-Aware Strategy Adapter Module

This module adapts trading strategies to be aware of market regimes, allowing them
to optimize their parameters and behavior based on the current market regime.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import pickle
from .market_regime_detector import MarketRegimeDetector

# Configure logging
logger = logging.getLogger(__name__)

class RegimeAwareStrategyAdapter:
    """
    Adapter class that makes trading strategies aware of market regimes.
    
    This class integrates market regime detection with trading strategies, allowing
    strategies to adjust their parameters and behavior based on the detected regime.
    """
    
    def __init__(self, strategy_path: str, regime_detector: Optional[MarketRegimeDetector] = None,
                 regime_config_path: Optional[str] = None):
        """
        Initialize the regime-aware strategy adapter.
        
        Args:
            strategy_path: Path to the strategy JSON file
            regime_detector: Optional MarketRegimeDetector instance
            regime_config_path: Path to the regime configuration file
        """
        self.strategy_path = strategy_path
        self.regime_detector = regime_detector or MarketRegimeDetector(method="hmm", n_regimes=4)
        self.regime_config_path = regime_config_path
        self.strategy_data = None
        self.regime_configs = {}
        self.current_regime = None
        self.regime_label = None
        
        # Load strategy
        self._load_strategy()
        
        # Load regime configurations if path provided
        if regime_config_path:
            self._load_regime_configs()
        else:
            # Generate default regime configurations
            self._generate_default_regime_configs()
    
    def _load_strategy(self):
        """Load strategy from JSON file."""
        try:
            with open(self.strategy_path, 'r') as f:
                self.strategy_data = json.load(f)
            logger.info(f"Loaded strategy from {self.strategy_path}")
        except Exception as e:
            logger.error(f"Error loading strategy: {str(e)}")
            raise
    
    def _load_regime_configs(self):
        """Load regime configurations from JSON file."""
        try:
            with open(self.regime_config_path, 'r') as f:
                self.regime_configs = json.load(f)
            logger.info(f"Loaded regime configurations from {self.regime_config_path}")
        except Exception as e:
            logger.error(f"Error loading regime configurations: {str(e)}")
            # Generate default regime configurations as fallback
            self._generate_default_regime_configs()
    
    def _generate_default_regime_configs(self):
        """Generate default regime configurations based on the strategy."""
        logger.info("Generating default regime configurations")
        
        # Get strategy parameters
        strategy_name = self.strategy_data.get("strategy", {}).get("Strategy Name", "Unknown Strategy")
        params = self.strategy_data.get("parameters", {})
        
        # Create default configurations for each regime
        regimes = {
            0: "Bull Market",
            1: "Bear Market",
            2: "Sideways/Consolidation",
            3: "High Volatility",
            4: "Low Volatility",
            5: "Crisis/Extreme"
        }
        
        self.regime_configs = {}
        
        for regime_id, regime_label in regimes.items():
            # Create a copy of the strategy parameters
            regime_params = params.copy() if params else {}
            
            # Adjust parameters based on the regime
            if regime_label == "Bull Market":
                # More aggressive in bull markets
                self._adjust_params_for_bull_market(regime_params)
            elif regime_label == "Bear Market":
                # More conservative in bear markets
                self._adjust_params_for_bear_market(regime_params)
            elif regime_label == "Sideways/Consolidation":
                # Optimize for range-bound markets
                self._adjust_params_for_sideways_market(regime_params)
            elif regime_label == "High Volatility":
                # Adjust for high volatility
                self._adjust_params_for_high_volatility(regime_params)
            elif regime_label == "Low Volatility":
                # Adjust for low volatility
                self._adjust_params_for_low_volatility(regime_params)
            elif regime_label == "Crisis/Extreme":
                # Very conservative in crisis
                self._adjust_params_for_crisis(regime_params)
            
            # Save regime configuration
            self.regime_configs[str(regime_id)] = {
                "label": regime_label,
                "parameters": regime_params,
                "position_sizing": self._get_position_sizing_for_regime(regime_label),
                "risk_management": self._get_risk_management_for_regime(regime_label)
            }
        
        logger.info(f"Generated default regime configurations for {len(self.regime_configs)} regimes")
        
        # Save the generated configurations if a path was provided
        if self.regime_config_path:
            self._save_regime_configs()
    
    def _adjust_params_for_bull_market(self, params: Dict[str, Any]):
        """Adjust strategy parameters for bull market regime."""
        # Looser entry criteria, tighter exit criteria
        if "entry_threshold" in params:
            params["entry_threshold"] *= 0.9  # Lower entry threshold
        
        if "exit_threshold" in params:
            params["exit_threshold"] *= 1.1  # Higher exit threshold
        
        # Longer holding periods
        if "max_holding_period" in params:
            params["max_holding_period"] = int(params["max_holding_period"] * 1.5)
        
        # More aggressive trend following
        if "trend_strength" in params:
            params["trend_strength"] *= 0.8  # Lower requirement for trend strength
    
    def _adjust_params_for_bear_market(self, params: Dict[str, Any]):
        """Adjust strategy parameters for bear market regime."""
        # Tighter entry criteria, looser exit criteria
        if "entry_threshold" in params:
            params["entry_threshold"] *= 1.2  # Higher entry threshold
        
        if "exit_threshold" in params:
            params["exit_threshold"] *= 0.9  # Lower exit threshold
        
        # Shorter holding periods
        if "max_holding_period" in params:
            params["max_holding_period"] = int(params["max_holding_period"] * 0.7)
        
        # More conservative trend following
        if "trend_strength" in params:
            params["trend_strength"] *= 1.2  # Higher requirement for trend strength
    
    def _adjust_params_for_sideways_market(self, params: Dict[str, Any]):
        """Adjust strategy parameters for sideways/consolidation market regime."""
        # Optimize for mean reversion strategies
        if "mean_reversion_strength" in params:
            params["mean_reversion_strength"] *= 1.2
        
        # Tighter stop losses
        if "stop_loss_atr_multiple" in params:
            params["stop_loss_atr_multiple"] *= 0.8
        
        # Shorter holding periods
        if "max_holding_period" in params:
            params["max_holding_period"] = int(params["max_holding_period"] * 0.8)
    
    def _adjust_params_for_high_volatility(self, params: Dict[str, Any]):
        """Adjust strategy parameters for high volatility market regime."""
        # Wider stop losses to account for volatility
        if "stop_loss_atr_multiple" in params:
            params["stop_loss_atr_multiple"] *= 1.5
        
        # Tighter profit targets
        if "profit_target_atr_multiple" in params:
            params["profit_target_atr_multiple"] *= 0.8
        
        # Shorter holding periods
        if "max_holding_period" in params:
            params["max_holding_period"] = int(params["max_holding_period"] * 0.6)
    
    def _adjust_params_for_low_volatility(self, params: Dict[str, Any]):
        """Adjust strategy parameters for low volatility market regime."""
        # Tighter stop losses due to lower volatility
        if "stop_loss_atr_multiple" in params:
            params["stop_loss_atr_multiple"] *= 0.7
        
        # Wider profit targets
        if "profit_target_atr_multiple" in params:
            params["profit_target_atr_multiple"] *= 1.2
        
        # Longer holding periods
        if "max_holding_period" in params:
            params["max_holding_period"] = int(params["max_holding_period"] * 1.3)
    
    def _adjust_params_for_crisis(self, params: Dict[str, Any]):
        """Adjust strategy parameters for crisis/extreme market regime."""
        # Very conservative entry criteria
        if "entry_threshold" in params:
            params["entry_threshold"] *= 1.5
        
        # Very quick exit criteria
        if "exit_threshold" in params:
            params["exit_threshold"] *= 0.7
        
        # Very short holding periods
        if "max_holding_period" in params:
            params["max_holding_period"] = int(params["max_holding_period"] * 0.4)
        
        # Much wider stop losses to account for extreme volatility
        if "stop_loss_atr_multiple" in params:
            params["stop_loss_atr_multiple"] *= 2.0
    
    def _get_position_sizing_for_regime(self, regime_label: str) -> Dict[str, Any]:
        """Get position sizing parameters for the given regime."""
        # Default position sizing
        position_sizing = {
            "max_position_size": 0.05,  # 5% of portfolio
            "risk_per_trade": 0.01,     # 1% risk per trade
            "sizing_method": "risk"     # Risk-based sizing
        }
        
        # Adjust based on regime
        if regime_label == "Bull Market":
            position_sizing["max_position_size"] = 0.08  # 8% of portfolio
            position_sizing["risk_per_trade"] = 0.015    # 1.5% risk per trade
        elif regime_label == "Bear Market":
            position_sizing["max_position_size"] = 0.03  # 3% of portfolio
            position_sizing["risk_per_trade"] = 0.007    # 0.7% risk per trade
        elif regime_label == "Sideways/Consolidation":
            position_sizing["max_position_size"] = 0.04  # 4% of portfolio
            position_sizing["risk_per_trade"] = 0.008    # 0.8% risk per trade
        elif regime_label == "High Volatility":
            position_sizing["max_position_size"] = 0.03  # 3% of portfolio
            position_sizing["risk_per_trade"] = 0.005    # 0.5% risk per trade
        elif regime_label == "Low Volatility":
            position_sizing["max_position_size"] = 0.07  # 7% of portfolio
            position_sizing["risk_per_trade"] = 0.012    # 1.2% risk per trade
        elif regime_label == "Crisis/Extreme":
            position_sizing["max_position_size"] = 0.02  # 2% of portfolio
            position_sizing["risk_per_trade"] = 0.003    # 0.3% risk per trade
        
        return position_sizing
    
    def _get_risk_management_for_regime(self, regime_label: str) -> Dict[str, Any]:
        """Get risk management parameters for the given regime."""
        # Default risk management
        risk_management = {
            "stop_loss_atr_multiple": 2.0,  # 2 ATR units for stop loss
            "trailing_stop": True,          # Use trailing stops
            "max_drawdown": 0.15,           # 15% max drawdown
            "correlation_threshold": 0.7,   # 0.7 correlation threshold
            "max_open_positions": 10        # Maximum 10 open positions
        }
        
        # Adjust based on regime
        if regime_label == "Bull Market":
            risk_management["stop_loss_atr_multiple"] = 2.5   # Wider stops
            risk_management["trailing_stop"] = True
            risk_management["max_drawdown"] = 0.18            # Higher drawdown tolerance
            risk_management["max_open_positions"] = 15        # More positions
        elif regime_label == "Bear Market":
            risk_management["stop_loss_atr_multiple"] = 1.8   # Tighter stops
            risk_management["trailing_stop"] = True
            risk_management["max_drawdown"] = 0.12            # Lower drawdown tolerance
            risk_management["max_open_positions"] = 7         # Fewer positions
        elif regime_label == "Sideways/Consolidation":
            risk_management["stop_loss_atr_multiple"] = 1.5   # Tighter stops
            risk_management["trailing_stop"] = False          # Fixed stops
            risk_management["max_open_positions"] = 8         # Moderate number of positions
        elif regime_label == "High Volatility":
            risk_management["stop_loss_atr_multiple"] = 3.0   # Wider stops
            risk_management["trailing_stop"] = True
            risk_management["max_drawdown"] = 0.20            # Higher drawdown tolerance
            risk_management["max_open_positions"] = 5         # Few positions
        elif regime_label == "Low Volatility":
            risk_management["stop_loss_atr_multiple"] = 1.2   # Tighter stops
            risk_management["trailing_stop"] = False          # Fixed stops
            risk_management["max_open_positions"] = 12        # More positions
        elif regime_label == "Crisis/Extreme":
            risk_management["stop_loss_atr_multiple"] = 4.0   # Much wider stops
            risk_management["trailing_stop"] = True
            risk_management["max_drawdown"] = 0.10            # Lower drawdown tolerance
            risk_management["max_open_positions"] = 3         # Very few positions
        
        return risk_management
    
    def _save_regime_configs(self):
        """Save regime configurations to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.regime_config_path), exist_ok=True)
            with open(self.regime_config_path, 'w') as f:
                json.dump(self.regime_configs, f, indent=2)
            logger.info(f"Saved regime configurations to {self.regime_config_path}")
        except Exception as e:
            logger.error(f"Error saving regime configurations: {str(e)}")
    
    def detect_regime(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect the current market regime.
        
        Args:
            price_data: DataFrame with market data
            
        Returns:
            Dictionary with regime detection results
        """
        # Detect regime
        regime_results = self.regime_detector.load_or_train(price_data)
        
        # Store current regime
        self.current_regime = regime_results.get("current_regime")
        self.regime_label = regime_results.get("regime_label")
        
        return regime_results
    
    def get_regime_adapted_strategy(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a strategy adapted to the current market regime.
        
        Args:
            price_data: DataFrame with market data
            
        Returns:
            Dictionary with the adapted strategy data
        """
        # Detect regime if not already detected
        if self.current_regime is None:
            self.detect_regime(price_data)
        
        # Get the regime configuration
        regime_config = self.regime_configs.get(str(self.current_regime), {})
        
        if not regime_config:
            logger.warning(f"No configuration found for regime {self.current_regime}. Using default strategy.")
            return self.strategy_data
        
        # Create a copy of the strategy data
        adapted_strategy = self.strategy_data.copy()
        
        # Apply regime-specific parameters
        if "parameters" in regime_config:
            adapted_strategy["parameters"] = regime_config["parameters"]
        
        # Apply regime-specific position sizing
        if "position_sizing" in regime_config:
            adapted_strategy["position_sizing"] = regime_config["position_sizing"]
        
        # Apply regime-specific risk management
        if "risk_management" in regime_config:
            adapted_strategy["risk_management"] = regime_config["risk_management"]
        
        # Add regime information
        adapted_strategy["regime"] = {
            "id": self.current_regime,
            "label": self.regime_label,
            "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Adapted strategy '{adapted_strategy.get('strategy', {}).get('Strategy Name')}' to regime {self.regime_label}")
        
        return adapted_strategy
    
    def save_adapted_strategy(self, output_path: Optional[str] = None) -> str:
        """
        Save the regime-adapted strategy to a JSON file.
        
        Args:
            output_path: Path to save the adapted strategy
            
        Returns:
            Path to the saved strategy file
        """
        if self.current_regime is None:
            logger.error("No regime detected. Please call get_regime_adapted_strategy() first.")
            raise ValueError("No regime detected")
        
        # Get adapted strategy
        adapted_strategy = self.get_regime_adapted_strategy(None)
        
        # Generate output path if not provided
        if output_path is None:
            strategy_name = adapted_strategy.get("strategy", {}).get("Strategy Name", "Unknown")
            strategy_name = strategy_name.replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"strategies/{strategy_name}_regime_{self.current_regime}_{timestamp}.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save strategy
        with open(output_path, 'w') as f:
            json.dump(adapted_strategy, f, indent=2)
        
        logger.info(f"Saved adapted strategy to {output_path}")
        
        return output_path
    
    def visualize_regime_impact(self, price_data: pd.DataFrame, output_path: Optional[str] = None):
        """
        Visualize the impact of market regimes on strategy performance.
        
        Args:
            price_data: DataFrame with market data
            output_path: Path to save the visualization
        """
        # Ensure regime is detected
        if self.current_regime is None:
            self.detect_regime(price_data)
        
        # Get regime statistics
        regime_stats = self.regime_detector.get_regime_returns(price_data)
        
        # Create a plot
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(15, 10))
        
        # Plot price chart with regimes
        self.regime_detector.plot_regimes(price_data, output_path=None)
        
        # Create performance metrics by regime
        plt.figure(figsize=(15, 8))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Annualized Returns by Regime
        regime_ids = list(regime_stats.keys())
        regime_labels = [regime_stats[r]["label"] for r in regime_ids]
        annualized_returns = [regime_stats[r]["annualized_return"] * 100 for r in regime_ids]
        
        sns.barplot(x=regime_labels, y=annualized_returns, ax=axes[0, 0], palette="viridis")
        axes[0, 0].set_title("Annualized Returns by Regime (%)")
        axes[0, 0].set_ylabel("Annualized Return (%)")
        axes[0, 0].set_xlabel("Market Regime")
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha="right")
        
        # Sharpe Ratio by Regime
        sharpe_ratios = [regime_stats[r]["sharpe_ratio"] for r in regime_ids]
        
        sns.barplot(x=regime_labels, y=sharpe_ratios, ax=axes[0, 1], palette="viridis")
        axes[0, 1].set_title("Sharpe Ratio by Regime")
        axes[0, 1].set_ylabel("Sharpe Ratio")
        axes[0, 1].set_xlabel("Market Regime")
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha="right")
        
        # Win Rate by Regime
        win_rates = [regime_stats[r]["positive_returns"] * 100 for r in regime_ids]
        
        sns.barplot(x=regime_labels, y=win_rates, ax=axes[1, 0], palette="viridis")
        axes[1, 0].set_title("Win Rate by Regime (%)")
        axes[1, 0].set_ylabel("Win Rate (%)")
        axes[1, 0].set_xlabel("Market Regime")
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha="right")
        
        # Annualized Volatility by Regime
        volatilities = [regime_stats[r]["annualized_volatility"] * 100 for r in regime_ids]
        
        sns.barplot(x=regime_labels, y=volatilities, ax=axes[1, 1], palette="viridis")
        axes[1, 1].set_title("Annualized Volatility by Regime (%)")
        axes[1, 1].set_ylabel("Annualized Volatility (%)")
        axes[1, 1].set_xlabel("Market Regime")
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha="right")
        
        # Add strategy name and current regime
        strategy_name = self.strategy_data.get("strategy", {}).get("Strategy Name", "Unknown Strategy")
        fig.suptitle(f"Regime Impact on {strategy_name}\nCurrent Regime: {self.regime_label}", fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save or show the plot
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved visualization to {output_path}")
        else:
            plt.show()


def main():
    """
    Example usage of the RegimeAwareStrategyAdapter.
    """
    import yfinance as yf
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Adapt trading strategies to market regimes")
    parser.add_argument("--strategy", required=True, help="Path to strategy JSON file")
    parser.add_argument("--output", help="Path to save adapted strategy")
    parser.add_argument("--regime-config", help="Path to regime configuration file")
    parser.add_argument("--method", default="hmm", help="Regime detection method (hmm, gmm, kmeans, hierarchical)")
    parser.add_argument("--n-regimes", type=int, default=4, help="Number of regimes to detect")
    parser.add_argument("--symbol", default="SPY", help="Symbol to download data for")
    parser.add_argument("--start", default="2018-01-01", help="Start date for data download")
    parser.add_argument("--end", default=None, help="End date for data download")
    parser.add_argument("--visualize", action="store_true", help="Visualize regime impact")
    parser.add_argument("--visualization-output", help="Path to save visualization")
    
    args = parser.parse_args()
    
    # Download data
    end_date = args.end or datetime.now().strftime("%Y-%m-%d")
    data = yf.download(args.symbol, start=args.start, end=end_date)
    
    # Create regime detector
    detector = MarketRegimeDetector(method=args.method, n_regimes=args.n_regimes)
    
    # Create adapter
    adapter = RegimeAwareStrategyAdapter(
        strategy_path=args.strategy,
        regime_detector=detector,
        regime_config_path=args.regime_config
    )
    
    # Detect regime
    regime_results = adapter.detect_regime(data)
    print(f"Detected regime: {regime_results['regime_label']} (ID: {regime_results['current_regime']})")
    
    # Get adapted strategy
    adapted_strategy = adapter.get_regime_adapted_strategy(data)
    
    # Save adapted strategy
    output_path = adapter.save_adapted_strategy(args.output)
    print(f"Saved adapted strategy to: {output_path}")
    
    # Visualize regime impact
    if args.visualize:
        adapter.visualize_regime_impact(data, args.visualization_output)


if __name__ == "__main__":
    main()