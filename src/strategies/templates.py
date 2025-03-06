"""
Trading strategy templates for strategy generation.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class StrategyParameters:
    """Base class for strategy parameters."""
    name: str = field(default="Unnamed Strategy")
    description: str = field(default="No description provided")
    version: str = field(default="1.0.0")
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class BaseStrategy(ABC):
    """Base class for all strategy templates."""
    
    def __init__(self, parameters: StrategyParameters):
        """Initialize strategy."""
        self.parameters = parameters
        self.signals = []
        self.positions = []
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals."""
        pass
        
    @abstractmethod
    def calculate_position_sizes(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Calculate position sizes."""
        pass

@dataclass
class MomentumParameters(StrategyParameters):
    """Parameters for momentum strategy."""
    lookback: int = field(default=20)
    threshold: float = field(default=0.02)
    position_size: float = field(default=1.0)
    stop_loss: Optional[float] = field(default=None)
    take_profit: Optional[float] = field(default=None)

class MomentumStrategy(BaseStrategy):
    """
    Momentum trading strategy.
    
    Signals based on price momentum over specified lookback period.
    """
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum signals."""
        # Calculate returns
        returns = data['close'].pct_change(self.parameters.lookback)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[returns > self.parameters.threshold] = 1
        signals[returns < -self.parameters.threshold] = -1
        
        return signals
        
    def calculate_position_sizes(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Calculate position sizes."""
        return signals * self.parameters.position_size

@dataclass
class MeanReversionParameters(StrategyParameters):
    """Parameters for mean reversion strategy."""
    window: int = field(default=20)
    std_dev: float = field(default=2.0)
    position_size: float = field(default=1.0)
    mean_type: str = field(default="simple")  # or "exponential"
    stop_loss: Optional[float] = field(default=None)

class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy.
    
    Signals based on deviation from moving average.
    """
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals."""
        # Calculate moving average
        if self.parameters.mean_type == "simple":
            ma = data['close'].rolling(self.parameters.window).mean()
        else:
            ma = data['close'].ewm(span=self.parameters.window).mean()
            
        # Calculate standard deviation
        std = data['close'].rolling(self.parameters.window).std()
        
        # Generate signals
        z_score = (data['close'] - ma) / std
        
        signals = pd.Series(0, index=data.index)
        signals[z_score > self.parameters.std_dev] = -1  # Sell when above upper band
        signals[z_score < -self.parameters.std_dev] = 1  # Buy when below lower band
        
        return signals
        
    def calculate_position_sizes(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Calculate position sizes."""
        return signals * self.parameters.position_size

@dataclass
class TrendFollowingParameters(StrategyParameters):
    """Parameters for trend following strategy."""
    fast_ma: int = field(default=10)
    slow_ma: int = field(default=30)
    position_size: float = field(default=1.0)
    trend_strength: float = field(default=0.01)
    stop_loss: Optional[float] = field(default=None)

class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy.
    
    Signals based on moving average crossovers.
    """
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trend following signals."""
        # Calculate moving averages
        fast_ma = data['close'].rolling(self.parameters.fast_ma).mean()
        slow_ma = data['close'].rolling(self.parameters.slow_ma).mean()
        
        # Calculate trend strength
        trend_strength = (fast_ma - slow_ma) / slow_ma
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[(fast_ma > slow_ma) & (trend_strength > self.parameters.trend_strength)] = 1
        signals[(fast_ma < slow_ma) & (trend_strength < -self.parameters.trend_strength)] = -1
        
        return signals
        
    def calculate_position_sizes(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Calculate position sizes."""
        return signals * self.parameters.position_size

@dataclass
class VolatilityParameters(StrategyParameters):
    """Parameters for volatility-based strategy."""
    calculation_window: int = field(default=20)
    entry_threshold: float = field(default=0.02)
    position_size: float = field(default=1.0)
    volatility_scaling: bool = field(default=True)
    stop_loss: Optional[float] = field(default=None)

class VolatilityStrategy(BaseStrategy):
    """
    Volatility-based trading strategy.
    
    Signals based on volatility patterns.
    """
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate volatility-based signals."""
        # Calculate volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(self.parameters.calculation_window).std()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[volatility > self.parameters.entry_threshold] = 1
        
        return signals
        
    def calculate_position_sizes(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Calculate position sizes with volatility scaling."""
        if self.parameters.volatility_scaling:
            # Scale position size inversely with volatility
            volatility = data['close'].pct_change().rolling(
                self.parameters.calculation_window
            ).std()
            position_sizes = signals * self.parameters.position_size * (
                1 / volatility
            ).fillna(0)
            return position_sizes.clip(-1, 1)  # Limit position sizes
        else:
            return signals * self.parameters.position_size

class StrategyFactory:
    """Factory for creating strategy instances."""
    
    @staticmethod
    def create_strategy(config: Dict[str, Any]) -> BaseStrategy:
        """
        Create strategy instance from configuration.
        
        Args:
            config: Strategy configuration dictionary
            
        Returns:
            Strategy instance
        """
        strategy_type = config['type']
        
        if strategy_type == 'momentum':
            parameters = MomentumParameters(
                name=config.get('name', 'Momentum Strategy'),
                description=config.get('description', 'Momentum-based trading strategy'),
                lookback=config.get('lookback', 20),
                threshold=config.get('threshold', 0.02),
                position_size=config.get('position_size', 1.0),
                stop_loss=config.get('stop_loss'),
                take_profit=config.get('take_profit')
            )
            return MomentumStrategy(parameters)
            
        elif strategy_type == 'mean_reversion':
            parameters = MeanReversionParameters(
                name=config.get('name', 'Mean Reversion Strategy'),
                description=config.get('description', 'Mean reversion trading strategy'),
                window=config.get('window', 20),
                std_dev=config.get('std_dev', 2.0),
                position_size=config.get('position_size', 1.0),
                mean_type=config.get('mean_type', 'simple'),
                stop_loss=config.get('stop_loss')
            )
            return MeanReversionStrategy(parameters)
            
        elif strategy_type == 'trend_following':
            parameters = TrendFollowingParameters(
                name=config.get('name', 'Trend Following Strategy'),
                description=config.get('description', 'Trend following trading strategy'),
                fast_ma=config.get('fast_ma', 10),
                slow_ma=config.get('slow_ma', 30),
                position_size=config.get('position_size', 1.0),
                trend_strength=config.get('trend_strength', 0.01),
                stop_loss=config.get('stop_loss')
            )
            return TrendFollowingStrategy(parameters)
            
        elif strategy_type == 'volatility':
            parameters = VolatilityParameters(
                name=config.get('name', 'Volatility Strategy'),
                description=config.get('description', 'Volatility-based trading strategy'),
                calculation_window=config.get('calculation_window', 20),
                entry_threshold=config.get('entry_threshold', 0.02),
                position_size=config.get('position_size', 1.0),
                volatility_scaling=config.get('volatility_scaling', True),
                stop_loss=config.get('stop_loss')
            )
            return VolatilityStrategy(parameters)
            
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

def create_strategy(config: Dict[str, Any]) -> BaseStrategy:
    """Convenience function for creating strategies."""
    return StrategyFactory.create_strategy(config)