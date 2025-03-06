"""
Safety checker implementation for ensuring ethical trading strategies.
"""
import logging
import enum
from datetime import datetime
from typing import Dict, List, Any, Optional, Protocol, Set, Tuple, Callable
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SafetyLevel(str, enum.Enum):
    """Safety check level."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SafetyCheck(BaseModel):
    """
    A safety check definition.
    
    Attributes:
        name: The name of the check
        description: Description of what the check does
        level: The severity level of the check
        enabled: Whether the check is enabled
    """
    name: str
    description: str
    level: SafetyLevel
    enabled: bool = True


class SafetyViolation(BaseModel):
    """
    A safety check violation.
    
    Attributes:
        check: The safety check that was violated
        message: Detailed message about the violation
        context: Additional context about the violation
        timestamp: When the violation occurred
    """
    check: SafetyCheck
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class SafetyChecker:
    """
    Safety checker for ensuring ethical trading strategies.
    
    This class provides a framework for checking trading strategies
    against various safety and ethical guidelines.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the safety checker.
        
        Args:
            config: Optional configuration for the safety checker
        """
        self.config = config or {}
        
        # Initialize safety checks
        self.checks: Dict[str, SafetyCheck] = {}
        self.check_handlers: Dict[str, Callable[[Dict[str, Any]], List[SafetyViolation]]] = {}
        
        # Configure from settings
        self.max_concentration = self.config.get("max_concentration", 0.2)  # 20% maximum concentration
        self.max_leverage = self.config.get("max_leverage", 1.5)  # 1.5x maximum leverage
        
        # Register default safety checks
        self._register_default_checks()
        
    def register_check(
        self,
        name: str,
        description: str,
        level: SafetyLevel,
        handler: Callable[[Dict[str, Any]], List[SafetyViolation]],
        enabled: bool = True
    ) -> None:
        """
        Register a safety check.
        
        Args:
            name: The name of the check
            description: Description of what the check does
            level: The severity level of the check
            handler: Function to call to perform the check
            enabled: Whether the check is enabled
        """
        check = SafetyCheck(
            name=name,
            description=description,
            level=level,
            enabled=enabled
        )
        
        self.checks[name] = check
        self.check_handlers[name] = handler
        
        logger.debug(f"Registered safety check: {name}")
        
    def check_strategy(
        self,
        strategy: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[SafetyViolation]:
        """
        Check a trading strategy for safety violations.
        
        Args:
            strategy: The strategy to check
            context: Optional additional context for the checks
            
        Returns:
            A list of safety violations
        """
        context = context or {}
        all_violations = []
        
        logger.debug(f"Checking strategy for safety violations")
        
        # Run all enabled checks
        for name, check in self.checks.items():
            if not check.enabled:
                continue
                
            try:
                # Create check context
                check_context = {
                    "strategy": strategy,
                    "context": context,
                    "config": self.config
                }
                
                # Run check
                handler = self.check_handlers[name]
                violations = handler(check_context)
                
                if violations:
                    all_violations.extend(violations)
                    severity = len(violations)
                    log_level = "error" if check.level in [SafetyLevel.ERROR, SafetyLevel.CRITICAL] else "warning"
                    
                    getattr(logger, log_level)(
                        f"Safety check '{name}' found {severity} violation(s)"
                    )
                    
            except Exception as e:
                logger.error(f"Error running safety check '{name}': {str(e)}")
        
        return all_violations
    
    def is_strategy_safe(
        self,
        strategy: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        allow_warnings: bool = True
    ) -> bool:
        """
        Check if a strategy is safe to execute.
        
        Args:
            strategy: The strategy to check
            context: Optional additional context for the checks
            allow_warnings: Whether warning-level violations are acceptable
            
        Returns:
            True if the strategy is safe, False otherwise
        """
        violations = self.check_strategy(strategy, context)
        
        if not violations:
            return True
            
        # Check for critical or error level violations
        for violation in violations:
            if violation.check.level in [SafetyLevel.ERROR, SafetyLevel.CRITICAL]:
                return False
                
            if violation.check.level == SafetyLevel.WARNING and not allow_warnings:
                return False
                
        return True
    
    def get_enabled_checks(self) -> List[SafetyCheck]:
        """Get a list of all enabled safety checks."""
        return [check for check in self.checks.values() if check.enabled]
    
    def enable_check(self, name: str) -> bool:
        """
        Enable a safety check.
        
        Args:
            name: The name of the check to enable
            
        Returns:
            True if the check was enabled, False if it wasn't found
        """
        if name in self.checks:
            self.checks[name].enabled = True
            logger.debug(f"Enabled safety check: {name}")
            return True
        else:
            logger.warning(f"Safety check not found: {name}")
            return False
    
    def disable_check(self, name: str) -> bool:
        """
        Disable a safety check.
        
        Args:
            name: The name of the check to disable
            
        Returns:
            True if the check was disabled, False if it wasn't found
        """
        if name in self.checks:
            self.checks[name].enabled = False
            logger.debug(f"Disabled safety check: {name}")
            return True
        else:
            logger.warning(f"Safety check not found: {name}")
            return False
    
    def _register_default_checks(self) -> None:
        """Register the default safety checks."""
        # Market manipulation check
        self.register_check(
            name="market_manipulation",
            description="Check for potential market manipulation",
            level=SafetyLevel.CRITICAL,
            handler=self._check_market_manipulation,
            enabled=self.config.get("market_manipulation_checks", True)
        )
        
        # Position limits check
        self.register_check(
            name="position_limits",
            description="Check for excessive position sizes",
            level=SafetyLevel.ERROR,
            handler=self._check_position_limits,
            enabled=self.config.get("position_limits", True)
        )
        
        # Trading frequency check
        self.register_check(
            name="trading_frequency",
            description="Check for excessive trading frequency",
            level=SafetyLevel.WARNING,
            handler=self._check_trading_frequency,
            enabled=self.config.get("trading_frequency_limits", True)
        )
        
        # Price impact check
        self.register_check(
            name="price_impact",
            description="Check for excessive price impact",
            level=SafetyLevel.WARNING,
            handler=self._check_price_impact,
            enabled=self.config.get("price_impact_monitoring", True)
        )
        
        # Concentration check
        self.register_check(
            name="concentration",
            description="Check for excessive concentration in a single asset",
            level=SafetyLevel.ERROR,
            handler=self._check_concentration,
            enabled=True
        )
        
        # Leverage check
        self.register_check(
            name="leverage",
            description="Check for excessive leverage",
            level=SafetyLevel.ERROR,
            handler=self._check_leverage,
            enabled=True
        )
    
    def _check_market_manipulation(self, context: Dict[str, Any]) -> List[SafetyViolation]:
        """
        Check for potential market manipulation.
        
        This includes:
        - Wash trading
        - Spoofing
        - Layering
        - Pump and dump
        
        Args:
            context: Check context
            
        Returns:
            A list of safety violations
        """
        violations = []
        strategy = context["strategy"]
        
        # Get the safety check
        check = self.checks["market_manipulation"]
        
        # Check for wash trading (buying and selling the same asset in a short time)
        if "trades" in strategy:
            trades = strategy["trades"]
            for i in range(len(trades) - 1):
                current_trade = trades[i]
                next_trade = trades[i + 1]
                
                if (current_trade.get("symbol") == next_trade.get("symbol") and
                    current_trade.get("direction") != next_trade.get("direction") and
                    (next_trade.get("timestamp", 0) - current_trade.get("timestamp", 0) < 60)):  # 60 seconds
                    
                    violations.append(SafetyViolation(
                        check=check,
                        message="Potential wash trading detected (rapid buy/sell of same asset)",
                        context={
                            "trade1": current_trade,
                            "trade2": next_trade
                        }
                    ))
        
        # Check for potential pump and dump
        if "holdings" in strategy and "target_prices" in strategy:
            holdings = strategy["holdings"]
            target_prices = strategy["target_prices"]
            
            for symbol, holding in holdings.items():
                if symbol in target_prices:
                    current_price = holding.get("current_price", 0)
                    target_price = target_prices[symbol]
                    
                    # If target price is significantly higher than current price (potential pump)
                    if target_price > current_price * 1.5:
                        violations.append(SafetyViolation(
                            check=check,
                            message="Potential pump and dump pattern detected",
                            context={
                                "symbol": symbol,
                                "current_price": current_price,
                                "target_price": target_price,
                                "percent_increase": (target_price / current_price - 1) * 100
                            }
                        ))
        
        return violations
    
    def _check_position_limits(self, context: Dict[str, Any]) -> List[SafetyViolation]:
        """
        Check for excessive position sizes.
        
        Args:
            context: Check context
            
        Returns:
            A list of safety violations
        """
        violations = []
        strategy = context["strategy"]
        
        # Get the safety check
        check = self.checks["position_limits"]
        
        # Get portfolio value
        portfolio_value = strategy.get("portfolio_value", 0)
        if portfolio_value <= 0:
            return violations  # Cannot check without portfolio value
        
        # Check position sizes
        if "positions" in strategy:
            positions = strategy["positions"]
            for symbol, position in positions.items():
                position_value = position.get("value", 0)
                position_pct = position_value / portfolio_value
                
                if position_pct > self.max_concentration:
                    violations.append(SafetyViolation(
                        check=check,
                        message=f"Position size exceeds limit for {symbol}",
                        context={
                            "symbol": symbol,
                            "position_value": position_value,
                            "position_percent": position_pct * 100,
                            "limit_percent": self.max_concentration * 100
                        }
                    ))
        
        return violations
    
    def _check_trading_frequency(self, context: Dict[str, Any]) -> List[SafetyViolation]:
        """
        Check for excessive trading frequency.
        
        Args:
            context: Check context
            
        Returns:
            A list of safety violations
        """
        violations = []
        strategy = context["strategy"]
        
        # Get the safety check
        check = self.checks["trading_frequency"]
        
        # Check trade frequency
        if "trades" in strategy and "time_period" in strategy:
            trades = strategy["trades"]
            time_period_days = strategy.get("time_period", 1)  # Default 1 day
            
            # Calculate trades per day
            trades_count = len(trades)
            trades_per_day = trades_count / time_period_days
            
            # Excessive trading threshold - depends on strategy but we'll use 20 trades/day as a warning
            if trades_per_day > 20:
                violations.append(SafetyViolation(
                    check=check,
                    message="Excessive trading frequency detected",
                    context={
                        "trades_count": trades_count,
                        "time_period_days": time_period_days,
                        "trades_per_day": trades_per_day,
                        "threshold": 20
                    }
                ))
        
        return violations
    
    def _check_price_impact(self, context: Dict[str, Any]) -> List[SafetyViolation]:
        """
        Check for excessive price impact.
        
        Args:
            context: Check context
            
        Returns:
            A list of safety violations
        """
        violations = []
        strategy = context["strategy"]
        
        # Get the safety check
        check = self.checks["price_impact"]
        
        # Check for price impact
        if "trades" in strategy and "market_volumes" in strategy:
            trades = strategy["trades"]
            market_volumes = strategy["market_volumes"]
            
            for trade in trades:
                symbol = trade.get("symbol")
                quantity = trade.get("quantity", 0)
                
                if symbol in market_volumes:
                    volume = market_volumes[symbol]
                    
                    # Calculate trade volume as percentage of market volume
                    if volume > 0:
                        impact_pct = quantity / volume
                        
                        # If trade is more than 1% of daily volume, flag it
                        if impact_pct > 0.01:
                            violations.append(SafetyViolation(
                                check=check,
                                message=f"Potential price impact for {symbol} trade",
                                context={
                                    "symbol": symbol,
                                    "trade_quantity": quantity,
                                    "market_volume": volume,
                                    "impact_percent": impact_pct * 100,
                                    "threshold_percent": 1.0
                                }
                            ))
        
        return violations
    
    def _check_concentration(self, context: Dict[str, Any]) -> List[SafetyViolation]:
        """
        Check for excessive concentration in a single asset.
        
        Args:
            context: Check context
            
        Returns:
            A list of safety violations
        """
        violations = []
        strategy = context["strategy"]
        
        # Get the safety check
        check = self.checks["concentration"]
        
        # Check for concentration
        if "allocations" in strategy:
            allocations = strategy["allocations"]
            
            for symbol, allocation in allocations.items():
                if allocation > self.max_concentration:
                    violations.append(SafetyViolation(
                        check=check,
                        message=f"Excessive concentration in {symbol}",
                        context={
                            "symbol": symbol,
                            "allocation": allocation * 100,
                            "threshold": self.max_concentration * 100
                        }
                    ))
        
        return violations
    
    def _check_leverage(self, context: Dict[str, Any]) -> List[SafetyViolation]:
        """
        Check for excessive leverage.
        
        Args:
            context: Check context
            
        Returns:
            A list of safety violations
        """
        violations = []
        strategy = context["strategy"]
        
        # Get the safety check
        check = self.checks["leverage"]
        
        # Check for leverage
        if "leverage" in strategy:
            leverage = strategy["leverage"]
            
            if leverage > self.max_leverage:
                violations.append(SafetyViolation(
                    check=check,
                    message="Excessive leverage detected",
                    context={
                        "leverage": leverage,
                        "threshold": self.max_leverage
                    }
                ))
        
        return violations