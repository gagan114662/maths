"""
Safety checker for validating LLM interactions and trading operations.
"""
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..utils.config import load_config

logger = logging.getLogger(__name__)

class SafetyChecker:
    """
    Safety verification system for LLM interactions and trading operations.
    
    Attributes:
        config: Configuration dictionary
        violation_history: List of safety violations
    """
    
    def __init__(self, config_path: str = None):
        """Initialize safety checker."""
        self.config = load_config(config_path) if config_path else {}
        self.violation_history = []
        
        # Load safety rules
        self.content_rules = self._load_content_rules()
        self.trading_rules = self._load_trading_rules()
        self.risk_limits = self._load_risk_limits()
        
    def verify_prompt(self, prompt: str, context: Dict[str, Any]) -> bool:
        """
        Verify if prompt is safe to send to LLM.
        
        Args:
            prompt: Input prompt
            context: Context dictionary
            
        Returns:
            bool: Whether prompt is safe
        """
        try:
            # Check for dangerous patterns
            if self._contains_dangerous_patterns(prompt):
                self._log_violation("dangerous_pattern", prompt)
                return False
                
            # Check for sensitive information
            if self._contains_sensitive_info(prompt):
                self._log_violation("sensitive_info", prompt)
                return False
                
            # Verify context safety
            if not self.verify_context(context):
                return False
                
            # Check token limits
            if len(prompt.split()) > self.config.get('max_prompt_tokens', 4096):
                self._log_violation("token_limit", prompt)
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in prompt verification: {str(e)}")
            return False
            
    def verify_response(self, response: str) -> bool:
        """
        Verify if LLM response is safe.
        
        Args:
            response: LLM response
            
        Returns:
            bool: Whether response is safe
        """
        try:
            # Check for dangerous patterns
            if self._contains_dangerous_patterns(response):
                self._log_violation("dangerous_pattern", response)
                return False
                
            # Check for trading violations
            if self._contains_trading_violations(response):
                self._log_violation("trading_violation", response)
                return False
                
            # Verify response format
            if not self._verify_response_format(response):
                self._log_violation("invalid_format", response)
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in response verification: {str(e)}")
            return False
            
    def verify_context(self, context: Dict[str, Any]) -> bool:
        """
        Verify if context is safe.
        
        Args:
            context: Context dictionary
            
        Returns:
            bool: Whether context is safe
        """
        try:
            # Check required fields
            if not self._verify_required_fields(context):
                self._log_violation("missing_fields", str(context))
                return False
                
            # Check context size
            if not self._verify_context_size(context):
                self._log_violation("context_size", str(context))
                return False
                
            # Verify trading constraints
            if not self._verify_trading_constraints(context):
                self._log_violation("trading_constraints", str(context))
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in context verification: {str(e)}")
            return False
            
    def verify_trading_action(self, action: Dict[str, Any]) -> bool:
        """
        Verify if trading action is safe.
        
        Args:
            action: Trading action dictionary
            
        Returns:
            bool: Whether action is safe
        """
        try:
            # Verify position sizes
            if not self._verify_position_size(action):
                self._log_violation("position_size", str(action))
                return False
                
            # Check risk limits
            if not self._verify_risk_limits(action):
                self._log_violation("risk_limits", str(action))
                return False
                
            # Verify trading rules
            if not self._verify_trading_rules(action):
                self._log_violation("trading_rules", str(action))
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in trading action verification: {str(e)}")
            return False
            
    def _load_content_rules(self) -> Dict[str, List[str]]:
        """Load content safety rules."""
        return {
            'dangerous_patterns': [
                r'exec\(',
                r'eval\(',
                r'system\(',
                r'subprocess',
                r'os\.',
                r'delete.*database',
                r'drop.*table'
            ],
            'sensitive_info': [
                r'api[_-]key',
                r'password',
                r'secret',
                r'token',
                r'credential'
            ]
        }
        
    def _load_trading_rules(self) -> Dict[str, Any]:
        """Load trading safety rules."""
        return {
            'max_position_size': self.config.get('max_position_size', 0.1),
            'max_leverage': self.config.get('max_leverage', 1.0),
            'restricted_assets': self.config.get('restricted_assets', []),
            'trading_hours': self.config.get('trading_hours', {'start': '09:30', 'end': '16:00'}),
            'min_liquidity': self.config.get('min_liquidity', 1000000)
        }
        
    def _load_risk_limits(self) -> Dict[str, float]:
        """Load risk limits."""
        return {
            'max_drawdown': self.config.get('max_drawdown', 0.2),
            'var_limit': self.config.get('var_limit', 0.05),
            'position_limit': self.config.get('position_limit', 0.1),
            'concentration_limit': self.config.get('concentration_limit', 0.2)
        }
        
    def _contains_dangerous_patterns(self, text: str) -> bool:
        """Check for dangerous patterns in text."""
        for pattern in self.content_rules['dangerous_patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
        
    def _contains_sensitive_info(self, text: str) -> bool:
        """Check for sensitive information in text."""
        for pattern in self.content_rules['sensitive_info']:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
        
    def _contains_trading_violations(self, text: str) -> bool:
        """Check for trading rule violations in text."""
        # Implementation depends on specific trading rules
        return False
        
    def _verify_response_format(self, response: str) -> bool:
        """Verify response format."""
        # Implementation depends on expected format
        return True
        
    def _verify_required_fields(self, context: Dict[str, Any]) -> bool:
        """Verify required fields in context."""
        required_fields = ['system', 'memory', 'market', 'tools']
        return all(field in context for field in required_fields)
        
    def _verify_context_size(self, context: Dict[str, Any]) -> bool:
        """Verify context size is within limits."""
        # Simple size check based on string representation
        return len(str(context)) <= self.config.get('max_context_size', 32768)
        
    def _verify_trading_constraints(self, context: Dict[str, Any]) -> bool:
        """Verify trading constraints in context."""
        if 'trading' not in context:
            return True
            
        trading = context['trading']
        
        # Check position sizes
        if 'position_size' in trading:
            if trading['position_size'] > self.trading_rules['max_position_size']:
                return False
                
        # Check leverage
        if 'leverage' in trading:
            if trading['leverage'] > self.trading_rules['max_leverage']:
                return False
                
        return True
        
    def _verify_position_size(self, action: Dict[str, Any]) -> bool:
        """Verify position size is within limits."""
        if 'size' not in action:
            return False
            
        return action['size'] <= self.trading_rules['max_position_size']
        
    def _verify_risk_limits(self, action: Dict[str, Any]) -> bool:
        """Verify risk limits are not exceeded."""
        if 'risk_metrics' not in action:
            return False
            
        metrics = action['risk_metrics']
        
        # Check against risk limits
        for metric, limit in self.risk_limits.items():
            if metric in metrics and metrics[metric] > limit:
                return False
                
        return True
        
    def _verify_trading_rules(self, action: Dict[str, Any]) -> bool:
        """Verify trading rules are followed."""
        if 'asset' in action:
            if action['asset'] in self.trading_rules['restricted_assets']:
                return False
                
        if 'time' in action:
            if not self._is_within_trading_hours(action['time']):
                return False
                
        return True
        
    def _is_within_trading_hours(self, time_str: str) -> bool:
        """Check if time is within trading hours."""
        try:
            time = datetime.strptime(time_str, '%H:%M').time()
            start = datetime.strptime(self.trading_rules['trading_hours']['start'], '%H:%M').time()
            end = datetime.strptime(self.trading_rules['trading_hours']['end'], '%H:%M').time()
            return start <= time <= end
        except:
            return False
            
    def _log_violation(self, violation_type: str, content: str) -> None:
        """Log safety violation."""
        violation = {
            'timestamp': datetime.now().isoformat(),
            'type': violation_type,
            'content': content[:1000],  # Truncate for storage
            'context': {}
        }
        
        self.violation_history.append(violation)
        logger.warning(f"Safety violation: {violation_type}")
        
    def get_violations(self) -> List[Dict[str, Any]]:
        """Get violation history."""
        return self.violation_history.copy()
        
    def clear_violations(self) -> None:
        """Clear violation history."""
        self.violation_history = []