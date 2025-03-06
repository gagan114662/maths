"""
Model Context Protocol (MCP) for managing LLM context and interactions in scientific workflows.

The MCP is a key component of the AI Co-Scientist framework, providing:
1. Structured context management for LLM interactions
2. Memory and knowledge persistence across research cycles
3. Efficient context optimization to maximize LLM performance
4. Safety verification for all context data
5. Scientific workflow tracking and reproducibility
"""
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime
from enum import Enum

from ..utils.config import load_config
from .safety_checker import SafetyChecker

logger = logging.getLogger(__name__)

class ContextType(str, Enum):
    """Enumeration of context types."""
    SYSTEM = "system"
    MEMORY = "memory" 
    MARKET = "market"
    AGENT_STATE = "agent_state"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    STRATEGY = "strategy"
    BACKTEST = "backtest"
    ANALYSIS = "analysis"
    SCIENTIFIC_LITERATURE = "scientific_literature"
    USER_GOAL = "user_goal"

class Context:
    """Context container with type information."""
    def __init__(
        self, 
        type: ContextType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.type = type
        self.data = data
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": self.type,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Context':
        """Create from dictionary representation."""
        return cls(
            type=data["type"],
            data=data["data"],
            metadata=data["metadata"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

class ModelContextProtocol:
    """
    Model Context Protocol for scientific research workflows.
    
    The MCP serves as the central nervous system of the AI Co-Scientist framework,
    managing context, memory, and knowledge flow between agents and the LLM.
    
    Key responsibilities:
    1. Context preparation and optimization for LLM interactions
    2. Scientific workflow state tracking
    3. Cross-agent knowledge sharing
    4. Memory management for long-term learning
    5. Safety and ethical verification
    
    Attributes:
        config: Configuration dictionary
        safety_checker: Safety verification instance
        context_history: History of past contexts
        active_contexts: Currently active contexts by type
        context_stats: Usage statistics and performance metrics
    """
    
    def __init__(self, config_path: str = None):
        """Initialize Model Context Protocol."""
        self.config = load_config(config_path) if config_path else {}
        self.safety_checker = SafetyChecker()
        self.context_history = []
        self.active_contexts = {context_type: None for context_type in ContextType}
        self.context_stats = {
            'total_contexts_created': 0,
            'total_tokens_used': 0,
            'context_type_counts': {context_type: 0 for context_type in ContextType},
            'average_tokens_per_type': {context_type: 0 for context_type in ContextType}
        }
        self.max_context_length = self.config.get('max_context_length', 4096)
        self.scientific_workflow_state = {
            'active_hypotheses': [],
            'active_experiments': [],
            'validated_hypotheses': [],
            'rejected_hypotheses': []
        }
        
    def update_context(self, context: Context) -> None:
        """
        Update or add a specific context by type.
        
        Args:
            context: Context object to update or add
        """
        # Validate context with safety checker
        if not self.safety_checker.verify_context(context.data):
            raise ValueError(f"Context of type {context.type} failed safety verification")
            
        # Update active context for this type
        self.active_contexts[context.type] = context
        
        # Update statistics
        self.context_stats['total_contexts_created'] += 1
        self.context_stats['context_type_counts'][context.type] += 1
        
        # Track token usage
        tokens = self._estimate_tokens(context.data)
        self.context_stats['total_tokens_used'] += tokens
        
        # Update average tokens for this type
        current_avg = self.context_stats['average_tokens_per_type'][context.type]
        current_count = self.context_stats['context_type_counts'][context.type]
        new_avg = (current_avg * (current_count - 1) + tokens) / current_count
        self.context_stats['average_tokens_per_type'][context.type] = new_avg
        
        # Update scientific workflow state if appropriate
        if context.type == ContextType.HYPOTHESIS:
            self._update_hypothesis_state(context)
        elif context.type == ContextType.EXPERIMENT:
            self._update_experiment_state(context)
            
        # Track context in history
        self._track_context(context)
        
        logger.debug(f"Updated context of type {context.type}")
    
    def prepare_context(self, contexts: Optional[List[Context]] = None) -> Dict[str, Any]:
        """
        Prepare optimized context for model interaction from active contexts.
        
        Args:
            contexts: Optional additional contexts to include
            
        Returns:
            Prepared context dictionary
        """
        # Base context structure
        base_context = {
            "system": {
                "role": "ai_co_scientist",
                "capabilities": self._get_capabilities(),
                "constraints": self._get_constraints(),
                "timestamp": datetime.now().isoformat()
            },
            "memory": self._get_memory_context(),
            "market": self._get_market_context(),
            "tools": self._get_tool_context(),
            "scientific_workflow": self._get_scientific_workflow_context()
        }
        
        # Add active contexts
        for context_type, context in self.active_contexts.items():
            if context:
                base_context[context_type] = context.data
                
        # Add additional contexts if provided
        if contexts:
            for context in contexts:
                if context.type in base_context:
                    # Merge with existing context of same type
                    base_context[context.type] = self._merge_contexts(
                        base_context[context.type], context.data
                    )
                else:
                    # Add new context type
                    base_context[context.type] = context.data
            
        # Validate context
        if not self.safety_checker.verify_context(base_context):
            raise ValueError("Context failed safety verification")
            
        # Optimize context size
        optimized_context = self._optimize_context(base_context)
        
        # Track full context usage
        self._track_full_context(optimized_context)
        
        return optimized_context
        
    def get_current_context(self) -> Dict[str, Any]:
        """Get the current active context."""
        return self.prepare_context()
    
    def _update_hypothesis_state(self, context: Context) -> None:
        """Update scientific workflow state with hypothesis information."""
        hypothesis_data = context.data
        hypothesis_id = hypothesis_data.get('id')
        
        if not hypothesis_id:
            return
            
        # Check if this hypothesis is already tracked
        existing_ids = [h.get('id') for h in self.scientific_workflow_state['active_hypotheses']]
        
        if hypothesis_id in existing_ids:
            # Update existing hypothesis
            for i, h in enumerate(self.scientific_workflow_state['active_hypotheses']):
                if h.get('id') == hypothesis_id:
                    self.scientific_workflow_state['active_hypotheses'][i] = hypothesis_data
                    break
        else:
            # Add new hypothesis
            self.scientific_workflow_state['active_hypotheses'].append(hypothesis_data)
            
        # Check for state transitions
        state = hypothesis_data.get('state')
        if state == 'validated':
            # Move to validated hypotheses
            self.scientific_workflow_state['validated_hypotheses'].append(hypothesis_data)
            self.scientific_workflow_state['active_hypotheses'] = [
                h for h in self.scientific_workflow_state['active_hypotheses'] 
                if h.get('id') != hypothesis_id
            ]
        elif state == 'rejected':
            # Move to rejected hypotheses
            self.scientific_workflow_state['rejected_hypotheses'].append(hypothesis_data)
            self.scientific_workflow_state['active_hypotheses'] = [
                h for h in self.scientific_workflow_state['active_hypotheses'] 
                if h.get('id') != hypothesis_id
            ]
    
    def _update_experiment_state(self, context: Context) -> None:
        """Update scientific workflow state with experiment information."""
        experiment_data = context.data
        experiment_id = experiment_data.get('id')
        
        if not experiment_id:
            return
            
        # Check if this experiment is already tracked
        existing_ids = [e.get('id') for e in self.scientific_workflow_state['active_experiments']]
        
        if experiment_id in existing_ids:
            # Update existing experiment
            for i, e in enumerate(self.scientific_workflow_state['active_experiments']):
                if e.get('id') == experiment_id:
                    self.scientific_workflow_state['active_experiments'][i] = experiment_data
                    break
        else:
            # Add new experiment
            self.scientific_workflow_state['active_experiments'].append(experiment_data)
            
        # Check for completed experiments
        status = experiment_data.get('status')
        if status == 'completed':
            # Remove from active experiments
            self.scientific_workflow_state['active_experiments'] = [
                e for e in self.scientific_workflow_state['active_experiments'] 
                if e.get('id') != experiment_id
            ]
            
    def _get_scientific_workflow_context(self) -> Dict[str, Any]:
        """Get scientific workflow context with hypotheses and experiments."""
        # Limit the number of items to include
        max_items = 5
        
        return {
            "active_hypotheses": self.scientific_workflow_state['active_hypotheses'][:max_items],
            "active_experiments": self.scientific_workflow_state['active_experiments'][:max_items],
            "validated_hypotheses_count": len(self.scientific_workflow_state['validated_hypotheses']),
            "rejected_hypotheses_count": len(self.scientific_workflow_state['rejected_hypotheses']),
            "recent_validated": self.scientific_workflow_state['validated_hypotheses'][-3:] 
                if self.scientific_workflow_state['validated_hypotheses'] else []
        }
        
    def _get_capabilities(self) -> List[str]:
        """Get system capabilities."""
        return [
            "strategy_generation",
            "risk_assessment",
            "backtesting",
            "optimization",
            "market_analysis"
        ]
        
    def _get_constraints(self) -> List[str]:
        """Get system constraints."""
        return [
            "ethical_trading",
            "position_limits",
            "risk_limits",
            "regulatory_compliance",
            "market_impact"
        ]
        
    def _get_memory_context(self) -> Dict[str, Any]:
        """Get memory context."""
        return {
            "short_term": self._get_recent_history(),
            "long_term": self._get_learned_patterns(),
            "working": self._get_current_context()
        }
        
    def _get_market_context(self) -> Dict[str, Any]:
        """Get market context."""
        return {
            "current_state": self._get_market_state(),
            "indicators": self._get_market_indicators(),
            "sentiment": self._get_market_sentiment()
        }
        
    def _get_tool_context(self) -> Dict[str, Any]:
        """Get tool availability context."""
        return {
            "available": self._get_available_tools(),
            "permissions": self._get_tool_permissions(),
            "constraints": self._get_tool_constraints()
        }
        
    def _get_recent_history(self) -> List[Dict[str, Any]]:
        """Get recent context history."""
        return self.context_history[-10:] if self.context_history else []
        
    def _get_learned_patterns(self) -> Dict[str, Any]:
        """Get learned patterns from long-term memory."""
        # Implement memory retrieval logic
        return {}
        
    def _get_current_context(self) -> Dict[str, Any]:
        """Get current working context."""
        return {
            "active_strategies": self._get_active_strategies(),
            "pending_decisions": self._get_pending_decisions(),
            "recent_actions": self._get_recent_actions()
        }
        
    def _get_market_state(self) -> Dict[str, Any]:
        """Get current market state."""
        # Implement market state retrieval
        return {}
        
    def _get_market_indicators(self) -> Dict[str, Any]:
        """Get current market indicators."""
        # Implement indicator retrieval
        return {}
        
    def _get_market_sentiment(self) -> Dict[str, Any]:
        """Get current market sentiment."""
        # Implement sentiment analysis
        return {}
        
    def _get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return [
            "data_analysis",
            "backtesting",
            "risk_assessment",
            "optimization"
        ]
        
    def _get_tool_permissions(self) -> Dict[str, List[str]]:
        """Get tool permissions."""
        return {
            "read": ["market_data", "indicators", "analysis"],
            "write": ["strategy_params", "risk_limits"],
            "execute": ["backtests", "simulations"]
        }
        
    def _get_tool_constraints(self) -> Dict[str, Any]:
        """Get tool constraints."""
        return {
            "position_limits": self.config.get('position_limits', {}),
            "risk_limits": self.config.get('risk_limits', {}),
            "rate_limits": self.config.get('rate_limits', {})
        }
        
    def _merge_contexts(self, base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Merge contexts with conflict resolution."""
        merged = base.copy()
        
        for key, value in new.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_contexts(merged[key], value)
            else:
                merged[key] = value
                
        return merged
        
    def _optimize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize context to fit within token limits."""
        # Calculate current size
        context_size = self._estimate_tokens(context)
        
        # If within limits, return as is
        if context_size <= self.max_context_length:
            return context
            
        # Optimize by removing less critical information
        optimized = context.copy()
        
        # Reduce history length
        if 'memory' in optimized:
            optimized['memory']['short_term'] = optimized['memory']['short_term'][-5:]
            
        # Remove detailed market data if needed
        if context_size > self.max_context_length:
            if 'market' in optimized:
                optimized['market'] = {
                    'current_state': optimized['market'].get('current_state', {}),
                    'summary': "Detailed market data omitted for context optimization"
                }
                
        return optimized
        
    def _estimate_tokens(self, context: Dict[str, Any]) -> int:
        """Estimate token count for context."""
        # Simple estimation based on character count
        # In practice, implement more accurate token counting
        return len(str(context)) // 4
        
    def _track_context(self, context: Dict[str, Any]) -> None:
        """Track context usage."""
        self.context_history.append({
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'tokens': self._estimate_tokens(context)
        })
        
        # Maintain history size
        max_history = self.config.get('max_history', 100)
        if len(self.context_history) > max_history:
            self.context_history = self.context_history[-max_history:]
            
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context usage statistics."""
        return {
            'total_contexts': len(self.context_history),
            'average_tokens': sum(c['tokens'] for c in self.context_history) / len(self.context_history) if self.context_history else 0,
            'last_context_tokens': self.context_history[-1]['tokens'] if self.context_history else 0
        }
        
    def clear_history(self) -> None:
        """Clear context history."""
        self.context_history = []