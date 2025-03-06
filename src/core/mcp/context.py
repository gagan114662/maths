"""
Context module for the Model Context Protocol.

This module defines the core context structures used throughout the system.
"""
import enum
from datetime import datetime
from typing import Dict, List, Any, Optional, Protocol, Set
from pydantic import BaseModel, Field


class ContextType(str, enum.Enum):
    """Types of context that can be provided to the LLM."""
    SYSTEM = "system"
    MARKET_DATA = "market_data"
    TECHNICAL_INDICATORS = "technical_indicators"
    FUNDAMENTALS = "fundamentals"
    SENTIMENT = "sentiment"
    STRATEGY = "strategy"
    BACKTEST_RESULTS = "backtest_results"
    RISK_ANALYSIS = "risk_analysis"
    PORTFOLIO = "portfolio"
    AGENT_STATE = "agent_state"
    MEMORY = "memory"
    USER_PREFERENCES = "user_preferences"
    USER_GOAL = "user_goal"


class Context(BaseModel):
    """
    A context structure that can be provided to an LLM.
    
    Attributes:
        type: The type of context
        data: The context data
        metadata: Additional metadata about the context
        timestamp: When the context was created
    """
    type: ContextType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    def update(self, new_data: Dict[str, Any]) -> 'Context':
        """Update the context with new data."""
        self.data.update(new_data)
        self.timestamp = datetime.now()
        return self
    
    def merge(self, other_context: 'Context') -> 'Context':
        """Merge another context into this one."""
        if self.type != other_context.type:
            raise ValueError(f"Cannot merge contexts of different types: {self.type} vs {other_context.type}")
        
        self.data.update(other_context.data)
        self.metadata.update(other_context.metadata)
        self.timestamp = datetime.now()
        return self


class ContextProvider(Protocol):
    """Protocol for objects that can provide context."""
    
    def get_context(self, context_type: ContextType, **kwargs) -> Context:
        """
        Get context of the specified type.
        
        Args:
            context_type: The type of context to retrieve
            **kwargs: Additional parameters for context retrieval
            
        Returns:
            A Context object
        """
        ...