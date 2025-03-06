"""
Model Context Protocol (MCP) implementation.

This module implements the core MCP that standardizes context between LLMs and tools.
"""
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from .context import Context, ContextType, ContextProvider

logger = logging.getLogger(__name__)

class ModelContextProtocol:
    """
    Model Context Protocol for standardizing LLM context.
    
    The MCP provides a standardized way to manage context for LLM interactions,
    including context retrieval, updates, and integration with various tools.
    """
    
    def __init__(self):
        """Initialize the Model Context Protocol."""
        self.providers: Dict[ContextType, List[ContextProvider]] = {}
        self.current_contexts: Dict[ContextType, Context] = {}
        self.context_history: List[Dict[ContextType, Context]] = []
        self.max_history_size = 10  # Store last 10 context states
    
    def register_provider(self, provider: ContextProvider, context_types: List[ContextType]) -> None:
        """
        Register a context provider for specific context types.
        
        Args:
            provider: The context provider to register
            context_types: The types of context this provider can provide
        """
        for context_type in context_types:
            if context_type not in self.providers:
                self.providers[context_type] = []
            
            self.providers[context_type].append(provider)
            logger.debug(f"Registered provider for context type: {context_type}")
    
    def get_context(self, context_type: ContextType, **kwargs) -> Optional[Context]:
        """
        Get context of the specified type from registered providers.
        
        Args:
            context_type: The type of context to retrieve
            **kwargs: Additional parameters to pass to providers
            
        Returns:
            A Context object, or None if no context could be retrieved
        """
        # Check if we have a current context of this type
        if context_type in self.current_contexts:
            context = self.current_contexts[context_type]
            logger.debug(f"Using existing context for type: {context_type}")
            return context
        
        # If no current context, try to get one from providers
        if context_type not in self.providers or not self.providers[context_type]:
            logger.warning(f"No providers registered for context type: {context_type}")
            return None
        
        # Try each provider in order
        for provider in self.providers[context_type]:
            try:
                context = provider.get_context(context_type, **kwargs)
                if context:
                    self.current_contexts[context_type] = context
                    logger.debug(f"Got context from provider for type: {context_type}")
                    return context
            except Exception as e:
                logger.error(f"Error getting context from provider: {str(e)}")
        
        logger.warning(f"No context could be retrieved for type: {context_type}")
        return None
    
    def update_context(self, context: Context) -> None:
        """
        Update the current context of a specific type.
        
        Args:
            context: The new context to use
        """
        # Save current state to history before updating
        self._save_to_history()
        
        # Update the context
        self.current_contexts[context.type] = context
        logger.debug(f"Updated context for type: {context.type}")
    
    def merge_context(self, context: Context) -> None:
        """
        Merge a new context with the existing context of the same type.
        
        Args:
            context: The context to merge
        """
        # Save current state to history before merging
        self._save_to_history()
        
        # Check if we have an existing context of this type
        if context.type in self.current_contexts:
            existing_context = self.current_contexts[context.type]
            existing_context.merge(context)
            logger.debug(f"Merged context for type: {context.type}")
        else:
            # If no existing context, just update
            self.update_context(context)
    
    def get_all_contexts(self) -> Dict[str, Any]:
        """
        Get all current contexts as a single dictionary.
        
        Returns:
            A dictionary containing all current contexts
        """
        result = {}
        
        for context_type, context in self.current_contexts.items():
            result[context_type.value] = {
                "data": context.data,
                "metadata": context.metadata,
                "timestamp": context.timestamp.isoformat(),
            }
        
        return result
    
    def clear_context(self, context_type: Optional[ContextType] = None) -> None:
        """
        Clear the current context.
        
        Args:
            context_type: Specific context type to clear, or None to clear all
        """
        # Save current state to history before clearing
        self._save_to_history()
        
        if context_type:
            if context_type in self.current_contexts:
                del self.current_contexts[context_type]
                logger.debug(f"Cleared context for type: {context_type}")
        else:
            self.current_contexts = {}
            logger.debug("Cleared all contexts")
    
    def rollback_context(self) -> bool:
        """
        Rollback to the previous context state.
        
        Returns:
            True if successful, False if no history to rollback to
        """
        if not self.context_history:
            logger.warning("No context history to rollback to")
            return False
        
        self.current_contexts = self.context_history.pop()
        logger.debug("Rolled back to previous context state")
        return True
    
    def _save_to_history(self) -> None:
        """Save the current context state to history."""
        # Create a deep copy of current contexts
        current_copy = {k: v.model_copy(deep=True) for k, v in self.current_contexts.items()}
        
        # Add to history
        self.context_history.append(current_copy)
        
        # Trim history if needed
        if len(self.context_history) > self.max_history_size:
            self.context_history.pop(0)
            
        logger.debug(f"Saved context state to history (size: {len(self.context_history)})")