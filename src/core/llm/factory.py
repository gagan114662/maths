"""
Factory for creating LLM providers based on configuration.
"""
import logging
from typing import Dict, Any, Optional

from .interface import LLMInterface, LLMProvider
from .providers import OllamaProvider

logger = logging.getLogger(__name__)

def create_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """
    Create an LLM provider based on configuration.
    
    Args:
        config: LLM configuration dictionary
        
    Returns:
        An LLM provider instance
        
    Raises:
        ValueError: If the provider is not supported
    """
    provider_name = config.get("provider", "ollama").lower()
    
    if provider_name == "ollama":
        logger.info("Creating Ollama provider for DeepSeek R1")
        return OllamaProvider(
            api_url=config.get("api_url"),
            config=config
        )
    else:
        # Default to Ollama provider if specified provider is not supported
        logger.warning(f"Provider '{provider_name}' not supported, defaulting to Ollama for DeepSeek R1")
        return OllamaProvider(
            api_url=config.get("api_url"),
            config=config
        )

def create_llm_interface(config: Dict[str, Any]) -> LLMInterface:
    """
    Create an LLM interface with the configured provider.
    
    Args:
        config: LLM configuration dictionary
        
    Returns:
        An LLM interface instance
    """
    provider = create_llm_provider(config)
    return LLMInterface(provider=provider, config=config)