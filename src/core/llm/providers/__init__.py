"""
LLM Provider implementations for the AI co-scientist system.
"""
# Import only the Ollama provider for DeepSeek R1
from .ollama_provider import OllamaProvider

# Import DeepSeek GOD MODE
try:
    from .deepseek_god_mode import DeepSeekGodMode
except ImportError:
    # DeepSeek GOD MODE not available
    pass
