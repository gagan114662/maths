
# Using Ollama with DeepSeek R1 and Simplified Memory

This README provides instructions for using Ollama with the DeepSeek R1 model and the simplified memory system that doesn't depend on SQLAlchemy.

## Overview

This implementation replaces:
1. The Anthropic LLM provider with an Ollama provider using DeepSeek R1
2. The SQLAlchemy-based memory system with a simple JSON file-based memory system

## Files Added/Modified

- `src/core/llm/providers/ollama_provider.py` - New Ollama provider implementation
- `src/core/llm/providers/__init__.py` - Updated to include the Ollama provider
- `src/core/llm/factory.py` - New factory for creating LLM providers based on config
- `src/core/simple_memory.py` - New simplified memory manager that doesn't use SQLAlchemy
- `src/config/system_config.py` - Updated to use Ollama with DeepSeek R1 as default
- `examples/standalone_ollama_example.py` - Standalone example that doesn't depend on the project structure

## Usage

### Standalone Example

The simplest way to use the new implementation is with the standalone example:

```bash
python3 examples/standalone_ollama_example.py
```

This example includes a self-contained implementation that:
1. Uses Ollama with DeepSeek R1
2. Implements a simple JSON file-based memory system
3. Demonstrates basic conversation and memory operations

### Integrated Example

If you want to use the integrated implementation with the existing codebase:

```bash
# Make sure Ollama is running and DeepSeek R1 is downloaded
ollama run deepseek-r1

# Configure to use Ollama provider with DeepSeek R1
# (This is already done in system_config.py)

# Use the SimpleMemoryManager instead of the SQLAlchemy-based one
from src.core.simple_memory import SimpleMemoryManager
memory = SimpleMemoryManager(memory_dir="memory")

# Use the factory to create the LLM interface
from src.core.llm import create_llm_interface
from src.config.system_config import LLM_CONFIG
llm = create_llm_interface(LLM_CONFIG)
```

## Prerequisites

- Ollama must be installed and running
- DeepSeek R1 model must be pulled in Ollama (`ollama pull deepseek-r1`)
- Python packages:
  - aiohttp
  - asyncio

## Configuration

You can configure the Ollama provider in `system_config.py`:

```python
LLM_CONFIG = {
    "provider": "ollama",  # Use Ollama provider
    "model": "deepseek-r1",  # Use DeepSeek R1 model
    "temperature": 0.2,
    "max_tokens": 4000,
    "retry_attempts": 3,
    "backoff_factor": 1.5,
    "api_url": "http://localhost:11434/api"  # Ollama API URL
}
```

## Memory System

The new `SimpleMemoryManager` provides a simpler alternative to the SQLAlchemy-based memory manager:

- Uses JSON files instead of a database
- Implements the same basic API (store, retrieve, search, update, delete)
- Maintains a simple index for efficient searching
- Includes a memory cache to reduce file I/O
- Does not support vector search or separate storage for large objects

## Troubleshooting

- Make sure Ollama is running: `ollama serve`
- Check that DeepSeek R1 is downloaded: `ollama list`
- If you get network errors, verify the API URL in the configuration
- If the model doesn't respond as expected, try adjusting temperature or max_tokens

## Limitations

- The Ollama provider does not support all features of the Anthropic provider, such as tool calls
- The simple memory manager does not support vector similarity search
- Ollama models may have different capabilities compared to Anthropic Claude models