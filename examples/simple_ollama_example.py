"""
Simple example demonstrating the use of Ollama with DeepSeek R1 model
and the simplified memory manager.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.llm import (
    LLMInterface, Message, MessageRole,
    create_llm_interface
)
from src.core.simple_memory import SimpleMemoryManager
from src.config.system_config import LLM_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    """Run a simple example with Ollama and DeepSeek R1."""
    # Initialize LLM interface using DeepSeek R1 via Ollama
    llm = create_llm_interface(LLM_CONFIG)
    
    # Initialize the simple memory manager
    memory = SimpleMemoryManager(memory_dir="example_memory")
    
    # Store a simple memory entry
    memory_id = memory.store(
        memory_type="example",
        content={"text": "This is an example memory entry."},
        metadata={"source": "simple_example", "importance": "high"}
    )
    logger.info(f"Stored memory with ID: {memory_id}")
    
    # Retrieve the memory entry
    entry = memory.retrieve(memory_id)
    logger.info(f"Retrieved memory: {entry}")
    
    # Create a conversation with the LLM
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful AI assistant that provides concise answers."),
        Message(role=MessageRole.USER, content="What's the capital of France?")
    ]
    
    try:
        # Generate a response
        response = await llm.generate(messages=messages)
        
        # Print the response
        logger.info(f"LLM Response: {response.message.content}")
        
        # Store the response in memory
        memory_id = memory.store(
            memory_type="llm_response",
            content={"query": "What's the capital of France?", "response": response.message.content},
            metadata={"model": response.model}
        )
        logger.info(f"Stored LLM response with ID: {memory_id}")
        
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
    
    # Search for memory entries
    results = memory.search(memory_type="llm_response")
    logger.info(f"Search results: {results}")
    
    # Clean up
    logger.info("Clearing memory")
    memory.clear()

if __name__ == "__main__":
    asyncio.run(main())