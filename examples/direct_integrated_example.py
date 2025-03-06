#!/usr/bin/env python3
"""
Direct integration example that uses the core components without relying on
the existing project imports. This directly imports our new components.
"""
import os
import sys
import json
import logging
import asyncio
import aiohttp
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import our directly created components
from src.core.llm.providers.ollama_provider import OllamaProvider
from src.core.llm.interface import LLMInterface, Message, MessageRole
from src.core.simple_memory import SimpleMemoryManager

# ===== Main Function =====

async def main():
    """Run the integrated example."""
    logger.info("Starting direct integrated example")
    
    # Initialize memory manager
    memory = SimpleMemoryManager(memory_dir="direct_memory")
    logger.info("Initialized simple memory manager")
    
    # Initialize Ollama provider configuration
    config = {
        "provider": "ollama",
        "model": "deepseek-r1",
        "temperature": 0.2,
        "max_tokens": 4000,
        "retry_attempts": 3,
        "backoff_factor": 1.5,
        "api_url": "http://localhost:11434/api"
    }
    
    # Create Ollama provider
    ollama_provider = OllamaProvider(api_url=config["api_url"], config=config)
    logger.info(f"Created Ollama provider for {config['model']}")
    
    # Create LLM interface with provider
    llm = LLMInterface(provider=ollama_provider, config=config)
    logger.info("Created LLM interface")
    
    # Store example content in memory
    content_id = memory.store(
        memory_type="example",
        content={"text": "This is an example from the integrated solution."},
        metadata={"source": "direct_integrated_example", "version": "1.0"}
    )
    logger.info(f"Stored content in memory with ID: {content_id}")
    
    # Create conversation
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful AI assistant that provides concise answers."),
        Message(role=MessageRole.USER, content="What are the three primary colors?")
    ]
    
    try:
        # Generate response
        logger.info("Generating response from DeepSeek R1 via Ollama...")
        response = await llm.generate(messages=messages)
        
        # Process response
        response_text = response.message.content
        logger.info(f"LLM Response: {response_text}")
        
        # Store in memory
        response_id = memory.store(
            memory_type="llm_response",
            content={"query": "What are the three primary colors?", "response": response_text},
            metadata={"model": config["model"]}
        )
        logger.info(f"Stored response in memory with ID: {response_id}")
        
        # Retrieve and display stored response
        stored = memory.retrieve(response_id)
        logger.info(f"Retrieved from memory: {stored['content']['response']}")
        
        # Update with additional metadata
        memory.update(
            memory_id=response_id,
            metadata={"importance": "high", "category": "educational"}
        )
        logger.info("Updated memory entry with additional metadata")
        
        # Search memory
        results = memory.search(memory_type="llm_response")
        logger.info(f"Found {len(results)} LLM responses in memory")
        
    except Exception as e:
        logger.error(f"Error during LLM processing: {str(e)}")
    
    # Add a follow-up question
    if 'response' in locals():
        # Add the assistant's response to the conversation
        messages.append(
            Message(
                role=MessageRole.ASSISTANT, 
                content=response.message.content
            )
        )
        
        # Add a follow-up question
        messages.append(
            Message(
                role=MessageRole.USER, 
                content="Can you explain why these colors are considered primary?"
            )
        )
        
        try:
            # Generate response to follow-up
            logger.info("Generating response to follow-up question...")
            follow_up_response = await llm.generate(messages=messages)
            
            # Display response
            logger.info(f"Follow-up response: {follow_up_response.message.content}")
            
            # Store in memory
            memory.store(
                memory_type="conversation",
                content={
                    "messages": [msg.to_dict() for msg in messages],
                    "final_response": follow_up_response.message.content
                },
                metadata={"type": "follow_up", "model": config["model"]}
            )
            logger.info("Stored complete conversation in memory")
            
        except Exception as e:
            logger.error(f"Error during follow-up: {str(e)}")
    
    # Clean up
    logger.info("Clearing memory")
    memory.clear()
    
    logger.info("Direct integrated example completed successfully")

if __name__ == "__main__":
    asyncio.run(main())