"""
Example showing how to integrate Ollama with DeepSeek R1 into the existing codebase.
This example uses the proper factory method and system configuration.
"""
import asyncio
import logging
import os
import sys
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

# Import from the system codebase
from src.core.llm import (
    create_llm_interface, 
    Message, 
    MessageRole
)
from src.core.simple_memory import SimpleMemoryManager
from src.config.system_config import LLM_CONFIG, get_full_config

async def main():
    """Run an integrated example with the core codebase."""
    logger.info("Starting integrated example with Ollama/DeepSeek R1")
    
    # Create LLM interface using the factory
    llm = create_llm_interface(LLM_CONFIG)
    logger.info(f"Created LLM interface with {LLM_CONFIG['provider']} provider and {LLM_CONFIG['model']} model")
    
    # Initialize simple memory manager
    memory = SimpleMemoryManager(memory_dir="integrated_memory")
    logger.info("Initialized simple memory manager")
    
    # Store example content in memory
    content_id = memory.store(
        memory_type="example",
        content={"text": "This is an example from the integrated solution."},
        metadata={"source": "integrated_example", "version": "1.0"}
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
            metadata={"model": LLM_CONFIG["model"]}
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
                metadata={"type": "follow_up", "model": LLM_CONFIG["model"]}
            )
            logger.info("Stored complete conversation in memory")
            
        except Exception as e:
            logger.error(f"Error during follow-up: {str(e)}")
    
    # Clean up
    logger.info("Clearing memory")
    memory.clear()
    
    logger.info("Integrated example completed successfully")

if __name__ == "__main__":
    asyncio.run(main())