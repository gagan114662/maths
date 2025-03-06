#\!/usr/bin/env python3
"""
Test script to verify that agents initialize correctly.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.core.llm import create_llm_interface
from src.core.mcp import ModelContextProtocol
from src.core.safety import SafetyChecker
from src.core.simple_memory import SimpleMemoryManager
from src.agents.base_agent import AgentType
from src.agents.supervisor.supervisor_agent import SupervisorAgent

async def main():
    """Test agent initialization."""
    try:
        # Initialize components
        llm = create_llm_interface({"provider": "ollama", "model": "deepseek-r1"})
        mcp = ModelContextProtocol()
        memory = SimpleMemoryManager(memory_dir="test_memory")
        safety = SafetyChecker()
        
        # Initialize supervisor agent
        logger.info("Initializing supervisor agent...")
        supervisor = SupervisorAgent(
            name="supervisor",
            config={},
            llm=llm,
            mcp=mcp,
            safety=safety,
            memory=memory
        )
        
        # Initialize the agent
        init_result = await supervisor.initialize()
        logger.info(f"Supervisor initialization result: {init_result}")
        
        # Start the agent
        start_result = await supervisor.start()
        logger.info(f"Supervisor start result: {start_result}")
        logger.info(f"Supervisor state: {supervisor.state}")
        
        # Test sending a simple message
        logger.info("Sending test message to supervisor...")
        response = await supervisor.send_message({
            "type": "query",
            "content": {"query_type": "get_status"}
        })
        logger.info(f"Response: {response}")
        
        # Stop the agent
        supervisor.stop()
        logger.info("Agent test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in agent test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
