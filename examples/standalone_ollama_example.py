"""
Standalone example of using Ollama with DeepSeek R1, without 
relying on the existing project structure.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ===== Simple Memory Manager =====

class SimpleMemoryManager:
    """A simple memory manager that stores entries as JSON files."""
    
    def __init__(self, memory_dir: str = "memory"):
        """Initialize with path to memory directory."""
        self.memory_dir = Path(memory_dir)
        self.entries_dir = self.memory_dir / "entries"
        self.index_path = self.memory_dir / "index.json"
        self.cache = {}
        
        # Create directories
        self.memory_dir.mkdir(exist_ok=True, parents=True)
        self.entries_dir.mkdir(exist_ok=True)
        
        # Create or load index
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {"entries": [], "next_id": 1}
            self._save_index()
    
    def _save_index(self):
        """Save the index to disk."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def store(self, memory_type: str, content: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> int:
        """Store a memory entry."""
        # Get next ID
        memory_id = self.index["next_id"]
        self.index["next_id"] += 1
        
        # Create entry
        timestamp = datetime.now().isoformat()
        entry = {
            "id": memory_id,
            "type": memory_type,
            "timestamp": timestamp,
            "content": content,
            "metadata": metadata or {}
        }
        
        # Save to file
        entry_path = self.entries_dir / f"{memory_id}.json"
        with open(entry_path, 'w') as f:
            json.dump(entry, f, indent=2)
        
        # Update index
        self.index["entries"].append({
            "id": memory_id,
            "type": memory_type,
            "timestamp": timestamp
        })
        self._save_index()
        
        # Update cache
        self.cache[memory_id] = entry
        
        return memory_id
    
    def retrieve(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a memory entry by ID."""
        # Check cache
        if memory_id in self.cache:
            return self.cache[memory_id]
        
        # Load from file
        entry_path = self.entries_dir / f"{memory_id}.json"
        if not entry_path.exists():
            return None
        
        with open(entry_path, 'r') as f:
            entry = json.load(f)
        
        # Update cache
        self.cache[memory_id] = entry
        
        return entry
    
    def search(self, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for memory entries by type."""
        results = []
        
        for idx_entry in self.index["entries"]:
            if memory_type and idx_entry["type"] != memory_type:
                continue
            
            entry = self.retrieve(idx_entry["id"])
            if entry:
                results.append(entry)
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return results
    
    def clear(self):
        """Clear all memory entries."""
        # Delete all entry files
        for entry_file in self.entries_dir.glob("*.json"):
            entry_file.unlink()
        
        # Reset index
        self.index = {"entries": [], "next_id": 1}
        self._save_index()
        
        # Clear cache
        self.cache = {}

# ===== Ollama Provider =====

class MessageRole(str, Enum):
    """Message roles for LLM conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class OllamaClient:
    """Simple client for interacting with Ollama API."""
    
    def __init__(self, model: str = "deepseek-r1", api_url: str = "http://localhost:11434/api"):
        """Initialize with model name and API URL."""
        self.model = model
        self.api_url = api_url
        logger.info(f"Initialized Ollama client with model {model}")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate a response from the LLM."""
        # Prepare parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        
        # Add max tokens if specified
        if max_tokens is not None:
            params["num_predict"] = max_tokens
        
        try:
            # Make API call
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/chat", json=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Ollama API returned status {response.status}: {error_text}")
                    
                    result = await response.json()
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {str(e)}")
            raise

# ===== Main Function =====

async def main():
    """Run the example."""
    # Initialize memory manager
    memory = SimpleMemoryManager(memory_dir="standalone_memory")
    
    # Initialize Ollama client
    ollama = OllamaClient(model="deepseek-r1")
    
    # Create a simple memory entry
    memory_id = memory.store(
        memory_type="example",
        content={"text": "This is a test memory entry"},
        metadata={"source": "standalone_example"}
    )
    logger.info(f"Created memory entry with ID: {memory_id}")
    
    # Retrieve the entry
    entry = memory.retrieve(memory_id)
    logger.info(f"Retrieved entry: {entry}")
    
    # Create messages for Ollama
    messages = [
        {"role": MessageRole.SYSTEM.value, "content": "You are a helpful AI assistant that provides concise answers."},
        {"role": MessageRole.USER.value, "content": "What's the capital of France?"}
    ]
    
    try:
        # Generate response
        response = await ollama.generate(messages=messages)
        
        # Print full response for debugging
        logger.info(f"Full Ollama response: {response}")
        
        # Parse response text from message content
        message = response.get('message', {})
        response_text = message.get('content', '')
        
        # Strip <think> blocks from the response
        if "<think>" in response_text and "</think>" in response_text:
            response_text = response_text.split("</think>")[-1].strip()
            
        logger.info(f"Ollama response text: {response_text}")
        
        # Store in memory
        memory.store(
            memory_type="llm_response",
            content={"query": "What's the capital of France?", "response": response_text},
            metadata={"model": ollama.model}
        )
        
        # Search memory
        responses = memory.search(memory_type="llm_response")
        logger.info(f"Found {len(responses)} LLM responses")
        
        # Display the stored response
        if responses:
            stored_response = responses[0]
            logger.info(f"Stored response: {stored_response['content']['response']}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    
    # Clean up
    memory.clear()
    logger.info("Memory cleared")

if __name__ == "__main__":
    asyncio.run(main())