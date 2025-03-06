#!/usr/bin/env python3
"""
Completely standalone example that includes all the necessary components
without relying on imports from the existing project structure.

This file contains:
1. A simple memory manager implementation
2. An Ollama client implementation for DeepSeek R1
3. An example using these components together
"""
import os
import sys
import json
import logging
import asyncio
import aiohttp
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Protocol
from datetime import datetime
from pathlib import Path
import shutil
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ===== Message and LLM Interface =====

class MessageRole(str, Enum):
    """Role types for messages in LLM conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class Message(BaseModel):
    """A message in an LLM conversation."""
    role: MessageRole
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by LLM APIs."""
        result = {"role": self.role.value}
        
        if self.content is not None:
            result["content"] = self.content
            
        if self.name is not None:
            result["name"] = self.name
            
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
            
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
            
        return result


class LLMResponse(BaseModel):
    """Response from an LLM."""
    message: Message
    usage: Dict[str, int] = Field(default_factory=dict)
    model: str
    created_at: datetime = Field(default_factory=datetime.now)
    finish_reason: Optional[str] = None
    raw_response: Dict[str, Any] = Field(default_factory=dict)


class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    async def generate(
        self,
        messages: List[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        ...


class OllamaProvider:
    """Ollama provider for the LLM interface."""
    
    def __init__(self, api_url: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the Ollama provider."""
        self.config = config or {}
        
        # Get API URL from environment, config, or default
        self.api_url = (
            api_url or 
            os.environ.get("OLLAMA_API_URL") or 
            self.config.get("api_url") or 
            "http://localhost:11434/api"
        )
        
        # Set default model
        self.default_model = self.config.get("model", "deepseek-r1")
        
        logger.info(f"Ollama provider initialized with API URL {self.api_url} and model {self.default_model}")
    
    def _convert_to_ollama_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert our message format to Ollama's format."""
        ollama_messages = []
        
        for message in messages:
            # Create base message
            ollama_message = {
                "role": message.role.value
            }
            
            # Add content if present
            if message.content is not None:
                ollama_message["content"] = message.content
                
            # Add name for function/tool messages if present (may not be supported)
            if message.name is not None and message.role in [MessageRole.TOOL, MessageRole.FUNCTION]:
                ollama_message["name"] = message.name
            
            ollama_messages.append(ollama_message)
            
        return ollama_messages
    
    def _parse_ollama_response(
        self, 
        response: Dict[str, Any],
        model: str
    ) -> LLMResponse:
        """Parse Ollama response into our format."""
        # Convert response message to our format 
        message_content = ""
        if "message" in response and "content" in response["message"]:
            message_content = response["message"]["content"]
        elif "response" in response:
            message_content = response["response"]
        
        # Strip <think> blocks from the response
        if "<think>" in message_content and "</think>" in message_content:
            message_content = message_content.split("</think>")[-1].strip()
            
        message = Message(
            role=MessageRole.ASSISTANT,
            content=message_content
        )
        
        # Create usage dictionary if available
        usage = {}
        eval_count = response.get("eval_count", 0)
        prompt_eval_count = response.get("prompt_eval_count", 0)
        
        if eval_count or prompt_eval_count:
            usage = {
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count - prompt_eval_count,
                "total_tokens": eval_count
            }
        
        # Create LLM response
        llm_response = LLMResponse(
            message=message,
            usage=usage,
            model=model,
            finish_reason="stop",  # Ollama doesn't provide this, assume "stop"
            raw_response=response
        )
        
        return llm_response
    
    async def generate(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        # Use default model if not specified
        model = model or self.default_model
        
        # Convert messages to Ollama format
        ollama_messages = self._convert_to_ollama_messages(messages)
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": ollama_messages,
            "temperature": temperature,
            "stream": False
        }
        
        # Add max tokens if specified
        if max_tokens is not None:
            params["num_predict"] = max_tokens
            
        # Add other parameters from kwargs
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value
        
        try:
            # Make API call
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/chat", json=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Ollama API returned status {response.status}: {error_text}")
                    
                    result = await response.json()
            
            # Parse response
            llm_response = self._parse_ollama_response(result, model)
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {str(e)}")
            raise


class LLMInterface:
    """Interface for interacting with LLMs from different providers."""
    
    def __init__(self, 
                 provider: Optional[LLMProvider] = None,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the LLM interface."""
        self.provider = provider
        self.config = config or {}
        self.default_model = self.config.get("model", "deepseek-r1")
        self.default_temperature = self.config.get("temperature", 0.7)
        self.default_max_tokens = self.config.get("max_tokens", 4000)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.backoff_factor = self.config.get("backoff_factor", 1.5)
    
    async def generate(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        if not self.provider:
            raise ValueError("No LLM provider configured")
        
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens
        
        # Implement retry logic with exponential backoff
        attempt = 0
        last_error = None
        
        while attempt < self.retry_attempts:
            try:
                logger.debug(f"Generating LLM response with model {model}, attempt {attempt+1}")
                response = await self.provider.generate(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    **kwargs
                )
                
                logger.debug(f"LLM response generated successfully")
                return response
                
            except Exception as e:
                attempt += 1
                last_error = e
                
                if attempt < self.retry_attempts:
                    # Exponential backoff
                    sleep_time = self.backoff_factor ** attempt
                    logger.warning(f"LLM request failed, retrying in {sleep_time:.2f}s: {str(e)}")
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"LLM request failed after {self.retry_attempts} attempts: {str(e)}")
        
        # If we get here, all attempts failed
        if last_error:
            raise last_error
        else:
            raise RuntimeError(f"LLM request failed after {self.retry_attempts} attempts")

# ===== Simple Memory Manager =====

class SimpleMemoryManager:
    """A simple memory manager that stores entries as JSON files."""
    
    def __init__(self, memory_dir: str = "memory"):
        """Initialize with path to memory directory."""
        self.memory_dir = Path(memory_dir)
        self.entries_dir = self.memory_dir / "entries"
        self.index_path = self.memory_dir / "index.json"
        self.cache = {}
        self.cache_size = 1000
        
        # Create directories
        self._create_dirs()
        
        # Load or create index
        self._load_index()
    
    def _create_dirs(self):
        """Create required directories."""
        self.memory_dir.mkdir(exist_ok=True, parents=True)
        self.entries_dir.mkdir(exist_ok=True)
    
    def _load_index(self):
        """Load or create the index file."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    self.index = json.load(f)
            except Exception as e:
                logger.error(f"Error loading memory index: {str(e)}")
                self.index = {"entries": [], "next_id": 1}
        else:
            self.index = {"entries": [], "next_id": 1}
            self._save_index()
    
    def _save_index(self):
        """Save the index to disk."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _get_entry_path(self, memory_id: int) -> Path:
        """Get the file path for a memory entry."""
        return self.entries_dir / f"{memory_id}.json"
    
    def store(
        self,
        memory_type: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        vectors: Optional[Dict[str, List[float]]] = None
    ) -> int:
        """Store a memory entry."""
        try:
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
            
            # Add vectors if provided
            if vectors:
                entry["vectors"] = vectors
            
            # Save to file
            entry_path = self._get_entry_path(memory_id)
            with open(entry_path, 'w') as f:
                json.dump(entry, f, indent=2)
            
            # Update index
            index_entry = {
                "id": memory_id,
                "type": memory_type,
                "timestamp": timestamp
            }
            self.index["entries"].append(index_entry)
            self._save_index()
            
            # Update cache
            self.cache[memory_id] = entry
            self._trim_cache()
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            raise
    
    def retrieve(
        self,
        memory_id: int,
        include_vectors: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a memory entry by ID."""
        try:
            # Check cache first
            if memory_id in self.cache:
                entry = self.cache[memory_id].copy()
                if not include_vectors and "vectors" in entry:
                    entry.pop("vectors")
                return entry
            
            # Load from file
            entry_path = self._get_entry_path(memory_id)
            if not entry_path.exists():
                return None
                
            with open(entry_path, 'r') as f:
                entry = json.load(f)
            
            # Update cache
            self.cache[memory_id] = entry.copy()
            self._trim_cache()
            
            if not include_vectors and "vectors" in entry:
                entry.pop("vectors")
                
            return entry
            
        except Exception as e:
            logger.error(f"Error retrieving memory: {str(e)}")
            return None
    
    def update(
        self,
        memory_id: int,
        content: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        vectors: Optional[Dict[str, List[float]]] = None
    ) -> bool:
        """Update a memory entry."""
        try:
            # Load existing entry
            entry = self.retrieve(memory_id, include_vectors=True)
            if not entry:
                return False
            
            # Update fields
            if content is not None:
                entry["content"] = content
                
            if metadata is not None:
                if "metadata" not in entry:
                    entry["metadata"] = {}
                entry["metadata"].update(metadata)
                
            if vectors is not None:
                entry["vectors"] = vectors
            
            # Save back to file
            entry_path = self._get_entry_path(memory_id)
            with open(entry_path, 'w') as f:
                json.dump(entry, f, indent=2)
            
            # Update cache
            self.cache[memory_id] = entry.copy()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory: {str(e)}")
            return False
    
    def search(
        self,
        memory_type: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search memory entries."""
        try:
            # Filter entries in index
            results = []
            
            for idx_entry in self.index["entries"]:
                # Type filter
                if memory_type and idx_entry["type"] != memory_type:
                    continue
                
                # We need to load the full entry for metadata filtering
                if metadata_filter:
                    entry = self.retrieve(idx_entry["id"])
                    if not entry:
                        continue
                        
                    # Apply metadata filter
                    match = True
                    for key, value in metadata_filter.items():
                        if "metadata" not in entry or key not in entry["metadata"] or entry["metadata"][key] != value:
                            match = False
                            break
                            
                    if not match:
                        continue
                        
                    results.append(entry)
                else:
                    # No metadata filter, just add the entry ID
                    results.append(self.retrieve(idx_entry["id"]))
            
            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Return top k
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching memory: {str(e)}")
            return []
    
    def delete(self, memory_id: int) -> bool:
        """Delete a memory entry."""
        try:
            # Check if entry exists
            entry_path = self._get_entry_path(memory_id)
            if not entry_path.exists():
                return False
            
            # Delete file
            entry_path.unlink()
            
            # Update index
            self.index["entries"] = [e for e in self.index["entries"] if e["id"] != memory_id]
            self._save_index()
            
            # Remove from cache
            self.cache.pop(memory_id, None)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """Clear all memory entries."""
        try:
            # Delete all entry files
            for entry_file in self.entries_dir.glob("*.json"):
                entry_file.unlink()
            
            # Reset index
            self.index = {"entries": [], "next_id": 1}
            self._save_index()
            
            # Clear cache
            self.cache = {}
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
            return False
    
    def _trim_cache(self):
        """Trim cache to maximum size."""
        if len(self.cache) > self.cache_size:
            # Remove oldest entries (by timestamp)
            sorted_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k]["timestamp"] if "timestamp" in self.cache[k] else "",
                reverse=False
            )
            
            # Remove oldest entries
            excess = len(self.cache) - self.cache_size
            for k in sorted_keys[:excess]:
                self.cache.pop(k, None)

# ===== Main Function =====

async def main():
    """Run the completely standalone example."""
    logger.info("Starting completely standalone example")
    
    # Initialize memory manager
    memory = SimpleMemoryManager(memory_dir="complete_memory")
    logger.info("Initialized simple memory manager")
    
    # Initialize config
    config = {
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
    
    # Create LLM interface
    llm = LLMInterface(provider=ollama_provider, config=config)
    logger.info("Created LLM interface")
    
    # Store example content in memory
    content_id = memory.store(
        memory_type="example",
        content={"text": "This is an example from the complete standalone solution."},
        metadata={"source": "complete_standalone", "version": "1.0"}
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
    
    logger.info("Complete standalone example completed successfully")

if __name__ == "__main__":
    asyncio.run(main())