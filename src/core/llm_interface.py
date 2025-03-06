"""
LLM interface for interacting with language models.
"""
import os
import json
import logging
import asyncio
import requests
import aiohttp
from typing import Dict, List, Any, Optional, Union, Tuple, AsyncGenerator
from datetime import datetime
from enum import Enum

# Import tenacity for retries
from tenacity import retry, stop_after_attempt, wait_exponential

from .mcp import ModelContextProtocol, Context, ContextType
from .safety_checker import SafetyChecker
from .memory_manager import MemoryManager, MemoryType, MemoryImportance
from ..utils.config import load_config
from ..utils.credentials import load_credentials

logger = logging.getLogger(__name__)

class MessageRole(str, Enum):
    """Message role enumeration."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

class Message:
    """Message structure for LLM conversations."""
    def __init__(
        self, 
        role: MessageRole, 
        content: Optional[str] = None,
        name: Optional[str] = None,
        function_call: Optional[Dict[str, Any]] = None
    ):
        self.role = role
        self.content = content
        self.name = name
        self.function_call = function_call
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        result = {"role": self.role.value}
        if self.content is not None:
            result["content"] = self.content
        if self.name is not None:
            result["name"] = self.name
        if self.function_call is not None:
            result["function_call"] = self.function_call
        return result

class LLMResponse:
    """Structure for LLM responses."""
    def __init__(
        self,
        message: Message,
        model: str,
        usage: Dict[str, int],
        timestamp: datetime
    ):
        self.message = message
        self.model = model
        self.usage = usage
        self.timestamp = timestamp

class LLMInterface:
    """
    Interface for interacting with Language Models.
    
    Attributes:
        config: Configuration dictionary
        mcp: Model Context Protocol instance
        safety_checker: Safety verification instance
        memory_manager: Memory management instance
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize LLM interface."""
        self.config = config or {}
        self.credentials = load_credentials()
        self.mcp = ModelContextProtocol()
        self.safety_checker = SafetyChecker()
        self.memory_manager = MemoryManager()
        
        # Set up API keys if needed
        self.anthropic_api_key = self.credentials.get('ANTHROPIC_API_KEY')
        
        # Get provider and model
        self.provider = self.config.get('provider', 'ollama')
        self.model = self.config.get('model', 'deepseek:latest')
        
        # Ollama config
        self.ollama_url = self.config.get('ollama_url', 'http://localhost:11434')
        
        # Initialize state
        self.conversation_history = []
        self.last_response = None
        self.total_tokens = 0
        self.total_cost = 0.0
        
    def create_system_message(self, content: str) -> Message:
        """Create a system message."""
        return Message(role=MessageRole.SYSTEM, content=content)
        
    def create_user_message(self, content: str) -> Message:
        """Create a user message."""
        return Message(role=MessageRole.USER, content=content)
        
    def create_assistant_message(self, content: str) -> Message:
        """Create an assistant message."""
        return Message(role=MessageRole.ASSISTANT, content=content)
        
    def create_function_message(self, name: str, content: str) -> Message:
        """Create a function message."""
        return Message(role=MessageRole.FUNCTION, name=name, content=content)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def chat(
        self,
        messages: List[Message],
        contexts: Optional[List[Context]] = None,
        max_tokens: int = 1000,
        temperature: Optional[float] = None,
        stream: bool = False,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, Any]]] = None
    ) -> Union[LLMResponse, AsyncGenerator[Dict[str, Any], None]]:
        """
        Chat with the LLM using a conversation.
        
        Args:
            messages: List of message objects
            contexts: Optional list of context objects
            max_tokens: Maximum tokens in response
            temperature: Response randomness (provider default if None)
            stream: Whether to stream response
            functions: Optional functions to expose to the LLM
            function_call: Optional function call instruction
            
        Returns:
            LLMResponse object or stream of chunks
        """
        try:
            # Prepare context from MCP if available
            if contexts:
                for context in contexts:
                    self.mcp.update_context(context)
            
            processed_context = self.mcp.get_current_context()
            
            # Convert messages to dictionaries
            message_dicts = [msg.to_dict() for msg in messages]
            
            # Verify safety of all user messages
            for message in messages:
                if message.role == MessageRole.USER and message.content:
                    if not self.safety_checker.verify_prompt(message.content, processed_context):
                        raise ValueError(f"Safety check failed for message: {message.content[:50]}...")
            
            # Generate response based on provider
            if self.provider == 'ollama':
                return await self._ollama_chat(
                    message_dicts, max_tokens, temperature, stream, functions, function_call
                )
            elif self.provider == 'anthropic':
                return await self._anthropic_chat(
                    message_dicts, max_tokens, temperature, stream
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            self.memory_manager.store(
                memory_type=MemoryType.SYSTEM_EVENT,
                content={
                    "event": "llm_error",
                    "error": str(e),
                    "provider": self.provider,
                    "model": self.model
                },
                importance=MemoryImportance.HIGH,
                tags=["error", "llm", self.provider]
            )
            raise
            
    async def _ollama_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: Optional[float],
        stream: bool,
        functions: Optional[List[Dict[str, Any]]],
        function_call: Optional[Union[str, Dict[str, Any]]]
    ) -> Union[LLMResponse, AsyncGenerator[Dict[str, Any], None]]:
        """Handle Ollama chat completion."""
        # Prepare parameters
        params = {
            "model": self.model,
            "messages": messages,
            "max_length": max_tokens,
            "stream": stream
        }
        
        # Add optional parameters
        if temperature is not None:
            params["temperature"] = temperature
            
        # Function calling is not natively supported in Ollama models,
        # but we can simulate it with proper prompting if needed
        if functions:
            # Add functions to system message
            functions_description = "Available functions:\n"
            for func in functions:
                functions_description += f"- {func['name']}: {func['description']}\n"
                if 'parameters' in func:
                    functions_description += f"  Parameters: {json.dumps(func['parameters'])}\n"
            
            # Add or modify system message
            system_message_found = False
            for i, msg in enumerate(messages):
                if msg['role'] == 'system':
                    messages[i]['content'] = messages[i]['content'] + "\n\n" + functions_description
                    system_message_found = True
                    break
            
            if not system_message_found:
                messages.insert(0, {'role': 'system', 'content': functions_description})
            
            params["messages"] = messages
        
        # Call API
        endpoint = f"{self.ollama_url}/api/chat"
        
        if stream:
            # Use async streaming request
            async def ollama_stream():
                async with aiohttp.ClientSession() as session:
                    async with session.post(endpoint, json=params) as response:
                        content_chunks = []
                        total_tokens = 0
                        
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if not line:
                                continue
                                
                            try:
                                chunk = json.loads(line)
                                if 'message' in chunk and chunk['message'].get('content'):
                                    content_chunk = chunk['message']['content']
                                    content_chunks.append(content_chunk)
                                    total_tokens += 1
                                    
                                    # Safety check chunk
                                    if not self.safety_checker.verify_response(content_chunk):
                                        raise ValueError("Response chunk failed safety check")
                                        
                                    yield {
                                        'type': 'content',
                                        'chunk': content_chunk,
                                        'tokens': total_tokens,
                                        'timestamp': datetime.now().isoformat()
                                    }
                            except json.JSONDecodeError:
                                continue
                        
                # Update total tokens
                self.total_tokens += total_tokens
                
                # Create response object
                full_content = ''.join(content_chunks)
                message = Message(role=MessageRole.ASSISTANT, content=full_content)
                
                llm_response = LLMResponse(
                    message=message,
                    model=self.model,
                    usage={'total_tokens': total_tokens},
                    timestamp=datetime.now()
                )
                
                # Update history
                self.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'message': message.to_dict(),
                    'tokens': total_tokens,
                    'model': self.model
                })
                
                # Store in memory
                self._store_interaction(llm_response)
                
                # Store last response
                self.last_response = llm_response
                
                yield {
                    'type': 'complete',
                    'response': llm_response,
                    'timestamp': datetime.now().isoformat()
                }
                
            return ollama_stream()
        else:
            # Use synchronous request
            try:
                response = requests.post(endpoint, json=params)
                response.raise_for_status()
                return self._process_ollama_response(response.json())
            except requests.RequestException as e:
                logger.error(f"Error in Ollama API call: {str(e)}")
                raise
    
    async def _anthropic_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: Optional[float],
        stream: bool
    ) -> Union[LLMResponse, AsyncGenerator[Dict[str, Any], None]]:
        """Handle Anthropic chat completion."""
        # Convert message format to Anthropic's format
        prompt = self._convert_to_anthropic_prompt(messages)
        
        # Prepare parameters
        params = {
            "prompt": prompt,
            "model": self.model,
            "max_tokens_to_sample": max_tokens,
            "stop_sequences": ["\n\nHuman:"],
            "stream": stream
        }
        
        # Add optional parameters
        if temperature is not None:
            params["temperature"] = temperature
            
        # Create Anthropic client
        client = anthropic.Client(api_key=self.anthropic_api_key)
        
        # Call API
        if stream:
            return self._handle_anthropic_stream(client.completion_stream(**params))
        else:
            response = client.completion(**params)
            return self._process_anthropic_response(response)
    
    def _convert_to_anthropic_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert OpenAI message format to Anthropic prompt format."""
        prompt = ""
        
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            
            if role == "system":
                # Add system message as a preamble
                prompt = f"{content}\n\n"
            elif role == "user":
                prompt += f"\n\nHuman: {content}"
            elif role == "assistant":
                prompt += f"\n\nAssistant: {content}"
            elif role == "function":
                # Format function responses for Anthropic
                prompt += f"\n\nHuman: Function {message['name']} returned: {content}"
                
        # Add final assistant prompt
        prompt += "\n\nAssistant:"
        return prompt
    
    def _process_ollama_response(self, response: Dict[str, Any]) -> LLMResponse:
        """Process Ollama non-streaming response."""
        # Ollama doesn't provide detailed token counts, so we estimate
        message_content = response.get('message', {}).get('content', '')
        
        # Estimate tokens (approximately 4 chars per token)
        prompt_tokens = sum(len(m.get('content', '')) // 4 for m in response.get('prompt', []))
        completion_tokens = len(message_content) // 4
        total_tokens = prompt_tokens + completion_tokens
        
        self.total_tokens += total_tokens
        
        # Create message object
        message = Message(
            role=MessageRole.ASSISTANT,
            content=message_content
        )
        
        # Safety check response
        if message.content and not self.safety_checker.verify_response(message.content):
            raise ValueError("Response failed safety check")
            
        # Create response object
        llm_response = LLMResponse(
            message=message,
            model=self.model,
            usage={
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens
            },
            timestamp=datetime.now()
        )
        
        # Update history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'message': message.to_dict(),
            'tokens': total_tokens,
            'model': self.model
        })
        
        # Store in memory
        self._store_interaction(llm_response)
        
        # Store last response
        self.last_response = llm_response
        
        return llm_response
        
    async def _handle_openai_stream(self, response_stream) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle OpenAI streaming response."""
        content = []
        function_call_parts = []
        total_tokens = 0
        
        async for chunk in response_stream:
            delta = chunk.choices[0].delta
            
            # Handle content chunks
            if hasattr(delta, 'content') and delta.content:
                content_chunk = delta.content
                content.append(content_chunk)
                total_tokens += 1
                
                # Safety check chunk
                if not self.safety_checker.verify_response(content_chunk):
                    raise ValueError("Response chunk failed safety check")
                    
                yield {
                    'type': 'content',
                    'chunk': content_chunk,
                    'tokens': total_tokens,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Handle function call chunks
            if hasattr(delta, 'function_call'):
                function_call_chunks = delta.function_call
                if hasattr(function_call_chunks, 'name'):
                    yield {
                        'type': 'function_call',
                        'name': function_call_chunks.name,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                if hasattr(function_call_chunks, 'arguments'):
                    function_call_parts.append(function_call_chunks.arguments)
                    yield {
                        'type': 'function_args',
                        'chunk': function_call_chunks.arguments,
                        'timestamp': datetime.now().isoformat()
                    }
        
        # Update total tokens
        self.total_tokens += total_tokens
        
        # Create final response
        if content:
            full_content = ''.join(content)
            message = Message(role=MessageRole.ASSISTANT, content=full_content)
        else:
            # Handle function call
            function_args = ''.join(function_call_parts)
            message = Message(
                role=MessageRole.ASSISTANT,
                function_call={
                    'name': chunk.choices[0].delta.function_call.name,
                    'arguments': function_args
                }
            )
        
        # Create response object
        llm_response = LLMResponse(
            message=message,
            model=self.model,
            usage={'total_tokens': total_tokens},
            timestamp=datetime.now()
        )
        
        # Update history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'message': message.to_dict(),
            'tokens': total_tokens,
            'model': self.model
        })
        
        # Store in memory
        self._store_interaction(llm_response)
        
        # Store last response
        self.last_response = llm_response
        
        yield {
            'type': 'complete',
            'response': llm_response,
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_anthropic_response(self, response: Dict[str, Any]) -> LLMResponse:
        """Process Anthropic non-streaming response."""
        # Update token count and cost
        prompt_tokens = len(response['prompt']) // 4  # Approximate
        completion_tokens = len(response['completion']) // 4  # Approximate
        total_tokens = prompt_tokens + completion_tokens
        
        self.total_tokens += total_tokens
        self.total_cost += self._calculate_cost(
            prompt_tokens, 
            completion_tokens, 
            self.model
        )
        
        # Create message object
        message = Message(
            role=MessageRole.ASSISTANT,
            content=response['completion']
        )
        
        # Safety check response
        if not self.safety_checker.verify_response(message.content):
            raise ValueError("Response failed safety check")
            
        # Create response object
        llm_response = LLMResponse(
            message=message,
            model=self.model,
            usage={
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens
            },
            timestamp=datetime.now()
        )
        
        # Update history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'message': message.to_dict(),
            'tokens': total_tokens,
            'model': self.model
        })
        
        # Store in memory
        self._store_interaction(llm_response)
        
        # Store last response
        self.last_response = llm_response
        
        return llm_response
    
    async def _handle_anthropic_stream(self, response_stream) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle Anthropic streaming response."""
        content = []
        total_tokens = 0
        
        async for chunk in response_stream:
            if chunk.completion:
                content_chunk = chunk.completion
                content.append(content_chunk)
                total_tokens += 1
                
                # Safety check chunk
                if not self.safety_checker.verify_response(content_chunk):
                    raise ValueError("Response chunk failed safety check")
                    
                yield {
                    'type': 'content',
                    'chunk': content_chunk,
                    'tokens': total_tokens,
                    'timestamp': datetime.now().isoformat()
                }
        
        # Update total tokens
        prompt_tokens = len(chunk.prompt) // 4  # Approximate
        self.total_tokens += total_tokens + prompt_tokens
        
        # Combine chunks
        full_content = ''.join(content)
        
        # Create message object
        message = Message(role=MessageRole.ASSISTANT, content=full_content)
        
        # Create response object
        llm_response = LLMResponse(
            message=message,
            model=self.model,
            usage={
                'prompt_tokens': prompt_tokens,
                'completion_tokens': total_tokens,
                'total_tokens': prompt_tokens + total_tokens
            },
            timestamp=datetime.now()
        )
        
        # Update history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'message': message.to_dict(),
            'tokens': prompt_tokens + total_tokens,
            'model': self.model
        })
        
        # Store in memory
        self._store_interaction(llm_response)
        
        # Store last response
        self.last_response = llm_response
        
        yield {
            'type': 'complete',
            'response': llm_response,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate approximate cost based on token usage."""
        # Pricing as of 2023 (update as needed)
        pricing = {
            # OpenAI models
            'gpt-4': {'prompt': 0.03, 'completion': 0.06},
            'gpt-4-32k': {'prompt': 0.06, 'completion': 0.12},
            'gpt-3.5-turbo': {'prompt': 0.0015, 'completion': 0.002},
            'gpt-3.5-turbo-16k': {'prompt': 0.003, 'completion': 0.004},
            # Anthropic models
            'claude-2': {'prompt': 0.01102, 'completion': 0.03268},
            'claude-instant-1': {'prompt': 0.00163, 'completion': 0.00551}
        }
        
        if model in pricing:
            return (prompt_tokens / 1000 * pricing[model]['prompt'] + 
                   completion_tokens / 1000 * pricing[model]['completion'])
        else:
            # Default cost calculation
            return (prompt_tokens + completion_tokens) / 1000 * 0.002
    
    def _store_interaction(self, response: LLMResponse) -> None:
        """Store interaction in memory."""
        # Store conversation in memory
        self.memory_manager.store(
            memory_type=MemoryType.LLM_INTERACTION,
            content={
                'message': response.message.to_dict(),
                'model': response.model,
                'usage': response.usage
            },
            metadata={
                'timestamp': response.timestamp.isoformat(),
                'tokens': response.usage.get('total_tokens', 0)
            },
            importance=MemoryImportance.MEDIUM,
            tags=['llm', 'conversation', self.provider]
        )
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get token usage statistics."""
        return {
            'total_tokens': self.total_tokens,
            'estimated_cost': self.total_cost,
            'history_length': len(self.conversation_history),
            'last_response_tokens': self.last_response.usage.get('total_tokens', 0) if self.last_response else 0,
            'provider': self.provider,
            'model': self.model
        }
        
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        self.last_response = None
        
    def get_last_response(self) -> Optional[LLMResponse]:
        """Get last response."""
        return self.last_response
        
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history.copy()
        
    def save_history(self, filepath: str) -> None:
        """Save conversation history to file."""
        serialized_history = []
        for item in self.conversation_history:
            serialized_item = item.copy()
            serialized_history.append(serialized_item)
            
        with open(filepath, 'w') as f:
            json.dump(serialized_history, f, indent=2)
            
    def load_history(self, filepath: str) -> None:
        """Load conversation history from file."""
        with open(filepath, 'r') as f:
            self.conversation_history = json.load(f)