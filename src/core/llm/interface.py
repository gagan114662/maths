"""
LLM interface implementation for integrating with different LLM providers.
"""
import logging
import time
import enum
import json
from typing import Dict, List, Any, Optional, Protocol, Union, Callable
from datetime import datetime
from pydantic import BaseModel, Field

from ..mcp import Context, ContextType

logger = logging.getLogger(__name__)


class MessageRole(str, enum.Enum):
    """Role types for messages in LLM conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class Message(BaseModel):
    """
    A message in an LLM conversation.
    
    Attributes:
        role: The role of the message sender
        content: The content of the message
        name: Optional name for function/tool messages
        tool_calls: Optional tool calls for assistant messages
        tool_call_id: Optional tool call ID for tool messages
    """
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
    """
    Response from an LLM.
    
    Attributes:
        message: The message from the LLM
        usage: Token usage statistics
        model: The model used
        created_at: When the response was created
        finish_reason: Reason for finishing (e.g., "stop", "length", "tool_calls")
        raw_response: The raw response from the LLM provider
    """
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
        """
        Generate a response from the LLM.
        
        Args:
            messages: The conversation history
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional provider-specific parameters
            
        Returns:
            A response from the LLM
        """
        ...


class LLMInterface:
    """
    Interface for interacting with large language models from different providers.
    
    This class provides a unified interface for making requests to different LLM
    providers and handling responses.
    """
    
    def __init__(self, 
                 provider: Optional[LLMProvider] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM interface.
        
        Args:
            provider: The LLM provider to use
            config: Configuration for the interface
        """
        self.provider = provider
        self.config = config or {}
        self.default_model = self.config.get("model", "claude-3-sonnet-20240229")
        self.default_temperature = self.config.get("temperature", 0.7)
        self.default_max_tokens = self.config.get("max_tokens", 4000)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.backoff_factor = self.config.get("backoff_factor", 1.5)
        
        # Set up tool registry
        self.tools: Dict[str, Dict[str, Any]] = {}
        
    def register_tool(self, 
                      name: str, 
                      description: str, 
                      parameters: Dict[str, Any],
                      handler: Callable[[Dict[str, Any]], Any]) -> None:
        """
        Register a tool that can be used by LLMs.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: Tool parameters schema
            handler: Function to call when the tool is used
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler
        }
        logger.debug(f"Registered tool: {name}")
        
    def create_system_message(self, content: str) -> Message:
        """Create a system message."""
        return Message(role=MessageRole.SYSTEM, content=content)
    
    def create_user_message(self, content: str) -> Message:
        """Create a user message."""
        return Message(role=MessageRole.USER, content=content)
    
    def create_assistant_message(self, content: str) -> Message:
        """Create an assistant message."""
        return Message(role=MessageRole.ASSISTANT, content=content)
    
    def create_tool_message(self, content: str, tool_call_id: str) -> Message:
        """Create a tool message."""
        return Message(
            role=MessageRole.TOOL, 
            content=content, 
            tool_call_id=tool_call_id
        )
    
    def context_to_message(self, context: Context) -> Message:
        """
        Convert a context to a message.
        
        Args:
            context: The context to convert
            
        Returns:
            A message containing the context
        """
        if context.type == ContextType.SYSTEM:
            return self.create_system_message(json.dumps(context.data))
        else:
            message_content = f"Context type: {context.type.value}\n"
            message_content += json.dumps(context.data, indent=2)
            return self.create_user_message(message_content)
    
    async def generate(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: The conversation history
            model: The model to use (defaults to configured default)
            temperature: Sampling temperature (defaults to configured default)
            max_tokens: Maximum tokens to generate (defaults to configured default)
            tools: Available tools
            **kwargs: Additional provider-specific parameters
            
        Returns:
            A response from the LLM
        """
        if not self.provider:
            raise ValueError("No LLM provider configured")
        
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens
        
        # Handle tools
        available_tools = None
        if tools:
            available_tools = tools
        elif self.tools:
            # Convert registered tools to format expected by provider
            available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                } for tool in self.tools.values()
            ]
        
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
                    tools=available_tools,
                    **kwargs
                )
                
                logger.debug(f"LLM response generated successfully, finish reason: {response.finish_reason}")
                return response
                
            except Exception as e:
                attempt += 1
                last_error = e
                
                if attempt < self.retry_attempts:
                    # Exponential backoff
                    sleep_time = self.backoff_factor ** attempt
                    logger.warning(f"LLM request failed, retrying in {sleep_time:.2f}s: {str(e)}")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"LLM request failed after {self.retry_attempts} attempts: {str(e)}")
        
        # If we get here, all attempts failed
        if last_error:
            raise last_error
        else:
            raise RuntimeError(f"LLM request failed after {self.retry_attempts} attempts")
    
    async def chat(
        self,
        messages: List[Message],
        contexts: Optional[List[Context]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a chat response from the LLM.
        
        Args:
            messages: The conversation history
            contexts: Optional contexts to include
            **kwargs: Additional parameters for generate()
            
        Returns:
            A response from the LLM
        """
        # Add contexts as system and user messages
        all_messages = []
        
        # Add contexts if provided
        if contexts:
            for context in contexts:
                all_messages.append(self.context_to_message(context))
        
        # Add conversation messages
        all_messages.extend(messages)
        
        # Generate response
        return await self.generate(messages=all_messages, **kwargs)
    
    async def handle_tool_calls(
        self,
        response: LLMResponse
    ) -> List[Message]:
        """
        Handle tool calls in an LLM response.
        
        Args:
            response: The LLM response containing tool calls
            
        Returns:
            A list of tool messages with the results
        """
        if not response.message.tool_calls:
            return []
        
        tool_messages = []
        
        for tool_call in response.message.tool_calls:
            function = tool_call.get("function", {})
            name = function.get("name")
            tool_call_id = tool_call.get("id")
            
            if not name or not tool_call_id:
                logger.warning(f"Invalid tool call: {tool_call}")
                continue
            
            if name not in self.tools:
                error_message = f"Tool not found: {name}"
                logger.warning(error_message)
                tool_messages.append(self.create_tool_message(
                    content=error_message,
                    tool_call_id=tool_call_id
                ))
                continue
            
            try:
                # Parse arguments
                arguments = json.loads(function.get("arguments", "{}"))
                
                # Call handler
                handler = self.tools[name]["handler"]
                result = handler(arguments)
                
                # Create tool message with result
                result_str = json.dumps(result) if not isinstance(result, str) else result
                tool_messages.append(self.create_tool_message(
                    content=result_str,
                    tool_call_id=tool_call_id
                ))
                
                logger.debug(f"Successfully handled tool call: {name}")
                
            except Exception as e:
                error_message = f"Error handling tool call {name}: {str(e)}"
                logger.error(error_message)
                tool_messages.append(self.create_tool_message(
                    content=error_message,
                    tool_call_id=tool_call_id
                ))
        
        return tool_messages