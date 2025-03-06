"""
Ollama provider implementation for LLM interface.
"""
import os
import logging
import json
import asyncio
import importlib.util
import aiohttp
from typing import Dict, List, Any, Optional, Union

from ..interface import LLMProvider, LLMResponse, Message, MessageRole

logger = logging.getLogger(__name__)

# Check if GOD MODE module is available
GOD_MODE_AVAILABLE = False
DeepSeekGodMode = None

try:
    # Try to import the GOD MODE module from our new location
    if importlib.util.find_spec("src.core.llm.providers.deepseek_god_mode"):
        from src.core.llm.providers.deepseek_god_mode import DeepSeekGodMode
        DeepSeekGodMode = DeepSeekGodMode()
        GOD_MODE_AVAILABLE = True
        logger.info("DeepSeek R1 GOD MODE module is available")
    # Backward compatibility check
    elif importlib.util.find_spec("src.god_mode"):
        from src.god_mode import DeepSeekGodMode
        GOD_MODE_AVAILABLE = True
        logger.info("DeepSeek R1 GOD MODE module is available (legacy path)")
except ImportError:
    logger.warning("DeepSeek R1 GOD MODE module is not available")
except Exception as e:
    logger.error(f"Error importing GOD MODE module: {str(e)}")


class OllamaProvider(LLMProvider):
    """
    Ollama provider for the LLM interface.
    
    This provider integrates with Ollama's local API for running models like DeepSeek R1.
    """
    
    def __init__(self, api_url: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Ollama provider.
        
        Args:
            api_url: Optional API URL (defaults to http://localhost:11434/api)
            config: Optional configuration dictionary
        """
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
        
        # Initialize GOD MODE if enabled and available
        self.god_mode = None
        self.god_mode_enabled = self.config.get("god_mode", False)
        
        if self.god_mode_enabled and GOD_MODE_AVAILABLE and DeepSeekGodMode is not None:
            try:
                logger.info("Initializing DeepSeek R1 GOD MODE")
                god_mode_config = self.config.get("god_mode_config", {})
                # DeepSeekGodMode is already instantiated
                self.god_mode = DeepSeekGodMode
                logger.info(f"DeepSeek R1 GOD MODE initialized with {len(self.god_mode.enhancements)} active enhancements")
                
                # Set specific enhancements if configured
                if "enhancements" in god_mode_config:
                    self.god_mode.enhancements = god_mode_config["enhancements"]
                        
                logger.info(f"Active GOD MODE enhancements: {', '.join(self.god_mode.enhancements)}")
            except Exception as e:
                logger.error(f"Error initializing GOD MODE: {str(e)}")
                self.god_mode_enabled = False
                
        elif self.god_mode_enabled and not GOD_MODE_AVAILABLE:
            logger.warning("GOD MODE requested but module is not available")
            self.god_mode_enabled = False
        
        logger.info(f"Ollama provider initialized with API URL {self.api_url} and model {self.default_model} (GOD MODE: {'ENABLED' if self.god_mode_enabled else 'DISABLED'})")
    
    def _convert_to_ollama_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert our message format to Ollama's format.
        
        Args:
            messages: List of Message objects
            
        Returns:
            List of messages in Ollama format
        """
        ollama_messages = []
        
        for message in messages:
            # Create base message
            ollama_message = {
                "role": message.role.value
            }
            
            # Add content if present
            if message.content is not None:
                ollama_message["content"] = message.content
                
            # Add name for tool/function messages if present (may not be supported)
            if message.name is not None and message.role in [MessageRole.TOOL, MessageRole.FUNCTION]:
                ollama_message["name"] = message.name
            
            ollama_messages.append(ollama_message)
            
        return ollama_messages
    
    def _parse_ollama_response(
        self, 
        response: Dict[str, Any],
        model: str
    ) -> LLMResponse:
        """
        Parse Ollama response into our format.
        
        Args:
            response: Ollama response
            model: The model used
            
        Returns:
            LLMResponse object
        """
        # Convert response message to our format 
        message_content = ""
        if "message" in response and "content" in response["message"]:
            message_content = response["message"]["content"]
        elif "response" in response:
            message_content = response["response"]
        
        # Strip <think> blocks from the response (DeepSeek R1 specific)
        if "<think>" in message_content and "</think>" in message_content:
            parts = message_content.split("</think>")
            message_content = parts[-1].strip()
            
        # Remove internal reasoning and specific formats (DeepSeek R1)
        if "Let me think through this step by step" in message_content:
            lines = message_content.split("\n")
            filtered_lines = []
            skip_mode = False
            
            for line in lines:
                if "Let me think through this step by step" in line:
                    skip_mode = True
                elif skip_mode and (line.strip() == "" or line.startswith("Alright")):
                    skip_mode = False
                
                if not skip_mode:
                    filtered_lines.append(line)
            
            message_content = "\n".join(filtered_lines).strip()
            
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
        retry_attempts: int = 3,
        backoff_factor: float = 1.5,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM with retry capability.
        
        Args:
            messages: The conversation history
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools (not supported by Ollama)
            retry_attempts: Number of retry attempts
            backoff_factor: Backoff factor for retries
            **kwargs: Additional parameters
            
        Returns:
            A response from the LLM
        """
        # Use default model if not specified
        model = model or self.default_model
        
        # Get retry config from class config if not specified
        if retry_attempts == 3 and "retry_attempts" in self.config:
            retry_attempts = self.config["retry_attempts"]
        if backoff_factor == 1.5 and "backoff_factor" in self.config:
            backoff_factor = self.config["backoff_factor"]
        
        # Apply GOD MODE enhancements if enabled
        if self.god_mode_enabled and self.god_mode is not None and "deepseek-r1" in model:
            try:
                # Create context for GOD MODE processing
                context = {
                    "purpose": kwargs.get("purpose", "general"),
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tools": tools,
                    **kwargs
                }
                
                # Apply GOD MODE enhancements to messages
                logger.debug(f"Applying GOD MODE enhancements to {len(messages)} messages")
                
                # Create enhanced copies of the messages
                enhanced_messages = []
                for msg in messages:
                    if msg.role == MessageRole.USER and msg.content:
                        # Apply enhancements to user messages
                        enhanced_content = self.god_mode.enhance_prompt(msg.content)
                        enhanced_msg = Message(
                            role=msg.role,
                            content=enhanced_content,
                            name=msg.name
                        )
                        enhanced_messages.append(enhanced_msg)
                    else:
                        # Keep other messages as is
                        enhanced_messages.append(msg)
                
                # Log the transformation
                original_tokens = sum(len(m.content or "") for m in messages)
                enhanced_tokens = sum(len(m.content or "") for m in enhanced_messages)
                logger.debug(f"GOD MODE transformation: {original_tokens} tokens → {enhanced_tokens} tokens")
                
                # Use the enhanced messages
                messages = enhanced_messages
                
                # Add GOD MODE enhancement info to the messages
                system_message = Message(
                    role=MessageRole.SYSTEM,
                    content="You are operating in DeepSeek R1 GOD MODE with advanced capabilities for financial analysis. Remember to leverage your enhanced features for this interaction."
                )
                if messages and messages[0].role == MessageRole.SYSTEM:
                    # Merge with existing system message
                    messages[0].content = f"{system_message.content}\n\n{messages[0].content}"
                else:
                    # Add new system message at the beginning
                    messages.insert(0, system_message)
                
                # Apply GOD MODE enhancements to model parameters
                model_params = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "purpose": kwargs.get("purpose", "general"),
                    **kwargs
                }
                # Our simple implementation doesn't enhance parameters, just pass through
                enhanced_params = model_params
                
                # Update parameters
                if "temperature" in enhanced_params:
                    temperature = enhanced_params["temperature"]
                    logger.debug(f"GOD MODE adjusted temperature: {temperature}")
                
                # Add GOD MODE specific parameters to kwargs
                for key, value in enhanced_params.items():
                    if key not in ["temperature", "max_tokens"] and key not in kwargs:
                        kwargs[key] = value
                
            except Exception as e:
                logger.error(f"Error applying GOD MODE enhancements: {str(e)}")
                # Continue with original messages if enhancement fails
        
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
            
        # Add DeepSeek R1 specific parameters if available
        if "deepseek-r1" in model and "deepseek_config" in self.config:
            deepseek_config = self.config["deepseek_config"]
            # Map config keys to Ollama parameter names
            param_mapping = {
                "top_k": "top_k",
                "top_p": "top_p",
                "repeat_penalty": "repeat_penalty",
                "mirostat": "mirostat",
                "num_ctx": "num_ctx"
            }
            
            for config_key, param_key in param_mapping.items():
                if config_key in deepseek_config:
                    params[param_key] = deepseek_config[config_key]
            
            # Use raw format to avoid XML tags in the output
            if "raw_format" not in params:
                params["raw_format"] = True
                
            # In GOD MODE, adjust specific parameters
            if self.god_mode_enabled:
                # Increase context window
                params["num_ctx"] = 8192
                # Adjust top_p for more deterministic outputs in analytical contexts
                if kwargs.get("purpose") in ["analysis", "scientific"]:
                    params["top_p"] = 0.8
                    params["repeat_penalty"] = 1.2
                else:
                    # More creative settings for other contexts
                    params["top_p"] = 0.95
                
            # Log DeepSeek R1 specific parameters
            logger.debug(f"Using DeepSeek R1 specific parameters: {deepseek_config}")
            
        # Add other parameters from kwargs
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value
        
        attempt = 0
        last_error = None
        
        while attempt <= retry_attempts:
            try:
                # Wait with exponential backoff if this is a retry
                if attempt > 0:
                    wait_time = backoff_factor ** (attempt - 1)
                    logger.info(f"Retrying after {wait_time:.2f} seconds (attempt {attempt}/{retry_attempts})")
                    await asyncio.sleep(wait_time)
                
                # Make API call
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.api_url}/chat", json=params) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise ValueError(f"Ollama API returned status {response.status}: {error_text}")
                        
                        result = await response.json()
                
                # Check if DeepSeek R1 returned a valid response
                if "message" not in result or "content" not in result.get("message", {}):
                    raise ValueError("Invalid response format from DeepSeek R1")
                
                # Check if response is empty
                content = result["message"]["content"]
                if not content.strip():
                    raise ValueError("Empty response from DeepSeek R1")
                
                # Parse response
                llm_response = self._parse_ollama_response(result, model)
                
                # Add retry count to metadata if we had to retry
                if attempt > 0:
                    if "metadata" not in llm_response.raw_response:
                        llm_response.raw_response["metadata"] = {}
                    llm_response.raw_response["metadata"]["retry_count"] = attempt
                
                return llm_response
                
            except Exception as e:
                last_error = e
                logger.warning(f"Error generating response from Ollama (attempt {attempt+1}/{retry_attempts+1}): {str(e)}")
                attempt += 1
        
        # If we reach here, all attempts failed
        logger.error(f"All {retry_attempts+1} attempts to generate response failed")
        raise last_error