"""
vLLM client module for CiteTube.
Handles vLLM server communication using OpenAI-compatible API.
"""

import time
from typing import Dict, Any, Optional
from openai import OpenAI

from ..core.config import (
    get_vllm_base_url, 
    get_vllm_api_key, 
    get_llm_model,
    get_temperature,
    get_max_tokens
)
from ..core.logging_config import get_logger, log_execution_time, log_error_with_context

logger = get_logger("citetube.vllm")

# Global vLLM client
_vllm_client = None

def get_vllm_client() -> OpenAI:
    """Get or create vLLM client using OpenAI-compatible API."""
    global _vllm_client
    if _vllm_client is None:
        try:
            base_url = get_vllm_base_url()
            api_key = get_vllm_api_key() or "EMPTY"  # vLLM doesn't require API key by default
            
            _vllm_client = OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
            
            # Test connection by listing models
            models = _vllm_client.models.list()
            logger.info(f"Connected to vLLM server at {base_url}")
            logger.info(f"Available models: {[model.id for model in models.data]}")
            
        except Exception as e:
            logger.error(f"Failed to connect to vLLM server: {e}")
            raise ConnectionError(f"vLLM server is not running or not accessible. Please start vLLM server first. Error: {e}")
    
    return _vllm_client

def ensure_model_available(model_name: str) -> bool:
    """
    Check if the model is available on the vLLM server.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model is available, False otherwise
    """
    try:
        client = get_vllm_client()
        models = client.models.list()
        available_models = [model.id for model in models.data]
        
        if model_name in available_models:
            logger.info(f"Model {model_name} is available on vLLM server")
            return True
        else:
            logger.warning(f"Model {model_name} not found on vLLM server. Available models: {available_models}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking model availability: {e}")
        return False

@log_execution_time("citetube.vllm")
def call_vllm(
    messages: list,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Call vLLM server with the given messages.
    
    Args:
        messages: List of message dictionaries
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional parameters
        
    Returns:
        Response from vLLM server
    """
    # Use config defaults if not provided
    if model is None:
        model = get_llm_model()
    if temperature is None:
        temperature = get_temperature()
    if max_tokens is None:
        max_tokens = get_max_tokens()
    
    start_time = time.time()
    
    try:
        # Ensure model is available
        if not ensure_model_available(model):
            raise ValueError(f"Model {model} is not available on vLLM server")
        
        # Get vLLM client
        client = get_vllm_client()
        
        # Generate response
        logger.info(f"Generating response with vLLM model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        logger.info(f"vLLM call completed in {time.time() - start_time:.2f}s")
        
        return {
            "content": response.choices[0].message.content,
            "usage": response.usage.dict() if response.usage else None,
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason
        }
        
    except Exception as e:
        log_error_with_context(
            e, 
            f"vLLM call failed with model {model}, temperature {temperature}, max_tokens {max_tokens}", 
            "citetube.vllm"
        )
        raise

def check_vllm_health() -> Dict[str, Any]:
    """
    Check vLLM server health and return status information.
    
    Returns:
        Dictionary with health status information
    """
    try:
        client = get_vllm_client()
        models = client.models.list()
        
        return {
            "status": "healthy",
            "base_url": get_vllm_base_url(),
            "available_models": [model.id for model in models.data],
            "model_count": len(models.data)
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "base_url": get_vllm_base_url()
        }