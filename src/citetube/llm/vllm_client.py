"""
vLLM client module for CiteTube.
Handles communication with vLLM server.
"""

import requests
import json
from typing import Dict, List, Any, Optional
from ..core.logging_config import get_logger

logger = get_logger("citetube.llm.vllm_client")

VLLM_BASE_URL = "http://localhost:8000"

def check_vllm_health() -> bool:
    """
    Check if vLLM server is healthy and accessible.
    
    Returns:
        True if vLLM is healthy, False otherwise
    """
    try:
        response = requests.get(f"{VLLM_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"vLLM health check failed: {e}")
        return False

def get_vllm_client():
    """Get vLLM client connection."""
    try:
        # Simple connection test
        response = requests.get(f"{VLLM_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True
        else:
            raise Exception("Connection error.")
    except Exception as e:
        logger.error(f"Failed to connect to vLLM server: {e}")
        raise Exception("Connection error.")

def ensure_model_available(model_name: str) -> bool:
    """
    Ensure the specified model is available on vLLM server.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model is available, False otherwise
    """
    try:
        # Try to get model info
        response = requests.get(f"{VLLM_BASE_URL}/v1/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model.get("id", "") for model in models_data.get("data", [])]
            if model_name in available_models:
                return True
        
        raise Exception(f"Model {model_name} is not available on vLLM server")
        
    except Exception as e:
        logger.error(f"Error checking model availability: {e}")
        raise Exception(f"Model {model_name} is not available on vLLM server")

def call_vllm(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs
) -> Dict[str, Any]:
    """
    Call vLLM server with chat completion request.
    
    Args:
        messages: List of message dictionaries
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters
        
    Returns:
        Response dictionary with 'content' key
    """
    try:
        # Ensure model is available
        ensure_model_available(model)
        
        # Prepare request
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Make request
        response = requests.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"content": content}
        else:
            raise Exception(f"vLLM request failed with status {response.status_code}")
            
    except Exception as e:
        logger.error(f"vLLM call failed: {e}")
        raise e