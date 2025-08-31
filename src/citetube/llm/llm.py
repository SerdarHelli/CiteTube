"""
Minimal LLM client module for CiteTube health checks and compatibility.
"""

from typing import List, Dict, Optional, Any
from ..core.logging_config import get_logger
try:
    from .vllm_client import check_vllm_health, call_vllm
except ImportError:
    # Fallback if vllm_client is not available
    def check_vllm_health():
        return False
    def call_vllm(*args, **kwargs):
        raise Exception("vLLM client not available")
from ..core.config import get_llm_model

logger = get_logger("citetube.llm")

def test_llm_connection() -> bool:
    """
    Test if the LLM connection is working.
    
    Returns:
        True if connection is working, False otherwise
    """
    try:
        # Check vLLM health
        if not check_vllm_health():
            return False
        
        # Try a simple test call
        test_response = call_vllm(
            messages=[{"role": "user", "content": "Hello, respond with 'OK'"}],
            model=get_llm_model(),
            temperature=0.1,
            max_tokens=10
        )
        
        return test_response is not None and 'content' in test_response
        
    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return False

def test_agent_connection() -> bool:
    """
    Test if the LangChain agent is working.
    
    Returns:
        True if agent is working, False otherwise
    """
    try:
        from .agent import CiteTubeAgent
        
        agent = CiteTubeAgent()
        if not agent:
            return False
        
        # Try a simple test question
        test_response = agent.ask("Hello, please respond with 'Agent OK'")
        return "Agent OK" in test_response.get("answer", "")
        
    except Exception as e:
        logger.error(f"Agent connection test failed: {e}")
        return False

def get_agent_tools() -> List[Dict[str, str]]:
    """
    Get information about available agent tools.
    
    Returns:
        List of tool information dictionaries
    """
    try:
        from .agent import CiteTubeAgent
        
        agent = CiteTubeAgent()
        if agent:
            return agent.get_available_tools()
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting agent tools: {e}")
        return []