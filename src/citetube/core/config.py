"""
Configuration module for CiteTube.
Handles environment variables and application settings.
"""

import os
from pathlib import Path
from typing import Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Database - PostgreSQL with pgvector
def get_db_host() -> str:
    """Get database host from environment."""
    return os.getenv("DB_HOST", "localhost")

def get_db_port() -> int:
    """Get database port from environment."""
    return int(os.getenv("DB_PORT", "5432"))

def get_db_name() -> str:
    """Get database name from environment."""
    return os.getenv("DB_NAME", "citetube")

def get_db_user() -> str:
    """Get database user from environment."""
    return os.getenv("DB_USER", "postgres")

def get_db_password() -> str:
    """Get database password from environment."""
    return os.getenv("DB_PASSWORD", "")

# Model settings
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # sentence-transformers embedding model
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"  # Keep sentence-transformers for reranker
DEFAULT_LLM_MODEL = "meta-llama/Llama-3.2-8B-Instruct"  # vLLM LLM model

# Search settings
DEFAULT_TOP_K = 8
DEFAULT_VECTOR_TOP_K = 30
DEFAULT_BM25_TOP_K = 30

# Environment variables
def get_embedding_model_name() -> str:
    """Get the embedding model name from environment or default."""
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

def get_reranker_model_name() -> str:
    """Get the reranker model name from environment or default."""
    return os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)

def use_reranker() -> bool:
    """Check if reranker should be used."""
    return os.getenv("USE_RERANKER", "true").lower() in ("true", "1", "yes")

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY")

def get_anthropic_api_key() -> Optional[str]:
    """Get Anthropic API key from environment."""
    return os.getenv("ANTHROPIC_API_KEY")

def get_llm_provider() -> str:
    """Get the LLM provider from environment or default to vLLM."""
    return os.getenv("LLM_PROVIDER", "vllm").lower()

def get_llm_model() -> str:
    """Get the LLM model name from environment."""
    provider = get_llm_provider()
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4")
    elif provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
    elif provider == "vllm":
        return os.getenv("VLLM_MODEL", DEFAULT_LLM_MODEL)
    elif provider == "ollama":
        return os.getenv("OLLAMA_MODEL", DEFAULT_LLM_MODEL)
    else:
        return os.getenv("MODEL_NAME", DEFAULT_LLM_MODEL)

def get_temperature() -> float:
    """Get the LLM temperature from environment."""
    return float(os.getenv("TEMPERATURE", "0.1"))

def get_max_tokens() -> int:
    """Get the max tokens from environment."""
    return int(os.getenv("MAX_TOKENS", "1024"))

# vLLM specific configuration
def get_vllm_host() -> str:
    """Get vLLM server host from environment."""
    return os.getenv("VLLM_HOST", "localhost")

def get_vllm_port() -> int:
    """Get vLLM server port from environment."""
    return int(os.getenv("VLLM_PORT", "8000"))

def get_vllm_api_key() -> Optional[str]:
    """Get vLLM API key from environment (if authentication is enabled)."""
    return os.getenv("VLLM_API_KEY")

def get_vllm_base_url() -> str:
    """Get vLLM base URL."""
    host = get_vllm_host()
    port = get_vllm_port()
    return f"http://{host}:{port}/v1"

# Ensure directories exist
def ensure_directories():
    """Ensure all required directories exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize directories when module is imported
ensure_directories()