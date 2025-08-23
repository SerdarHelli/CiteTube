"""
CiteTube: Local YouTube Transcript QA Application.

A Python package for ingesting YouTube videos and answering questions about their content
using hybrid search (vector + BM25) and large language models.
"""

try:
    from ._version import version as __version__
except ImportError:
    # Fallback for development installations
    __version__ = "0.1.0-dev"

__author__ = "CiteTube Team"
__description__ = "Local YouTube Transcript QA Application"

# Public API - import key functions for easy access
from .core.config import (
    get_embedding_model_name,
    get_llm_model,
    get_llm_provider,
    ensure_directories,
)

from .core.db import test_connection, init_db

__all__ = [
    "__version__",
    "__author__", 
    "__description__",
    "get_embedding_model_name",
    "get_llm_model", 
    "get_llm_provider",
    "ensure_directories",
    "test_connection",
    "init_db",
]