"""
Models module for CiteTube.
Handles loading and caching of embedding and reranker models using sentence-transformers.
"""

import os
import numpy as np
from typing import Dict, Any, List, Union
from sentence_transformers import SentenceTransformer, CrossEncoder
from .logging_config import get_logger, log_execution_time, log_error_with_context

# Global model cache
_models = {}

# Logger for this module
logger = get_logger("citetube.models")

@log_execution_time("citetube.models")
def get_embedding_model(model_name: str = None):
    """
    Load and cache the sentence-transformers embedding model.
    
    Args:
        model_name: Name of the embedding model to load
        
    Returns:
        SentenceTransformer instance
    """
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    global _models
    model_key = f"embedding_{model_name}"
    
    if model_key not in _models:
        logger.info(f"Loading sentence-transformers embedding model: {model_name}")
        try:
            _models[model_key] = SentenceTransformer(model_name)
            logger.success(f"Successfully loaded embedding model: {model_name}")
        except Exception as e:
            log_error_with_context(e, f"Failed to load embedding model: {model_name}", "citetube.models")
            raise
    else:
        logger.debug(f"Using cached embedding model: {model_name}")
    
    return _models[model_key]

@log_execution_time("citetube.models")
def get_reranker_model(model_name: str = None) -> CrossEncoder:
    """
    Load and cache the reranker model.
    
    Args:
        model_name: Name of the reranker model to load
        
    Returns:
        CrossEncoder model
    """
    if model_name is None:
        model_name = os.getenv("RERANKER_MODEL", "bge-reranker-base")
    
    global _models
    model_key = f"reranker_{model_name}"
    
    if model_key not in _models:
        logger.info(f"Loading reranker model: {model_name}")
        try:
            _models[model_key] = CrossEncoder(model_name)
            logger.success(f"Successfully loaded reranker model: {model_name}")
        except Exception as e:
            log_error_with_context(e, f"Failed to load reranker model: {model_name}", "citetube.models")
            raise
    else:
        logger.debug(f"Using cached reranker model: {model_name}")
    
    return _models[model_key]

def normalize_embedding(embedding):
    """
    Normalize embedding vector for inner product similarity.
    
    Args:
        embedding: Vector to normalize
        
    Returns:
        Normalized vector
    """
    import numpy as np
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding