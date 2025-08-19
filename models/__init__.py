"""
Models module for CiteTube.
Handles loading and caching of embedding and reranker models.
"""

import os
from typing import Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
import logging

# Global model cache
_models = {}

def get_embedding_model(model_name: str = None) -> SentenceTransformer:
    """
    Load and cache the embedding model.
    
    Args:
        model_name: Name of the embedding model to load
        
    Returns:
        SentenceTransformer model
    """
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", "bge-m3")
    
    global _models
    model_key = f"embedding_{model_name}"
    
    if model_key not in _models:
        logging.info(f"Loading embedding model: {model_name}")
        _models[model_key] = SentenceTransformer(model_name)
    
    return _models[model_key]

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
        logging.info(f"Loading reranker model: {model_name}")
        _models[model_key] = CrossEncoder(model_name)
    
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