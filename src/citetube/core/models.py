"""
Models module for CiteTube.
Handles loading and caching of embedding and reranker models using Ollama.
"""

import os
import numpy as np
from typing import Dict, Any, List, Union
import logging
import ollama
from sentence_transformers import CrossEncoder

# Global model cache
_models = {}
_ollama_client = None

def get_ollama_client():
    """Get or create Ollama client."""
    global _ollama_client
    if _ollama_client is None:
        try:
            _ollama_client = ollama.Client()
            # Test connection
            _ollama_client.list()
        except Exception as e:
            logging.error(f"Failed to connect to Ollama: {e}")
            raise ConnectionError(f"Ollama is not running or not accessible. Please start Ollama first. Error: {e}")
    return _ollama_client

class OllamaEmbeddingModel:
    """Wrapper class for Ollama embedding models to match SentenceTransformer interface."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = get_ollama_client()
        
        # Ensure the model is available
        try:
            # Try to get model info to check if it exists
            self.client.show(model_name)
            logging.info(f"Ollama embedding model {model_name} is available")
        except Exception as e:
            logging.warning(f"Model {model_name} not found, attempting to pull...")
            try:
                self.client.pull(model_name)
                logging.info(f"Successfully pulled model {model_name}")
            except Exception as pull_error:
                logging.error(f"Failed to pull model {model_name}: {pull_error}")
                raise
    
    def encode(self, sentences: Union[str, List[str]], show_progress_bar: bool = False, **kwargs) -> np.ndarray:
        """
        Encode sentences using Ollama embeddings.
        
        Args:
            sentences: Single sentence or list of sentences to encode
            show_progress_bar: Whether to show progress bar (ignored for Ollama)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        embeddings = []
        total = len(sentences)
        
        for i, sentence in enumerate(sentences):
            if show_progress_bar and i % 10 == 0:
                logging.info(f"Encoding progress: {i}/{total}")
            
            try:
                response = self.client.embeddings(
                    model=self.model_name,
                    prompt=sentence
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                logging.error(f"Error encoding sentence '{sentence[:50]}...': {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * 1024)  # Default dimension, will be adjusted
        
        if show_progress_bar:
            logging.info(f"Encoding complete: {total}/{total}")
        
        return np.array(embeddings, dtype=np.float32)

def get_embedding_model(model_name: str = None):
    """
    Load and cache the Ollama embedding model.
    
    Args:
        model_name: Name of the embedding model to load
        
    Returns:
        OllamaEmbeddingModel instance
    """
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    global _models
    model_key = f"embedding_{model_name}"
    
    if model_key not in _models:
        logging.info(f"Loading Ollama embedding model: {model_name}")
        _models[model_key] = OllamaEmbeddingModel(model_name)
    
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