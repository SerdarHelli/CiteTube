"""
Retrieval module for CiteTube.
Implements hybrid search combining FAISS and BM25.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from rank_bm25 import BM25Okapi
import time

from ..core.models import get_embedding_model, get_reranker_model, normalize_embedding
from ..core import db
from ..core.config import DEFAULT_TOP_K, DEFAULT_VECTOR_TOP_K, use_reranker

# Import centralized logging config
from ..core.logging_config import get_logger

logger = get_logger("citetube.retrieve")

def reciprocal_rank_fusion(
    results_lists: List[List[int]], 
    k: int = 60
) -> List[int]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    Args:
        results_lists: List of lists containing segment IDs
        k: Constant to prevent division by zero and reduce impact of high rankings
        
    Returns:
        Combined and re-ranked list of segment IDs
    """
    # Create a dictionary to store the RRF scores for each document
    rrf_scores = {}
    
    # Process each results list
    for results in results_lists:
        # Process each document in the results list
        for rank, doc_id in enumerate(results):
            # Calculate the RRF score for this document in this results list
            # Add 1 to rank because RRF uses 1-based ranking
            contribution = 1.0 / (k + rank + 1)
            
            # Add the contribution to the document's total RRF score
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += contribution
    
    # Sort documents by their RRF scores in descending order
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return just the document IDs
    return [doc_id for doc_id, _ in sorted_docs]

def search_vector(
    query: str, 
    video_id: int, 
    top_k: int = 30
) -> List[int]:
    """
    Search for segments using pgvector similarity.
    
    Args:
        query: Search query
        video_id: Database ID of the video
        top_k: Number of results to return
        
    Returns:
        List of segment IDs
    """
    try:
        # Get embedding model and encode query
        model = get_embedding_model()
        query_embedding = model.encode(query)
        
        # Normalize query embedding
        query_embedding = normalize_embedding(query_embedding)
        
        # Search using pgvector
        segment_ids = db.vector_similarity_search(query_embedding, video_id, top_k)
        
        return segment_ids
        
    except Exception as e:
        logger.error(f"Vector search failed for video {video_id}: {e}")
        return []

def search_bm25(
    query: str, 
    video_id: int, 
    top_k: int = 30
) -> List[int]:
    """
    Search for segments using BM25 keyword search.
    
    Args:
        query: Search query
        video_id: Database ID of the video
        top_k: Number of results to return
        
    Returns:
        List of segment IDs
    """
    # Get segments
    segments = db.get_video_segments(video_id)
    
    # Prepare corpus
    corpus = [segment["text"] for segment in segments]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    
    # Create BM25 model
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Tokenize query
    tokenized_query = query.lower().split()
    
    # Get scores
    scores = bm25.get_scores(tokenized_query)
    
    # Get top_k segment IDs
    top_indices = np.argsort(scores)[::-1][:top_k]
    segment_ids = [segments[idx]["id"] for idx in top_indices]
    
    return segment_ids

def rerank_results(
    query: str, 
    segment_ids: List[int], 
    top_k: int = DEFAULT_TOP_K
) -> List[int]:
    """
    Rerank results using a cross-encoder model.
    
    Args:
        query: Search query
        segment_ids: List of segment IDs to rerank
        top_k: Number of results to return
        
    Returns:
        Reranked list of segment IDs
    """
    if not use_reranker() or not segment_ids:
        return segment_ids[:top_k]
    
    # Get segments
    segments = db.get_segments_by_ids(segment_ids)
    
    # Prepare pairs for reranking
    pairs = [(query, segment["text"]) for segment in segments]
    
    # Get reranker model
    model = get_reranker_model()
    
    # Get scores
    scores = model.predict(pairs)
    
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    reranked_ids = [segment_ids[idx] for idx in sorted_indices[:top_k]]
    
    return reranked_ids

def hybrid_search(
    query: str, 
    video_id: int, 
    top_k: int = DEFAULT_TOP_K
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining pgvector and BM25.
    
    Args:
        query: Search query
        video_id: Database ID of the video
        top_k: Number of results to return
        
    Returns:
        List of segment dictionaries
    """
    start_time = time.time()
    
    # Search using pgvector
    vector_results = search_vector(query, video_id, DEFAULT_VECTOR_TOP_K)
    
    # Search using BM25
    bm25_results = search_bm25(query, video_id)
    
    # Combine results using RRF
    combined_ids = reciprocal_rank_fusion([vector_results, bm25_results])
    
    # Rerank results
    reranked_ids = rerank_results(query, combined_ids, top_k)
    
    # Get segment details
    segments = db.get_segments_by_ids(reranked_ids)
    
    # Add formatted timestamps
    from ..ingestion.ingest import format_timestamp
    for segment in segments:
        segment["timestamp"] = format_timestamp(segment["start_s"])
    
    logger.info(f"Hybrid search for '{query}' completed in {time.time() - start_time:.2f}s")
    
    return segments