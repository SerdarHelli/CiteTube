"""
Transcript ingestion module for CiteTube.
Handles fetching, chunking, and embedding of YouTube transcripts.
"""

import os
import re
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from ..core.models import get_embedding_model, normalize_embedding
from ..core import db

# Import centralized logging config
from ..core.logging_config import setup_logging

logger = logging.getLogger("citetube.ingest")

# Constants
CHUNK_OVERLAP_SECONDS = 5
MIN_CHUNK_LENGTH = 50
MAX_CHUNK_LENGTH = 500

def extract_youtube_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        YouTube video ID or None if not found
    """
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def fetch_transcript(video_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch transcript from YouTube.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Tuple of (transcript_items, metadata)
    
    Raises:
        Exception: If transcript cannot be fetched
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript first, then manual, then any available
        transcript = None
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            try:
                # Try to get manually created transcript
                for t in transcript_list:
                    if t.is_generated is False:
                        transcript = t
                        break
            except:
                # Get any available transcript
                transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
        
        if not transcript:
            # If still no transcript, get the first available one
            transcript = list(transcript_list)[0]
        
        # Get transcript data
        transcript_data = transcript.fetch()
        
        # Get video metadata
        metadata = {
            "language": transcript.language,
            "is_generated": transcript.is_generated,
            "title": "Unknown",  # We don't have access to title via transcript API
            "channel": "Unknown",  # We don't have access to channel via transcript API
            "duration_s": transcript_data[-1]["start"] + transcript_data[-1]["duration"] if transcript_data else 0
        }
        
        return transcript_data, metadata
        
    except TranscriptsDisabled:
        raise Exception("Transcripts are disabled for this video")
    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {str(e)}")

def chunk_transcript(transcript_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunk transcript into segments with overlap.
    
    Args:
        transcript_items: List of transcript items from YouTube API
        
    Returns:
        List of chunked segments
    """
    chunks = []
    current_chunk = {
        "start_s": 0,
        "end_s": 0,
        "text": "",
    }
    
    for item in transcript_items:
        # If adding this item would make the chunk too long, finalize the current chunk
        if len(current_chunk["text"]) + len(item["text"]) > MAX_CHUNK_LENGTH and len(current_chunk["text"]) >= MIN_CHUNK_LENGTH:
            current_chunk["end_s"] = item["start"]
            chunks.append(current_chunk)
            
            # Start a new chunk with overlap
            overlap_start = max(0, item["start"] - CHUNK_OVERLAP_SECONDS)
            current_chunk = {
                "start_s": overlap_start,
                "end_s": 0,
                "text": "",
            }
        
        # Add the current item to the chunk
        if current_chunk["text"]:
            current_chunk["text"] += " "
        current_chunk["text"] += item["text"]
        
        # Update end time
        current_chunk["end_s"] = item["start"] + item["duration"]
    
    # Add the last chunk if it's not empty
    if current_chunk["text"] and len(current_chunk["text"]) >= MIN_CHUNK_LENGTH:
        chunks.append(current_chunk)
    
    return chunks

def format_timestamp(seconds: float) -> str:
    """
    Format seconds as mm:ss.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def generate_and_store_embeddings(video_id: int, segments: List[Dict[str, Any]]) -> None:
    """
    Generate embeddings for video segments and store them in PostgreSQL with pgvector.
    
    Args:
        video_id: Database ID of the video
        segments: List of segment dictionaries
    """
    # Get embedding model
    model = get_embedding_model()
    
    # Extract texts for embedding
    texts = [segment["text"] for segment in segments]
    
    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Normalize embeddings for cosine similarity
    normalized_embeddings = np.array([normalize_embedding(emb) for emb in embeddings], dtype=np.float32)
    
    # Store embeddings in database
    db.update_segment_embeddings(video_id, normalized_embeddings)
    
    logger.info(f"Generated and stored embeddings for video {video_id} with {len(segments)} segments")

def ingest_video(url: str) -> Tuple[int, Dict[str, Any]]:
    """
    Ingest a YouTube video: fetch transcript, chunk, embed, and store.
    
    Args:
        url: YouTube URL
        
    Returns:
        Tuple of (video_id, video_metadata)
    
    Raises:
        Exception: If ingestion fails
    """
    # Extract YouTube ID
    yt_id = extract_youtube_id(url)
    if not yt_id:
        raise ValueError("Invalid YouTube URL")
    
    logger.info(f"Ingesting video: {yt_id}")
    
    # Check if video already exists
    existing_video = db.get_video_by_yt_id(yt_id)
    
    # Fetch transcript
    transcript_items, metadata = fetch_transcript(yt_id)
    
    # Generate transcript hash
    transcript_text = " ".join(item["text"] for item in transcript_items)
    transcript_hash = hashlib.md5(transcript_text.encode()).hexdigest()
    
    # If video exists and transcript hasn't changed, return existing video
    if existing_video and existing_video["transcript_hash"] == transcript_hash:
        logger.info(f"Video {yt_id} already ingested with same transcript")
        return existing_video["id"], existing_video
    
    # Chunk transcript
    chunks = chunk_transcript(transcript_items)
    
    # Store video metadata
    video_id = db.store_video_metadata(
        yt_id=yt_id,
        title=metadata.get("title", "Unknown"),
        channel=metadata.get("channel", "Unknown"),
        duration_s=int(metadata.get("duration_s", 0)),
        language=metadata.get("language", "unknown"),
        transcript_hash=transcript_hash,
        source="youtube"
    )
    
    # Store segments
    db.store_segments(video_id, chunks)
    
    # Generate and store embeddings
    segments = db.get_video_segments(video_id)
    generate_and_store_embeddings(video_id, segments)
    
    # Get updated video metadata
    video_metadata = db.get_video_by_yt_id(yt_id)
    
    logger.info(f"Successfully ingested video {yt_id} with {len(chunks)} segments")
    
    return video_id, video_metadata