"""
PostgreSQL database utilities for CiteTube application with pgvector support.
Handles PostgreSQL operations for storing video metadata, transcript segments, and embeddings.
"""

import os
import logging
import psycopg2
import psycopg2.extras
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from pgvector.psycopg2 import register_vector

from .config import get_db_host, get_db_port, get_db_name, get_db_user, get_db_password

# Configure logging
logger = logging.getLogger(__name__)

def get_db_connection() -> psycopg2.extensions.connection:
    """Get a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=get_db_host(),
            port=get_db_port(),
            database=get_db_name(),
            user=get_db_user(),
            password=get_db_password()
        )
        # Register pgvector extension
        register_vector(conn)
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

def init_db() -> None:
    """Initialize the database with required tables and extensions if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create videos table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id SERIAL PRIMARY KEY,
            yt_id TEXT UNIQUE NOT NULL,
            title TEXT,
            channel TEXT,
            duration_s INTEGER,
            language TEXT,
            last_synced_at TIMESTAMP,
            transcript_hash TEXT,
            source TEXT
        )
        ''')
        
        # Create segments table with vector column
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS segments (
            id SERIAL PRIMARY KEY,
            video_id INTEGER NOT NULL,
            start_s REAL NOT NULL,
            end_s REAL NOT NULL,
            text TEXT NOT NULL,
            embedding vector(768),
            FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
        )
        ''')
        
        # Create index on embedding column for faster similarity search
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS segments_embedding_idx 
        ON segments USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        ''')
        
        # Create index on video_id for faster joins
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS segments_video_id_idx 
        ON segments (video_id);
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")
        
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Failed to initialize database: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def store_video_metadata(
    yt_id: str, 
    title: str, 
    channel: str, 
    duration_s: int, 
    language: str, 
    transcript_hash: str,
    source: str = "youtube"
) -> int:
    """
    Store video metadata in the database.
    
    Args:
        yt_id: YouTube video ID
        title: Video title
        channel: Channel name
        duration_s: Duration in seconds
        language: Language code
        transcript_hash: Hash of the transcript content
        source: Source of the video (default: "youtube")
        
    Returns:
        The ID of the inserted or updated video
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    try:
        # Check if video already exists
        cursor.execute("SELECT id FROM videos WHERE yt_id = %s", (yt_id,))
        result = cursor.fetchone()
        
        now = datetime.now()
        
        if result:
            # Update existing video
            video_id = result['id']
            cursor.execute('''
            UPDATE videos 
            SET title = %s, channel = %s, duration_s = %s, language = %s, 
                last_synced_at = %s, transcript_hash = %s, source = %s
            WHERE id = %s
            ''', (title, channel, duration_s, language, now, transcript_hash, source, video_id))
        else:
            # Insert new video
            cursor.execute('''
            INSERT INTO videos (yt_id, title, channel, duration_s, language, last_synced_at, transcript_hash, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            ''', (yt_id, title, channel, duration_s, language, now, transcript_hash, source))
            video_id = cursor.fetchone()['id']
        
        conn.commit()
        return video_id
        
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Failed to store video metadata: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def store_segments(video_id: int, segments: List[Dict[str, Any]], embeddings: Optional[np.ndarray] = None) -> None:
    """
    Store transcript segments for a video with optional embeddings.
    
    Args:
        video_id: The ID of the video in the database
        segments: List of segment dictionaries with start_s, end_s, and text
        embeddings: Optional numpy array of embeddings for each segment
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Delete existing segments for this video
        cursor.execute("DELETE FROM segments WHERE video_id = %s", (video_id,))
        
        # Insert new segments
        for i, segment in enumerate(segments):
            embedding = embeddings[i] if embeddings is not None else None
            cursor.execute('''
            INSERT INTO segments (video_id, start_s, end_s, text, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ''', (video_id, segment['start_s'], segment['end_s'], segment['text'], embedding))
        
        conn.commit()
        logger.info(f"Stored {len(segments)} segments for video {video_id}")
        
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Failed to store segments: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def update_segment_embeddings(video_id: int, embeddings: np.ndarray) -> None:
    """
    Update embeddings for all segments of a video.
    
    Args:
        video_id: The ID of the video in the database
        embeddings: Numpy array of embeddings for each segment
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get segment IDs in order
        cursor.execute("SELECT id FROM segments WHERE video_id = %s ORDER BY start_s", (video_id,))
        segment_ids = [row[0] for row in cursor.fetchall()]
        
        if len(segment_ids) != len(embeddings):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) doesn't match number of segments ({len(segment_ids)})")
        
        # Update embeddings
        for segment_id, embedding in zip(segment_ids, embeddings):
            cursor.execute(
                "UPDATE segments SET embedding = %s WHERE id = %s",
                (embedding, segment_id)
            )
        
        conn.commit()
        logger.info(f"Updated embeddings for {len(segment_ids)} segments of video {video_id}")
        
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Failed to update embeddings: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_video_by_yt_id(yt_id: str) -> Optional[Dict[str, Any]]:
    """
    Get video metadata by YouTube ID.
    
    Args:
        yt_id: YouTube video ID
        
    Returns:
        Dictionary with video metadata or None if not found
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    try:
        cursor.execute("SELECT * FROM videos WHERE yt_id = %s", (yt_id,))
        result = cursor.fetchone()
        
        if result:
            return dict(result)
        return None
        
    except psycopg2.Error as e:
        logger.error(f"Failed to get video by yt_id: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_video_segments(video_id: int) -> List[Dict[str, Any]]:
    """
    Get all segments for a video.
    
    Args:
        video_id: The ID of the video in the database
        
    Returns:
        List of segment dictionaries
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    try:
        cursor.execute("SELECT * FROM segments WHERE video_id = %s ORDER BY start_s", (video_id,))
        results = cursor.fetchall()
        
        return [dict(row) for row in results]
        
    except psycopg2.Error as e:
        logger.error(f"Failed to get video segments: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_segments_by_ids(segment_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Get segments by their IDs.
    
    Args:
        segment_ids: List of segment IDs
        
    Returns:
        List of segment dictionaries
    """
    if not segment_ids:
        return []
        
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    try:
        cursor.execute('''
        SELECT s.*, v.yt_id, v.title 
        FROM segments s 
        JOIN videos v ON s.video_id = v.id 
        WHERE s.id = ANY(%s)
        ''', (segment_ids,))
        results = cursor.fetchall()
        
        return [dict(row) for row in results]
        
    except psycopg2.Error as e:
        logger.error(f"Failed to get segments by IDs: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def vector_similarity_search(
    query_embedding: np.ndarray, 
    video_id: int, 
    top_k: int = 30
) -> List[int]:
    """
    Perform vector similarity search using pgvector.
    
    Args:
        query_embedding: Query embedding vector
        video_id: Database ID of the video
        top_k: Number of results to return
        
    Returns:
        List of segment IDs ordered by similarity
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Use cosine similarity for search
        cursor.execute('''
        SELECT id, 1 - (embedding <=> %s) as similarity
        FROM segments 
        WHERE video_id = %s AND embedding IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT %s
        ''', (query_embedding, video_id, query_embedding, top_k))
        
        results = cursor.fetchall()
        return [row[0] for row in results]
        
    except psycopg2.Error as e:
        logger.error(f"Failed to perform vector similarity search: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

# Note: Database initialization is now done explicitly via setup_postgres.py or app startup
# This prevents import-time failures when PostgreSQL is not available