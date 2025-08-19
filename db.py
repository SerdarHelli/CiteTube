"""
Database utilities for CiteTube application.
Handles SQLite operations for storing video metadata and transcript segments.
"""

import os
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Ensure data directory exists
DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "meta.db"

def get_db_connection() -> sqlite3.Connection:
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initialize the database with required tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create videos table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    
    # Create segments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS segments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER NOT NULL,
        start_s REAL NOT NULL,
        end_s REAL NOT NULL,
        text TEXT NOT NULL,
        FOREIGN KEY (video_id) REFERENCES videos (id)
    )
    ''')
    
    conn.commit()
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
    cursor = conn.cursor()
    
    # Check if video already exists
    cursor.execute("SELECT id FROM videos WHERE yt_id = ?", (yt_id,))
    result = cursor.fetchone()
    
    now = datetime.now().isoformat()
    
    if result:
        # Update existing video
        video_id = result['id']
        cursor.execute('''
        UPDATE videos 
        SET title = ?, channel = ?, duration_s = ?, language = ?, 
            last_synced_at = ?, transcript_hash = ?, source = ?
        WHERE id = ?
        ''', (title, channel, duration_s, language, now, transcript_hash, source, video_id))
    else:
        # Insert new video
        cursor.execute('''
        INSERT INTO videos (yt_id, title, channel, duration_s, language, last_synced_at, transcript_hash, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (yt_id, title, channel, duration_s, language, now, transcript_hash, source))
        video_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    return video_id

def store_segments(video_id: int, segments: List[Dict[str, Any]]) -> None:
    """
    Store transcript segments for a video.
    
    Args:
        video_id: The ID of the video in the database
        segments: List of segment dictionaries with start_s, end_s, and text
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Delete existing segments for this video
    cursor.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))
    
    # Insert new segments
    for segment in segments:
        cursor.execute('''
        INSERT INTO segments (video_id, start_s, end_s, text)
        VALUES (?, ?, ?, ?)
        ''', (video_id, segment['start_s'], segment['end_s'], segment['text']))
    
    conn.commit()
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
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM videos WHERE yt_id = ?", (yt_id,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return dict(result)
    return None

def get_video_segments(video_id: int) -> List[Dict[str, Any]]:
    """
    Get all segments for a video.
    
    Args:
        video_id: The ID of the video in the database
        
    Returns:
        List of segment dictionaries
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM segments WHERE video_id = ? ORDER BY start_s", (video_id,))
    results = cursor.fetchall()
    
    conn.close()
    
    return [dict(row) for row in results]

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
    cursor = conn.cursor()
    
    placeholders = ','.join('?' for _ in segment_ids)
    query = f"SELECT s.*, v.yt_id, v.title FROM segments s JOIN videos v ON s.video_id = v.id WHERE s.id IN ({placeholders})"
    
    cursor.execute(query, segment_ids)
    results = cursor.fetchall()
    
    conn.close()
    
    return [dict(row) for row in results]

# Initialize the database when the module is imported
init_db()