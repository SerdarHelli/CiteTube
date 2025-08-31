"""
LangChain tools for CiteTube YouTube video analysis.
"""

import json
from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..core import db
from ..retrieval import retrieve
from ..ingestion import ingest
from ..core.logging_config import get_logger

logger = get_logger("citetube.agents.tools")


class VideoSearchInput(BaseModel):
    """Input for video search tool."""
    query: str = Field(description="Search query for finding relevant transcript segments")
    video_id: Optional[str] = Field(default=None, description="Specific video ID to search in (optional)")


class VideoMetadataInput(BaseModel):
    """Input for video metadata tool."""
    video_id: Optional[str] = Field(default=None, description="Video ID to get metadata for (optional, uses current video if not provided)")


class TranscriptSearchInput(BaseModel):
    """Input for transcript search tool."""
    query: str = Field(description="Search query for transcript content")
    video_id: Optional[str] = Field(default=None, description="Video ID to search in (optional)")
    top_k: int = Field(default=5, description="Number of top results to return")


class VideoSummaryInput(BaseModel):
    """Input for video summary tool."""
    video_id: Optional[str] = Field(default=None, description="Video ID to summarize (optional)")
    max_segments: int = Field(default=10, description="Maximum number of segments to include in summary")


class TimestampInput(BaseModel):
    """Input for timestamp tool."""
    timestamp: str = Field(description="Timestamp in format mm:ss or hh:mm:ss")
    video_id: Optional[str] = Field(default=None, description="Video ID (optional)")


class VideoSearchTool(BaseTool):
    """Tool for searching video transcript content."""
    
    name: str = "video_search"
    description: str = """Search for relevant segments in YouTube video transcripts. 
    Use this when you need to find specific information or topics within video content.
    Returns ranked segments with timestamps and relevance scores."""
    args_schema: type[BaseModel] = VideoSearchInput
    current_video_id: Optional[int] = None
    
    def __init__(self, current_video_id: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.current_video_id = current_video_id
    
    def _run(self, query: str, video_id: Optional[str] = None, **kwargs) -> str:
        """Execute the video search."""
        try:
            # Use provided video_id or fall back to current video
            if video_id:
                # Try to convert string video_id to int, or use as-is if it's already int
                try:
                    target_video_id = int(video_id) if isinstance(video_id, str) else video_id
                except ValueError:
                    target_video_id = self.current_video_id
            else:
                target_video_id = self.current_video_id
            
            if not target_video_id:
                return "Error: No video ID provided and no current video set."
            
            # Perform hybrid search
            segments = retrieve.hybrid_search(query, target_video_id)
            
            if not segments:
                return f"No relevant segments found for query: '{query}'"
            
            # Format results
            results = []
            for i, segment in enumerate(segments[:5], 1):
                timestamp = segment.get("timestamp", "00:00")
                text = segment.get("text", "")[:200] + "..." if len(segment.get("text", "")) > 200 else segment.get("text", "")
                score = segment.get("score", 0.0)
                
                results.append(f"{i}. [{timestamp}] (Score: {score:.3f})\n{text}")
            
            return f"Found {len(segments)} relevant segments for '{query}':\n\n" + "\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Error in video search: {e}")
            return f"Error searching video: {str(e)}"


class VideoMetadataTool(BaseTool):
    """Tool for getting video metadata and information."""
    
    name: str = "video_metadata"
    description: str = """Get metadata and information about a YouTube video including title, duration, 
    description, and ingestion details. Use this to understand what video you're working with."""
    args_schema: type[BaseModel] = VideoMetadataInput
    current_video_id: Optional[int] = None
    
    def __init__(self, current_video_id: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.current_video_id = current_video_id
    
    def _run(self, video_id: Optional[str] = None, **kwargs) -> str:
        """Get video metadata."""
        try:
            # Use provided video_id or fall back to current video
            if video_id:
                # Try to convert string video_id to int, or use as-is if it's already int
                try:
                    target_video_id = int(video_id) if isinstance(video_id, str) else video_id
                except ValueError:
                    target_video_id = self.current_video_id
            else:
                target_video_id = self.current_video_id
            
            if not target_video_id:
                return "Error: No video ID provided and no current video set."
            
            # Get video metadata from database
            video_data = db.get_video_by_id(target_video_id)
            
            if not video_data:
                return f"Error: Video with ID {target_video_id} not found."
            
            # Format metadata
            metadata = {
                "title": video_data.get("title", "Unknown"),
                "youtube_id": video_data.get("yt_id", "Unknown"),
                "duration": f"{video_data.get('duration_s', 0) // 60} minutes {video_data.get('duration_s', 0) % 60} seconds",
                "description": video_data.get("description", "No description")[:300] + "..." if len(video_data.get("description", "")) > 300 else video_data.get("description", ""),
                "ingested_at": video_data.get("created_at", "Unknown"),
                "segment_count": video_data.get("segment_count", 0),
                "url": f"https://www.youtube.com/watch?v={video_data.get('yt_id', '')}"
            }
            
            return f"""Video Metadata:
Title: {metadata['title']}
YouTube ID: {metadata['youtube_id']}
Duration: {metadata['duration']}
Segments: {metadata['segment_count']}
Ingested: {metadata['ingested_at']}
URL: {metadata['url']}

Description: {metadata['description']}"""
            
        except Exception as e:
            logger.error(f"Error getting video metadata: {e}")
            return f"Error getting video metadata: {str(e)}"


class TranscriptSearchTool(BaseTool):
    """Tool for detailed transcript search with more control."""
    
    name: str = "transcript_search"
    description: str = """Advanced transcript search with customizable parameters. 
    Use this for more detailed searches when you need specific control over results."""
    args_schema: type[BaseModel] = TranscriptSearchInput
    current_video_id: Optional[int] = None
    
    def __init__(self, current_video_id: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.current_video_id = current_video_id
    
    def _run(self, query: str, video_id: Optional[str] = None, top_k: int = 5, **kwargs) -> str:
        """Execute detailed transcript search."""
        try:
            # Use provided video_id or fall back to current video
            if video_id:
                # Try to convert string video_id to int, or use as-is if it's already int
                try:
                    target_video_id = int(video_id) if isinstance(video_id, str) else video_id
                except ValueError:
                    target_video_id = self.current_video_id
            else:
                target_video_id = self.current_video_id
            
            if not target_video_id:
                return "Error: No video ID provided and no current video set."
            
            # Perform search with custom top_k
            segments = retrieve.hybrid_search(query, target_video_id, top_k=top_k)
            
            if not segments:
                return f"No relevant segments found for query: '{query}'"
            
            # Detailed formatting
            results = []
            for segment in segments:
                timestamp = segment.get("timestamp", "00:00")
                text = segment.get("text", "")
                score = segment.get("score", 0.0)
                segment_id = segment.get("id", "unknown")
                
                results.append({
                    "segment_id": segment_id,
                    "timestamp": timestamp,
                    "text": text,
                    "relevance_score": score
                })
            
            return json.dumps({
                "query": query,
                "total_results": len(segments),
                "segments": results
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error in transcript search: {e}")
            return f"Error in transcript search: {str(e)}"


class VideoSummaryTool(BaseTool):
    """Tool for generating video summaries."""
    
    name: str = "video_summary"
    description: str = """Generate a summary of the video content by analyzing transcript segments. 
    Use this to get an overview of what the video is about."""
    args_schema: type[BaseModel] = VideoSummaryInput
    current_video_id: Optional[int] = None
    
    def __init__(self, current_video_id: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.current_video_id = current_video_id
    
    def _run(self, video_id: Optional[str] = None, max_segments: int = 10, **kwargs) -> str:
        """Generate video summary."""
        try:
            # Use provided video_id or fall back to current video
            if video_id:
                # Try to convert string video_id to int, or use as-is if it's already int
                try:
                    target_video_id = int(video_id) if isinstance(video_id, str) else video_id
                except ValueError:
                    target_video_id = self.current_video_id
            else:
                target_video_id = self.current_video_id
            
            if not target_video_id:
                return "Error: No video ID provided and no current video set."
            
            # Get video metadata
            video_data = db.get_video_by_id(target_video_id)
            if not video_data:
                return f"Error: Video with ID {target_video_id} not found."
            
            # Get representative segments (using a broad search)
            segments = retrieve.hybrid_search("summary overview main points", target_video_id, top_k=max_segments)
            
            if not segments:
                # Fallback: get first few segments
                segments = db.get_segments_by_video_id(target_video_id, limit=max_segments)
            
            # Create summary
            title = video_data.get("title", "Unknown Video")
            duration = f"{video_data.get('duration_s', 0) // 60} minutes"
            
            summary_parts = [f"Video: {title} ({duration})"]
            
            if segments:
                summary_parts.append("\nKey Content:")
                for i, segment in enumerate(segments[:5], 1):
                    timestamp = segment.get("timestamp", "00:00")
                    text = segment.get("text", "")[:150] + "..." if len(segment.get("text", "")) > 150 else segment.get("text", "")
                    summary_parts.append(f"{i}. [{timestamp}] {text}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating video summary: {e}")
            return f"Error generating video summary: {str(e)}"


class TimestampTool(BaseTool):
    """Tool for working with video timestamps."""
    
    name: str = "timestamp_lookup"
    description: str = """Look up content at a specific timestamp in the video. 
    Use this when you need to find what was said at a particular time."""
    args_schema: type[BaseModel] = TimestampInput
    current_video_id: Optional[int] = None
    
    def __init__(self, current_video_id: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.current_video_id = current_video_id
    
    def _run(self, timestamp: str, video_id: Optional[str] = None, **kwargs) -> str:
        """Look up content at timestamp."""
        try:
            # Use provided video_id or fall back to current video
            if video_id:
                # Try to convert string video_id to int, or use as-is if it's already int
                try:
                    target_video_id = int(video_id) if isinstance(video_id, str) else video_id
                except ValueError:
                    target_video_id = self.current_video_id
            else:
                target_video_id = self.current_video_id
            
            if not target_video_id:
                return "Error: No video ID provided and no current video set."
            
            # Convert timestamp to seconds for comparison
            try:
                time_parts = timestamp.split(":")
                if len(time_parts) == 2:  # mm:ss
                    minutes, seconds = map(int, time_parts)
                    target_seconds = minutes * 60 + seconds
                elif len(time_parts) == 3:  # hh:mm:ss
                    hours, minutes, seconds = map(int, time_parts)
                    target_seconds = hours * 3600 + minutes * 60 + seconds
                else:
                    return f"Invalid timestamp format: {timestamp}. Use mm:ss or hh:mm:ss"
            except ValueError:
                return f"Invalid timestamp format: {timestamp}. Use mm:ss or hh:mm:ss"
            
            # Find segment closest to timestamp
            segment = db.get_segment_by_timestamp(target_video_id, target_seconds)
            
            if not segment:
                return f"No content found near timestamp {timestamp}"
            
            return f"Content at {timestamp}:\n[{segment.get('timestamp', 'Unknown')}] {segment.get('text', 'No text available')}"
            
        except Exception as e:
            logger.error(f"Error looking up timestamp: {e}")
            return f"Error looking up timestamp: {str(e)}"


def create_tools(current_video_id: Optional[int] = None) -> List[BaseTool]:
    """Create all CiteTube tools with optional current video context."""
    return [
        VideoSearchTool(current_video_id),
        VideoMetadataTool(current_video_id),
        TranscriptSearchTool(current_video_id),
        VideoSummaryTool(current_video_id),
        TimestampTool(current_video_id)
    ]