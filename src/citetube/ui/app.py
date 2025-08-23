"""
CiteTube: Local YouTube Transcript QA Application.
Gradio UI for ingesting YouTube videos and answering questions about their content.
"""

import os
import time
import gradio as gr
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import local modules
from ..ingestion import ingest
from ..retrieval import retrieve
from ..llm import llm
from ..core import db

# Import centralized logging config
from ..core.logging_config import get_logger

logger = get_logger("citetube.app")

# Application state management
class AppState:
    """Manages application state for the current session."""
    def __init__(self):
        self.current_video_id = None
        self.current_video_metadata = None
    
    def set_video(self, video_id: int, metadata: Dict[str, Any]):
        """Set the current video."""
        self.current_video_id = video_id
        self.current_video_metadata = metadata
    
    def clear_video(self):
        """Clear the current video."""
        self.current_video_id = None
        self.current_video_metadata = None
    
    def has_video(self) -> bool:
        """Check if a video is currently loaded."""
        return self.current_video_id is not None

# Global app state instance
app_state = AppState()

def process_youtube_url(url: str) -> Tuple[str, Dict[str, Any]]:
    """
    Process a YouTube URL: extract ID, fetch transcript, chunk, and index.
    
    Args:
        url: YouTube URL
        
    Returns:
        Tuple of (status_message, video_metadata)
    """
    try:
        # Extract YouTube ID
        yt_id = ingest.extract_youtube_id(url)
        if not yt_id:
            return "❌ Invalid YouTube URL", None
        
        # Check if video already exists
        existing_video = db.get_video_by_yt_id(yt_id)
        if existing_video:
            app_state.set_video(existing_video["id"], existing_video)
            return f"✅ Video already ingested: {existing_video.get('title', 'Unknown')}", existing_video
        
        # Ingest video
        start_time = time.time()
        video_id, video_metadata = ingest.ingest_video(url)
        elapsed_time = time.time() - start_time
        
        # Update app state
        app_state.set_video(video_id, video_metadata)
        
        return f"✅ Successfully ingested video in {elapsed_time:.2f}s", video_metadata
        
    except Exception as e:
        logger.error(f"Error processing YouTube URL: {str(e)}")
        return f"❌ Error: {str(e)}", None

def answer_question(question: str) -> Tuple[str, str, str]:
    """
    Answer a question about the current video.
    
    Args:
        question: User's question
        
    Returns:
        Tuple of (answer_html, debug_info, raw_response)
    """
    if not app_state.has_video():
        return "Please ingest a YouTube video first.", "", ""
    
    try:
        # Start timing
        start_time = time.time()
        
        # Retrieve relevant segments
        segments = retrieve.hybrid_search(question, app_state.current_video_id)
        retrieve_time = time.time() - start_time
        
        if not segments:
            return "No relevant segments found in the transcript.", "", ""
        
        # Answer question
        llm_start_time = time.time()
        response = llm.answer_question(question, segments)
        llm_time = time.time() - llm_start_time
        total_time = time.time() - start_time
        
        # Format answer with citations
        answer = response.get("answer", "No answer generated.")
        bullets = response.get("bullets", [])
        confidence = response.get("confidence", 0.0)
        
        # Format answer HTML
        answer_html = f"<h3>Answer:</h3><p>{answer}</p>"
        
        if bullets:
            answer_html += "<h3>Key Points:</h3><ul>"
            for bullet in bullets:
                answer_html += f"<li>{bullet}</li>"
            answer_html += "</ul>"
        
        # Add video info
        video_title = app_state.current_video_metadata.get("title", "Unknown")
        video_id = app_state.current_video_metadata.get("yt_id", "")
        
        answer_html += f"<h3>Source:</h3><p><a href='https://www.youtube.com/watch?v={video_id}' target='_blank'>{video_title}</a></p>"
        
        # Debug info
        debug_info = f"""
        <b>Performance:</b>
        - Retrieval: {retrieve_time:.2f}s
        - LLM: {llm_time:.2f}s
        - Total: {total_time:.2f}s
        
        <b>Metrics:</b>
        - Segments retrieved: {len(segments)}
        - Confidence: {confidence:.2f}
        
        <b>Video Info:</b>
        - ID: {video_id}
        - Title: {video_title}
        - Duration: {app_state.current_video_metadata.get("duration_s", 0) // 60} minutes
        """
        
        # Return formatted answer, debug info, and raw response
        return answer_html, debug_info, response.get("raw_response", "")
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return f"Error: {str(e)}", "", ""

# Create Gradio interface
def create_app():
    """Create and return the Gradio app."""
    with gr.Blocks(title="CiteTube - YouTube Transcript QA") as app:
        gr.Markdown("# 📺 CiteTube - YouTube Transcript QA")
        gr.Markdown("Ask questions about YouTube videos using their transcripts.")
        
        with gr.Tab("Ingest Video"):
            with gr.Row():
                youtube_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    scale=4
                )
                ingest_button = gr.Button("Ingest Video", scale=1)
            
            ingest_status = gr.Markdown("Paste a YouTube URL and click 'Ingest Video'")
            video_info = gr.JSON(label="Video Information", visible=False)
            
            ingest_button.click(
                process_youtube_url,
                inputs=[youtube_url],
                outputs=[ingest_status, video_info]
            )
        
        with gr.Tab("Ask Questions"):
            with gr.Row():
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Ask a question about the video...",
                    scale=4
                )
                ask_button = gr.Button("Ask", scale=1)
            
            answer_output = gr.HTML(label="Answer")
            
            with gr.Accordion("Debug Information", open=False):
                debug_info = gr.HTML()
                raw_response = gr.Textbox(label="Raw LLM Response", lines=10)
            
            ask_button.click(
                answer_question,
                inputs=[question_input],
                outputs=[answer_output, debug_info, raw_response]
            )
        
        gr.Markdown("## How to use CiteTube")
        gr.Markdown("""
        1. **Ingest a YouTube video**: Paste a YouTube URL and click 'Ingest Video'
        2. **Ask questions**: Switch to the 'Ask Questions' tab and ask questions about the video content
        3. **View answers**: The app will retrieve relevant transcript segments and generate an answer with citations
        
        Note: This app only processes the transcript, not the audio or video content.
        """)
    
    return app

# Health check endpoint
def health_check():
    """Simple health check for the application."""
    try:
        # Check database connection
        if not db.test_connection():
            return {"status": "unhealthy", "reason": "database connection failed"}
        
        # Check if vLLM is accessible
        try:
            from ..llm.llm import test_llm_connection
            if not test_llm_connection():
                return {"status": "unhealthy", "reason": "vLLM connection failed"}
        except Exception as e:
            return {"status": "unhealthy", "reason": f"vLLM test failed: {str(e)}"}
        
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}

# Launch function
def launch_app(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False
):
    """Launch the Gradio app with configurable parameters."""
    # Logging is initialized in main.py
    logger.info(f"Launching Gradio app on {server_name}:{server_port}")
    
    # Create and launch the app
    app = create_app()
    
    logger.info("Starting Gradio server")
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        debug=debug,
        show_error=debug
    )

# For backward compatibility
if __name__ == "__main__":
    launch_app()