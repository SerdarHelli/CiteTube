"""
CiteTube: Simplified YouTube Transcript QA Application with LangChain Agent.
Focuses on chatbot inference and YouTube URL investigation.
"""

import os
import time
import gradio as gr
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import core modules
from ..core.logging_config import get_logger
from ..core import db
from ..ingestion import ingest
from ..llm.agent import CiteTubeAgent

logger = get_logger("citetube.app")

# Global agent instance
global_agent = None

def get_agent() -> CiteTubeAgent:
    """Get or create the global agent instance."""
    global global_agent
    if global_agent is None:
        global_agent = CiteTubeAgent()
        logger.info("Created new CiteTube agent instance")
    return global_agent

def process_youtube_url(url: str) -> Tuple[str, str]:
    """
    Process a YouTube URL and make it available to the agent.
    
    Args:
        url: YouTube URL
        
    Returns:
        Tuple of (status_message, video_info_html)
    """
    if not url.strip():
        return "Please enter a YouTube URL", ""
    
    try:
        # Extract YouTube ID
        yt_id = ingest.extract_youtube_id(url)
        if not yt_id:
            return "❌ Invalid YouTube URL", ""
        
        # Check if video already exists
        existing_video = db.get_video_by_yt_id(yt_id)
        if existing_video:
            # Update agent with existing video
            agent = get_agent()
            agent.set_current_video(existing_video["id"])
            
            video_info = f"""
            <div style="background-color: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h3>✅ Video Already Available</h3>
                <p><strong>Title:</strong> {existing_video.get('title', 'Unknown')}</p>
                <p><strong>Duration:</strong> {existing_video.get('duration_s', 0) // 60} minutes</p>
                <p><strong>YouTube ID:</strong> {existing_video.get('yt_id', '')}</p>
                <p><strong>URL:</strong> <a href="{url}" target="_blank">{url}</a></p>
            </div>
            """
            return "✅ Video loaded and ready for questions!", video_info
        
        # Ingest new video
        start_time = time.time()
        logger.info(f"Starting ingestion of video: {yt_id}")
        
        video_id, video_metadata = ingest.ingest_video(url)
        elapsed_time = time.time() - start_time
        
        # Update agent with new video
        agent = get_agent()
        agent.set_current_video(video_id)
        
        video_info = f"""
        <div style="background-color: #f0fff0; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h3>✅ Video Successfully Ingested</h3>
            <p><strong>Title:</strong> {video_metadata.get('title', 'Unknown')}</p>
            <p><strong>Duration:</strong> {video_metadata.get('duration_s', 0) // 60} minutes</p>
            <p><strong>Processing Time:</strong> {elapsed_time:.2f} seconds</p>
            <p><strong>YouTube ID:</strong> {video_metadata.get('yt_id', '')}</p>
            <p><strong>URL:</strong> <a href="{url}" target="_blank">{url}</a></p>
        </div>
        """
        
        logger.info(f"Video ingested successfully in {elapsed_time:.2f}s")
        return "✅ Video ingested and ready for questions!", video_info
        
    except Exception as e:
        logger.error(f"Error processing YouTube URL: {str(e)}")
        return f"❌ Error: {str(e)}", ""

def chat_with_agent(message: str, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
    """
    Chat with the LangChain agent. The agent will decide when to use tools.
    
    Args:
        message: User's message
        history: Chat history
        
    Returns:
        Tuple of (updated_history, empty_string_for_textbox)
    """
    if not message.strip():
        return history, ""
    
    try:
        agent = get_agent()
        
        # Check if agent has a video loaded
        if not agent.current_video_id:
            response_text = """
            🤖 **CiteTube Assistant**: Hello! I'm ready to help you analyze YouTube videos.
            
            To get started, please:
            1. **Paste a YouTube URL** in the "YouTube URL" field above
            2. **Click "Process Video"** to ingest the video
            3. **Ask me questions** about the video content
            
            I have access to specialized tools for:
            - 🔍 Searching video transcripts
            - 📊 Getting video metadata
            - 📋 Generating summaries
            - ⏰ Looking up specific timestamps
            
            Once you load a video, I'll automatically use the right tools to answer your questions!
            """
        else:
            # Get response from agent
            start_time = time.time()
            logger.info(f"Processing chat message: {message[:100]}...")
            
            response = agent.ask(message)
            elapsed_time = time.time() - start_time
            
            answer = response.get("answer", "I couldn't generate a response.")
            agent_steps = response.get("agent_steps", [])
            
            # Format response with agent information
            response_text = f"🤖 **CiteTube Assistant**: {answer}"
            
            # Add tool usage information if available
            if agent_steps:
                tools_used = []
                for step in agent_steps:
                    if hasattr(step, 'action') and hasattr(step.action, 'tool'):
                        tools_used.append(step.action.tool)
                
                if tools_used:
                    unique_tools = list(set(tools_used))
                    response_text += f"\n\n🛠️ *Tools used: {', '.join(unique_tools)}*"
            
            response_text += f"\n\n⏱️ *Response time: {elapsed_time:.2f}s*"
            
            logger.info(f"Agent response generated in {elapsed_time:.2f}s using {len(agent_steps)} steps")
        
        # Update history with message format
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response_text})
        return history, ""
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        error_response = f"❌ **Error**: Sorry, I encountered an error: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_response})
        return history, ""

def clear_chat():
    """Clear the chat history."""
    return [], ""

def get_agent_info() -> str:
    """Get information about the current agent state."""
    try:
        agent = get_agent()
        
        if not agent.current_video_id:
            return """
            <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h3>🤖 Agent Status</h3>
                <p><strong>Status:</strong> Ready (No video loaded)</p>
                <p><strong>Available Tools:</strong> 5 tools ready</p>
                <p>Load a YouTube video to start asking questions!</p>
            </div>
            """
        
        # Get current video info
        video_info = agent.get_current_video_info()
        tools = agent.get_available_tools()
        
        tools_list = "<ul>" + "".join([f"<li><strong>{tool['name']}</strong>: {tool['description'][:100]}...</li>" for tool in tools]) + "</ul>"
        
        return f"""
        <div style="background-color: #d4edda; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h3>🤖 Agent Status</h3>
            <p><strong>Status:</strong> Active with video loaded</p>
            <p><strong>Current Video:</strong> ID {agent.current_video_id}</p>
            <p><strong>Available Tools:</strong> {len(tools)} tools</p>
            
            <h4>🛠️ Available Tools:</h4>
            {tools_list}
            
            <h4>📹 Current Video Info:</h4>
            <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 4px; font-size: 12px;">{video_info}</pre>
        </div>
        """
        
    except Exception as e:
        logger.error(f"Error getting agent info: {e}")
        return f"""
        <div style="background-color: #f8d7da; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h3>❌ Agent Error</h3>
            <p>Error getting agent information: {str(e)}</p>
        </div>
        """

def create_app():
    """Create and return the Gradio app."""
    with gr.Blocks(
        title="CiteTube - AI Agent for YouTube Analysis",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-container {
            height: 600px !important;
        }
        """
    ) as app:
        
        gr.Markdown("""
        # 🎥 CiteTube - AI Agent for YouTube Analysis
        
        **Intelligent YouTube video analysis powered by LangChain agents**
        
        Simply paste a YouTube URL, process the video, and start chatting! The AI agent will automatically use the right tools to answer your questions about the video content.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # YouTube URL Processing Section
                with gr.Group():
                    gr.Markdown("### 📺 YouTube Video Processing")
                    with gr.Row():
                        youtube_url = gr.Textbox(
                            label="YouTube URL",
                            placeholder="https://www.youtube.com/watch?v=...",
                            scale=4
                        )
                        process_btn = gr.Button("Process Video", variant="primary", scale=1)
                    
                    url_status = gr.Markdown("Enter a YouTube URL and click 'Process Video' to get started.")
                    video_info_display = gr.HTML()
                
                # Chat Interface
                with gr.Group():
                    gr.Markdown("### 💬 Chat with AI Agent")
                    
                    chatbot = gr.Chatbot(
                        label="CiteTube Assistant",
                        height=500,
                        show_label=True,
                        container=True,
                        bubble_full_width=False,
                        type="messages"
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Your message",
                            placeholder="Ask me anything about the video...",
                            scale=4,
                            lines=1
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            with gr.Column(scale=1):
                # Agent Status Panel
                with gr.Group():
                    gr.Markdown("### 🤖 Agent Status")
                    agent_status = gr.HTML()
                    refresh_status_btn = gr.Button("Refresh Status", variant="secondary")
                
                # Quick Actions
                with gr.Group():
                    gr.Markdown("### ⚡ Quick Actions")
                    
                    def quick_summary():
                        return "Please provide a comprehensive summary of this video.", []
                    
                    def quick_topics():
                        return "What are the main topics covered in this video?", []
                    
                    def quick_timestamps():
                        return "Can you provide key timestamps for the most important parts?", []
                    
                    summary_btn = gr.Button("📋 Get Summary", variant="secondary")
                    topics_btn = gr.Button("📝 Main Topics", variant="secondary")
                    timestamps_btn = gr.Button("⏰ Key Timestamps", variant="secondary")
        
        # Event handlers
        process_btn.click(
            process_youtube_url,
            inputs=[youtube_url],
            outputs=[url_status, video_info_display]
        ).then(
            get_agent_info,
            outputs=[agent_status]
        )
        
        # Chat functionality
        def handle_send(message, history):
            return chat_with_agent(message, history)
        
        send_btn.click(
            handle_send,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            handle_send,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, msg_input]
        )
        
        refresh_status_btn.click(
            get_agent_info,
            outputs=[agent_status]
        )
        
        # Quick action buttons
        summary_btn.click(
            quick_summary,
            outputs=[msg_input, chatbot]
        )
        
        topics_btn.click(
            quick_topics,
            outputs=[msg_input, chatbot]
        )
        
        timestamps_btn.click(
            quick_timestamps,
            outputs=[msg_input, chatbot]
        )
        
        # Load initial agent status
        app.load(
            get_agent_info,
            outputs=[agent_status]
        )
        
        # Instructions
        gr.Markdown("""
        ## 🚀 How to Use CiteTube
        
        1. **Process a Video**: Paste any YouTube URL and click "Process Video"
        2. **Start Chatting**: Ask questions about the video content in natural language
        3. **Agent Intelligence**: The AI agent automatically chooses the right tools for your questions
        4. **Get Insights**: Receive detailed answers with precise timestamp citations
        
        ### 🛠️ Agent Capabilities
        
        The AI agent has access to these specialized tools:
        - **🔍 Video Search**: Find specific content within transcripts
        - **📊 Metadata Tool**: Get video information and details
        - **🔎 Transcript Search**: Advanced search with detailed results
        - **📋 Summary Tool**: Generate comprehensive video summaries
        - **⏰ Timestamp Tool**: Look up content at specific times
        
        ### 💡 Example Questions
        
        - "What is this video about?"
        - "Find mentions of artificial intelligence"
        - "Summarize the key points from 5:00 to 10:00"
        - "What does the speaker say about machine learning?"
        - "Give me timestamps for the most important parts"
        
        The agent will automatically decide which tools to use based on your question!
        """)
    
    return app

def health_check():
    """Simple health check for the application."""
    try:
        # Check database connection
        if not db.test_connection():
            return {"status": "unhealthy", "reason": "database connection failed"}
        
        # Check if agent can be created
        try:
            agent = get_agent()
            tools = agent.get_available_tools()
            return {
                "status": "healthy", 
                "features": ["database", "langchain_agent", "youtube_processing"],
                "agent_tools": len(tools)
            }
        except Exception as e:
            return {"status": "degraded", "reason": f"agent initialization failed: {str(e)}"}
        
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}

def launch_app(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False
):
    """Launch the Gradio app with configurable parameters."""
    logger.info(f"Launching CiteTube app on {server_name}:{server_port}")
    
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

if __name__ == "__main__":
    launch_app()