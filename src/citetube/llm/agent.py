"""
CiteTube LangChain Agent for enhanced YouTube video analysis.
"""

import json
import hashlib
from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from .tools import create_tools
from ..core.config import get_llm_model, get_temperature, get_max_tokens
from ..core.logging_config import get_logger

logger = get_logger("citetube.agents.agent")


class CiteTubeAgent:
    """
    LangChain-based agent for CiteTube with YouTube-specific tools.
    """
    
    def __init__(self, current_video_id: Optional[int] = None, vllm_base_url: str = "http://localhost:8000/v1"):
        """
        Initialize the CiteTube agent.
        
        Args:
            current_video_id: ID of the currently loaded video
            vllm_base_url: Base URL for vLLM OpenAI-compatible API
        """
        self.current_video_id = current_video_id
        self.vllm_base_url = vllm_base_url
        
        # Simple response cache to avoid redundant calls
        self._response_cache = {}
        
        # Initialize LLM client pointing to vLLM with optimizations
        self.llm = ChatOpenAI(
            model=get_llm_model(),
            temperature=get_temperature(),
            max_tokens=get_max_tokens(),
            base_url=vllm_base_url,
            api_key="dummy-key",  # vLLM doesn't require real API key
            max_retries=3,  # Add retry logic
            timeout=30.0,   # Set reasonable timeout
            streaming=False  # Disable streaming for better reliability
        )
        
        # Create tools with current video context
        self.tools = create_tools(current_video_id)
        
        # Create agent prompt
        self.prompt = self._create_prompt()
        
        # Create agent
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create agent executor with optimizations
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True,  # For debugging and optimization
            early_stopping_method="generate"  # Stop early when possible
        )
        
        logger.info(f"CiteTube agent initialized with {len(self.tools)} tools")
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the agent prompt template."""
        system_message = """You are CiteTube Assistant, an AI agent specialized in analyzing YouTube video content through transcripts.

You have access to several tools for working with YouTube videos:
- video_search: Search for specific content within video transcripts
- video_metadata: Get information about videos (title, duration, description, etc.)
- transcript_search: Advanced transcript search with detailed results
- video_summary: Generate summaries of video content
- timestamp_lookup: Find content at specific timestamps

IMPORTANT GUIDELINES:
1. Always cite timestamps [mm:ss] when referencing specific content
2. Use tools to gather information before answering questions
3. Be precise and factual - only use information from the actual transcript
4. If information is not available in the transcript, clearly state this
5. When possible, provide multiple relevant timestamps for comprehensive answers
6. Format your responses clearly with proper citations

Current video context: {current_video_info}

Remember: You can only work with videos that have been ingested into the CiteTube system."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        return prompt
    
    def set_current_video(self, video_id: int):
        """Update the current video context for the agent."""
        self.current_video_id = video_id
        
        # Clear cache when video changes
        self._response_cache.clear()
        
        # Recreate tools with new video context
        self.tools = create_tools(video_id)
        
        # Recreate agent with updated tools
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Recreate agent executor with optimizations
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True,  # For debugging and optimization
            early_stopping_method="generate"  # Stop early when possible
        )
        
        logger.info(f"Agent updated with video ID: {video_id}")
    
    def get_current_video_info(self) -> str:
        """Get information about the current video for context."""
        if not self.current_video_id:
            return "No video currently loaded"
        
        # Use the metadata tool to get current video info
        metadata_tool = next((tool for tool in self.tools if tool.name == "video_metadata"), None)
        if metadata_tool:
            try:
                return metadata_tool._run()
            except Exception as e:
                logger.error(f"Error getting current video info: {e}")
                return f"Video ID: {self.current_video_id} (error getting details)"
        
        return f"Video ID: {self.current_video_id}"
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask the agent a question about the video content.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Create cache key based on question and current video
            cache_key = hashlib.md5(f"{question}_{self.current_video_id}".encode()).hexdigest()
            
            # Check cache first
            if cache_key in self._response_cache:
                logger.info("Returning cached response")
                return self._response_cache[cache_key]
            
            # Get current video info for context
            current_video_info = self.get_current_video_info()
            
            # Execute the agent
            result = self.agent_executor.invoke({
                "input": question,
                "current_video_info": current_video_info
            })
            
            # Extract the answer
            answer = result.get("output", "No answer generated")
            
            # Try to extract citations from the answer
            citations = self._extract_citations(answer)
            
            # Format response
            response = {
                "answer": answer,
                "citations": citations,
                "confidence": 0.8,  # Default confidence for agent responses
                "raw_response": answer,
                "agent_steps": result.get("intermediate_steps", []),
                "current_video_id": self.current_video_id
            }
            
            # Cache the response for future use
            self._response_cache[cache_key] = response
            
            logger.info("Agent response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error in agent execution: {e}")
            return {
                "answer": f"Error: Failed to process question. {str(e)}",
                "citations": [],
                "confidence": 0.0,
                "raw_response": str(e),
                "agent_steps": [],
                "current_video_id": self.current_video_id
            }
    
    def _extract_citations(self, text: str) -> List[Dict[str, str]]:
        """Extract timestamp citations from the response text."""
        import re
        
        # Pattern to match timestamps like [mm:ss] or [hh:mm:ss]
        timestamp_pattern = r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]'
        matches = re.findall(timestamp_pattern, text)
        
        citations = []
        for i, timestamp in enumerate(matches):
            citations.append({
                "seg_id": i,  # Placeholder segment ID
                "ts": timestamp
            })
        
        return citations
    
    def summarize_video(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of the current video."""
        if not self.current_video_id:
            return {
                "answer": "No video currently loaded for summarization.",
                "citations": [],
                "confidence": 0.0
            }
        
        summary_question = "Please provide a comprehensive summary of this video, including the main topics, key points, and important timestamps."
        return self.ask(summary_question)
    
    def search_video(self, query: str) -> Dict[str, Any]:
        """Search for specific content in the current video."""
        if not self.current_video_id:
            return {
                "answer": "No video currently loaded for searching.",
                "citations": [],
                "confidence": 0.0
            }
        
        search_question = f"Search for information about: {query}. Provide relevant content with timestamps."
        return self.ask(search_question)
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get information about available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools
        ]