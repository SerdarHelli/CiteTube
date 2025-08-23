"""
LLM client module for CiteTube.
Handles LLM inference using vLLM and prompt construction.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from dotenv import load_dotenv

from ..core.config import get_llm_model, get_llm_provider, get_temperature, get_max_tokens
from .vllm_client import call_vllm, ensure_model_available, check_vllm_health

# Load environment variables
load_dotenv()

# Import centralized logging config
from ..core.logging_config import get_logger

logger = get_logger("citetube.llm")

def build_prompt(question: str, segments: List[Dict[str, Any]]) -> str:
    """
    Build prompt for the LLM.
    
    Args:
        question: User's question
        segments: List of transcript segments
        
    Returns:
        Formatted prompt string
    """
    context_parts = []
    
    for segment in segments:
        timestamp = segment.get("timestamp", "00:00")
        text = segment.get("text", "")
        context_parts.append(f"[{timestamp}] {text}")
    
    context = "\n".join(context_parts)
    
    prompt = f"""You are a careful assistant. 
Answer ONLY using the provided transcript chunks. 
Every claim must cite one or more timestamps in the form [mm:ss]. 
If info is missing, say so.

Question: {question}

Context:
{context}

Return JSON with keys:
- answer (string)
- bullets (list of strings)
- citations (list of {{seg_id:int, ts:string}})
- confidence (float 0-1)
"""
    
    return prompt

def call_llm(
    prompt: str, 
    temperature: float = None,
    max_tokens: int = None
) -> Dict[str, Any]:
    """
    Call vLLM with the given prompt.
    
    Args:
        prompt: Prompt to send to the LLM
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        LLM response as a dictionary
    """
    # Use config defaults if not provided
    if temperature is None:
        temperature = get_temperature()
    if max_tokens is None:
        max_tokens = get_max_tokens()
    
    start_time = time.time()
    
    try:
        # Get model name and ensure it's available
        model_name = get_llm_model()
        ensure_model_available(model_name)
        
        # Prepare system and user messages
        system_prompt = """You are a careful assistant that answers questions about YouTube videos based on their transcripts.
Always cite timestamps [mm:ss] for every claim you make. Return your answer in the requested JSON format."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Generate response using vLLM
        logger.info("Generating response with vLLM...")
        response = call_vllm(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract generated text
        content = response['content'].strip()
        
        # Try to parse as JSON
        try:
            parsed_content = json.loads(content)
            parsed_content["raw_response"] = content
        except json.JSONDecodeError:
            # Fallback to raw string if not valid JSON
            parsed_content = {
                "answer": content,
                "bullets": [],
                "citations": [],
                "confidence": 0.5,
                "raw_response": content
            }
        
        logger.info(f"vLLM call completed in {time.time() - start_time:.2f}s")
        
        return parsed_content
        
    except Exception as e:
        logger.error(f"Error calling vLLM: {str(e)}")
        return {
            "answer": f"Error: Failed to get response from vLLM. {str(e)}",
            "bullets": [],
            "citations": [],
            "confidence": 0.0,
            "raw_response": str(e)
        }

def answer_question(question: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Answer a question using the LLM.
    
    Args:
        question: User's question
        segments: List of transcript segments
        
    Returns:
        LLM response as a dictionary
    """
    # Build prompt
    prompt = build_prompt(question, segments)
    
    # Call LLM
    response = call_llm(prompt)
    
    # Add segment information to response
    segment_map = {segment["id"]: segment for segment in segments}
    
    # Process citations if they exist
    if "citations" in response and isinstance(response["citations"], list):
        for citation in response["citations"]:
            if isinstance(citation, dict) and "seg_id" in citation:
                seg_id = citation.get("seg_id")
                if seg_id in segment_map:
                    citation["segment"] = segment_map[seg_id]
    
    return response

def test_llm_connection() -> bool:
    """
    Test if the LLM connection is working.
    
    Returns:
        True if connection is working, False otherwise
    """
    try:
        # Check vLLM health
        if not check_vllm_health():
            return False
        
        # Try a simple test call
        test_response = call_vllm(
            messages=[{"role": "user", "content": "Hello, respond with 'OK'"}],
            model=get_llm_model(),
            temperature=0.1,
            max_tokens=10
        )
        
        return test_response is not None and 'content' in test_response
        
    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return False