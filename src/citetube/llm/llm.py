"""
LLM client module for CiteTube.
Handles local LLM inference using Ollama and prompt construction.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from dotenv import load_dotenv
import ollama

from ..core.config import get_llm_model, get_llm_provider, get_temperature, get_max_tokens

# Load environment variables
load_dotenv()

# Import centralized logging config
from ..core.logging_config import setup_logging

logger = logging.getLogger("citetube.llm")

# Global Ollama client
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

def ensure_model_available(model_name: str):
    """
    Ensure the Ollama model is available, pull if necessary.
    
    Args:
        model_name: Name of the Ollama model
    """
    client = get_ollama_client()
    
    try:
        # Try to get model info to check if it exists
        client.show(model_name)
        logger.info(f"Ollama model {model_name} is available")
    except Exception as e:
        logger.warning(f"Model {model_name} not found, attempting to pull...")
        try:
            client.pull(model_name)
            logger.info(f"Successfully pulled model {model_name}")
        except Exception as pull_error:
            logger.error(f"Failed to pull model {model_name}: {pull_error}")
            raise

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
    Call Ollama LLM with the given prompt.
    
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
        
        # Get Ollama client
        client = get_ollama_client()
        
        # Prepare system and user messages
        system_prompt = """You are a careful assistant that answers questions about YouTube videos based on their transcripts.
Always cite timestamps [mm:ss] for every claim you make. Return your answer in the requested JSON format."""
        
        # Generate response
        logger.info("Generating response with Ollama...")
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        )
        
        # Extract generated text
        content = response['message']['content'].strip()
        
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
        
        logger.info(f"Ollama LLM call completed in {time.time() - start_time:.2f}s")
        
        return parsed_content
        
    except Exception as e:
        logger.error(f"Error calling Ollama LLM: {str(e)}")
        return {
            "answer": f"Error: Failed to get response from Ollama LLM. {str(e)}",
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