"""
LLM client module for CiteTube.
Handles communication with vLLM server and prompt construction.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "data" / "logs" / "llm.log")
    ]
)
logger = logging.getLogger("llm")

# Constants
VLLM_API = os.getenv("VLLM_API", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instruct")
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1024

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

def call_vllm(
    prompt: str, 
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> Dict[str, Any]:
    """
    Call vLLM API with the given prompt.
    
    Args:
        prompt: Prompt to send to the LLM
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        LLM response as a dictionary
    """
    start_time = time.time()
    
    # Prepare system and user messages
    system_prompt = """You are a careful assistant that answers questions about YouTube videos based on their transcripts.
Always cite timestamps [mm:ss] for every claim you make. Return your answer in the requested JSON format."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Prepare request payload
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Call API
    try:
        response = requests.post(
            f"{VLLM_API}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
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
        logger.error(f"Error calling vLLM API: {str(e)}")
        return {
            "answer": f"Error: Failed to get response from LLM server. {str(e)}",
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
    response = call_vllm(prompt)
    
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