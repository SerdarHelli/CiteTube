#!/usr/bin/env python3
"""
Test script to verify Ollama integration with CiteTube.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from citetube.core.models import get_embedding_model
from citetube.llm.llm import call_llm
import numpy as np

def test_embedding_model():
    """Test the Ollama embedding model."""
    print("Testing Ollama embedding model...")
    
    try:
        # Get embedding model
        model = get_embedding_model()
        model_name = getattr(model, 'model_name', getattr(model, '_model_name', 'Unknown'))
        print(f"✓ Successfully loaded embedding model: {model_name}")
        
        # Test encoding
        test_sentences = [
            "This is a test sentence.",
            "Another test sentence for embeddings."
        ]
        
        embeddings = model.encode(test_sentences, show_progress_bar=True)
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
        print(f"✓ Embedding dimension: {embeddings.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing embedding model: {e}")
        return False

def test_llm_model():
    """Test the Ollama LLM model."""
    print("\nTesting Ollama LLM model...")
    
    try:
        # Test LLM call
        test_prompt = """You are a helpful assistant. Please respond with a simple JSON object.

Question: What is the capital of France?

Return JSON with keys:
- answer (string)
- confidence (float 0-1)
"""
        
        response = call_llm(test_prompt, temperature=0.1, max_tokens=100)
        print(f"✓ Successfully got LLM response")
        print(f"✓ Response keys: {list(response.keys())}")
        print(f"✓ Answer: {response.get('answer', 'N/A')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing LLM model: {e}")
        return False

def main():
    """Run all tests."""
    print("CiteTube Ollama Integration Test")
    print("=" * 40)
    
    # Test embedding model
    embedding_success = test_embedding_model()
    
    # Test LLM model
    llm_success = test_llm_model()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Embedding Model: {'✓ PASS' if embedding_success else '✗ FAIL'}")
    print(f"LLM Model: {'✓ PASS' if llm_success else '✗ FAIL'}")
    
    if embedding_success and llm_success:
        print("\n🎉 All tests passed! Ollama integration is working.")
    else:
        print("\n❌ Some tests failed. Please check Ollama installation and models.")
        print("\nTo install required models:")
        print("  ollama pull nomic-embed-text")
        print("  ollama pull llama3.2")

if __name__ == "__main__":
    main()