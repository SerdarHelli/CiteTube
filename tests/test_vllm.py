#!/usr/bin/env python3
"""
Test script for vLLM integration in CiteTube.
Tests both the vLLM server connection and the embedding models.
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

def test_vllm_connection():
    """Test vLLM server connection."""
    print("🔍 Testing vLLM server connection...")
    
    try:
        from citetube.llm.vllm_client import check_vllm_health, get_vllm_client
        
        # Check server health
        health = check_vllm_health()
        
        if health["status"] == "healthy":
            print("✅ vLLM server is healthy")
            print(f"   Base URL: {health['base_url']}")
            print(f"   Available models: {health['available_models']}")
            print(f"   Model count: {health['model_count']}")
            return True
        else:
            print("❌ vLLM server is not healthy")
            print(f"   Error: {health.get('error', 'Unknown error')}")
            print(f"   Base URL: {health['base_url']}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing vLLM connection: {e}")
        return False

def test_vllm_generation():
    """Test vLLM text generation."""
    print("\n🔍 Testing vLLM text generation...")
    
    try:
        from citetube.llm.vllm_client import call_vllm
        from citetube.core.config import get_llm_model
        
        model_name = get_llm_model()
        
        # Test simple generation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello and explain what you are in one sentence."}
        ]
        
        print(f"   Using model: {model_name}")
        print("   Generating response...")
        
        response = call_vllm(
            messages=messages,
            model=model_name,
            temperature=0.1,
            max_tokens=100
        )
        
        print("✅ vLLM generation successful")
        print(f"   Response: {response['content'][:100]}...")
        print(f"   Model used: {response.get('model', 'Unknown')}")
        print(f"   Finish reason: {response.get('finish_reason', 'Unknown')}")
        
        if response.get('usage'):
            usage = response['usage']
            print(f"   Tokens - Prompt: {usage.get('prompt_tokens', 0)}, "
                  f"Completion: {usage.get('completion_tokens', 0)}, "
                  f"Total: {usage.get('total_tokens', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing vLLM generation: {e}")
        return False

def test_embedding_model():
    """Test sentence-transformers embedding model."""
    print("\n🔍 Testing embedding model...")
    
    try:
        from citetube.core.models import get_embedding_model
        from citetube.core.config import get_embedding_model_name
        
        model_name = get_embedding_model_name()
        print(f"   Loading model: {model_name}")
        
        model = get_embedding_model()
        
        # Test embedding generation
        test_texts = [
            "This is a test sentence for embedding.",
            "Another sentence to test the embedding model."
        ]
        
        print("   Generating embeddings...")
        embeddings = model.encode(test_texts)
        
        print("✅ Embedding model working")
        print(f"   Model: {model_name}")
        print(f"   Embedding shape: {embeddings.shape}")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing embedding model: {e}")
        return False

def test_llm_integration():
    """Test the full LLM integration."""
    print("\n🔍 Testing LLM integration...")
    
    try:
        from citetube.llm.llm import call_llm, build_prompt
        
        # Create test data
        test_segments = [
            {
                "id": 1,
                "timestamp": "01:30",
                "text": "This is the first segment of the video transcript."
            },
            {
                "id": 2,
                "timestamp": "02:15",
                "text": "This is the second segment discussing the main topic."
            }
        ]
        
        question = "What is discussed in the video?"
        
        # Build prompt
        prompt = build_prompt(question, test_segments)
        print("   Built prompt successfully")
        
        # Call LLM
        print("   Calling LLM...")
        response = call_llm(prompt, temperature=0.1, max_tokens=200)
        
        print("✅ LLM integration working")
        print(f"   Answer: {response.get('answer', 'No answer')[:100]}...")
        print(f"   Confidence: {response.get('confidence', 'Unknown')}")
        print(f"   Citations count: {len(response.get('citations', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing LLM integration: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 CiteTube vLLM Integration Tests")
    print("=" * 50)
    
    tests = [
        ("vLLM Connection", test_vllm_connection),
        ("vLLM Generation", test_vllm_generation),
        ("Embedding Model", test_embedding_model),
        ("LLM Integration", test_llm_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 All tests passed! vLLM integration is working correctly.")
        print("\nYou can now run CiteTube:")
        print("  python main.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure vLLM server is running:")
        print("   python setup_vllm.py")
        print("2. Check your .env configuration")
        print("3. Verify the model is available on the vLLM server")

if __name__ == "__main__":
    main()