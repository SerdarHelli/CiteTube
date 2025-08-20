#!/usr/bin/env python3
"""
Test script for pgvector functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_database_connection():
    """Test database connection."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from citetube.core.db import test_connection
        
        print("Testing database connection...")
        if test_connection():
            print("✅ Database connection successful!")
            return True
        else:
            print("❌ Database connection failed!")
            return False
    except Exception as e:
        print(f"❌ Error testing connection: {e}")
        return False

def test_embedding_model():
    """Test embedding model."""
    try:
        from citetube.core.models import get_embedding_model
        
        print("Testing embedding model...")
        model = get_embedding_model()
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        
        print(f"✅ Embedding model working! Embedding shape: {embedding.shape}")
        return True
    except Exception as e:
        print(f"❌ Error testing embedding model: {e}")
        return False

def test_vector_search():
    """Test vector search functionality."""
    try:
        from citetube.core.db import vector_similarity_search
        from citetube.core.models import get_embedding_model, normalize_embedding
        
        print("Testing vector search...")
        
        # Create a test embedding
        model = get_embedding_model()
        query_embedding = model.encode("test query")
        query_embedding = normalize_embedding(query_embedding)
        
        # Test search (should return empty list since no data exists yet)
        results = vector_similarity_search(query_embedding, video_id=1, top_k=5)
        
        print(f"✅ Vector search working! Results: {results}")
        return True
    except Exception as e:
        print(f"❌ Error testing vector search: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing pgvector setup...")
    print("=" * 50)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Embedding Model", test_embedding_model),
        ("Vector Search", test_vector_search),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        print("-" * 30)
    
    print(f"\n{'=' * 50}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! pgvector setup is working correctly.")
    else:
        print("❌ Some tests failed. Please check the setup.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)