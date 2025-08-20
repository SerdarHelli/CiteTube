#!/usr/bin/env python3
"""
Consolidated setup script for CiteTube.
This script sets up both Ollama models and PostgreSQL database.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_ollama():
    """Check if Ollama is installed and running."""
    print("🔍 Checking Ollama...")
    try:
        result = subprocess.run("ollama --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama is not installed or not in PATH")
            return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def setup_ollama_models():
    """Setup required Ollama models."""
    if not check_ollama():
        print("\n📋 To install Ollama:")
        print("1. Visit: https://ollama.ai/")
        print("2. Download and install Ollama")
        print("3. Run this script again")
        return False
    
    models = [
        ("nomic-embed-text", "Embedding model"),
        ("llama3.2", "Language model")
    ]
    
    print(f"\n📦 Installing {len(models)} Ollama models...")
    success_count = 0
    
    for model_name, description in models:
        print(f"\n  Installing {model_name} ({description})...")
        try:
            result = subprocess.run(f"ollama pull {model_name}", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ {model_name} installed successfully")
                success_count += 1
            else:
                print(f"  ❌ Failed to install {model_name}: {result.stderr.strip()}")
        except Exception as e:
            print(f"  ❌ Error installing {model_name}: {e}")
        time.sleep(1)
    
    return success_count == len(models)

def setup_postgres():
    """Setup PostgreSQL database."""
    print("\n🐘 Setting up PostgreSQL...")
    
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        from citetube.core.config import get_db_host, get_db_port, get_db_user, get_db_password, get_db_name
        
        print(f"Database: {get_db_name()} at {get_db_host()}:{get_db_port()}")
        
        # Test connection
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        # Connect to default postgres database
        conn = psycopg2.connect(
            host=get_db_host(),
            port=get_db_port(),
            database="postgres",
            user=get_db_user(),
            password=get_db_password()
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (get_db_name(),))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f'CREATE DATABASE "{get_db_name()}"')
            print(f"✅ Created database '{get_db_name()}'")
        else:
            print(f"✅ Database '{get_db_name()}' already exists")
        
        cursor.close()
        conn.close()
        
        # Setup pgvector and tables
        from citetube.core.db import init_db, test_connection
        
        if test_connection():
            init_db()
            print("✅ Database tables initialized")
            return True
        else:
            print("❌ Database connection test failed")
            return False
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Run: pip install psycopg2-binary pgvector python-dotenv")
        return False
    except Exception as e:
        print(f"❌ PostgreSQL setup failed: {e}")
        print("\n📋 To setup PostgreSQL:")
        print("1. Install PostgreSQL with pgvector extension")
        print("2. Update .env file with correct database credentials")
        print("3. Ensure PostgreSQL service is running")
        return False

def main():
    """Main setup function."""
    print("🚀 CiteTube Setup")
    print("=" * 50)
    
    # Setup Ollama models
    ollama_success = setup_ollama_models()
    
    # Setup PostgreSQL
    postgres_success = setup_postgres()
    
    # Results
    print("\n" + "=" * 50)
    print("📊 Setup Results:")
    print(f"Ollama Models: {'✅ SUCCESS' if ollama_success else '❌ FAILED'}")
    print(f"PostgreSQL:    {'✅ SUCCESS' if postgres_success else '❌ FAILED'}")
    
    if ollama_success and postgres_success:
        print("\n🎉 Setup completed successfully!")
        print("\nYou can now run CiteTube:")
        print("  python main.py")
    else:
        print("\n⚠️  Some components failed to setup.")
        print("Please check the errors above and fix them before running CiteTube.")
    
    return ollama_success and postgres_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)