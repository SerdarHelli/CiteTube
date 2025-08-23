"""
Pytest configuration and fixtures for CiteTube tests.
"""

import pytest
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session")
def test_video_url():
    """Sample YouTube video URL for testing."""
    return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - short video


@pytest.fixture(scope="session")
def test_question():
    """Sample question for testing."""
    return "What is this video about?"


@pytest.fixture
def mock_video_metadata():
    """Mock video metadata for testing."""
    return {
        "id": 1,
        "yt_id": "dQw4w9WgXcQ",
        "title": "Test Video",
        "duration_s": 212,
        "created_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def mock_segments():
    """Mock transcript segments for testing."""
    return [
        {
            "id": 1,
            "video_id": 1,
            "start_time": 0.0,
            "end_time": 10.0,
            "text": "This is a test segment about machine learning.",
            "embedding": [0.1] * 384  # Mock embedding vector
        },
        {
            "id": 2,
            "video_id": 1,
            "start_time": 10.0,
            "end_time": 20.0,
            "text": "Another segment discussing artificial intelligence.",
            "embedding": [0.2] * 384  # Mock embedding vector
        }
    ]