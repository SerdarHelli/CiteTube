"""
Tests for core CiteTube functionality.
"""

import pytest
from unittest.mock import patch, MagicMock

from citetube.core import config, db


class TestConfig:
    """Test configuration management."""
    
    def test_get_db_config(self):
        """Test database configuration retrieval."""
        assert config.get_db_host() is not None
        assert config.get_db_port() > 0
        assert config.get_db_name() is not None
        assert config.get_db_user() is not None
    
    def test_get_model_config(self):
        """Test model configuration retrieval."""
        assert config.get_embedding_model_name() is not None
        assert config.get_llm_model() is not None
        assert config.get_llm_provider() in ["vllm", "openai", "anthropic", "ollama"]
    
    def test_ensure_directories(self):
        """Test directory creation."""
        # This should not raise an exception
        config.ensure_directories()


class TestDatabase:
    """Test database operations."""
    
    @patch('citetube.core.db.psycopg2.connect')
    def test_get_db_connection(self, mock_connect):
        """Test database connection."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        conn = db.get_db_connection()
        
        assert conn == mock_conn
        mock_connect.assert_called_once()
    
    @patch('citetube.core.db.get_db_connection')
    def test_test_connection(self, mock_get_conn):
        """Test connection testing."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        result = db.test_connection()
        
        assert result is True
        mock_cursor.execute.assert_called_once_with("SELECT 1")
    
    @patch('citetube.core.db.get_db_connection')
    def test_test_connection_failure(self, mock_get_conn):
        """Test connection testing with failure."""
        mock_get_conn.side_effect = Exception("Connection failed")
        
        result = db.test_connection()
        
        assert result is False