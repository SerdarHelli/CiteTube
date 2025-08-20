"""
Centralized logging configuration for CiteTube.
"""

import logging
import os
from pathlib import Path

def setup_logging():
    """Setup centralized logging configuration for CiteTube."""
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent.parent.parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "citetube.log")
        ]
    )
    
    # Set specific log levels for external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

# Initialize logging when module is imported
setup_logging()