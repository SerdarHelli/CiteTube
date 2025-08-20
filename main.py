#!/usr/bin/env python3
"""
Main entry point for CiteTube application.
This script provides backward compatibility with the old structure.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and launch the app
from citetube.ui.app import launch_app

if __name__ == "__main__":
    launch_app()