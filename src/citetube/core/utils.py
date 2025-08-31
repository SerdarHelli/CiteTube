"""
Utility functions for CiteTube.
"""

import shutil
from pathlib import Path
from typing import Optional

from rich.console import Console
import typer

console = Console()


def ensure_env_file(project_root: Path) -> None:
    """Ensure .env file exists, create from example if needed."""
    env_file = project_root / ".env"
    if not env_file.exists():
        console.print("⚠️ .env file not found. Creating from .env.example...", style="yellow")
        example_file = project_root / ".env.example"
        if example_file.exists():
            shutil.copy(example_file, env_file)
            console.print("✅ Created .env file. Please review and modify if needed.", style="green")
        else:
            console.print("❌ .env.example not found. Please create .env manually.", style="red")
            raise typer.Exit(1)


def get_project_info() -> dict:
    """Get project information."""
    try:
        from citetube import __version__, __author__, __description__
        return {
            "version": __version__,
            "author": __author__,
            "description": __description__,
        }
    except ImportError:
        return {
            "version": "0.1.0-dev",
            "author": "CiteTube Team",
            "description": "Local YouTube Transcript QA Application",
        }


def format_service_status(status: dict) -> str:
    """Format service status for display."""
    status_symbols = {
        True: "✅ Running",
        False: "❌ Not running"
    }
    
    formatted = []
    service_names = {
        "postgresql": "PostgreSQL",
        "vllm": "vLLM",
        "citetube_app": "CiteTube App"
    }
    
    for service, running in status.items():
        name = service_names.get(service, service)
        symbol = status_symbols[running]
        formatted.append(f"{name}: {symbol}")
    
    return "\n".join(formatted)