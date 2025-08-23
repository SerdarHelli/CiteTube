#!/usr/bin/env python3
"""
CiteTube - YouTube Transcript QA Application
Smart CLI interface that handles everything automatically.
"""

import sys
import os
import subprocess
import time
import signal
import socket
from pathlib import Path
from typing import Optional

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("Missing dependencies. Please install with: pip install typer rich")
    sys.exit(1)

from citetube.ui.app import launch_app
from citetube.core.config import ensure_directories
from citetube.core.db import test_connection, init_db
from citetube.core.logging_config import setup_logging, setup_vllm_logging, get_logger

app = typer.Typer(
    name="citetube",
    help="CiteTube - YouTube Transcript QA Application",
    add_completion=False
)
console = Console()

# Initialize logging system
def init_logging(debug: bool = False):
    """Initialize the logging system with appropriate settings."""
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(
        log_level=log_level,
        max_file_size="10 MB",
        retention="7 days",
        console_output=True,
        file_output=True
    )
    setup_vllm_logging()
    return get_logger("citetube.main")

# Project root and service files
PROJECT_ROOT = Path(__file__).parent
LOGS_DIR = PROJECT_ROOT / "logs"
VLLM_PID_FILE = LOGS_DIR / "vllm.pid"
VLLM_LOG_FILE = LOGS_DIR / "vllm.log"


def check_service(host: str, port: int, timeout: int = 30) -> bool:
    """Check if a service is running on the given host:port."""
    for _ in range(timeout):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                if result == 0:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def load_env() -> dict:
    """Load environment variables from .env file."""
    env_file = PROJECT_ROOT / ".env"
    env_vars = {}
    
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    
    return env_vars


def ensure_env_file():
    """Ensure .env file exists, create from example if needed."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        console.print("⚠️ .env file not found. Creating from .env.example...", style="yellow")
        example_file = PROJECT_ROOT / ".env.example"
        if example_file.exists():
            import shutil
            shutil.copy(example_file, env_file)
            console.print("✅ Created .env file. Please review and modify if needed.", style="green")
        else:
            console.print("❌ .env.example not found. Please create .env manually.", style="red")
            raise typer.Exit(1)


def start_vllm() -> Optional[int]:
    """Start vLLM server and return the process ID."""
    logger = get_logger("citetube.main.vllm")
    env_vars = load_env()
    
    model = env_vars.get('VLLM_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')
    host = env_vars.get('VLLM_HOST', 'localhost')
    port = env_vars.get('VLLM_PORT', '8000')
    
    logger.info(f"Starting vLLM server with model: {model} on {host}:{port}")
    console.print(f"🤖 Starting vLLM server with model: {model}")
    
    cmd = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', model,
        '--host', host,
        '--port', port,
        '--max-model-len', env_vars.get('VLLM_MAX_MODEL_LEN', '8192'),
        '--gpu-memory-utilization', env_vars.get('VLLM_GPU_MEMORY_UTILIZATION', '0.85'),
        '--tensor-parallel-size', env_vars.get('VLLM_TENSOR_PARALLEL_SIZE', '1'),
        '--enable-prefix-caching',
        '--enable-chunked-prefill',
        '--disable-sliding-window'
    ]
    
    try:
        # Ensure logs directory exists
        LOGS_DIR.mkdir(exist_ok=True)
        logger.debug(f"vLLM command: {' '.join(cmd)}")
        
        with open(VLLM_LOG_FILE, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
        
        # Save PID
        with open(VLLM_PID_FILE, 'w') as f:
            f.write(str(process.pid))
        
        logger.success(f"vLLM server started with PID: {process.pid}")
        return process.pid
    except Exception as e:
        logger.error(f"Failed to start vLLM: {e}")
        console.print(f"❌ Failed to start vLLM: {e}", style="red")
        return None


def stop_vllm():
    """Stop the vLLM server."""
    logger = get_logger("citetube.main.vllm")
    
    if VLLM_PID_FILE.exists():
        try:
            with open(VLLM_PID_FILE) as f:
                pid = int(f.read().strip())
            
            logger.info(f"Stopping vLLM server with PID: {pid}")
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            logger.success("vLLM server stopped successfully")
            console.print("✅ Stopped vLLM server", style="green")
            VLLM_PID_FILE.unlink()
        except (ValueError, ProcessLookupError, FileNotFoundError) as e:
            logger.warning(f"vLLM server was not running: {e}")
            console.print("⚠️ vLLM server was not running", style="yellow")
            if VLLM_PID_FILE.exists():
                VLLM_PID_FILE.unlink()
    else:
        logger.warning("vLLM PID file not found")
        console.print("⚠️ vLLM PID file not found", style="yellow")


def is_vllm_running() -> bool:
    """Check if vLLM is already running."""
    return check_service("localhost", 8000, timeout=1)


@app.command()
def run(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(7860, "--port", "-p", help="Port to bind to"),
    share: bool = typer.Option(False, "--share", "-s", help="Create public link"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    auto_start: bool = typer.Option(True, "--auto-start/--no-auto-start", help="Automatically start vLLM if needed"),
):
    """Launch the CiteTube web application (smart mode - handles everything automatically)."""
    # Initialize logging first
    logger = init_logging(debug=debug)
    logger.info("Starting CiteTube application")
    
    console.print(Panel.fit("🚀 Starting CiteTube (Smart Mode)", style="bold blue"))
    
    # Ensure .env file exists
    logger.debug("Ensuring .env file exists")
    ensure_env_file()
    
    # Ensure directories exist
    logger.debug("Ensuring required directories exist")
    ensure_directories()
    
    # Test database connection
    logger.info("Testing database connection")
    if not test_connection():
        logger.error("Database connection failed")
        console.print("❌ Database connection failed. Please check your configuration.", style="red")
        console.print("💡 Try running: python main.py init", style="yellow")
        raise typer.Exit(1)
    
    logger.success("Database connection successful")
    console.print("✅ Database connection successful", style="green")
    
    # Smart vLLM handling
    if auto_start and not is_vllm_running():
        logger.info("vLLM not running, starting automatically")
        console.print("🤖 vLLM not running, starting automatically...")
        vllm_pid = start_vllm()
        if vllm_pid:
            logger.info(f"vLLM started with PID: {vllm_pid}")
            console.print("⏳ Waiting for vLLM to start...")
            if check_service("localhost", 8000, timeout=60):
                logger.success("vLLM server is ready")
                console.print("✅ vLLM server is ready", style="green")
            else:
                logger.error("vLLM server failed to start within timeout")
                console.print("❌ vLLM server failed to start", style="red")
                console.print("💡 Try running with --no-auto-start to skip vLLM", style="yellow")
                raise typer.Exit(1)
        else:
            logger.error("Failed to start vLLM server")
            console.print("❌ Failed to start vLLM", style="red")
            raise typer.Exit(1)
    elif is_vllm_running():
        logger.info("vLLM server already running")
        console.print("✅ vLLM server already running", style="green")
    else:
        logger.warning("vLLM not running (auto-start disabled)")
        console.print("⚠️ vLLM not running (auto-start disabled)", style="yellow")
    
    # Launch the app
    try:
        logger.info(f"Launching CiteTube app on {host}:{port}")
        launch_app(server_name=host, server_port=port, share=share, debug=debug)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        console.print("\n⏹️ Shutting down...")
        if auto_start:
            logger.info("Stopping vLLM server")
            stop_vllm()
        logger.info("CiteTube shutdown complete")


@app.command()
def start(
    skip_vllm: bool = typer.Option(False, "--skip-vllm", help="Skip starting vLLM server"),
    port: int = typer.Option(7860, "--port", "-p", help="Port for CiteTube app"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host for CiteTube app"),
):
    """Start all CiteTube services (vLLM + app)."""
    console.print(Panel.fit("🚀 Starting All CiteTube Services", style="bold blue"))
    
    # Ensure .env file exists
    ensure_env_file()
    
    # Start vLLM if requested
    if not skip_vllm:
        if is_vllm_running():
            console.print("✅ vLLM server already running", style="green")
        else:
            vllm_pid = start_vllm()
            if vllm_pid:
                console.print("⏳ Waiting for vLLM to start...")
                if check_service("localhost", 8000, timeout=60):
                    console.print("✅ vLLM server is ready", style="green")
                else:
                    console.print("❌ vLLM server failed to start", style="red")
                    return
    
    # Start CiteTube app
    console.print("🌐 Starting CiteTube application...")
    
    try:
        subprocess.run([
            sys.executable, "main.py", "run",
            "--host", host,
            "--port", str(port),
            "--no-auto-start"  # Don't auto-start vLLM again
        ], check=True)
    except KeyboardInterrupt:
        console.print("\n⏹️ Shutting down...")
    except subprocess.CalledProcessError as e:
        console.print(f"❌ Failed to start CiteTube: {e}", style="red")
    finally:
        if not skip_vllm:
            stop_vllm()


@app.command()
def stop():
    """Stop all CiteTube services."""
    console.print("🛑 Stopping CiteTube services...")
    stop_vllm()
    console.print("✅ All services stopped", style="green")


@app.command()
def status():
    """Check the status of all services."""
    console.print("📊 Service Status", style="bold blue")
    
    table = Table()
    table.add_column("Service", style="cyan")
    table.add_column("Port", style="magenta")
    table.add_column("Status", style="green")
    
    # Check PostgreSQL
    if check_service("localhost", 5432, timeout=1):
        table.add_row("PostgreSQL", "5432", "✅ Running")
    else:
        table.add_row("PostgreSQL", "5432", "❌ Not running")
    
    # Check vLLM
    if check_service("localhost", 8000, timeout=1):
        table.add_row("vLLM", "8000", "✅ Running")
    else:
        table.add_row("vLLM", "8000", "❌ Not running")
    
    # Check CiteTube app
    if check_service("localhost", 7860, timeout=1):
        table.add_row("CiteTube App", "7860", "✅ Running")
    else:
        table.add_row("CiteTube App", "7860", "❌ Not running")
    
    console.print(table)


@app.command()
def logs():
    """View vLLM logs."""
    if VLLM_LOG_FILE.exists():
        console.print("📋 vLLM Logs (press Ctrl+C to exit):")
        try:
            subprocess.run(["tail", "-f", str(VLLM_LOG_FILE)])
        except KeyboardInterrupt:
            pass
    else:
        console.print("⚠️ vLLM log file not found", style="yellow")


@app.command()
def init():
    """Initialize the database with required tables and extensions."""
    # Initialize logging
    logger = init_logging(debug=False)
    logger.info("Starting database initialization")
    
    console.print("🗄️ Initializing database...", style="blue")
    
    try:
        init_db()
        logger.success("Database initialized successfully")
        console.print("✅ Database initialized successfully!", style="green")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        console.print(f"❌ Database initialization failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def health():
    """Check the health of all services."""
    console.print("🔍 Checking service health...", style="blue")
    
    # Check database
    if test_connection():
        console.print("✅ Database: Healthy", style="green")
    else:
        console.print("❌ Database: Unhealthy", style="red")
    
    # Check vLLM (optional)
    try:
        from citetube.llm.llm import test_llm_connection
        if test_llm_connection():
            console.print("✅ vLLM: Healthy", style="green")
        else:
            console.print("❌ vLLM: Unhealthy", style="red")
    except Exception as e:
        console.print(f"⚠️ vLLM: Could not test ({e})", style="yellow")


@app.command()
def version():
    """Show version information."""
    from citetube import __version__
    console.print(f"CiteTube version: {__version__}", style="blue")


if __name__ == "__main__":
    app()