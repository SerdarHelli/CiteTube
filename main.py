#!/usr/bin/env python3
"""
CiteTube - YouTube Transcript QA Application
Smart CLI interface that handles everything automatically.
"""

import subprocess
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    print("Missing dependencies. Please install with: pip install typer rich")
    sys.exit(1)

from citetube.ui.app import launch_app
from citetube.core.config import ensure_directories
from citetube.core.db import test_connection, init_db
from citetube.core.logging_config import setup_logging, setup_vllm_logging, get_logger
from citetube.core.services import ServiceManager
from citetube.core.utils import ensure_env_file, get_project_info

app = typer.Typer(
    name="citetube",
    help="CiteTube - YouTube Transcript QA Application",
    add_completion=False
)
console = Console()

# Project root and service manager
PROJECT_ROOT = Path(__file__).parent
service_manager = ServiceManager(PROJECT_ROOT)


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
    ensure_env_file(PROJECT_ROOT)
    
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
    if auto_start and not service_manager.is_vllm_running():
        logger.info("vLLM not running, starting automatically")
        console.print("🤖 vLLM not running, starting automatically...")
        env_vars = service_manager.load_env()
        model = env_vars.get('VLLM_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')
        console.print(f"🤖 Starting vLLM server with model: {model}")
        
        vllm_pid = service_manager.start_vllm()
        if vllm_pid:
            logger.info(f"vLLM started with PID: {vllm_pid}")
            console.print("⏳ Waiting for vLLM to start...")
            if service_manager.check_service("localhost", 8000, timeout=60):
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
    elif service_manager.is_vllm_running():
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
            service_manager.stop_vllm()
            console.print("✅ Stopped vLLM server", style="green")
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
    ensure_env_file(PROJECT_ROOT)
    
    # Start vLLM if requested
    if not skip_vllm:
        if service_manager.is_vllm_running():
            console.print("✅ vLLM server already running", style="green")
        else:
            env_vars = service_manager.load_env()
            model = env_vars.get('VLLM_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')
            console.print(f"🤖 Starting vLLM server with model: {model}")
            
            vllm_pid = service_manager.start_vllm()
            if vllm_pid:
                console.print("⏳ Waiting for vLLM to start...")
                if service_manager.check_service("localhost", 8000, timeout=60):
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
            service_manager.stop_vllm()


@app.command()
def stop():
    """Stop all CiteTube services."""
    console.print("🛑 Stopping CiteTube services...")
    service_manager.stop_vllm()
    console.print("✅ All services stopped", style="green")


@app.command()
def status():
    """Check the status of all services."""
    console.print("📊 Service Status", style="bold blue")
    
    table = Table()
    table.add_column("Service", style="cyan")
    table.add_column("Port", style="magenta")
    table.add_column("Status", style="green")
    
    status_data = service_manager.get_service_status()
    
    # Add service rows
    services = [
        ("PostgreSQL", "5432", status_data["postgresql"]),
        ("vLLM", "8000", status_data["vllm"]),
        ("CiteTube App", "7860", status_data["citetube_app"]),
    ]
    
    for name, port, running in services:
        status_text = "✅ Running" if running else "❌ Not running"
        table.add_row(name, port, status_text)
    
    console.print(table)


@app.command()
def logs():
    """View vLLM logs."""
    if service_manager.vllm_log_file.exists():
        console.print("📋 vLLM Logs (press Ctrl+C to exit):")
        try:
            subprocess.run(["tail", "-f", str(service_manager.vllm_log_file)])
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
    console.print("🏥 [bold blue]Checking CiteTube Health[/bold blue]")
    
    # Check database
    console.print("📊 Checking database connection...")
    if test_connection():
        console.print("✅ Database: [green]Connected[/green]")
    else:
        console.print("❌ Database: [red]Connection failed[/red]")
    
    # Check vLLM
    console.print("🤖 Checking vLLM server...")
    try:
        from src.citetube.llm.llm import test_llm_connection
        if test_llm_connection():
            console.print("✅ vLLM: [green]Running[/green]")
        else:
            console.print("❌ vLLM: [red]Not accessible[/red]")
    except Exception as e:
        console.print(f"❌ vLLM: [red]Error - {e}[/red]")
    
    # Check LangChain Agent
    console.print("🔧 Checking LangChain Agent...")
    try:
        from src.citetube.llm.llm import test_agent_connection, get_agent_tools
        agent_ok = test_agent_connection()
        if agent_ok:
            console.print("✅ Agent: [green]Working[/green]")
            tools = get_agent_tools()
            console.print(f"🛠️  Agent tools: [cyan]{len(tools)} available[/cyan]")
            for tool in tools:
                console.print(f"   - {tool['name']}")
        else:
            console.print("❌ Agent: [yellow]Not working (vLLM required)[/yellow]")
    except Exception as e:
        console.print(f"❌ Agent: [red]Error - {e}[/red]")
    
    # Check models
    console.print("📦 Checking models...")
    try:
        from src.citetube.core.models import get_embedding_model
        model = get_embedding_model()
        model_name = getattr(model, 'model_name', getattr(model, '_model_name', 'sentence-transformers model'))
        console.print(f"✅ Embedding model: [green]{model_name}[/green]")
    except Exception as e:
        console.print(f"❌ Embedding model: [red]Error - {e}[/red]")
    
    console.print("🎉 [bold green]Health check complete![/bold green]")


@app.command()
def version():
    """Show version information."""
    info = get_project_info()
    console.print(f"CiteTube version: {info['version']}", style="blue")
    console.print(f"Author: {info['author']}", style="cyan")
    console.print(f"Description: {info['description']}", style="green")


if __name__ == "__main__":
    app()