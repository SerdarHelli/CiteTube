"""
Centralized logging configuration for CiteTube using loguru.
"""

import os
import sys
import time
import functools
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from loguru import logger

def get_logs_directory() -> Path:
    """Get the logs directory path."""
    return Path(__file__).parent.parent.parent.parent / "logs"

def setup_logging(
    log_level: str = "INFO", 
    max_file_size: str = "10 MB", 
    retention: str = "7 days",
    console_output: bool = True,
    file_output: bool = True
):
    """
    Setup centralized logging configuration for CiteTube using loguru.
    
    Args:
        log_level: Logging level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
        max_file_size: Maximum size of log file before rotation (e.g., "10 MB", "100 KB")
        retention: How long to keep log files (e.g., "7 days", "1 week", "1 month")
        console_output: Whether to enable console logging
        file_output: Whether to enable file logging
    """
    # Create logs directory if it doesn't exist
    log_dir = get_logs_directory()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler with colored output
    if console_output:
        logger.add(
            sys.stderr,
            level=log_level.upper(),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True,
            enqueue=True  # Thread-safe logging
        )
    
    # Main application log file with rotation
    if file_output:
        main_log_file = log_dir / "citetube.log"
        logger.add(
            main_log_file,
            level=log_level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[name]}:{function}:{line} - {message}",
            rotation=max_file_size,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
            encoding="utf-8",
            enqueue=True  # Thread-safe logging
        )
    
    # Suppress noisy external libraries
    external_loggers = [
        "httpx", "httpcore", "urllib3", "requests", "transformers", 
        "sentence_transformers", "torch", "gradio", "uvicorn", "fastapi",
        "asyncio", "multipart", "PIL", "openai", "anthropic"
    ]
    
    for logger_name in external_loggers:
        logger.disable(logger_name)

def get_logger(name: str = None):
    """
    Get a logger instance. With loguru, we use the global logger with contextualized names.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Loguru logger instance with bound context
    """
    if name:
        return logger.bind(name=name)
    return logger.bind(name="citetube")

def setup_module_logger(module_name: str, log_file: Optional[str] = None):
    """
    Setup a dedicated logger for a specific module with optional separate log file.
    
    Args:
        module_name: Name of the module
        log_file: Optional separate log file name (without extension)
        
    Returns:
        Loguru logger instance
    """
    module_logger = logger.bind(name=module_name)
    
    # If a separate log file is requested, add a file handler
    if log_file:
        log_dir = get_logs_directory()
        log_path = log_dir / f"{log_file}.log"
        
        # Add dedicated file handler for this module
        logger.add(
            log_path,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[name]}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            filter=lambda record: record["extra"].get("name") == module_name,
            backtrace=True,
            diagnose=True,
            encoding="utf-8"
        )
    
    return module_logger

def setup_vllm_logging():
    """Setup dedicated logging for vLLM with its own log file."""
    log_dir = get_logs_directory()
    vllm_log_file = log_dir / "vllm.log"
    
    # Add vLLM-specific log handler
    logger.add(
        vllm_log_file,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | vLLM - {message}",
        rotation="50 MB",
        retention="3 days",
        compression="zip",
        filter=lambda record: "vllm" in record["extra"].get("name", "").lower() or "vllm" in record.get("message", "").lower(),
        backtrace=True,
        diagnose=True,
        encoding="utf-8",
        enqueue=True  # Thread-safe logging
    )

def log_function_call(func_name: str, args: Dict[str, Any] = None, module_name: str = None):
    """
    Log function entry with parameters (useful for debugging).
    
    Args:
        func_name: Name of the function being called
        args: Function arguments to log
        module_name: Module name for context
    """
    module_logger = get_logger(module_name) if module_name else logger
    args_str = f" with args: {args}" if args else ""
    module_logger.debug(f"Entering function: {func_name}{args_str}")

def log_performance(func_name: str, duration: float, module_name: str = None):
    """
    Log performance metrics for functions.
    
    Args:
        func_name: Name of the function
        duration: Execution time in seconds
        module_name: Module name for context
    """
    module_logger = get_logger(module_name) if module_name else logger
    module_logger.info(f"Function {func_name} completed in {duration:.3f}s")

def log_error_with_context(error: Exception, context: str = None, module_name: str = None):
    """
    Log errors with additional context information.
    
    Args:
        error: The exception that occurred
        context: Additional context about when/where the error occurred
        module_name: Module name for context
    """
    module_logger = get_logger(module_name) if module_name else logger
    context_str = f" Context: {context}" if context else ""
    module_logger.error(f"Error: {str(error)}.{context_str}", exc_info=True)

def log_execution_time(module_name: str = None):
    """
    Decorator to log function execution time.
    
    Args:
        module_name: Module name for context
        
    Usage:
        @log_execution_time("my_module")
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                log_performance(func.__name__, duration, module_name)
                return result
            except Exception as e:
                duration = time.time() - start_time
                log_error_with_context(
                    e, 
                    f"Function {func.__name__} failed after {duration:.3f}s", 
                    module_name
                )
                raise
        return wrapper
    return decorator

# Note: Logging is initialized explicitly in main.py or other entry points
# This allows for proper configuration based on command-line arguments