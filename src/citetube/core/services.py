"""
Service management utilities for CiteTube.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .logging_config import get_logger


class ServiceManager:
    """Manages external services like vLLM."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logs_dir = project_root / "logs"
        self.vllm_pid_file = self.logs_dir / "vllm.pid"
        self.vllm_log_file = self.logs_dir / "vllm.log"
        self.logger = get_logger("citetube.services")
    
    def check_service(self, host: str, port: int, timeout: int = 30) -> bool:
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
    
    def load_env(self) -> dict:
        """Load environment variables from .env file."""
        env_file = self.project_root / ".env"
        env_vars = {}
        
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        
        return env_vars
    
    def start_vllm(self) -> Optional[int]:
        """Start vLLM server and return the process ID."""
        env_vars = self.load_env()
        
        model = env_vars.get('VLLM_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')
        host = env_vars.get('VLLM_HOST', 'localhost')
        port = env_vars.get('VLLM_PORT', '8000')
        
        self.logger.info(f"Starting vLLM server with model: {model} on {host}:{port}")
        
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
            '--disable-sliding-window',
            '--enable-auto-tool-choice',
            '--tool-call-parser', 'hermes'
        ]
        
        try:
            # Ensure logs directory exists
            self.logs_dir.mkdir(exist_ok=True)
            self.logger.debug(f"vLLM command: {' '.join(cmd)}")
            
            with open(self.vllm_log_file, 'w') as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid
                )
            
            # Save PID
            with open(self.vllm_pid_file, 'w') as f:
                f.write(str(process.pid))
            
            self.logger.success(f"vLLM server started with PID: {process.pid}")
            return process.pid
        except Exception as e:
            self.logger.error(f"Failed to start vLLM: {e}")
            return None
    
    def stop_vllm(self):
        """Stop the vLLM server."""
        if self.vllm_pid_file.exists():
            try:
                with open(self.vllm_pid_file) as f:
                    pid = int(f.read().strip())
                
                self.logger.info(f"Stopping vLLM server with PID: {pid}")
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                self.logger.success("vLLM server stopped successfully")
                self.vllm_pid_file.unlink()
            except (ValueError, ProcessLookupError, FileNotFoundError) as e:
                self.logger.warning(f"vLLM server was not running: {e}")
                if self.vllm_pid_file.exists():
                    self.vllm_pid_file.unlink()
        else:
            self.logger.warning("vLLM PID file not found")
    
    def is_vllm_running(self) -> bool:
        """Check if vLLM is already running."""
        return self.check_service("localhost", 8000, timeout=1)
    
    def get_service_status(self) -> dict:
        """Get status of all services."""
        return {
            "postgresql": self.check_service("localhost", 5432, timeout=1),
            "vllm": self.check_service("localhost", 8000, timeout=1),
            "citetube_app": self.check_service("localhost", 7860, timeout=1),
        }