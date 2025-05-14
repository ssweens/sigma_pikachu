import subprocess
import os
import sys
import shlex # For robust command parsing
import threading # Need RLock reference
import time
from ..constants import LLAMA_SERVER_LOG_FILE, CONFIG_FILE, OLLAMA_CMD # Reusing log file for now
from ..settings.config_manager import config_manager # Singleton instance

OLLAMA_ENV = {
    "OLLAMA_DEBUG": "1",
    "OLLAMA_LOAD_TIMEOUT": "15m0s",
    "OLLAMA_CONTEXT_LENGTH": "32768",
    "OLLAMA_FLASH_ATTENTION": "true",
    "OLLAMA_MAX_LOADED_MODELS": "1",
    "OLLAMA_HOST": "http://0.0.0.0:9999",
    "OLLAMA_MODELS": "/Users/ssweens/models",
}

class OllamaManager:
    def __init__(self, process_manager_instance):
        # Reference to the main ProcessManager instance to access shared state and lock
        self._pm = process_manager_instance

    def is_running(self):
        with self._pm._lock: # Use the lock from the main ProcessManager
            # Assuming process_manager will hold a reference like ollama_process
            return self._pm.ollama_process is not None and self._pm.ollama_process.poll() is None

    def start(self):
        with self._pm._lock:
            if self.is_running():
                print("Ollama server is already running.")
                return True

            if not os.path.exists(CONFIG_FILE):
                print(f"Error: {CONFIG_FILE} not found. Cannot start Ollama server.")
                return False

            current_config = config_manager.get_config()
            if not current_config:
                print(f"Error: Could not load {CONFIG_FILE}. Cannot start Ollama server.")
                return False

            ollama_config_path = current_config.get("ollama", {}).get("config_file", "config.yaml")
            listen_address = current_config.get("ollama", {}).get("listen", ":9999")

            command = [
                "ollama",  # Now just the executable, not the env string
                "serve",
                #"--config", ollama_config_path,
                #"--listen", listen_address,
            ]

            # Merge OLLAMA_ENV with the current environment
            env = os.environ.copy()
            env.update(OLLAMA_ENV)

            print(f"Starting Ollama server with command: {' '.join(shlex.quote(str(c)) for c in command)}")
            try:
                os.makedirs(os.path.dirname(LLAMA_SERVER_LOG_FILE), exist_ok=True)
                with open(LLAMA_SERVER_LOG_FILE, 'a') as log:
                    self._pm.ollama_process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env  # Pass the environment
                    )
                print(f"Ollama server started. PID: {self._pm.ollama_process.pid}. Logging to {LLAMA_SERVER_LOG_FILE}")
                return True
            except FileNotFoundError:
                print(f"Error: 'ollama' command not found. Is ollama installed and in your PATH?")
                self._pm.ollama_process = None
                return False
            except Exception as e:
                print(f"Failed to start Ollama server: {e}")
                self._pm.ollama_process = None
                return False

    def stop(self):
        process_to_stop = None
        pid_to_log = None

        with self._pm._lock: # Lock to safely access/modify shared state
            # Assuming process_manager will hold a reference like ollama_process
            if self.is_running():
                process_to_stop = self._pm.ollama_process
                if process_to_stop: # Should always be true if is_running was true
                    pid_to_log = process_to_stop.pid
                self._pm.ollama_process = None # Mark as stopping/stopped
            else:
                # print("Ollama server is not running (or already marked as stopped by internal check).")
                return True # Already stopped or stopping

        if process_to_stop:
            print(f"Stopping Ollama server (PID: {pid_to_log})...")
            try:
                process_to_stop.terminate()
                process_to_stop.wait(timeout=5)
                print("Ollama server terminated.")
            except subprocess.TimeoutExpired:
                print("Ollama server did not terminate gracefully, killing...")
                process_to_stop.kill()
                process_to_stop.wait()
                print("Ollama server killed.")
            except Exception as e:
                print(f"Error stopping Ollama server: {e}")
        else:
            print("Ollama server was already considered stopped or not found for termination.")
        return True
