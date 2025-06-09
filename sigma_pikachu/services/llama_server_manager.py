import subprocess
import os
import sys
import shlex # For robust command parsing
import threading # Need RLock reference
import time
import logging
from ..constants import LLAMA_SERVER_LOG_FILE, CONFIG_FILE, LLAMA_SERVER_CONFIG_FILE
from ..settings.config_manager import config_manager # Singleton instance

class LlamaServerManager:
    def __init__(self, process_manager_instance):
        # Reference to the main ProcessManager instance to access shared state and lock
        self._pm = process_manager_instance
        self.logger = logging.getLogger(__name__)

    def get_python_executable(self):
        """Determines the correct Python executable to use."""
        if getattr(sys, 'frozen', False):
            # In a PyInstaller bundle, 'python' or 'python3' might not be on PATH
            # or might refer to a system Python. sys.executable is the bundle's Python.
            # However, for llama_cpp.server, it might expect to be run as a module
            # by an external python. This needs careful consideration.
            # For now, let's assume 'python3' is available or we use sys.executable.
            # If llama_cpp.server is bundled, sys.executable is correct.
            # If llama_cpp.server is an external package, 'python3' or 'python' is better.
            # Given the original code used sys.executable for non-frozen and "python3" for frozen,
            # there might be a specific reason. Let's stick to sys.executable for now if bundled.
            # The original logic was:
            # python_executable = "python3" if getattr(sys, 'frozen', False) else sys.executable
            # This seems counter-intuitive if "python3" is for the bundled case.
            # Let's assume sys.executable is generally safer for bundled apps to call internal/bundled scripts.
            # For external commands like npx, it doesn't matter.
            return sys.executable # More robust for bundled python scripts
        else:
            return sys.executable # e.g., /usr/bin/python3 or venv python

    def is_running(self):
        with self._pm._lock: # Use the lock from the main ProcessManager
            return self._pm.llama_server_process is not None and self._pm.llama_server_process.poll() is None

    def start(self):
        with self._pm._lock: # Use the lock from the main ProcessManager
            if self.is_running():
                self.logger.info("Llama server is already running.")
                return True

            if not os.path.exists(CONFIG_FILE): # Check for main config, not specific llama config
                self.logger.error("Error: %s not found. Cannot start Llama server.", CONFIG_FILE)
                return False

            current_config = config_manager.get_config() # Use the singleton
            if not current_config:
                self.logger.error("Error: Could not load %s. Cannot start Llama server.", CONFIG_FILE)
                return False

            llama_config = current_config.get("llama", {})

            python_executable = self.get_python_executable()
            command = [
                python_executable,
                "-m", "llama_cpp.server",
                "--config_file", LLAMA_SERVER_CONFIG_FILE, # This is the main config file
            ]

            # Note: The original code had commented-out logic for adding
            # VALID_LLAMA_ARGS from the config. This logic is not included
            # in this refactored version as it was commented out in the source.
            # If needed, this logic should be added here.

            # Use shlex.quote for printing the command to handle spaces/special chars in arguments
            self.logger.info("Starting Llama server with command: %s", ' '.join(shlex.quote(str(c)) for c in command))
            try:
                # Ensure log directory exists (though constants.py should do this)
                os.makedirs(os.path.dirname(LLAMA_SERVER_LOG_FILE), exist_ok=True)
                with open(LLAMA_SERVER_LOG_FILE, 'a') as log:
                    self._pm.llama_server_process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                self.logger.info("Llama server started. PID: %s. Logging to %s", self._pm.llama_server_process.pid, LLAMA_SERVER_LOG_FILE)
                return True
            except Exception as e:
                self.logger.error("Failed to start Llama server: %s", e)
                self._pm.llama_server_process = None
                return False

    def stop(self):
        process_to_stop = None
        pid_to_log = None

        with self._pm._lock: # Lock to safely access/modify shared state
            if self.is_running():
                process_to_stop = self._pm.llama_server_process
                if process_to_stop: # Should always be true if is_running was true
                    pid_to_log = process_to_stop.pid
                self._pm.llama_server_process = None # Mark as stopping/stopped
            else:
                # print("Llama server is not running (or already marked as stopped by internal check).")
                return True # Already stopped or stopping

        if process_to_stop:
            self.logger.info("Stopping Llama server (PID: %s)...", pid_to_log)
            try:
                process_to_stop.terminate()
                process_to_stop.wait(timeout=5)
                self.logger.info("Llama server terminated.")
            except subprocess.TimeoutExpired:
                self.logger.warning("Llama server did not terminate gracefully, killing...")
                process_to_stop.kill()
                process_to_stop.wait()
                self.logger.info("Llama server killed.")
            except Exception as e:
                self.logger.error("Error stopping Llama server: %s", e)
        else:
            self.logger.info("Llama server was already considered stopped or not found for termination.")
        return True
