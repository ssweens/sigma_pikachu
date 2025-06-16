import subprocess
import os
import shlex # For robust command parsing
import threading # Need RLock reference
import time
import logging
from ..constants import MCP_LOGS_DIR, CONFIG_FILE, TOOLSHED_CMD
from ..settings.config_manager import config_manager # Singleton instance

class McpServerManager:
    def __init__(self, process_manager_instance):
        # Reference to the main ProcessManager instance to access shared state and lock
        self._pm = process_manager_instance
        self.logger = logging.getLogger(__name__)

    def is_running(self):
        with self._pm._lock: # Use the lock from the main ProcessManager
            # Following LlamaSwapManager pattern - single process management
            return self._pm.toolshed_process is not None and self._pm.toolshed_process.poll() is None

    def start(self):
        with self._pm._lock: # Use the lock from the main ProcessManager
            if self.is_running():
                self.logger.info("Toolshed MCP Gateway is already running.")
                return True

            if not os.path.exists(CONFIG_FILE): # Check for main config
                self.logger.error("Error: %s not found. Cannot start Toolshed MCP Gateway.", CONFIG_FILE)
                return False

            current_config = config_manager.get_config() # Use the singleton
            if not current_config:
                self.logger.error("Error: Could not load %s. Cannot start Toolshed MCP Gateway.", CONFIG_FILE)
                return False

            # Build command like LlamaSwapManager does - directly in code
            # Use the toolshed config path from user's home directory
            toolshed_config_path = os.path.expanduser("~/.config/toolshed/config.yaml")

            command = [
                TOOLSHED_CMD, # Use the full path to toolshed
                "--config", toolshed_config_path
            ]

            # Use shlex.quote for printing the command to handle spaces/special chars in arguments
            self.logger.info("Starting Toolshed MCP Gateway with command: %s", ' '.join(shlex.quote(str(c)) for c in command))
            
            try:
                # Ensure log directory exists
                log_file_path = os.path.join(MCP_LOGS_DIR, "toolshed_gateway.log")
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                with open(log_file_path, 'a') as log:
                    # Pass the current environment to ensure the PATH is correctly used
                    self._pm.toolshed_process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ)
                self.logger.info("Toolshed MCP Gateway started. PID: %s. Logging to %s", self._pm.toolshed_process.pid, log_file_path)
                return True
            except FileNotFoundError:
                 self.logger.error("Error: 'toolshed' command not found. Is toolshed binary present at %s?", TOOLSHED_CMD)
                 self._pm.toolshed_process = None
                 return False
            except Exception as e:
                self.logger.error("Failed to start Toolshed MCP Gateway: %s", e)
                self._pm.toolshed_process = None
                return False

    def stop(self):
        process_to_stop = None
        pid_to_log = None

        with self._pm._lock: # Lock to safely access/modify shared state
            if self.is_running():
                process_to_stop = self._pm.toolshed_process
                if process_to_stop: # Should always be true if is_running was true
                    pid_to_log = process_to_stop.pid
                self._pm.toolshed_process = None # Mark as stopping/stopped
            else:
                return True # Already stopped or stopping

        if process_to_stop:
            self.logger.info("Stopping Toolshed MCP Gateway (PID: %s)...", pid_to_log)
            try:
                process_to_stop.terminate()
                process_to_stop.wait(timeout=5)
                self.logger.info("Toolshed MCP Gateway terminated.")
            except subprocess.TimeoutExpired:
                self.logger.warning("Toolshed MCP Gateway did not terminate gracefully, killing...")
                process_to_stop.kill()
                process_to_stop.wait()
                self.logger.info("Toolshed MCP Gateway killed.")
            except Exception as e:
                self.logger.error("Error stopping Toolshed MCP Gateway: %s", e)
        else:
            self.logger.info("Toolshed MCP Gateway was already considered stopped or not found for termination.")
        return True

    def stop_all(self):
        """Stop all MCP services (unified gateway)."""
        self.logger.info("Stopping all MCP services (Toolshed Gateway)...")
        return self.stop()
