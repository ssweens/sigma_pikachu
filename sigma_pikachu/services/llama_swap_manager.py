import subprocess
import os
import sys
import shlex # For robust command parsing
import threading # Need RLock reference
import time
import logging
from ..constants import LLAMA_SERVER_LOG_FILE, CONFIG_FILE, LLAMA_SWAP_CMD # Reusing log file for now
from ..settings.config_manager import config_manager # Singleton instance

class LlamaSwapManager:
    def __init__(self, process_manager_instance):
        # Reference to the main ProcessManager instance to access shared state and lock
        self._pm = process_manager_instance
        self.logger = logging.getLogger(__name__)

    def is_running(self):
        with self._pm._lock: # Use the lock from the main ProcessManager
            # Assuming process_manager will hold a reference like llama_swap_process
            return self._pm.llama_swap_process is not None and self._pm.llama_swap_process.poll() is None

    def start(self):
        with self._pm._lock: # Use the lock from the main ProcessManager
            if self.is_running():
                self.logger.info("Llama-swap server is already running.")
                return True

            if not os.path.exists(CONFIG_FILE): # Check for main config
                self.logger.error("Error: %s not found. Cannot start Llama-swap server.", CONFIG_FILE)
                return False

            current_config = config_manager.get_config() # Use the singleton
            if not current_config:
                self.logger.error("Error: Could not load %s. Cannot start Llama-swap server.", CONFIG_FILE)
                return False

            # Assuming llama-swap config path is in the main config under 'corral'
            llama_swap_config_path = current_config.get("corral", {}).get("config_file", "config.yaml")
            listen_address = current_config.get("corral", {}).get("listen", ":9999")


            command = [
                LLAMA_SWAP_CMD, # Use the full path to corral
                "--config", llama_swap_config_path,
                "--listen", listen_address,
                "--watch-config"
            ]

            # Use shlex.quote for printing the command to handle spaces/special chars in arguments
            self.logger.info("Starting Corral server with command: %s", ' '.join(shlex.quote(str(c)) for c in command))
            
            # Debug: Log the PATH being passed to corral
            current_path = os.environ.get('PATH', '')
            if current_path:
                path_parts = current_path.split(os.pathsep)
                self.logger.info("PATH being passed to corral contains %d entries. First entry: %s", len(path_parts), path_parts[0] if path_parts else 'None')
            
            try:
                # Ensure log directory exists (though constants.py should do this)
                os.makedirs(os.path.dirname(LLAMA_SERVER_LOG_FILE), exist_ok=True) # Reusing log file for now
                with open(LLAMA_SERVER_LOG_FILE, 'a') as log:
                    # Assuming process_manager will hold a reference like llama_swap_process
                    # Pass the current environment to ensure the PATH is correctly used
                    self._pm.llama_swap_process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ)
                self.logger.info("Corral server started. PID: %s. Logging to %s", self._pm.llama_swap_process.pid, LLAMA_SERVER_LOG_FILE)
                return True
            except FileNotFoundError:
                 self.logger.error("Error: 'corral' command not found. Is corral installed and in your PATH?")
                 self._pm.llama_swap_process = None
                 return False
            except Exception as e:
                self.logger.error("Failed to start Corral server: %s", e)
                self._pm.llama_swap_process = None
                return False

    def stop(self):
        process_to_stop = None
        pid_to_log = None

        with self._pm._lock: # Lock to safely access/modify shared state
            # Assuming process_manager will hold a reference like llama_swap_process
            if self.is_running():
                process_to_stop = self._pm.llama_swap_process
                if process_to_stop: # Should always be true if is_running was true
                    pid_to_log = process_to_stop.pid
                self._pm.llama_swap_process = None # Mark as stopping/stopped
            else:
                # print("Llama-swap server is not running (or already marked as stopped by internal check).")
                return True # Already stopped or stopping

        if process_to_stop:
            self.logger.info("Stopping Corral server (PID: %s)...", pid_to_log)
            try:
                process_to_stop.terminate()
                process_to_stop.wait(timeout=5)
                self.logger.info("Corral server terminated.")
            except subprocess.TimeoutExpired:
                self.logger.warning("Corral server did not terminate gracefully, killing...")
                process_to_stop.kill()
                process_to_stop.wait()
                self.logger.info("Corral server killed.")
            except Exception as e:
                self.logger.error("Error stopping Corral server: %s", e)
        else:
            self.logger.info("Corral server was already considered stopped or not found for termination.")
        return True
