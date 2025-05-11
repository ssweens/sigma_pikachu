import subprocess
import os
import sys
import shlex # For robust command parsing
import threading # Need RLock reference
import time
from ..constants import LLAMA_SERVER_LOG_FILE, CONFIG_FILE, LLAMA_SWAP_CMD # Reusing log file for now
from ..settings.config_manager import config_manager # Singleton instance

class LlamaSwapManager:
    def __init__(self, process_manager_instance):
        # Reference to the main ProcessManager instance to access shared state and lock
        self._pm = process_manager_instance

    def is_running(self):
        with self._pm._lock: # Use the lock from the main ProcessManager
            # Assuming process_manager will hold a reference like llama_swap_process
            return self._pm.llama_swap_process is not None and self._pm.llama_swap_process.poll() is None

    def start(self):
        with self._pm._lock: # Use the lock from the main ProcessManager
            if self.is_running():
                print("Llama-swap server is already running.")
                return True

            if not os.path.exists(CONFIG_FILE): # Check for main config
                print(f"Error: {CONFIG_FILE} not found. Cannot start Llama-swap server.")
                return False

            current_config = config_manager.get_config() # Use the singleton
            if not current_config:
                print(f"Error: Could not load {CONFIG_FILE}. Cannot start Llama-swap server.")
                return False

            # Assuming llama-swap config path is in the main config under 'llama_swap'
            llama_swap_config_path = current_config.get("llama_swap", {}).get("config_file", "config.yaml")
            listen_address = current_config.get("llama_swap", {}).get("listen", ":9999")


            command = [
                LLAMA_SWAP_CMD,
                "--config", llama_swap_config_path,
                "--listen", listen_address,
            ]

            # Use shlex.quote for printing the command to handle spaces/special chars in arguments
            print(f"Starting Llama-swap server with command: {' '.join(shlex.quote(str(c)) for c in command)}")
            try:
                # Ensure log directory exists (though constants.py should do this)
                os.makedirs(os.path.dirname(LLAMA_SERVER_LOG_FILE), exist_ok=True) # Reusing log file for now
                with open(LLAMA_SERVER_LOG_FILE, 'a') as log:
                    # Assuming process_manager will hold a reference like llama_swap_process
                    self._pm.llama_swap_process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                print(f"Llama-swap server started. PID: {self._pm.llama_swap_process.pid}. Logging to {LLAMA_SERVER_LOG_FILE}")
                return True
            except FileNotFoundError:
                 print(f"Error: 'llama-swap' command not found. Is llama-swap installed and in your PATH?")
                 self._pm.llama_swap_process = None
                 return False
            except Exception as e:
                print(f"Failed to start Llama-swap server: {e}")
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
            print(f"Stopping Llama-swap server (PID: {pid_to_log})...")
            try:
                process_to_stop.terminate()
                process_to_stop.wait(timeout=5)
                print("Llama-swap server terminated.")
            except subprocess.TimeoutExpired:
                print("Llama-swap server did not terminate gracefully, killing...")
                process_to_stop.kill()
                process_to_stop.wait()
                print("Llama-swap server killed.")
            except Exception as e:
                print(f"Error stopping Llama-swap server: {e}")
        else:
            print("Llama-swap server was already considered stopped or not found for termination.")
        return True
