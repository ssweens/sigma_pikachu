import subprocess
import os
import shlex # For robust command parsing
import threading # Need RLock reference
import time
from ..constants import MCP_LOGS_DIR
from ..settings.config_manager import config_manager # Singleton instance

class McpServerManager:
    def __init__(self, process_manager_instance):
        # Reference to the main ProcessManager instance to access shared state and lock
        self._pm = process_manager_instance

    def is_running(self, alias):
        with self._pm._lock: # Use the lock from the main ProcessManager
            process = self._pm.mcp_server_processes.get(alias)
            return process is not None and process.poll() is None

    def start(self, mcp_config_entry):
        alias = mcp_config_entry.get("alias")
        command_str = mcp_config_entry.get("command")
        enabled = mcp_config_entry.get("enabled", False)

        if not alias or not command_str:
            print(f"MCP Server config missing alias or command: {mcp_config_entry}")
            return False

        if not enabled:
            print(f"MCP Server '{alias}' is disabled in config. Not starting.")
            return False # Not an error, but didn't start

        with self._pm._lock:
            if self.is_running(alias):
                print(f"MCP Server '{alias}' is already running.")
                return True

            # Ensure MCP logs directory exists
            os.makedirs(MCP_LOGS_DIR, exist_ok=True)
            log_file_path = os.path.join(MCP_LOGS_DIR, f"mcp_{alias.replace(' ', '_')}.log")

            # Parse command string safely
            try:
                command_list = shlex.split(command_str)
            except Exception as e:
                print(f"Error parsing command for MCP server '{alias}': {command_str}. Error: {e}")
                return False

            print(f"Starting MCP Server '{alias}' with command: {command_str}")
            try:
                with open(log_file_path, 'a') as log:
                    process = subprocess.Popen(command_list, stdout=log, stderr=subprocess.STDOUT)
                self._pm.mcp_server_processes[alias] = process
                print(f"MCP Server '{alias}' started. PID: {process.pid}. Logging to {log_file_path}")
                return True
            except Exception as e:
                print(f"Failed to start MCP server '{alias}': {e}")
                if alias in self._pm.mcp_server_processes: # Clean up if entry was made
                    del self._pm.mcp_server_processes[alias]
                return False

    def stop(self, alias):
        process_to_stop = None
        pid_to_log = None

        with self._pm._lock: # Lock to safely access/modify shared state
            if self.is_running(alias):
                process_to_stop = self._pm.mcp_server_processes.get(alias)
                if process_to_stop: # Should be true if is_running was true
                    pid_to_log = process_to_stop.pid
                # Remove from dict immediately while under lock
                if alias in self._pm.mcp_server_processes:
                    del self._pm.mcp_server_processes[alias]
            else:
                # print(f"MCP Server '{alias}' is not running (or already marked as stopped).")
                # Ensure it's removed if somehow still in dict but not running
                if alias in self._pm.mcp_server_processes:
                    del self._pm.mcp_server_processes[alias]
                return True # Already stopped or stopping

        if process_to_stop:
            print(f"Stopping MCP Server '{alias}' (PID: {pid_to_log})...")
            try:
                process_to_stop.terminate()
                process_to_stop.wait(timeout=5)
                print(f"MCP Server '{alias}' terminated.")
            except subprocess.TimeoutExpired:
                print(f"MCP Server '{alias}' did not terminate gracefully, killing...")
                process_to_stop.kill()
                process_to_stop.wait()
                print(f"MCP Server '{alias}' killed.")
            except Exception as e:
                print(f"Error stopping MCP server '{alias}': {e}")
        else:
            print(f"MCP Server '{alias}' was already considered stopped or not found for termination.")
        return True

    def stop_all(self):
        # Get a list of aliases to stop outside the main loop of stopping
        aliases_to_stop = []
        with self._pm._lock:
            aliases_to_stop = list(self._pm.mcp_server_processes.keys())

        if not aliases_to_stop:
            print("No running MCP servers to stop.")
            return

        print(f"Stopping MCP servers: {', '.join(aliases_to_stop)}...")
        for alias in aliases_to_stop:
            self.stop(alias) # This method now handles its own locking correctly
        print("All MCP servers stop sequence completed for specified aliases.")
