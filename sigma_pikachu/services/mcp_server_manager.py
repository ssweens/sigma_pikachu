import subprocess
import os
import shlex # For robust command parsing
import threading # Need RLock reference
import time
import logging
from ..constants import MCP_LOGS_DIR
from ..settings.config_manager import config_manager # Singleton instance

class McpServerManager:
    def __init__(self, process_manager_instance):
        # Reference to the main ProcessManager instance to access shared state and lock
        self._pm = process_manager_instance
        self.logger = logging.getLogger(__name__)

    def is_running(self, alias):
        with self._pm._lock: # Use the lock from the main ProcessManager
            process = self._pm.mcp_server_processes.get(alias)
            return process is not None and process.poll() is None

    def start(self, mcp_config_entry):
        alias = mcp_config_entry.get("alias")
        command_str = mcp_config_entry.get("command")
        enabled = mcp_config_entry.get("enabled", False)

        if not alias or not command_str:
            self.logger.error("MCP Server config missing alias or command: %s", mcp_config_entry)
            return False

        if not enabled:
            self.logger.info("MCP Server '%s' is disabled in config. Not starting.", alias)
            return False # Not an error, but didn't start

        with self._pm._lock:
            if self.is_running(alias):
                self.logger.info("MCP Server '%s' is already running.", alias)
                return True

            # Ensure MCP logs directory exists
            os.makedirs(MCP_LOGS_DIR, exist_ok=True)
            log_file_path = os.path.join(MCP_LOGS_DIR, f"mcp_{alias.replace(' ', '_')}.log")

            # Parse command string safely
            try:
                command_list = shlex.split(command_str)
            except Exception as e:
                self.logger.error("Error parsing command for MCP server '%s': %s. Error: %s", alias, command_str, e)
                return False

            self.logger.info("Starting MCP Server '%s' with command: %s", alias, command_str)
            try:
                with open(log_file_path, 'a') as log:
                    process = subprocess.Popen(command_list, stdout=log, stderr=subprocess.STDOUT)
                self._pm.mcp_server_processes[alias] = process
                self.logger.info("MCP Server '%s' started. PID: %s. Logging to %s", alias, process.pid, log_file_path)
                return True
            except Exception as e:
                self.logger.error("Failed to start MCP server '%s': %s", alias, e)
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
            self.logger.info("Stopping MCP Server '%s' (PID: %s)...", alias, pid_to_log)
            try:
                process_to_stop.terminate()
                process_to_stop.wait(timeout=5)
                self.logger.info("MCP Server '%s' terminated.", alias)
            except subprocess.TimeoutExpired:
                self.logger.warning("MCP Server '%s' did not terminate gracefully, killing...", alias)
                process_to_stop.kill()
                process_to_stop.wait()
                self.logger.info("MCP Server '%s' killed.", alias)
            except Exception as e:
                self.logger.error("Error stopping MCP server '%s': %s", alias, e)
        else:
            self.logger.info("MCP Server '%s' was already considered stopped or not found for termination.", alias)
        return True

    def stop_all(self):
        # Get a list of aliases to stop outside the main loop of stopping
        aliases_to_stop = []
        with self._pm._lock:
            aliases_to_stop = list(self._pm.mcp_server_processes.keys())

        if not aliases_to_stop:
            self.logger.info("No running MCP servers to stop.")
            return

        self.logger.info("Stopping MCP servers: %s...", ', '.join(aliases_to_stop))
        for alias in aliases_to_stop:
            self.stop(alias) # This method now handles its own locking correctly
        self.logger.info("All MCP servers stop sequence completed for specified aliases.")
