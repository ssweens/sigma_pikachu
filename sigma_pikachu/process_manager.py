import subprocess
import os
import sys
import shlex # For robust command parsing
import threading
import time
from .constants import LLAMA_SERVER_LOG_FILE, MCP_LOGS_DIR, CONFIG_FILE
from .config_manager import config_manager # Singleton instance

class ProcessManager:
    def __init__(self):
        self.llama_server_process = None
        self.mcp_server_processes = {} # Stores {'alias': Popen_object}
        self._lock = threading.RLock() # Use RLock for re-entrant lock

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

    # --- Llama Server Management ---
    def is_llama_server_running(self):
        with self._lock:
            return self.llama_server_process is not None and self.llama_server_process.poll() is None

    def start_llama_server(self):
        with self._lock:
            if self.is_llama_server_running():
                print("Llama server is already running.")
                return True

            if not os.path.exists(CONFIG_FILE): # Check for main config, not specific llama config
                print(f"Error: {CONFIG_FILE} not found. Cannot start Llama server.")
                return False
            
            current_config = config_manager.get_config() # Use the singleton
            if not current_config:
                print(f"Error: Could not load {CONFIG_FILE}. Cannot start Llama server.")
                return False

            python_executable = self.get_python_executable()
            command = [
                python_executable,
                "-m", "llama_cpp.server",
                "--config_file", CONFIG_FILE # llama_cpp.server uses its own config file path
            ]
            
            print(f"Starting Llama server with command: {' '.join(command)}")
            try:
                # Ensure log directory exists (though constants.py should do this)
                os.makedirs(os.path.dirname(LLAMA_SERVER_LOG_FILE), exist_ok=True)
                with open(LLAMA_SERVER_LOG_FILE, 'a') as log:
                    self.llama_server_process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                print(f"Llama server started. PID: {self.llama_server_process.pid}. Logging to {LLAMA_SERVER_LOG_FILE}")
                return True
            except Exception as e:
                print(f"Failed to start Llama server: {e}")
                self.llama_server_process = None
                return False

    def stop_llama_server(self):
        process_to_stop = None
        pid_to_log = None

        with self._lock: # Lock to safely access/modify shared state
            # is_llama_server_running() also uses the lock, RLock handles re-entrancy.
            if self.is_llama_server_running():
                process_to_stop = self.llama_server_process
                if process_to_stop: # Should always be true if is_llama_server_running was true
                    pid_to_log = process_to_stop.pid
                self.llama_server_process = None # Mark as stopping/stopped
            else:
                # print("Llama server is not running (or already marked as stopped by internal check).")
                return True # Already stopped or stopping

        if process_to_stop:
            print(f"Stopping Llama server (PID: {pid_to_log})...")
            try:
                process_to_stop.terminate()
                process_to_stop.wait(timeout=5) 
                print("Llama server terminated.")
            except subprocess.TimeoutExpired:
                print("Llama server did not terminate gracefully, killing...")
                process_to_stop.kill()
                process_to_stop.wait()
                print("Llama server killed.")
            except Exception as e:
                print(f"Error stopping Llama server: {e}")
        else:
            print("Llama server was already considered stopped or not found for termination.")
        return True

    # --- MCP Server Management ---
    def is_mcp_server_running(self, alias):
        with self._lock: # RLock allows re-entrant calls if needed
            process = self.mcp_server_processes.get(alias)
            return process is not None and process.poll() is None

    def start_mcp_server(self, mcp_config_entry):
        alias = mcp_config_entry.get("alias")
        command_str = mcp_config_entry.get("command")
        enabled = mcp_config_entry.get("enabled", False)

        if not alias or not command_str:
            print(f"MCP Server config missing alias or command: {mcp_config_entry}")
            return False
        
        if not enabled:
            print(f"MCP Server '{alias}' is disabled in config. Not starting.")
            return False # Not an error, but didn't start

        with self._lock:
            if self.is_mcp_server_running(alias):
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
                self.mcp_server_processes[alias] = process
                print(f"MCP Server '{alias}' started. PID: {process.pid}. Logging to {log_file_path}")
                return True
            except Exception as e:
                print(f"Failed to start MCP server '{alias}': {e}")
                if alias in self.mcp_server_processes: # Clean up if entry was made
                    del self.mcp_server_processes[alias]
                return False

    def stop_mcp_server(self, alias):
        process_to_stop = None
        pid_to_log = None

        with self._lock: # Lock to safely access/modify shared state
            # is_mcp_server_running also uses the lock.
            if self.is_mcp_server_running(alias):
                process_to_stop = self.mcp_server_processes.get(alias)
                if process_to_stop: # Should be true if is_mcp_server_running was true
                    pid_to_log = process_to_stop.pid
                # Remove from dict immediately while under lock
                if alias in self.mcp_server_processes:
                    del self.mcp_server_processes[alias]
            else:
                # print(f"MCP Server '{alias}' is not running (or already marked as stopped).")
                # Ensure it's removed if somehow still in dict but not running
                if alias in self.mcp_server_processes:
                    del self.mcp_server_processes[alias]
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
            
    def stop_all_mcp_servers(self):
        # Get a list of aliases to stop outside the main loop of stopping
        aliases_to_stop = []
        with self._lock:
            aliases_to_stop = list(self.mcp_server_processes.keys())
        
        if not aliases_to_stop:
            print("No running MCP servers to stop.")
            return

        print(f"Stopping MCP servers: {', '.join(aliases_to_stop)}...")
        for alias in aliases_to_stop:
            self.stop_mcp_server(alias) # This method now handles its own locking correctly
        print("All MCP servers stop sequence completed for specified aliases.")

    def stop_all_servers(self):
        """Stops Llama server and all MCP servers."""
        print("Stopping all servers...")
        self.stop_llama_server()
        self.stop_all_mcp_servers()
        print("All servers stop sequence initiated.")

# Singleton instance
process_manager = ProcessManager()

if __name__ == '__main__':
    # This is for basic testing of ProcessManager.
    # It requires a config.yaml to be set up correctly.
    print("ProcessManager Test")
    print(f"Make sure {CONFIG_FILE} is configured, especially with some MCP servers.")
    print(f"MCP Logs will be in: {MCP_LOGS_DIR}")
    print(f"Llama Server Log will be in: {LLAMA_SERVER_LOG_FILE}")

    # Test Llama Server
    # print("\nTesting Llama Server...")
    # if process_manager.start_llama_server():
    #     time.sleep(5) # Give it time to start
    #     print(f"Llama server running: {process_manager.is_llama_server_running()}")
    #     process_manager.stop_llama_server()
    #     time.sleep(1)
    #     print(f"Llama server running after stop: {process_manager.is_llama_server_running()}")
    # else:
    #     print("Failed to start Llama server. Check config and llama_cpp.server installation.")

    # Test MCP Servers
    print("\nTesting MCP Servers...")
    mcp_configs = config_manager.get_mcp_servers() # Uses the singleton config_manager
    if not mcp_configs:
        print("No MCP servers found in config.yaml. Add some to test.")
    else:
        test_alias = None
        for mcp_cfg in mcp_configs:
            if mcp_cfg.get("enabled", False): # Only test enabled ones
                test_alias = mcp_cfg.get("alias")
                print(f"\nAttempting to start MCP server: {test_alias}")
                if process_manager.start_mcp_server(mcp_cfg):
                    time.sleep(3) # Give it time to start
                    print(f"MCP server '{test_alias}' running: {process_manager.is_mcp_server_running(test_alias)}")
                    # Keep one running for stop_all_mcp_servers test
                    # process_manager.stop_mcp_server(test_alias)
                    # time.sleep(1)
                    # print(f"MCP server '{test_alias}' running after stop: {process_manager.is_mcp_server_running(test_alias)}")
                else:
                    print(f"Failed to start MCP server '{test_alias}'. Check command and config.")
                # break # Only test one for now to avoid too many processes

        # Test stop_all_mcp_servers
        if any(process_manager.is_mcp_server_running(alias) for alias in process_manager.mcp_server_processes.keys()):
            print("\nTesting Stop All MCP Servers...")
            process_manager.stop_all_mcp_servers()
            time.sleep(2)
            for alias in mcp_configs: # Check all configured, not just running ones
                if alias.get("alias"):
                     print(f"MCP server '{alias['alias']}' running after stop all: {process_manager.is_mcp_server_running(alias['alias'])}")
    
    print("\nProcessManager test finished.")
