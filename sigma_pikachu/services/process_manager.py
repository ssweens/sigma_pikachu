import subprocess
import os
import sys
import shlex # For robust command parsing
import threading
import time
import asyncio # Import asyncio
from ..constants import LLAMA_SERVER_LOG_FILE, MCP_LOGS_DIR, CONFIG_FILE, LLAMA_SERVER_CONFIG_FILE
from ..settings.config_manager import config_manager # Singleton instance
from . import socket_proxy # Import the socket_proxy script from the same directory

# Import the new manager classes
from .llama_server_manager import LlamaServerManager
from .mcp_server_manager import McpServerManager
from .proxy_server_manager import ProxyServerManager
from .llama_swap_manager import LlamaSwapManager # Import the new manager class
from .ollama_manager import OllamaManager # Import the OllamaManager class
# Removed import of ui_manager to break circular dependency

class ProcessManager:
    def __init__(self):
        self.llama_server_process = None
        self.llama_swap_process = None # Add process reference for llama-swap
        self.ollama_process = None
        self.mcp_server_processes = {} # Stores {'alias': Popen_object}
        self.proxy_server = None # To hold the asyncio server instance
        self._proxy_loop = None # To hold the asyncio event loop for the proxy
        self._proxy_thread = None # To run the proxy's asyncio loop in a separate thread
        self._last_detected_model = None # To track the last model name from the proxy
        self._lock = threading.RLock() # Use RLock for re-entrant lock

        # Instantiate the new manager classes
        self.llama_manager = LlamaServerManager(self)
        self.llama_swap_manager = LlamaSwapManager(self) # Instantiate LlamaSwapManager
        self.ollama_manager = OllamaManager(self)
        self.mcp_manager = McpServerManager(self)
        # ProxyServerManager instance is created only if enabled in config
        self.proxy_manager = None

        # Process monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_processes, daemon=True)
        self._monitor_thread.start()

        # Keep track of previous states to detect changes
        self._prev_llama_server_running = False
        self._prev_llama_swap_running = False
        self._prev_ollama_server_running = False
        self._prev_mcp_server_running_states = {} # {'alias': bool}

        # Callback for UI updates, set after initialization
        self._ui_update_callback = None

    def set_ui_update_callback(self, callback):
        """Sets the callback function to be called when a UI update is needed."""
        self._ui_update_callback = callback

    # --- Llama Server Management ---
    def is_llama_server_running(self):
        return self.llama_manager.is_running()

    def start_llama_server(self):
        return self.llama_manager.start()

    def stop_llama_server(self):
        return self.llama_manager.stop()

    # --- Llama-swap Server Management ---
    def is_llama_swap_running(self):
        return self.llama_swap_manager.is_running()

    def start_llama_swap(self):
        return self.llama_swap_manager.start()

    def stop_llama_swap(self):
        return self.llama_swap_manager.stop()
    
    # --- Ollama Server Management ---
    def is_ollama_server_running(self):
        return self.ollama_manager.is_running()
    
    def start_ollama_server(self):
        return self.ollama_manager.start()
    
    def stop_ollama_server(self):
        return self.ollama_manager.stop()

    # --- MCP Server Management ---
    def is_mcp_server_running(self, alias):
        return self.mcp_manager.is_running(alias)

    def start_mcp_server(self, mcp_config_entry):
        return self.mcp_manager.start(mcp_config_entry)

    def stop_mcp_server(self, alias):
        return self.mcp_manager.stop(alias)

    def stop_all_mcp_servers(self):
        return self.mcp_manager.stop_all()

    # --- Proxy Server Management ---
    # Note: _run_proxy_loop, _start_proxy_async, _stop_proxy_async, and _on_model_detected
    # are now internal to ProxyServerManager and called via the instance.
    # The public start/stop methods delegate to the manager.

    def start_proxy_server(self, listen_host, listen_port, downstream_host, downstream_port):
        return self.proxy_manager.start(listen_host, listen_port, downstream_host, downstream_port)

    def stop_proxy_server(self):
        return self.proxy_manager.stop()

    def start_all_servers(self):
        """Starts the configured server (Llama or Llama-swap), all enabled MCP servers, and the proxy server."""
        print("Starting all servers...")

        current_config = config_manager.get_config()
        if not current_config:
            print("Error: Could not load config. Cannot start servers.")
            return

        server_type = current_config.get("server_type", "llama_swap").lower() # Get server type from config

        # Start the configured server (Llama or Llama-swap)
        if server_type == "llama_cpp":
            print("Configured server type: llama_cpp. Starting Llama server...")
            self.llama_manager.start()
        elif server_type == "llama_swap":
            print("Configured server type: llama_swap. Starting Llama-swap server...")
            self.llama_swap_manager.start()
        elif server_type == "ollama":
            print("Configured server type: ollama. Starting Ollama server...")
            self.ollama_manager.start()
        else:
            print(f"Warning: Unknown server_type '{server_type}' in config.yaml. Not starting a model server.")


        # Start enabled MCP servers
        mcp_configs = config_manager.get_mcp_servers()
        if mcp_configs:
            for mcp_cfg in mcp_configs:
                self.mcp_manager.start(mcp_cfg)

        # Start the proxy server if enabled in config
        proxy_config = current_config.get("proxy", {})
        proxy_enabled = proxy_config.get("enabled", False)

        if proxy_enabled:
            PROXY_LISTEN_HOST = proxy_config.get("listen_host", '127.0.0.1')
            PROXY_LISTEN_PORT = proxy_config.get("listen_port", 8888)
            # The downstream host/port should point to the *configured* model server
            if server_type == "llama_cpp":
                 # Assuming default llama_cpp.server port is 5000 unless overridden in its config
                 # This might need refinement if llama_cpp.server port is configurable elsewhere
                 PROXY_DOWNSTREAM_HOST = proxy_config.get("downstream_host", '127.0.0.1')
                 PROXY_DOWNSTREAM_PORT = proxy_config.get("downstream_port", 5000)
            elif server_type == "llama_swap":
                 # Get listen address from llama_swap config, split host and port
                 llama_swap_listen = current_config.get("llama_swap", {}).get("listen", ":9999")
                 # Simple split, assumes format :port or host:port
                 if ":" in llama_swap_listen:
                     parts = llama_swap_listen.split(":")
                     PROXY_DOWNSTREAM_HOST = parts[0] if parts[0] else '127.0.0.1' # Default to localhost if empty
                     try:
                         PROXY_DOWNSTREAM_PORT = int(parts[1])
                     except (ValueError, IndexError):
                         print(f"Warning: Invalid llama_swap listen address format '{llama_swap_listen}'. Using default proxy downstream port 5000.")
                         PROXY_DOWNSTREAM_HOST = proxy_config.get("downstream_host", '127.0.0.1')
                         PROXY_DOWNSTREAM_PORT = proxy_config.get("downstream_port", 5000)
                 else:
                     print(f"Warning: Invalid llama_swap listen address format '{llama_swap_listen}'. Using default proxy downstream port 5000.")
                     PROXY_DOWNSTREAM_HOST = proxy_config.get("downstream_host", '127.0.0.1')
                     PROXY_DOWNSTREAM_PORT = proxy_config.get("downstream_port", 5000)
            else:
                 # If no valid server type, proxy can't connect downstream
                 print(f"Warning: Unknown server_type '{server_type}'. Proxy server cannot connect downstream.")
                 PROXY_DOWNSTREAM_HOST = None
                 PROXY_DOWNSTREAM_PORT = None


            if PROXY_DOWNSTREAM_HOST is not None and PROXY_DOWNSTREAM_PORT is not None:
                print("Proxy server is enabled in config. Starting proxy...")
                # Create the proxy manager instance if it doesn't exist
                if self.proxy_manager is None:
                     self.proxy_manager = ProxyServerManager(self)
                self.proxy_manager.start(PROXY_LISTEN_HOST, PROXY_LISTEN_PORT, PROXY_DOWNSTREAM_HOST, PROXY_DOWNSTREAM_PORT)
            else:
                 print("Proxy server cannot start due to invalid server configuration.")

        else:
            print("Proxy server is disabled in config. Not starting proxy.")
            # Ensure proxy manager is None if it was previously running and now disabled
            if self.proxy_manager is not None:
                 self.proxy_manager.stop() # Stop if it was running
                 self.proxy_manager = None


        print("All servers start sequence initiated.")


    def stop_all_servers(self):
        """Stops the configured server (Llama or Llama-swap), all MCP servers, and the proxy server."""
        print("Stopping all servers...")
        # Stop both managers; they handle if their process is running
        self.llama_manager.stop()
        self.llama_swap_manager.stop() # Stop llama-swap manager
        self.ollama_manager.stop() # Stop Ollama manager
        self.mcp_manager.stop_all()
        # Stop the proxy server only if it was started
        if self.proxy_manager is not None:
            self.proxy_manager.stop()
        print("All servers stop sequence initiated.")

    def _monitor_processes(self):
        """Background thread to monitor process states and trigger UI updates."""
        while True:
            time.sleep(2) # Check every 2 seconds

            ui_update_needed = False

            # Check Llama Server
            current_llama_server_running = self.is_llama_server_running()
            if current_llama_server_running != self._prev_llama_server_running:
                print(f"ProcessMonitor: Llama server state changed to {current_llama_server_running}. Requesting UI update.")
                ui_update_needed = True
                self._prev_llama_server_running = current_llama_server_running

            # Check Llama-swap Server
            current_llama_swap_running = self.is_llama_swap_running()
            if current_llama_swap_running != self._prev_llama_swap_running:
                print(f"ProcessMonitor: Llama-swap server state changed to {current_llama_swap_running}. Requesting UI update.")
                ui_update_needed = True
                self._prev_llama_swap_running = current_llama_swap_running
                
            # Check Ollama Server
            current_ollama_server_running = self.is_ollama_server_running()
            if current_ollama_server_running != self._prev_ollama_server_running:
                print(f"ProcessMonitor: Ollama server state changed to {current_ollama_server_running}. Requesting UI update.")
                ui_update_needed = True
                self._prev_ollama_server_running = current_ollama_server_running
            # Check MCP Servers
            current_mcp_server_running_states = {}
            with self._lock: # Lock when accessing mcp_server_processes
                for alias, process in self.mcp_server_processes.items():
                    current_mcp_server_running_states[alias] = process is not None and process.poll() is None

            # Detect changes in MCP server states
            # Check for newly stopped processes
            for alias, was_running in self._prev_mcp_server_running_states.items():
                is_running = current_mcp_server_running_states.get(alias, False) # Assume stopped if not in current list
                if was_running and not is_running:
                    print(f"ProcessMonitor: MCP server '{alias}' state changed to Stopped. Requesting UI update.")
                    ui_update_needed = True

            # Check for newly started processes (less likely to be missed by explicit start, but good for robustness)
            for alias, is_running in current_mcp_server_running_states.items():
                 was_running = self._prev_mcp_server_running_states.get(alias, False)
                 if not was_running and is_running:
                     print(f"ProcessMonitor: MCP server '{alias}' state changed to Running. Requesting UI update.")
                     ui_update_needed = True


            self._prev_mcp_server_running_states = current_mcp_server_running_states # Update previous state

            # Trigger UI update if needed
            if ui_update_needed and self._ui_update_callback:
                # Call the registered UI update callback
                self._ui_update_callback()


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
