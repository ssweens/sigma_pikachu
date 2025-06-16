import subprocess
import os
import sys
import shlex # For robust command parsing
import threading
import time
import asyncio # Import asyncio
import logging
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
        self.llama_swap_process = None # Add process reference for corral
        self.toolshed_process = None # Add process reference for toolshed MCP gateway
        self.ollama_process = None
        self.mcp_server_processes = {} # Stores {'alias': Popen_object} - DEPRECATED for toolshed
        self.proxy_server = None # To hold the asyncio server instance
        self._proxy_loop = None # To hold the asyncio event loop for the proxy
        self._proxy_thread = None # To run the proxy's asyncio loop in a separate thread
        self._last_detected_model = None # To track the last model name from the proxy
        self._lock = threading.RLock() # Use RLock for re-entrant lock
        self.logger = logging.getLogger(__name__)

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
        self._prev_toolshed_running = False
        self._prev_ollama_server_running = False
        self._prev_mcp_server_running_states = {} # {'alias': bool} - DEPRECATED for toolshed

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

    # --- corral Server Management ---
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

    # --- MCP Server Management (Unified Toolshed Gateway) ---
    def is_mcp_server_running(self, alias=None):
        """Check if MCP gateway is running. alias parameter kept for backward compatibility."""
        return self.mcp_manager.is_running()

    def start_mcp_server(self, mcp_config_entry=None):
        """Start MCP gateway. mcp_config_entry parameter kept for backward compatibility."""
        return self.mcp_manager.start()

    def stop_mcp_server(self, alias=None):
        """Stop MCP gateway. alias parameter kept for backward compatibility."""
        return self.mcp_manager.stop()

    def stop_all_mcp_servers(self):
        """Stop all MCP services (unified gateway)."""
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
        """Starts the configured server (Llama or corral), all enabled MCP servers, and the proxy server."""
        self.logger.info("Starting all servers...")

        current_config = config_manager.get_config()
        if not current_config:
            self.logger.error("Error: Could not load config. Cannot start servers.")
            return

        server_type = current_config.get("server_type", "corral").lower() # Get server type from config

        # Start the configured server (Llama or corral)
        if server_type == "llama_cpp":
            self.logger.info("Configured server type: llama_cpp. Starting Llama server...")
            self.llama_manager.start()
        elif server_type == "corral":
            self.logger.info("Configured server type: corral. Starting corral server...")
            self.llama_swap_manager.start()
        elif server_type == "ollama":
            self.logger.info("Configured server type: ollama. Starting Ollama server...")
            self.ollama_manager.start()
        else:
            self.logger.warning("Warning: Unknown server_type '%s' in config.yaml. Not starting a model server.", server_type)


        # Start enabled MCP servers (unified toolshed gateway)
        mcp_configs = config_manager.get_mcp_servers()
        if mcp_configs:
            # Check if any MCP server is enabled before starting the gateway
            enabled_mcp_found = any(mcp_cfg.get("enabled", False) for mcp_cfg in mcp_configs)
            if enabled_mcp_found:
                self.mcp_manager.start()

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
            elif server_type == "corral":
                 # Get listen address from corral config, split host and port
                 llama_swap_listen = current_config.get("corral", {}).get("listen", ":9999")
                 # Simple split, assumes format :port or host:port
                 if ":" in llama_swap_listen:
                     parts = llama_swap_listen.split(":")
                     PROXY_DOWNSTREAM_HOST = parts[0] if parts[0] else '127.0.0.1' # Default to localhost if empty
                     try:
                         PROXY_DOWNSTREAM_PORT = int(parts[1])
                     except (ValueError, IndexError):
                         self.logger.warning("Warning: Invalid corral listen address format '%s'. Using default proxy downstream port 5000.", llama_swap_listen)
                         PROXY_DOWNSTREAM_HOST = proxy_config.get("downstream_host", '127.0.0.1')
                         PROXY_DOWNSTREAM_PORT = proxy_config.get("downstream_port", 5000)
                 else:
                     self.logger.warning("Warning: Invalid corral listen address format '%s'. Using default proxy downstream port 5000.", llama_swap_listen)
                     PROXY_DOWNSTREAM_HOST = proxy_config.get("downstream_host", '127.0.0.1')
                     PROXY_DOWNSTREAM_PORT = proxy_config.get("downstream_port", 5000)
            else:
                 # If no valid server type, proxy can't connect downstream
                 self.logger.warning("Warning: Unknown server_type '%s'. Proxy server cannot connect downstream.", server_type)
                 PROXY_DOWNSTREAM_HOST = None
                 PROXY_DOWNSTREAM_PORT = None


            if PROXY_DOWNSTREAM_HOST is not None and PROXY_DOWNSTREAM_PORT is not None:
                self.logger.info("Proxy server is enabled in config. Starting proxy...")
                # Create the proxy manager instance if it doesn't exist
                if self.proxy_manager is None:
                     self.proxy_manager = ProxyServerManager(self)
                self.proxy_manager.start(PROXY_LISTEN_HOST, PROXY_LISTEN_PORT, PROXY_DOWNSTREAM_HOST, PROXY_DOWNSTREAM_PORT)
            else:
                 self.logger.error("Proxy server cannot start due to invalid server configuration.")

        else:
            self.logger.info("Proxy server is disabled in config. Not starting proxy.")
            # Ensure proxy manager is None if it was previously running and now disabled
            if self.proxy_manager is not None:
                 self.proxy_manager.stop() # Stop if it was running
                 self.proxy_manager = None


        self.logger.info("All servers start sequence initiated.")


    def stop_all_servers(self):
        """Stops the configured server (Llama or corral), all MCP servers, and the proxy server."""
        self.logger.info("Stopping all servers...")
        # Stop both managers; they handle if their process is running
        self.llama_manager.stop()
        self.llama_swap_manager.stop() # Stop corral manager
        self.ollama_manager.stop() # Stop Ollama manager
        self.mcp_manager.stop_all()
        # Stop the proxy server only if it was started
        if self.proxy_manager is not None:
            self.proxy_manager.stop()
        self.logger.info("All servers stop sequence initiated.")

    def _monitor_processes(self):
        """Background thread to monitor process states and trigger UI updates."""
        while True:
            time.sleep(2) # Check every 2 seconds

            ui_update_needed = False

            # Check Llama Server
            current_llama_server_running = self.is_llama_server_running()
            if current_llama_server_running != self._prev_llama_server_running:
                self.logger.info("ProcessMonitor: Llama server state changed to %s. Requesting UI update.", current_llama_server_running)
                ui_update_needed = True
                self._prev_llama_server_running = current_llama_server_running

            # Check corral Server
            current_llama_swap_running = self.is_llama_swap_running()
            if current_llama_swap_running != self._prev_llama_swap_running:
                self.logger.info("ProcessMonitor: corral server state changed to %s. Requesting UI update.", current_llama_swap_running)
                ui_update_needed = True
                self._prev_llama_swap_running = current_llama_swap_running
                
            # Check Ollama Server
            current_ollama_server_running = self.is_ollama_server_running()
            if current_ollama_server_running != self._prev_ollama_server_running:
                self.logger.info("ProcessMonitor: Ollama server state changed to %s. Requesting UI update.", current_ollama_server_running)
                ui_update_needed = True
                self._prev_ollama_server_running = current_ollama_server_running
            # Check Toolshed MCP Gateway
            current_toolshed_running = self.is_mcp_server_running()
            if current_toolshed_running != self._prev_toolshed_running:
                self.logger.info("ProcessMonitor: Toolshed MCP Gateway state changed to %s. Requesting UI update.", current_toolshed_running)
                ui_update_needed = True
                self._prev_toolshed_running = current_toolshed_running

            # Trigger UI update if needed
            if ui_update_needed and self._ui_update_callback:
                # Call the registered UI update callback
                self._ui_update_callback()


# Singleton instance
process_manager = ProcessManager()

if __name__ == '__main__':
    # This is for basic testing of ProcessManager.
    # It requires a config.yaml to be set up correctly.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("ProcessManager Test")
    logger.info("Make sure %s is configured, especially with some MCP servers.", CONFIG_FILE)
    logger.info("MCP Logs will be in: %s", MCP_LOGS_DIR)
    logger.info("Llama Server Log will be in: %s", LLAMA_SERVER_LOG_FILE)

    # Test Llama Server
    # logger.info("\nTesting Llama Server...")
    # if process_manager.start_llama_server():
    #     time.sleep(5) # Give it time to start
    #     logger.info("Llama server running: %s", process_manager.is_llama_server_running())
    #     process_manager.stop_llama_server()
    #     time.sleep(1)
    #     logger.info("Llama server running after stop: %s", process_manager.is_llama_server_running())
    # else:
    #     logger.error("Failed to start Llama server. Check config and llama_cpp.server installation.")

    # Test MCP Servers
    logger.info("\nTesting MCP Servers...")
    mcp_configs = config_manager.get_mcp_servers() # Uses the singleton config_manager
    if not mcp_configs:
        logger.info("No MCP servers found in config.yaml. Add some to test.")
    else:
        test_alias = None
        for mcp_cfg in mcp_configs:
            if mcp_cfg.get("enabled", False): # Only test enabled ones
                test_alias = mcp_cfg.get("alias")
                logger.info("\nAttempting to start MCP server: %s", test_alias)
                if process_manager.start_mcp_server(mcp_cfg):
                    time.sleep(3) # Give it time to start
                    logger.info("MCP server '%s' running: %s", test_alias, process_manager.is_mcp_server_running(test_alias))
                    # Keep one running for stop_all_mcp_servers test
                    # process_manager.stop_mcp_server(test_alias)
                    # time.sleep(1)
                    # logger.info("MCP server '%s' running after stop: %s", test_alias, process_manager.is_mcp_server_running(test_alias))
                else:
                    logger.error("Failed to start MCP server '%s'. Check command and config.", test_alias)
                # break # Only test one for now to avoid too many processes

        # Test stop_all_mcp_servers
        if any(process_manager.is_mcp_server_running(alias) for alias in process_manager.mcp_server_processes.keys()):
            logger.info("\nTesting Stop All MCP Servers...")
            process_manager.stop_all_mcp_servers()
            time.sleep(2)
            for alias in mcp_configs: # Check all configured, not just running ones
                if alias.get("alias"):
                     logger.info("MCP server '%s' running after stop all: %s", alias['alias'], process_manager.is_mcp_server_running(alias['alias']))

    logger.info("\nProcessManager test finished.")
