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

class ProcessManager:
    def __init__(self):
        self.llama_server_process = None
        self.mcp_server_processes = {} # Stores {'alias': Popen_object}
        self.proxy_server = None # To hold the asyncio server instance
        self._proxy_loop = None # To hold the asyncio event loop for the proxy
        self._proxy_thread = None # To run the proxy's asyncio loop in a separate thread
        self._last_detected_model = None # To track the last model name from the proxy
        self._lock = threading.RLock() # Use RLock for re-entrant lock

        # Instantiate the new manager classes
        self.llama_manager = LlamaServerManager(self)
        self.mcp_manager = McpServerManager(self)
        # ProxyServerManager instance is created only if enabled in config
        self.proxy_manager = None

    # --- Llama Server Management ---
    def is_llama_server_running(self):
        return self.llama_manager.is_running()

    def start_llama_server(self):
        return self.llama_manager.start()

    def stop_llama_server(self):
        return self.llama_manager.stop()

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
        """Starts Llama server, all enabled MCP servers, and the proxy server."""
        print("Starting all servers...")

        current_config = config_manager.get_config()
        if not current_config:
            print("Error: Could not load config. Cannot start servers.")
            return

        # Start Llama server
        self.llama_manager.start()

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
            PROXY_DOWNSTREAM_HOST = proxy_config.get("downstream_host", '127.0.0.1')
            PROXY_DOWNSTREAM_PORT = proxy_config.get("downstream_port", 5000) # Default to default model server port

            print("Proxy server is enabled in config. Starting proxy...")
            # Create the proxy manager instance if it doesn't exist
            if self.proxy_manager is None:
                 self.proxy_manager = ProxyServerManager(self)
            self.proxy_manager.start(PROXY_LISTEN_HOST, PROXY_LISTEN_PORT, PROXY_DOWNSTREAM_HOST, PROXY_DOWNSTREAM_PORT)
        else:
            print("Proxy server is disabled in config. Not starting proxy.")
            # Ensure proxy manager is None if it was previously running and now disabled
            if self.proxy_manager is not None:
                 self.proxy_manager.stop() # Stop if it was running
                 self.proxy_manager = None


        print("All servers start sequence initiated.")


    def stop_all_servers(self):
        """Stops Llama server, all MCP servers, and the proxy server."""
        print("Stopping all servers...")
        self.llama_manager.stop()
        self.mcp_manager.stop_all()
        # Stop the proxy server only if it was started
        if self.proxy_manager is not None:
            self.proxy_manager.stop()
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
