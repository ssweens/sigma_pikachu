import asyncio
import threading
import time
from .. import socket_proxy # Import the socket_proxy script from the same directory

class ProxyServerManager:
    def __init__(self, process_manager_instance):
        # Reference to the main ProcessManager instance to access shared state and lock
        self._pm = process_manager_instance

    def _run_proxy_loop(self):
        """Runs the asyncio event loop for the proxy in a separate thread."""
        self._pm._proxy_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._pm._proxy_loop)
        try:
            self._pm._proxy_loop.run_forever()
        finally:
            # Clean up the loop resources
            self._pm._proxy_loop.close()
            print("Proxy asyncio loop closed.")

    async def _start_proxy_async(self, listen_host, listen_port, downstream_host, downstream_port):
        """Starts the proxy server within the proxy's asyncio loop."""
        try:
            self._pm.proxy_server = await socket_proxy.start_proxy(
                listen_host, listen_port, downstream_host, downstream_port, self._on_model_detected
            )
            print(f"Proxy server started successfully on {listen_host}:{listen_port}")
        except Exception as e:
            print(f"Failed to start proxy server: {e}")
            self._pm.proxy_server = None # Ensure server is None on failure

    def start(self, listen_host, listen_port, downstream_host, downstream_port):
        """Starts the proxy server in a separate thread."""
        with self._pm._lock:
            if self._pm._proxy_thread and self._pm._proxy_thread.is_alive():
                print("Proxy server is already starting or running.")
                return True

            print(f"Attempting to start proxy server thread...")
            self._pm._proxy_thread = threading.Thread(target=self._run_proxy_loop, daemon=True)
            self._pm._proxy_thread.start()
            print(f"Proxy server thread started: {self._pm._proxy_thread.name}")

            # Wait a moment for the loop to start, then schedule the server start
            # A more robust way might involve a Future or Event, but sleep is simpler for now.
            time.sleep(0.1)
            if self._pm._proxy_loop is None or not self._pm._proxy_loop.is_running():
                 print("Error: Proxy asyncio loop did not start or is not running.")
                 self._pm._proxy_thread = None
                 self._pm._proxy_loop = None
                 return False

            print("Proxy asyncio loop is running. Scheduling server start.")
            asyncio.run_coroutine_threadsafe(
                self._start_proxy_async(listen_host, listen_port, downstream_host, downstream_port),
                self._pm._proxy_loop
            )
            print("Proxy server start scheduled.")
            return True

    async def _stop_proxy_async(self):
        """Stops the proxy server within its asyncio loop."""
        if self._pm.proxy_server:
            print("Stopping proxy server...")
            self._pm.proxy_server.close()
            await self._pm.proxy_server.wait_closed()
            print("Proxy server stopped.")
            self._pm.proxy_server = None
        if self._pm._proxy_loop:
             self._pm._proxy_loop.stop()
             print("Proxy asyncio loop stopped.")
             self._pm._proxy_loop = None

    def stop(self):
        """Stops the proxy server and its thread."""
        with self._pm._lock:
            if self._pm._proxy_thread and self._pm._proxy_thread.is_alive() and self._pm._proxy_loop and self._pm._proxy_loop.is_running():
                print("Attempting to stop proxy server thread...")
                # Schedule the async stop function in the proxy's loop
                asyncio.run_coroutine_threadsafe(self._stop_proxy_async(), self._pm._proxy_loop)
                # The loop.stop() in _stop_proxy_async will cause run_forever to exit
                self._pm._proxy_thread.join(timeout=5) # Wait for the thread to finish
                if self._pm._proxy_thread.is_alive():
                    print("Warning: Proxy thread did not join gracefully.")
                self._pm._proxy_thread = None
                print("Proxy server stop scheduled/initiated.")
            else:
                print("Proxy server thread is not running.")
        return True

    async def _on_model_detected(self, model_name):
        """Callback function executed when the proxy detects a model header."""
        print(f"ProxyServerManager received model detection callback: {model_name}")
        # This callback runs in the proxy's asyncio loop.
        # Use run_coroutine_threadsafe to call thread-safe methods of ProcessManager
        # or schedule coroutines back on the main loop if needed.

        # Basic logic to react to model change:
        with self._pm._lock: # Acquire the lock before accessing/modifying shared state
            print(f"ProxyServerManager._on_model_detected inside lock. Current model: {self._pm._last_detected_model}, Detected: {model_name}")
            if model_name and model_name != self._pm._last_detected_model:
                print(f"Model changed from {self._pm._last_detected_model} to {model_name}. Triggering server switch...")
                # In a real implementation, you would map model_name to a specific server config
                # and then stop the current server and start the new one.
                # This requires calling back to the main ProcessManager or a dedicated orchestrator.
                # For now, we'll just update the last detected model.
                # A more complete implementation would involve a mechanism to signal the main
                # ProcessManager to handle the server switch.
                print(f"Placeholder: Logic to signal main ProcessManager for model switch to '{model_name}' needs to be implemented here.")
                self._pm._last_detected_model = model_name
            elif model_name:
                 print(f"Detected model {model_name} (same as last).")
            else:
                 print("Detected model name is None or empty.")
