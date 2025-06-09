import asyncio
import threading
import time
import logging
from . import socket_proxy # Import the socket_proxy script from the same directory

class ProxyServerManager:
    def __init__(self, process_manager_instance):
        # Reference to the main ProcessManager instance to access shared state and lock
        self._pm = process_manager_instance
        self.logger = logging.getLogger(__name__)

    def _run_proxy_loop(self):
        """Runs the asyncio event loop for the proxy in a separate thread."""
        self._pm._proxy_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._pm._proxy_loop)
        try:
            self._pm._proxy_loop.run_forever()
        finally:
            # Clean up the loop resources
            self._pm._proxy_loop.close()
            self.logger.info("Proxy asyncio loop closed.")

    async def _start_proxy_async(self, listen_host, listen_port, downstream_host, downstream_port):
        """Starts the proxy server within the proxy's asyncio loop."""
        try:
            self._pm.proxy_server = await socket_proxy.start_proxy(
                listen_host, listen_port, downstream_host, downstream_port, self._on_model_detected
            )
            self.logger.info("Proxy server started successfully on %s:%s", listen_host, listen_port)
        except Exception as e:
            self.logger.error("Failed to start proxy server: %s", e)
            self._pm.proxy_server = None # Ensure server is None on failure

    def start(self, listen_host, listen_port, downstream_host, downstream_port):
        """Starts the proxy server in a separate thread."""
        with self._pm._lock:
            if self._pm._proxy_thread and self._pm._proxy_thread.is_alive():
                self.logger.info("Proxy server is already starting or running.")
                return True

            self.logger.info("Attempting to start proxy server thread...")
            self._pm._proxy_thread = threading.Thread(target=self._run_proxy_loop, daemon=True)
            self._pm._proxy_thread.start()
            self.logger.info("Proxy server thread started: %s", self._pm._proxy_thread.name)

            # Wait a moment for the loop to start, then schedule the server start
            # A more robust way might involve a Future or Event, but sleep is simpler for now.
            time.sleep(0.1)
            if self._pm._proxy_loop is None or not self._pm._proxy_loop.is_running():
                 self.logger.error("Error: Proxy asyncio loop did not start or is not running.")
                 self._pm._proxy_thread = None
                 self._pm._proxy_loop = None
                 return False

            self.logger.info("Proxy asyncio loop is running. Scheduling server start.")
            asyncio.run_coroutine_threadsafe(
                self._start_proxy_async(listen_host, listen_port, downstream_host, downstream_port),
                self._pm._proxy_loop
            )
            self.logger.info("Proxy server start scheduled.")
            return True

    async def _stop_proxy_async(self):
        """Stops the proxy server within its asyncio loop."""
        if self._pm.proxy_server:
            self.logger.info("Stopping proxy server...")
            self._pm.proxy_server.close()
            await self._pm.proxy_server.wait_closed()
            self.logger.info("Proxy server stopped.")
            self._pm.proxy_server = None
        if self._pm._proxy_loop:
             self._pm._proxy_loop.stop()
             self.logger.info("Proxy asyncio loop stopped.")
             self._pm._proxy_loop = None

    def stop(self):
        """Stops the proxy server and its thread."""
        with self._pm._lock:
            if self._pm._proxy_thread and self._pm._proxy_thread.is_alive() and self._pm._proxy_loop and self._pm._proxy_loop.is_running():
                self.logger.info("Attempting to stop proxy server thread...")
                # Schedule the async stop function in the proxy's loop
                asyncio.run_coroutine_threadsafe(self._stop_proxy_async(), self._pm._proxy_loop)
                # The loop.stop() in _stop_proxy_async will cause run_forever to exit
                self._pm._proxy_thread.join(timeout=5) # Wait for the thread to finish
                if self._pm._proxy_thread.is_alive():
                    self.logger.warning("Proxy thread did not join gracefully.")
                self._pm._proxy_thread = None
                self.logger.info("Proxy server stop scheduled/initiated.")
            else:
                self.logger.info("Proxy server thread is not running.")
        return True

    async def _on_model_detected(self, model_name):
        """Callback function executed when the proxy detects a model header."""
        self.logger.info("ProxyServerManager received model detection callback: %s", model_name)
        # This callback runs in the proxy's asyncio loop.
        # Use run_coroutine_threadsafe to call thread-safe methods of ProcessManager
        # or schedule coroutines back on the main loop if needed.

        # Basic logic to react to model change:
        with self._pm._lock: # Acquire the lock before accessing/modifying shared state
            self.logger.debug("ProxyServerManager._on_model_detected inside lock. Current model: %s, Detected: %s", self._pm._last_detected_model, model_name)
            if model_name and model_name != self._pm._last_detected_model:
                self.logger.info("Model changed from %s to %s. Triggering server switch...", self._pm._last_detected_model, model_name)
                # In a real implementation, you would map model_name to a specific server config
                # and then stop the current server and start the new one.
                # This requires calling back to the main ProcessManager or a dedicated orchestrator.
                # For now, we'll just update the last detected model.
                # A more complete implementation would involve a mechanism to signal the main
                # ProcessManager to handle the server switch.
                self.logger.warning("Placeholder: Logic to signal main ProcessManager for model switch to '%s' needs to be implemented here.", model_name)
                self._pm._last_detected_model = model_name
            elif model_name:
                 self.logger.debug("Detected model %s (same as last).", model_name)
            else:
                 self.logger.warning("Detected model name is None or empty.")
