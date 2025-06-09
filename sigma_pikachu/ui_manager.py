import pystray
from PIL import Image, ImageDraw
import threading
import time
import logging

from .constants import ICON_FILE, CONFIG_FILE
from .settings.config_manager import config_manager # Singleton
from .services.process_manager import process_manager # Singleton
from .utils import view_llama_server_logs, view_mcp_server_log, view_mcp_logs_directory, view_main_app_logs

class UIManager:
    def __init__(self):
        self.app_icon = None
        self._ui_update_lock = threading.Lock() # Lock for UI updates if needed from threads
        self.logger = logging.getLogger(__name__)

    def request_menu_update(self):
        """Public method for other modules (like ProcessManager) to request a UI menu update."""
        self.logger.info("UI: Received request to update menu.")
        self._update_menu_async()

    def _create_default_icon_image(self):
        """Creates a simple default PIL Image for the tray icon if main icon fails."""
        width = 64
        height = 64
        color1 = (0, 0, 0, 0)  # Transparent background
        color2 = (70, 130, 180, 255) # Steel Blue, Opaque
        image = Image.new("RGBA", (width, height), color1)
        dc = ImageDraw.Draw(image)
        dc.ellipse((8, 8, width - 8, height - 8), fill=color2)
        return image

    def _load_icon(self):
        try:
            image = Image.open(ICON_FILE)
            self.logger.info("Loaded icon from %s", ICON_FILE)
        except FileNotFoundError:
            self.logger.warning("%s not found, using default icon.", ICON_FILE)
            image = self._create_default_icon_image()
        except Exception as e:
            self.logger.error("Error loading %s: %s. Using default icon.", ICON_FILE, e)
            image = self._create_default_icon_image()
        return image

    # --- Menu Item Text/State Generators ---
    def _get_llama_status_text(self, item):
        current_config = config_manager.get_config()
        server_type = current_config.get("server_type", "corral").lower()

        if server_type == "llama_cpp":
            host, port = config_manager.get_llama_host_port() # Assuming this gets llama_cpp config
            if process_manager.is_llama_server_running():
                return f"Llama CPP: Running ({host}:{port})"
            return "Llama CPP: Stopped"
        elif server_type == "corral":
            llama_swap_listen = current_config.get("corral", {}).get("listen", ":9999")
            # Simple split for display, assumes format :port or host:port
            host, port_str = llama_swap_listen.split(":") if ":" in llama_swap_listen else ('127.0.0.1', llama_swap_listen)
            port = port_str if port_str else '9999' # Default port if not specified

            if process_manager.is_llama_swap_running():
                return f"Llama Swap: Running ({host}:{port})"
            return "Llama Swap: Stopped"
        elif server_type == "ollama":
            ollama_listen = current_config.get("ollama", {}).get("listen", ":11434")
            # Simple split for display, assumes format :port or host:port
            host, port_str = ollama_listen.split(":") if ":" in ollama_listen else ('127.0.0.1', ollama_listen)
            port = port_str if port_str else '11434' # Default port if not specified

            if process_manager.is_ollama_server_running():
                return f"Ollama: Running ({host}:{port})"
            return "Ollama: Stopped"
        else:
            return f"Model Server: Unknown Type ({server_type})"


    def _get_llama_toggle_text(self, item):
        current_config = config_manager.get_config()
        server_type = current_config.get("server_type", "corral").lower()

        if server_type == "llama_cpp":
            return "Stop Llama CPP Server" if process_manager.is_llama_server_running() else "Start Llama CPP Server"
        elif server_type == "corral":
            return "Stop Llama Swap Server" if process_manager.is_llama_swap_running() else "Start Llama Swap Server"
        elif server_type == "ollama":
            return "Stop Ollama Server" if process_manager.is_ollama_server_running() else "Start Ollama Server"
        else:
            return "Toggle Model Server (Unknown Type)"

    def _get_llama_models_menu(self, item):
        import threading

        current_config = config_manager.get_config()
        server_type = current_config.get("server_type", "corral").lower()

        # Helper to check if the relevant server is running
        def is_server_running():
            if server_type == "llama_cpp":
                return process_manager.is_llama_server_running()
            elif server_type == "corral":
                return process_manager.is_llama_swap_running()
            elif server_type == "ollama":
                return process_manager.is_ollama_server_running()
            return False

        # Helper to get models for the current server type
        def get_models():
            if server_type == "llama_cpp":
                return config_manager.get_llama_models()
            elif server_type == "corral":
                return config_manager.get_llama_swap_models()
            elif server_type == "ollama":
                return config_manager.get_ollama_models()
            return []

        # If server is not running, start a polling thread and show "Loading models..."
        if not is_server_running():
            self.logger.info("[UIManager] Server not running for model menu. Starting poll thread.")
            def poll_for_server():
                max_retries = 40
                retries = 0
                while retries < max_retries:
                    if is_server_running():
                        self.logger.info("[UIManager] Server is now running. Triggering menu update.")
                        threading.Timer(0.1, self._update_menu_async).start()
                        return
                    time.sleep(0.5)
                    retries += 1
                self.logger.warning("[UIManager] Server did not start in time for model menu.")
                threading.Timer(0.1, self._update_menu_async).start()
            # Start polling in background if not already polling
            threading.Thread(target=poll_for_server, daemon=True).start()
            return pystray.Menu(pystray.MenuItem("Loading models...", None, enabled=False))

        # Server is running, get models
        models = get_models()
        self.logger.info("[UIManager] Server running: %s. Models returned: %s", server_type, models)
        if not models:
            self.logger.info("[UIManager] No models configured or models list is empty.")
            return pystray.Menu(pystray.MenuItem("No models configured", None, enabled=False))

        model_sub_items = []
        for model_info in models:
            alias = model_info.get("model_alias", model_info.get("model", "Unknown Model"))
            self.logger.debug("[UIManager] Adding model to menu: %s", alias)
            model_sub_items.append(pystray.MenuItem(alias, None, enabled=False))
        return pystray.Menu(*model_sub_items)

    def _is_llama_models_menu_visible(self, item):
        current_config = config_manager.get_config()
        server_type = current_config.get("server_type", "corral").lower()

        if server_type == "llama_cpp":
            # Visible if llama_cpp server is running and models are configured under 'llama:'
            return process_manager.is_llama_server_running() and bool(config_manager.get_llama_models())
        elif server_type == "corral":
            # Visible if corral server is running and models are configured in corral config
            return process_manager.is_llama_swap_running() and bool(config_manager.get_llama_swap_models())
        elif server_type == "ollama":
            # Visible if ollama server is running and models are configured under 'ollama:'
            return process_manager.is_ollama_server_running() and bool(config_manager.get_ollama_models())
        else:
            return False # Not visible for unknown server types

    # --- MCP Menu Item Generators ---
    def _get_mcp_server_status_text(self, mcp_alias):
        # This needs to be a callable that pystray can use.
        # We use a lambda that captures mcp_alias.
        def status_text_generator(item):
            if process_manager.is_mcp_server_running(mcp_alias):
                return f"{mcp_alias}: Running"
            return f"{mcp_alias}: Stopped"
        return status_text_generator

    def _get_mcp_server_toggle_text(self, mcp_alias):
        def toggle_text_generator(item):
            return f"Stop {mcp_alias}" if process_manager.is_mcp_server_running(mcp_alias) else f"Start {mcp_alias}"
        return toggle_text_generator
    
    def _get_mcp_server_log_text(self, mcp_alias):
        return f"View {mcp_alias} Log"

    # This is now the primary method for generating the MCP servers menu.
    # It uses a flatter structure based on user's successful modification.
    def _get_mcp_servers_menu(self, item):
        mcp_server_configs = config_manager.get_mcp_servers()

        if not mcp_server_configs:
            return pystray.Menu(pystray.MenuItem("No MCP servers in config", None, enabled=False))

        mcp_sub_items = []
        for mcp_config in mcp_server_configs:
            alias = mcp_config.get("alias")
            enabled_in_config = mcp_config.get("enabled", False)
            if not alias:
                continue
            
            # Item to toggle the server: Text shows status, action toggles.
            mcp_sub_items.append(pystray.MenuItem(
                self._get_mcp_server_status_text(alias), # Dynamic text: "Alias: Status"
                # ACTION must be a callable (lambda) with (icon, item) signature. 'alias' is captured by closure.
                lambda icon, item: self._toggle_mcp_server_action(alias),
                enabled=enabled_in_config
            ))
            # Item to view log for this server
            mcp_sub_items.append(pystray.MenuItem(
                f"View Log for {alias}",
                # ACTION must be a callable (lambda) with (icon, item) signature. 'alias' is captured by closure.
                lambda icon, item: view_mcp_server_log(alias),
                enabled=enabled_in_config
            ))
            mcp_sub_items.append(pystray.Menu.SEPARATOR) # Separator after each server's items

        if not mcp_sub_items: # Should only happen if all configs had no alias
            return pystray.Menu(pystray.MenuItem("No valid MCP servers configured", None, enabled=False))

        # Remove last separator if it exists
        if mcp_sub_items and mcp_sub_items[-1] == pystray.Menu.SEPARATOR:
            mcp_sub_items.pop()

        # Add global actions
        mcp_sub_items.extend([
            pystray.Menu.SEPARATOR, # Separator before global actions
            pystray.MenuItem("Start All Enabled MCPs", self._start_all_mcp_action),
            pystray.MenuItem("Stop All Running MCPs", self._stop_all_mcp_action),
            pystray.MenuItem("View All MCP Logs (Directory)", lambda icon, menu_item_obj: view_mcp_logs_directory())
        ])
        
        return pystray.Menu(*mcp_sub_items)

    # --- Action Callbacks (must take icon, item) ---
    # Note: methods called directly by pystray (like _toggle_llama_server_action) take (self, icon, item)
    # Methods called by our lambdas (like _toggle_mcp_server_action) will take (self, captured_arg)
    def _update_menu_async(self):
        """Schedules a menu update on the icon's thread."""
        if self.app_icon and hasattr(self.app_icon, 'update_menu'):
            # pystray's update_menu should be called from the icon's own thread.
            # If calling from another thread (like a process monitor),
            # it might need special handling or a queue if pystray isn't thread-safe here.
            # For now, direct call, assuming pystray handles it or actions are on main/icon thread.
            # A common pattern is to have the icon run in its own thread, and use icon.loop.call_soon_threadsafe
            # For pystray, updates are often triggered by menu item clicks which are on its thread.
            # If background state changes (e.g. server crashes), a more robust update mechanism is needed.
            self.app_icon.update_menu()

    def _toggle_llama_server_action(self, icon, item):
        self.logger.info("UI: Toggle Model Server action triggered.")
        current_config = config_manager.get_config()
        server_type = current_config.get("server_type", "corral").lower()

        def task():
            if server_type == "llama_cpp":
                if process_manager.is_llama_server_running():
                    process_manager.stop_llama_server()
                else:
                    process_manager.start_llama_server()
            elif server_type == "corral":
                if process_manager.is_llama_swap_running():
                    process_manager.stop_llama_swap()
                else:
                    process_manager.start_llama_swap()
            elif server_type == "ollama":
                if process_manager.is_ollama_server_running():
                    process_manager.stop_ollama_server()
                else:
                    process_manager.start_ollama_server()
            else:
                self.logger.warning("Cannot toggle unknown server type: %s", server_type)

            # Schedule update after task completion
            threading.Timer(0.1, self._update_menu_async).start()

        threading.Thread(target=task, daemon=True).start()
        # self._update_menu_async() # Moved into the thread

    def _toggle_mcp_server_action(self, alias):
        # The lambda in menu creation will call this with just alias.
        # So, we need to make it compatible or adjust lambda.
        # Let's adjust the lambda to pass icon, item if needed, or make this method flexible.
        # For now, assuming the lambda passes what's needed or this method is adapted.
        # The lambda `lambda item, current_alias=alias: self._toggle_mcp_server_action(current_alias)`
        # will result in this method being called as `_toggle_mcp_server_action(self, current_alias_value)`
        # Pystray menu item action signature is `action(icon, item)`.
        # The lambda should be: `lambda icon, item, current_alias=alias: self.ui_manager_instance._toggle_mcp_server_action(current_alias)`
        # This is tricky with instance methods. Let's make the method take (self, icon, item, alias)
        # and adapt the lambda. Or, simpler: the lambda calls a method that *only* takes alias.

        self.logger.info("UI: Toggle MCP Server action triggered for: %s", alias)
        mcp_config_to_toggle = None
        for conf in config_manager.get_mcp_servers():
            if conf.get("alias") == alias:
                mcp_config_to_toggle = conf
                break

        if not mcp_config_to_toggle:
            self.logger.error("Error: No MCP config found for alias %s", alias)
            return

        if not mcp_config_to_toggle.get("enabled", False):
            self.logger.warning("MCP Server '%s' is disabled in config. Cannot toggle via UI menu for individual start/stop if disabled.", alias)
            return

        # Define the task to be run in a separate thread
        def task():
            # These calls can block, so they are inside the thread
            if process_manager.is_mcp_server_running(alias): # 'alias' is captured
                process_manager.stop_mcp_server(alias)
            else:
                # 'mcp_config_to_toggle' is captured from the outer scope
                process_manager.start_mcp_server(mcp_config_to_toggle)

            # Schedule UI update after the blocking task is complete
            threading.Timer(0.1, self._update_menu_async).start()

        # Start the task in a new daemon thread
        threading.Thread(target=task, daemon=True).start()

    def _start_all_mcp_action(self, icon, item):
        self.logger.info("UI: Start All Enabled MCPs action triggered.")
        def task():
            mcp_server_configs = config_manager.get_mcp_servers()
            for mcp_config in mcp_server_configs:
                if mcp_config.get("enabled", False):
                    if not process_manager.is_mcp_server_running(mcp_config.get("alias")):
                        process_manager.start_mcp_server(mcp_config)
                        time.sleep(0.2) # Slightly increased delay for multiple starts
            threading.Timer(0.1, self._update_menu_async).start()

        threading.Thread(target=task, daemon=True).start()
        # self._update_menu_async() # Moved into the thread

    def _stop_all_mcp_action(self, icon, item):
        self.logger.info("UI: Stop All Running MCPs action triggered.")
        def task():
            process_manager.stop_all_mcp_servers()
            threading.Timer(0.1, self._update_menu_async).start()

        threading.Thread(target=task, daemon=True).start()
        # self._update_menu_async() # Moved into the thread

    def _open_config_action(self, icon, item):
        self.logger.info("UI: Open config action triggered for %s", CONFIG_FILE)
        # This is usually a fast operation, but threading for consistency if desired, though likely not needed.
        config_manager.open_config_file_externally()

    def _view_llama_logs_action(self, icon, item):
        self.logger.info("UI: View Llama server logs action triggered.")
        # This is also usually fast.
        view_llama_server_logs()

    def _view_main_app_logs_action(self, icon, item):
        self.logger.info("UI: View main app logs action triggered.")
        view_main_app_logs()

    def _quit_action(self, icon, item):
        self.logger.info("UI: Quit action triggered.")
        # Stopping servers can take time, so definitely thread this.
        def task():
            self.logger.info("Quit task: Stopping all servers...")
            process_manager.stop_all_servers()
            self.logger.info("Quit task: All servers believed to be stopped.")
            if self.app_icon: # Check if icon still exists
                self.logger.info("Quit task: Requesting icon to stop.")
                self.app_icon.stop() # Request pystray to stop its loop
            # The main script (main.py) will handle sys.exit after icon.run() returns.

        # Don't make this a daemon thread if it's critical it finishes before app truly exits.
        # However, icon.stop() should make the main loop terminate, then main.py can exit.
        # If stop_all_servers is lengthy, UI might be gone before it finishes.
        # For quit, it might be okay for it to block for a few seconds.
        # Let's try direct first for quit, if it hangs too long, then thread.
        # On second thought, quit should also be responsive.

        # Threaded quit:
        quit_thread = threading.Thread(target=task, daemon=False) # Non-daemon for quit
        quit_thread.start()
        # Do not call icon.stop() here directly, let the thread do it.

    def setup_and_run_tray_icon(self):
        """Configures and runs the system tray icon."""
        icon_image = self._load_icon()

        # Define menu structure
        # Note: pystray menu items with dynamic text/visibility should be functions (callables)
        menu = pystray.Menu(
            pystray.MenuItem(self._get_llama_status_text, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(self._get_llama_toggle_text, self._toggle_llama_server_action),
            pystray.MenuItem(
                "Llama Models Loaded",
                self._get_llama_models_menu(None), # Static submenu for models
                visible=self._is_llama_models_menu_visible
            ),
            pystray.Menu.SEPARATOR,
            # Using the corrected _get_mcp_servers_menu, called once for a static structure
            pystray.MenuItem("MCP Servers", self._get_mcp_servers_menu(None)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open config.yaml", self._open_config_action),
            pystray.MenuItem("View Main App Log", self._view_main_app_logs_action),
            pystray.MenuItem("View Llama Server Log", self._view_llama_logs_action),
            # "View All MCP Logs (Directory)" is now under MCP Servers submenu
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit Sigma Pikachu", self._quit_action)
        )

        self.app_icon = pystray.Icon("sigma_pikachu_tray", icon_image, "Sigma Pikachu Control", menu)

        # Initial state update might be good if servers can auto-start
        # For now, assuming servers start based on user action or later logic in main
        # self._update_menu_async() # Call once to set initial text based on current state.
        # pystray calls the generator functions for text/visibility on first display.

        self.logger.info("Starting tray icon...")
        self.app_icon.run() # This is a blocking call

# Singleton instance
ui_manager = UIManager()

if __name__ == '__main__':
    # This test requires a valid config.yaml to be useful for MCP servers.
    # It also assumes that constants, config_manager, process_manager are available.
    logger = logging.getLogger(__name__)
    logger.info("UIManager Test: Setting up a dummy tray icon.")
    logger.info("Ensure %s exists and is configured if you want to test MCP server menus.", CONFIG_FILE)

    # For testing, we might need to mock process_manager and config_manager
    # or ensure they can run in a limited test mode.
    # For now, assume they initialize.

    # A simple way to test is to run it. You'll need to manually quit the icon.
    ui_manager.setup_and_run_tray_icon()
    logger.info("UIManager test finished (tray icon was closed).")
