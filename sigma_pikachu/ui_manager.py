import pystray
from PIL import Image, ImageDraw
import threading
import time

from .constants import ICON_FILE, CONFIG_FILE
from .settings.config_manager import config_manager # Singleton
from .services.process_manager import process_manager # Singleton
from .utils import view_llama_server_logs, view_mcp_server_log, view_mcp_logs_directory

class UIManager:
    def __init__(self):
        self.app_icon = None
        self._ui_update_lock = threading.Lock() # Lock for UI updates if needed from threads

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
            print(f"Loaded icon from {ICON_FILE}")
        except FileNotFoundError:
            print(f"{ICON_FILE} not found, using default icon.")
            image = self._create_default_icon_image()
        except Exception as e:
            print(f"Error loading {ICON_FILE}: {e}. Using default icon.")
            image = self._create_default_icon_image()
        return image

    # --- Menu Item Text/State Generators ---
    def _get_llama_status_text(self, item):
        host, port = config_manager.get_llama_host_port()
        if process_manager.is_llama_server_running():
            return f"Llama: Running ({host}:{port})"
        return "Llama: Stopped"

    def _get_llama_toggle_text(self, item):
        return "Stop Llama Server" if process_manager.is_llama_server_running() else "Start Llama Server"

    def _get_llama_models_menu(self, item):
        models = config_manager.get_llama_models()
        if not models:
            return pystray.Menu(pystray.MenuItem("No models in config", None, enabled=False))
        
        model_sub_items = []
        for model_info in models:
            alias = model_info.get("model_alias", "Unknown Model")
            model_sub_items.append(pystray.MenuItem(alias, None, enabled=False))
        return pystray.Menu(*model_sub_items)

    def _is_llama_models_menu_visible(self, item):
        # Only show "Loaded Models" if server is running and there are models
        return process_manager.is_llama_server_running() and bool(config_manager.get_llama_models())

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
            # If background process changes state, this needs care.
            # Let's assume for now that updates are mostly driven by user actions.
            # If background state changes (e.g. server crashes), a more robust update mechanism is needed.
            self.app_icon.update_menu()

    def _toggle_llama_server_action(self, icon, item):
        print("UI: Toggle Llama Server action triggered.")
        def task():
            if process_manager.is_llama_server_running():
                process_manager.stop_llama_server()
            else:
                process_manager.start_llama_server()
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
        
        # Corrected approach: The lambda in get_mcp_servers_menu is:
        # lambda item_passed_by_pystray, current_alias=alias: self._toggle_mcp_server_action(current_alias)
        # So this method receives `item_passed_by_pystray` as its first arg after `self`, then `current_alias`.
        # Let's rename `alias` to `item_or_alias` to reflect this.
        # No, the lambda is `lambda icon_obj, menu_item_obj, captured_alias=alias: self.actual_handler(captured_alias)`
        # The actual handler should just take `alias`.

        print(f"UI: Toggle MCP Server action triggered for: {alias}")
        mcp_config_to_toggle = None
        for conf in config_manager.get_mcp_servers():
            if conf.get("alias") == alias:
                mcp_config_to_toggle = conf
                break
        
        if not mcp_config_to_toggle:
            print(f"Error: No MCP config found for alias {alias}")
            return

        if not mcp_config_to_toggle.get("enabled", False):
            print(f"MCP Server '{alias}' is disabled in config. Cannot toggle via UI menu for individual start/stop if disabled.")
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
        print("UI: Start All Enabled MCPs action triggered.")
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
        print("UI: Stop All Running MCPs action triggered.")
        def task():
            process_manager.stop_all_mcp_servers()
            threading.Timer(0.1, self._update_menu_async).start()

        threading.Thread(target=task, daemon=True).start()
        # self._update_menu_async() # Moved into the thread

    def _open_config_action(self, icon, item):
        print(f"UI: Open config action triggered for {CONFIG_FILE}")
        # This is usually a fast operation, but threading for consistency if desired, though likely not needed.
        config_manager.open_config_file_externally()

    def _view_llama_logs_action(self, icon, item):
        print("UI: View Llama server logs action triggered.")
        # This is also usually fast.
        view_llama_server_logs()
        
    def _quit_action(self, icon, item):
        print("UI: Quit action triggered.")
        # Stopping servers can take time, so definitely thread this.
        def task():
            print("Quit task: Stopping all servers...")
            process_manager.stop_all_servers()
            print("Quit task: All servers believed to be stopped.")
            if self.app_icon: # Check if icon still exists
                print("Quit task: Requesting icon to stop.")
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

        print("Starting tray icon...")
        self.app_icon.run() # This is a blocking call

# Singleton instance
ui_manager = UIManager()

if __name__ == '__main__':
    # This test requires a valid config.yaml to be useful for MCP servers.
    # It also assumes that constants, config_manager, process_manager are available.
    print("UIManager Test: Setting up a dummy tray icon.")
    print(f"Ensure {CONFIG_FILE} exists and is configured if you want to test MCP server menus.")
    
    # For testing, we might need to mock process_manager and config_manager
    # or ensure they can run in a limited test mode.
    # For now, assume they initialize.

    # A simple way to test is to run it. You'll need to manually quit the icon.
    ui_manager.setup_and_run_tray_icon()
    print("UIManager test finished (tray icon was closed).")
