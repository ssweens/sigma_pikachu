import sys
import os

# Ensure the package directory is in the Python path if running as a script
if not getattr(sys, 'frozen', False): # Not running as a PyInstaller bundle
    # Get the directory of the current script (main.py)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (project root, assuming main.py is in sigma_pikachu/)
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    # Add the project root to sys.path to allow imports like 'from sigma_pikachu.module'
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    # Also add the package directory itself if needed for relative imports from sibling modules
    # if SCRIPT_DIR not in sys.path:
    #    sys.path.insert(0, SCRIPT_DIR)


# Import necessary modules after path adjustments
#try:
from sigma_pikachu.constants import CONFIG_FILE, USER_CONFIG_DIR, MCP_LOGS_DIR, LLAMA_SERVER_LOG_FILE
from sigma_pikachu.settings.config_manager import config_manager # Singleton instance
from sigma_pikachu.services.process_manager import process_manager # Singleton instance
from sigma_pikachu.ui_manager import ui_manager # Singleton instance
from PIL import Image, ImageDraw # Check for Pillow
# except ImportError as e:
#     print(f"Error importing application modules: {e}")
#     print("Please ensure all modules (constants, config_manager, process_manager, ui_manager) are in the 'sigma_pikachu' directory.")
#     print("If running from source, ensure the project root is in PYTHONPATH or you are running from the project root e.g. python -m sigma_pikachu.main")
#     sys.exit(1)


def initial_setup_checks():
    """Perform initial checks, create directories, and ensure Pillow is available."""
    print(f"User config directory: {USER_CONFIG_DIR}")
    print(f"Application config file: {CONFIG_FILE}")
    print(f"Llama server log file: {LLAMA_SERVER_LOG_FILE}")
    print(f"MCP logs directory: {MCP_LOGS_DIR}")

    # Ensure directories exist (constants.py should handle this, but double-check)
    if not os.path.exists(USER_CONFIG_DIR):
        os.makedirs(USER_CONFIG_DIR, exist_ok=True)
        print(f"Created user config directory: {USER_CONFIG_DIR}")
    
    if not os.path.exists(MCP_LOGS_DIR):
        os.makedirs(MCP_LOGS_DIR, exist_ok=True)
        print(f"Created MCP logs directory: {MCP_LOGS_DIR}")

    # Check for config file and create default if missing (config_manager handles this on init)
    _ = config_manager.get_config() # This will trigger creation if missing

    # Ensure Pillow is available (critical for pystray icon)
    try:
        from PIL import Image, ImageDraw
        print("Pillow library found.")
    except ImportError:
        print("CRITICAL ERROR: Pillow (PIL) library not found.")
        print("Please install it: pip install Pillow")
        # For PyInstaller, Pillow should be included as a dependency.
        # If this error occurs in a bundled app, the PyInstaller spec might be missing Pillow.
        sys.exit("Pillow is required for icon generation and UI. Application cannot start.")

    print("Initial setup checks complete.")


def main():
    """Main function to initialize and run the application."""
    print("Starting Sigma Pikachu application...")
    initial_setup_checks()

    # The config_manager and process_manager are already initialized as singletons
    # when their modules are imported.

    # Optionally, attempt to start Llama server or MCP servers on startup based on config?
    # For now, let's keep it manual via tray icon.
    # Example: if config_manager.get_config().get("autostart_llama_server", False):
    # process_manager.start_llama_server()

    # Start the proxy server
    # Use placeholder ports for now, matching those in ProcessManager
    PROXY_LISTEN_PORT = 8888
    PROXY_DOWNSTREAM_PORT = 9999 # This should be the port of the default model server
    process_manager.start_proxy_server('127.0.0.1', PROXY_LISTEN_PORT, '127.0.0.1', PROXY_DOWNSTREAM_PORT)


    # The ui_manager will set up and run the tray icon, which is blocking.
    try:
        ui_manager.setup_and_run_tray_icon()
    except Exception as e:
        print(f"An unhandled error occurred in the UI manager: {e}")
        # Attempt to clean up any running servers before exiting
        process_manager.stop_all_servers()
        sys.exit(1)

    # After ui_manager.app_icon.run() finishes (i.e., icon is stopped/quit)
    print("Sigma Pikachu application finished.")
    # All servers should have been stopped by the quit action in ui_manager.
    # process_manager.stop_all_servers() # Ensure cleanup, though quit_action should handle it.
    sys.exit(0)

if __name__ == "__main__":
    main()
