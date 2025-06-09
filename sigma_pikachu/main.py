import os
import sys
import logging

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
from sigma_pikachu.constants import CONFIG_FILE, USER_CONFIG_DIR, MCP_LOGS_DIR, LLAMA_SERVER_LOG_FILE, MAIN_APP_LOG_FILE, USER_LOGS_DIR
from sigma_pikachu.settings.config_manager import config_manager # Singleton instance
from sigma_pikachu.services.process_manager import process_manager # Singleton instance
from sigma_pikachu.ui_manager import ui_manager # Singleton instance
from PIL import Image, ImageDraw # Check for Pillow
# except ImportError as e:
#     print(f"Error importing application modules: {e}")
#     print("Please ensure all modules (constants, config_manager, process_manager, ui_manager) are in the 'sigma_pikachu' directory.")
#     print("If running from source, ensure the project root is in PYTHONPATH or you are running from the project root e.g. python -m sigma_pikachu.main")
#     sys.exit(1)

# Add our local bin directory to PATH for subprocess calls, conditionally based on frozen status
if getattr(sys, 'frozen', False):
    # When frozen, the 'bin' directory is bundled under 'sigma_pikachu/bin' relative to _MEIPASS
    bin_path = os.path.join(sys._MEIPASS, "sigma_pikachu", "bin")
    lib_path = os.path.join(sys._MEIPASS, "sigma_pikachu", "lib") # Assuming lib is also under sigma_pikachu
else:
    # When running as a script, 'bin' is a sibling directory to main.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(script_dir, "bin")
    lib_path = os.path.join(script_dir, "lib")

os.environ["PATH"] = os.pathsep.join([bin_path, os.environ["PATH"]])
os.environ["DYLD_LIBRARY_PATH"] = os.pathsep.join([lib_path, os.environ.get("DYLD_LIBRARY_PATH", "")])

def setup_logging():
    """Configure logging for the main application."""
    # Ensure log directory exists
    if not os.path.exists(USER_LOGS_DIR):
        os.makedirs(USER_LOGS_DIR, exist_ok=True)
    
    # Create file handler with immediate flushing
    file_handler = logging.FileHandler(MAIN_APP_LOG_FILE, mode='a')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized - main application log file: %s", MAIN_APP_LOG_FILE)
    
    # Force flush
    file_handler.flush()
    
    return logger

def initial_setup_checks(logger):
    """Perform initial checks, create directories, and ensure Pillow is available."""
    logger.info("User config directory: %s", USER_CONFIG_DIR)
    logger.info("Application config file: %s", CONFIG_FILE)
    logger.info("Llama server log file: %s", LLAMA_SERVER_LOG_FILE)
    logger.info("Main app log file: %s", MAIN_APP_LOG_FILE)
    logger.info("MCP logs directory: %s", MCP_LOGS_DIR)

    # Ensure directories exist (constants.py should handle this, but double-check)
    if not os.path.exists(USER_CONFIG_DIR):
        os.makedirs(USER_CONFIG_DIR, exist_ok=True)
        logger.info("Created user config directory: %s", USER_CONFIG_DIR)
    
    if not os.path.exists(MCP_LOGS_DIR):
        os.makedirs(MCP_LOGS_DIR, exist_ok=True)
        logger.info("Created MCP logs directory: %s", MCP_LOGS_DIR)

    # Check for config file and create default if missing (config_manager handles this on init)
    _ = config_manager.get_config() # This will trigger creation if missing

    # Ensure Pillow is available (critical for pystray icon)
    try:
        from PIL import Image, ImageDraw
        logger.info("Pillow library found.")
    except ImportError:
        logger.critical("CRITICAL ERROR: Pillow (PIL) library not found.")
        logger.critical("Please install it: pip install Pillow")
        # For PyInstaller, Pillow should be included as a dependency.
        # If this error occurs in a bundled app, the PyInstaller spec might be missing Pillow.
        sys.exit("Pillow is required for icon generation and UI. Application cannot start.")

    logger.info("Initial setup checks complete.")


def main():
    """Main function to initialize and run the application."""
    # Setup logging first
    logger = setup_logging()
    logger.info("Starting Sigma Pikachu application...")
    
    initial_setup_checks(logger)

    # The config_manager and process_manager are already initialized as singletons
    # when their modules are imported.

    # Set the UI update callback in the process manager
    process_manager.set_ui_update_callback(ui_manager.request_menu_update)

    # Start all configured servers (model server, MCP servers, proxy)
    logger.info("Starting all configured servers...")
    process_manager.start_all_servers()

    # The ui_manager will set up and run the tray icon, which is blocking.
    try:
        logger.info("Setting up and running tray icon...")
        ui_manager.setup_and_run_tray_icon()
    except Exception as e:
        logger.error("An unhandled error occurred in the UI manager: %s", e, exc_info=True)
        # Attempt to clean up any running servers before exiting
        logger.info("Attempting to stop all servers before exit...")
        process_manager.stop_all_servers()
        sys.exit(1)

    # After ui_manager.app_icon.run() finishes (i.e., icon is stopped/quit)
    logger.info("Sigma Pikachu application finished.")
    # All servers should have been stopped by the quit action in ui_manager.
    # process_manager.stop_all_servers() # Ensure cleanup, though quit_action should handle it.
    sys.exit(0)

if __name__ == "__main__":
    main()
