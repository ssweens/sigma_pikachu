import pystray
from PIL import Image, ImageDraw
import subprocess
from io import StringIO
import os
import json
import threading
import sys
import webbrowser
import yaml

#import svgloader

# Determine where to look for external resources (config, logs, icons)
if getattr(sys, 'frozen', False):
    # Running as PyInstaller bundle: resources live in the temporary extraction directory
    RESOURCE_DIR = sys._MEIPASS
else:
    # Running as script: resources live beside this file
    RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

# Config and log files remain external and editable
if getattr(sys, 'frozen', False):
    EXTERNAL_DIR = os.path.dirname(sys.executable)
else:
    EXTERNAL_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"

CONFIG_FILE = os.path.join(EXTERNAL_DIR, "config.yaml")
LOG_FILE    = os.path.join(EXTERNAL_DIR, "server.log")

# Icon file is bundled
ICON_FILE   = os.path.join(RESOURCE_DIR, "pik64x64w.png")

server_process = None
app_icon = None
# app_exit_event removed

def get_config():
    """Loads the server configuration from config.yaml, converting to json."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            # # remove any comments from the JSON file
            # # json.load() does not support comments, so we need to strip them out
            # s = f.read()
            # s = '\n'.join([line for line in s.splitlines() if not line.strip().startswith('//')])
            # return json.load(StringIO(s))
            y = yaml.safe_load(f)
        if y is not None:
            return json.loads(json.dumps(y))
    except FileNotFoundError:
        print(f"Error: {CONFIG_FILE} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode {CONFIG_FILE}.")
        return None

def get_server_host_port():
    """Gets host and port from config."""
    config = get_config()
    if config:
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 8000) # Default llama-cpp-python port if not in config
        return host, port
    return "N/A", "N/A"

def is_server_running():
    """Checks if the server process is active."""
    return server_process is not None and server_process.poll() is None

# --- Dynamic Menu Item Generators ---

def get_status_item_text(item): # Add 'item' parameter
    host, port = get_server_host_port()
    if is_server_running():
        return f"Status: Running on {host}:{port}"
    return "Status: Server Stopped"

def get_server_toggle_item_text(item): # Add 'item' parameter
    if is_server_running():
        return "Stop Server"
    return "Start Server"

def toggle_server(icon, item):
    """Toggles the llama-cpp-python server on/off with debug logs."""
    print(f"toggle_server called; server running: {is_server_running()}")
    if is_server_running():
        print("toggle_server: stopping server")
        stop_server(icon, item)
    else:
        print("toggle_server: starting server")
        start_server(icon, item)

def get_models_menu():
    config = get_config()
    model_sub_items = []
    if config and config.get("models"):
        for model_info in config["models"]:
            alias = model_info.get("model_alias", "Unknown Model")
            model_sub_items.append(pystray.MenuItem(alias, None, enabled=False))
    if not model_sub_items:
        model_sub_items.append(pystray.MenuItem("No models in config", None, enabled=False))
    return pystray.Menu(*model_sub_items)

def get_models_item_visibility(item): # Add 'item' parameter
    return is_server_running()

# --- Server Actions ---

def update_menu_state_after_action():
    """Call this after start/stop server to ensure menu reflects new state."""
    if app_icon and hasattr(app_icon, 'update_menu'):
        app_icon.update_menu()
    # A short delay might sometimes be needed for UI to catch up on some systems
    # threading.Timer(0.1, lambda: app_icon.update_menu() if app_icon and hasattr(app_icon, 'update_menu') else None).start()


def start_server(icon=None, item=None): # Add default None for icon and item
    """Starts the llama-cpp-python server."""
    global server_process
    print("Starting server..")
    if is_server_running():
        print("Server is already running.")
        return

    if not os.path.exists(CONFIG_FILE):
        print(f"Error: {CONFIG_FILE} not found. Cannot start server.")
        return
    
    # Reload config to ensure it's up-to-date
    config = get_config()
    if config is None:
        print(f"Error: Could not load {CONFIG_FILE}. Cannot start server.")
        return

    if getattr(sys, 'frozen', False):
        python_executable = "python3"
    else:
        python_executable = sys.executable

    command = [
        python_executable,
        "-m", "llama_cpp.server",
        "--config_file", CONFIG_FILE
    ]
    
    print(f"Starting server with command: {' '.join(command)}")
    try:
        with open(LOG_FILE, 'a') as log:
            server_process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        print(f"Server started. PID: {server_process.pid}. Logging to {LOG_FILE}")
    except Exception as e:
        print(f"Failed to start server: {e}")
    update_menu_state_after_action()

def stop_server(icon, item):
    """Stops the llama-cpp-python server."""
    global server_process
    if not is_server_running():
        print("Server is not running.")
        return

    print(f"Stopping server (PID: {server_process.pid})...")
    try:
        server_process.terminate()
        server_process.wait(timeout=4)
        print("Server terminated.")
    except subprocess.TimeoutExpired:
        print("Server did not terminate gracefully, killing...")
        server_process.kill()
        server_process.wait()
        print("Server killed.")
    except Exception as e:
        print(f"Error stopping server: {e}")
    finally:
        server_process = None
    update_menu_state_after_action()

def open_config_file(icon, item):
    """Opens the config.json file."""
    try:
        if sys.platform == "win32":
            os.startfile(CONFIG_FILE)
        elif sys.platform == "darwin":
            subprocess.run(["open", CONFIG_FILE], check=True)
        else: # linux variants
            subprocess.run(["xdg-open", CONFIG_FILE], check=True)
        print(f"Opened {CONFIG_FILE}")
    except Exception as e:
        print(f"Failed to open {CONFIG_FILE}: {e}")

def view_logs(icon, item):
    """Opens the server.log file."""
    try:
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w') as f:
                f.write("Log file created.\n")
        
        if sys.platform == "win32":
            os.startfile(LOG_FILE)
        elif sys.platform == "darwin":
            subprocess.run(["open", LOG_FILE], check=True)
        else: # linux variants
            subprocess.run(["xdg-open", LOG_FILE], check=True)
        print(f"Opened {LOG_FILE}")
    except Exception as e:
        print(f"Failed to open {LOG_FILE}: {e}")

def quit_application(icon, item):
    """Stops the server and quits the application."""
    print("Quit requested.")
    if is_server_running():
        stop_server(icon, item) # This will call update_menu_state_after_action
    icon.stop()
    # app_exit_event.set() removed

def create_default_icon():
    """Creates a simple default PIL Image for the tray icon."""
    width = 64
    height = 64
    # Using a more distinct color for visibility, and transparent background
    color1 = (0, 0, 0, 0)  # Transparent background
    color2 = (70, 130, 180, 255) # Steel Blue, Opaque
    image = Image.new("RGBA", (width, height), color1)
    dc = ImageDraw.Draw(image)
    # Draw a simple shape, e.g., a filled circle
    dc.ellipse((8, 8, width - 8, height - 8), fill=color2)
    return image

def setup_tray_icon():
    global app_icon

    #image = load_svg_with_cairosvg('sigma_pikachu_svg.svg', output_format='png')

    try:
        image = Image.open(ICON_FILE)
        print(f"Loaded icon from {ICON_FILE}")
    except FileNotFoundError:
        print(f"{ICON_FILE} not found, using default icon.")
        image = create_default_icon()
    except Exception as e: # Catch more specific PIL errors if possible, e.g., UnidentifiedImageError
        print(f"Error loading {ICON_FILE}: {e}. Using default icon.")
        image = create_default_icon()

    menu_items = (
        pystray.MenuItem(get_status_item_text, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(get_server_toggle_item_text, toggle_server),
        pystray.MenuItem("Loaded Models", 
                         get_models_menu(), 
                         visible=get_models_item_visibility),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open config.json", open_config_file),
        pystray.MenuItem("View Server Logs", view_logs),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", quit_application)
    )

    app_icon = pystray.Icon("llama_server_tray", image, "Llama Server Control", menu_items)
    
    # Call update_menu_state_after_action once at the beginning to set initial menu state
    # Needs to be in a thread because app_icon isn't fully set up yet for update_menu
    # Or, pystray might handle initial state correctly with functional items.
    # Let's rely on pystray's initial call to the generator functions.
    # If issues persist, a threaded initial update_menu_state_after_action() might be needed.
    toggle_server(None, None) # Call to set initial state
    
    app_icon.run() # Reverted to blocking call

if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE):
        print(f"{CONFIG_FILE} not found. Please create it or ensure it's in the correct directory.")
        sys.exit(f"Error: {CONFIG_FILE} is missing. The application requires it to run.")
    
    # Ensure Pillow is available
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow library not found. Please install it: pip install Pillow")
        sys.exit("Pillow is required for icon generation.")

    setup_tray_icon() # This will now block until the icon is stopped.
    print("Application finished.") # This line will be reached after the tray icon quits.
