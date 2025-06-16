import os
import sys

# Determine where to look for external resources (config, logs, icons)
if getattr(sys, 'frozen', False):
    # Running as PyInstaller bundle: resources live in the temporary extraction directory
    RESOURCE_DIR = sys._MEIPASS
    # For PyInstaller, the executable is typically one level above the _MEIPASS dir
    # or in a known location relative to it.
    # Assuming the executable is in a directory like 'dist/app_name/'
    # and we want logs/config relative to that.
    # This might need adjustment based on the actual PyInstaller setup.
    # For now, let's assume EXTERNAL_DIR is where the executable is.
    EXTERNAL_DIR = os.path.dirname(sys.executable)
else:
    # Running as script: resources live beside the main script's directory
    RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXTERNAL_DIR = os.path.abspath(os.path.join(RESOURCE_DIR, ".."))


# Config files live under user directory in .config folder
if sys.platform == "win32":
    USER_CONFIG_DIR = os.path.join(os.getenv('APPDATA'), 'sigma_pikachu')
    USER_LOGS_DIR = os.path.join(os.getenv('LOCALAPPDATA'), 'sigma_pikachu')
# elif sys.platform == "darwin": # macOS specific path
# USER_CONFIG_DIR = os.path.join(os.getenv('HOME'), 'Library', 'Application Support', 'sigma_pikachu')
else: # Linux and other Unix-like systems (including macOS if not using the specific path)
    USER_CONFIG_DIR = os.path.join(os.getenv('HOME'), '.config', 'sigma_pikachu')
    USER_LOGS_DIR = os.path.join(os.getenv('HOME'), 'Library', 'Logs', 'sigma_pikachu')

# Ensure the user config directory exists
if not os.path.exists(USER_CONFIG_DIR):
    os.makedirs(USER_CONFIG_DIR, exist_ok=True)

CONFIG_FILE = os.path.join(USER_CONFIG_DIR, "config.yaml")
LLAMA_SERVER_LOG_FILE = os.path.join(USER_LOGS_DIR, "server.log")
MAIN_APP_LOG_FILE = os.path.join(USER_LOGS_DIR, "main.log")

LLAMA_SERVER_CONFIG_FILE = os.path.join(USER_CONFIG_DIR, "llama.cpp-python_config.json")

LLAMA_SWAP_CMD = os.path.join(RESOURCE_DIR, "bin", "corral")
TOOLSHED_CMD = os.path.join(RESOURCE_DIR, "bin", "toolshed")

OLLAMA_CMD = "ollama"

MCP_LOGS_DIR = os.path.join(USER_LOGS_DIR, "mcp_logs")

# Ensure the MCP logs directory exists
if not os.path.exists(MCP_LOGS_DIR):
    os.makedirs(MCP_LOGS_DIR, exist_ok=True)

# Icon file is bundled
ICON_FILE = os.path.join(RESOURCE_DIR, "pik64x64w.png")

# Default values
DEFAULT_HOST = "127.0.0.1"
DEFAULT_LLAMA_PORT = 8000 # Default llama-cpp-python port
