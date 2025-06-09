import os
import sys
import subprocess
import logging
from .constants import LLAMA_SERVER_LOG_FILE, MCP_LOGS_DIR, MAIN_APP_LOG_FILE

logger = logging.getLogger(__name__)

def open_file_externally(file_path):
    """Opens the specified file using the system's default application."""
    try:
        if not os.path.exists(file_path):
            # Create an empty file if it doesn't exist, especially for logs
            if file_path.endswith(".log"):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as f:
                    f.write(f"Log file created at {file_path}.\n")
                logger.info("Created empty log file: %s", file_path)
            else:
                logger.error("Error: File not found at %s", file_path)
                return

        if sys.platform == "win32":
            if file_path.endswith(".log"):
                # For Windows, opening in a new command prompt and tailing is complex.
                # We'll stick to os.startfile for logs, which might open in Notepad or default log viewer.
                # A more advanced solution might involve powershell:
                # subprocess.run(['powershell', '-Command', f'Start-Process cmd -ArgumentList "/k tail -f -n 40 {file_path}"'], check=True)
                # For now, keeping it simple.
                os.startfile(file_path)
            else:
                os.startfile(file_path)
        elif sys.platform == "darwin":
            if file_path.endswith(".log"):
                # Open log files in a new Terminal window with tail
                subprocess.run([
                    'osascript',
                    '-e', 'tell app "Terminal" to do script "clear; tail -f -n 40 ' + file_path.replace('"', '\\"') + '"'
                ], check=True)
            else:
                subprocess.run(["open", file_path], check=True)
        else:  # linux variants
            if file_path.endswith(".log"):
                # Attempt to open in a new terminal with tail for Linux
                # This depends on having a common terminal emulator like gnome-terminal, xterm, etc.
                # Using 'x-terminal-emulator' which is a Debian alternatives system link
                try:
                    subprocess.run([
                        'x-terminal-emulator', '-e', f'sh -c "tail -f -n 40 {file_path}; exec sh"'
                    ], check=True)
                except FileNotFoundError:
                    # Fallback if x-terminal-emulator is not found or fails
                    logger.warning("Could not open terminal with tail for %s. Opening with xdg-open.", file_path)
                    subprocess.run(["xdg-open", file_path], check=True)
                except Exception as e:
                    logger.warning("Failed to open terminal with tail for %s: %s. Opening with xdg-open.", file_path, e)
                    subprocess.run(["xdg-open", file_path], check=True)
            else:
                subprocess.run(["xdg-open", file_path], check=True)
        logger.info("Opened %s", file_path)
    except Exception as e:
        logger.error("Failed to open %s: %s", file_path, e)

def view_llama_server_logs():
    """Opens the main Llama server log file."""
    open_file_externally(LLAMA_SERVER_LOG_FILE)

def view_main_app_logs():
    """Opens the main application log file."""
    open_file_externally(MAIN_APP_LOG_FILE)

def view_mcp_server_log(alias):
    """Opens the log file for a specific MCP server."""
    log_file_name = f"mcp_{alias.replace(' ', '_')}.log"
    log_file_path = os.path.join(MCP_LOGS_DIR, log_file_name)
    open_file_externally(log_file_path)

def view_mcp_logs_directory():
    """Opens the directory containing all MCP server logs."""
    if not os.path.exists(MCP_LOGS_DIR):
        os.makedirs(MCP_LOGS_DIR, exist_ok=True)
        logger.info("Created MCP logs directory: %s", MCP_LOGS_DIR)
    
    open_file_externally(MCP_LOGS_DIR)

if __name__ == '__main__':
    # Basic test for utility functions
    logger.info("Utils Test")
    # Ensure dummy log files and directories exist for testing
    if not os.path.exists(LLAMA_SERVER_LOG_FILE):
        os.makedirs(os.path.dirname(LLAMA_SERVER_LOG_FILE), exist_ok=True)
        with open(LLAMA_SERVER_LOG_FILE, 'w') as f:
            f.write("Dummy Llama server log.\n")
    
    os.makedirs(MCP_LOGS_DIR, exist_ok=True)
    dummy_mcp_alias = "TestServer"
    dummy_mcp_log_path = os.path.join(MCP_LOGS_DIR, f"mcp_{dummy_mcp_alias.replace(' ', '_')}.log")
    with open(dummy_mcp_log_path, 'w') as f:
        f.write(f"Dummy MCP log for {dummy_mcp_alias}.\n")

    logger.info("Attempting to open Llama log: %s", LLAMA_SERVER_LOG_FILE)
    # view_llama_server_logs() # This would open the file

    logger.info("Attempting to open MCP log for '%s': %s", dummy_mcp_alias, dummy_mcp_log_path)
    # view_mcp_server_log(dummy_mcp_alias) # This would open the file
    
    logger.info("Attempting to open MCP logs directory: %s", MCP_LOGS_DIR)
    # view_mcp_logs_directory() # This would open the directory

    logger.info("Utils test finished. Check if files/directories opened (if uncommented).")
