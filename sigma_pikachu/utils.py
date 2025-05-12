import os
import sys
import subprocess
from .constants import LLAMA_SERVER_LOG_FILE, MCP_LOGS_DIR

def open_file_externally(file_path):
    """Opens the specified file using the system's default application."""
    try:
        if not os.path.exists(file_path):
            # Create an empty file if it doesn't exist, especially for logs
            if file_path.endswith(".log"):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as f:
                    f.write(f"Log file created at {file_path}.\n")
                print(f"Created empty log file: {file_path}")
            else:
                print(f"Error: File not found at {file_path}")
                return

        if sys.platform == "win32":
            os.startfile(file_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", file_path], check=True)
        else:  # linux variants
            subprocess.run(["xdg-open", file_path], check=True)
        print(f"Opened {file_path}")
    except Exception as e:
        print(f"Failed to open {file_path}: {e}")

def view_llama_server_logs():
    """Opens the main Llama server log file."""
    open_file_externally(LLAMA_SERVER_LOG_FILE)

def view_mcp_server_log(alias):
    """Opens the log file for a specific MCP server."""
    log_file_name = f"mcp_{alias.replace(' ', '_')}.log"
    log_file_path = os.path.join(MCP_LOGS_DIR, log_file_name)
    open_file_externally(log_file_path)

def view_mcp_logs_directory():
    """Opens the directory containing all MCP server logs."""
    if not os.path.exists(MCP_LOGS_DIR):
        os.makedirs(MCP_LOGS_DIR, exist_ok=True)
        print(f"Created MCP logs directory: {MCP_LOGS_DIR}")
    
    open_file_externally(MCP_LOGS_DIR)

if __name__ == '__main__':
    # Basic test for utility functions
    print("Utils Test")
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

    print(f"Attempting to open Llama log: {LLAMA_SERVER_LOG_FILE}")
    # view_llama_server_logs() # This would open the file

    print(f"Attempting to open MCP log for '{dummy_mcp_alias}': {dummy_mcp_log_path}")
    # view_mcp_server_log(dummy_mcp_alias) # This would open the file
    
    print(f"Attempting to open MCP logs directory: {MCP_LOGS_DIR}")
    # view_mcp_logs_directory() # This would open the directory

    print("Utils test finished. Check if files/directories opened (if uncommented).")
