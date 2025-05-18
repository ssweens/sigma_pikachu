import yaml
import json
import os
import sys
import subprocess
import time # Added for retry delay
from ..constants import CONFIG_FILE, DEFAULT_HOST, DEFAULT_LLAMA_PORT

import requests

class ConfigManager:
    DEFAULT_CONFIG = {
        "llama": {
            "host": DEFAULT_HOST,
            "port": DEFAULT_LLAMA_PORT,
            "models": []
        },
        "mcp_servers": []
    }

    def __init__(self):
        self.config_file_path = CONFIG_FILE
        self.config = self._load_config()

    def _load_config(self):
        """Loads the server configuration from config.yaml."""
        if not os.path.exists(self.config_file_path):
            print(f"Error: Configuration file {self.config_file_path} not found.")
            # Create a minimal default config if it doesn't exist
            try:
                with open(self.config_file_path, 'w') as f:
                    yaml.dump(self.DEFAULT_CONFIG, f, sort_keys=False)
                print(f"Created a default configuration file at {self.config_file_path}")
                return self.DEFAULT_CONFIG.copy()
            except Exception as e:
                print(f"Error creating default configuration file: {e}")
                return None # Or consider returning self.DEFAULT_CONFIG.copy()


        try:
            with open(self.config_file_path, 'r') as f:
                config_data = yaml.safe_load(f)
            if config_data is None: # Handle empty or invalid YAML
                print(f"Warning: {self.config_file_path} is empty or invalid. Using defaults.")
                return self.DEFAULT_CONFIG.copy()
            # Convert to JSON and back to ensure consistent object types (e.g., dicts not OrderedDicts)
            # although with safe_load, this is less of an issue.
            return json.loads(json.dumps(config_data))
        except FileNotFoundError:
            # This case should be handled by the os.path.exists check above,
            # but kept for robustness.
            print(f"Error: {self.config_file_path} not found (should not happen here).")
            return None
        except yaml.YAMLError as e:
            print(f"Error parsing YAML in {self.config_file_path}: {e}")
            return None
        except Exception as e: # Catch any other loading errors
            print(f"An unexpected error occurred while loading {self.config_file_path}: {e}")
            return None

    def reload_config(self):
        """Reloads the configuration from the file."""
        print("Reloading configuration...")
        self.config = self._load_config()
        if self.config is None:
            print("Failed to reload configuration. Previous configuration might still be in use if not overwritten.")
            # Potentially fall back to a known good state or a minimal default
            self.config = self.DEFAULT_CONFIG.copy()
        return self.config is not None


    def get_config(self):
        """Returns the current configuration."""
        if self.config is None: # Ensure there's always some config to return
             return self.DEFAULT_CONFIG.copy()
        return self.config

    def get_llama_host_port(self):
        """Gets host and port for the Llama server."""
        llama_config = self.get_config().get("llama", {})
        host = llama_config.get("host", DEFAULT_HOST)
        port = llama_config.get("port", DEFAULT_LLAMA_PORT)
        return host, port

    def get_llama_models(self):
        """Gets the list of Llama models."""
        llama_config = self.get_config().get("llama", {})
        return llama_config.get("models", [])

    def get_mcp_servers(self):
        """Gets the list of MCP server configurations."""
        return self.get_config().get("mcp_servers", [])

    def get_llama_swap_models(self):
        """
        Gets the list of Llama-swap models from the configured llama-swap config file.
        Returns an empty list if llama_swap is not configured or config file cannot be read.
        """
        main_config = self.get_config()
        llama_swap_config_path = main_config.get("llama_swap", {}).get("config_file")

        if not llama_swap_config_path:
            print("Warning: 'llama_swap.config_file' not specified in config.yaml.")
            return []

        if not os.path.exists(llama_swap_config_path):
            print(f"Error: Llama-swap configuration file {llama_swap_config_path} not found.")
            return []

        try:
            with open(llama_swap_config_path, 'r') as f:
                llama_swap_config_data = yaml.safe_load(f)

        except FileNotFoundError:
            print(f"Error: Llama-swap configuration file {llama_swap_config_path} not found during read.")
            return []
        except yaml.YAMLError as e:
            print(f"Error parsing YAML in {llama_swap_config_path}: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred while loading {llama_swap_config_path}: {e}")
            return []

        # If data was loaded successfully, process it
        if llama_swap_config_data is None:
            print(f"Warning: {llama_swap_config_path} is empty or invalid.")
            return [] # Return empty list for empty/invalid config

        # The 'models' key in llama-swap config is a dictionary, not a list.
        # We need to extract the keys (model aliases) from this dictionary.
        models_dict = llama_swap_config_data.get("models", {})
        if not isinstance(models_dict, dict):
             print(f"Warning: 'models' key in {llama_swap_config_path} is not a dictionary.")
             return [] # Return empty list if 'models' is not a dict

        # Return a list of model aliases (the keys of the models dictionary)
        # We can represent each model by its alias string for the UI menu.
        model_aliases = [{"model_alias": alias} for alias in models_dict.keys()]

        return model_aliases
    
    def get_ollama_models(self):
        """Gets the list of OLLAMA models by making attempts to the API with retries."""
        from requests.exceptions import RequestException

        # Ensure the process_manager is available to check if ollama is even supposed to be running
        # This import needs to be here to avoid circular dependencies at the module level
        from ..services.process_manager import process_manager

        if not process_manager.is_ollama_server_running():
            # print("Ollama server is not reported as running by ProcessManager. Skipping model fetch.")
            return []

        current_config = self.get_config()
        ollama_listen_address = current_config.get("ollama", {}).get("listen", "127.0.0.1:11434") # Default to common Ollama port
        
        if ollama_listen_address.startswith(":"):
            ollama_host = "127.0.0.1"
            ollama_port = ollama_listen_address[1:]
        else:
            parts = ollama_listen_address.split(":")
            ollama_host = parts[0] if parts[0] else "127.0.0.1"
            ollama_port = parts[1] if len(parts) > 1 else "11434"

        if ollama_host == "0.0.0.0":
            ollama_host = "127.0.0.1"

        # The Ollama API endpoint for listing local models is typically /api/tags or /v1/models
        # Using /api/tags as it's commonly seen in Ollama examples for local model listing.
        # The official API docs suggest /api/tags for listing local models.
        # The /v1/models endpoint might be for a different purpose or newer API versions.
        # Let's stick to /api/tags for now as per the original simpler version,
        # but ensure the response parsing matches its structure.
        url = f"http://{ollama_host}:{ollama_port}/api/tags"
        
        max_retries = 5
        retry_delay_seconds = 1
        request_timeout_seconds = 2

        for attempt in range(max_retries):
            try:
                r = requests.get(url, timeout=request_timeout_seconds)
                if r.status_code == 200:
                    json_response = r.json()
                    # Ollama /api/tags returns a list of models directly under a "models" key.
                    # Each model object has a "name" field (e.g., "llama2:latest").
                    return [{"model_alias": model["name"]} for model in json_response.get("models", [])]
                elif r.status_code == 503: # Service Unavailable
                    print(f"Ollama API at {url} returned status 503 (Attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay_seconds}s...")
                else:
                    print(f"Ollama API request failed. URL: {url}, Status: {r.status_code}, Response: {r.text[:100]}")
                    return [] # Non-retryable error or final attempt failed
            except (RequestException, ConnectionError, TimeoutError) as e:
                print(f"Could not connect to Ollama server at {url} (Attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay_seconds)
            else:
                print(f"Failed to fetch Ollama models from {url} after {max_retries} attempts.")
                return []
        
        return [] # Should be unreachable if loop logic is correct, but as a fallback.


def open_config_file_externally(self):
    """Opens the config.yaml file in the default system editor."""
    try:
        if sys.platform == "win32":
            os.startfile(self.config_file_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", self.config_file_path], check=True)
        else: # linux variants
            subprocess.run(["xdg-open", self.config_file_path], check=True)
        print(f"Opened {self.config_file_path}")
    except Exception as e:
        print(f"Failed to open {self.config_file_path}: {e}")

# For direct testing or use as a singleton instance
config_manager = ConfigManager()

if __name__ == '__main__':
    # Example usage:
    print("Current config file path:", CONFIG_FILE)
    if not os.path.exists(CONFIG_FILE):
        print(f"Config file {CONFIG_FILE} does not exist. Please create it.")
        # Create a dummy config for testing if it doesn't exist
        dummy_config = {
            "llama": {
                'host': '127.0.0.1',
                'port': 8080,
                'models': [{'model_alias': 'TestModel', 'model': '/path/to/model.gguf'}]
            },
            'mcp_servers': [{'alias': 'TestMCP', 'command': 'echo "MCP Test"', 'enabled': True}]
        }
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(dummy_config, f)
        print(f"Created dummy config file at {CONFIG_FILE}")
        config_manager.reload_config()


    print("Llama Server Host/Port:", config_manager.get_llama_host_port())
    print("Llama Models:", config_manager.get_llama_models())
    print("MCP Servers:", config_manager.get_mcp_servers())
    
    # Test reloading (e.g., after manually editing the config.yaml)
    # input("Edit config.yaml and press Enter to reload and see changes...")
    # config_manager.reload_config()
    # print("Reloaded MCP Servers:", config_manager.get_mcp_servers())
