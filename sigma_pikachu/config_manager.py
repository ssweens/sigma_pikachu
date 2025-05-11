import yaml
import json
import os
import sys
import subprocess
from .constants import CONFIG_FILE, DEFAULT_HOST, DEFAULT_LLAMA_PORT

class ConfigManager:
    def __init__(self):
        self.config_file_path = CONFIG_FILE
        self.config = self._load_config()

    def _load_config(self):
        """Loads the server configuration from config.yaml."""
        if not os.path.exists(self.config_file_path):
            print(f"Error: Configuration file {self.config_file_path} not found.")
            # Create a minimal default config if it doesn't exist
            default_cfg = {
                "host": DEFAULT_HOST,
                "port": DEFAULT_LLAMA_PORT,
                "models": [],
                "mcp_servers": []
            }
            try:
                with open(self.config_file_path, 'w') as f:
                    yaml.dump(default_cfg, f, sort_keys=False)
                print(f"Created a default configuration file at {self.config_file_path}")
                return default_cfg
            except Exception as e:
                print(f"Error creating default configuration file: {e}")
                return None


        try:
            with open(self.config_file_path, 'r') as f:
                config_data = yaml.safe_load(f)
            if config_data is None: # Handle empty or invalid YAML
                print(f"Warning: {self.config_file_path} is empty or invalid. Using defaults.")
                return {
                    "host": DEFAULT_HOST,
                    "port": DEFAULT_LLAMA_PORT,
                    "models": [],
                    "mcp_servers": []
                }
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
            self.config = {
                "host": DEFAULT_HOST,
                "port": DEFAULT_LLAMA_PORT,
                "models": [],
                "mcp_servers": []
            }
        return self.config is not None


    def get_config(self):
        """Returns the current configuration."""
        if self.config is None: # Ensure there's always some config to return
             return {
                "host": DEFAULT_HOST,
                "port": DEFAULT_LLAMA_PORT,
                "models": [],
                "mcp_servers": []
            }
        return self.config

    def get_llama_host_port(self):
        """Gets host and port for the Llama server."""
        cfg = self.get_config()
        host = cfg.get("host", DEFAULT_HOST)
        port = cfg.get("port", DEFAULT_LLAMA_PORT)
        return host, port

    def get_llama_models(self):
        """Gets the list of Llama models."""
        return self.get_config().get("models", [])

    def get_mcp_servers(self):
        """Gets the list of MCP server configurations."""
        return self.get_config().get("mcp_servers", [])

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
            'host': '127.0.0.1',
            'port': 8080,
            'models': [{'model_alias': 'TestModel', 'model': '/path/to/model.gguf'}],
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
