import yaml
import json
import os
import sys
import subprocess
import time # Added for retry delay
import logging
from ..constants import CONFIG_FILE, DEFAULT_HOST, DEFAULT_LLAMA_PORT

import requests
try:
    import tomli
except ImportError:
    tomli = None

try:
    import jinja2
except ImportError:
    jinja2 = None

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
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config()

    def _load_config(self):
        """Loads the server configuration from config.yaml."""
        if not os.path.exists(self.config_file_path):
            self.logger.error("Error: Configuration file %s not found.", self.config_file_path)
            # Create a minimal default config if it doesn't exist
            try:
                with open(self.config_file_path, 'w') as f:
                    yaml.dump(self.DEFAULT_CONFIG, f, sort_keys=False)
                self.logger.info("Created a default configuration file at %s", self.config_file_path)
                return self.DEFAULT_CONFIG.copy()
            except Exception as e:
                self.logger.error("Error creating default configuration file: %s", e)
                return None # Or consider returning self.DEFAULT_CONFIG.copy()


        try:
            with open(self.config_file_path, 'r') as f:
                config_data = yaml.safe_load(f)
            if config_data is None: # Handle empty or invalid YAML
                self.logger.warning("Warning: %s is empty or invalid. Using defaults.", self.config_file_path)
                return self.DEFAULT_CONFIG.copy()
            # Convert to JSON and back to ensure consistent object types (e.g., dicts not OrderedDicts)
            # although with safe_load, this is less of an issue.
            return json.loads(json.dumps(config_data))
        except FileNotFoundError:
            # This case should be handled by the os.path.exists check above,
            # but kept for robustness.
            self.logger.error("Error: %s not found (should not happen here).", self.config_file_path)
            return None
        except yaml.YAMLError as e:
            self.logger.error("Error parsing YAML in %s: %s", self.config_file_path, e)
            return None
        except Exception as e: # Catch any other loading errors
            self.logger.error("An unexpected error occurred while loading %s: %s", self.config_file_path, e)
            return None

    def reload_config(self):
        """Reloads the configuration from the file."""
        self.logger.info("Reloading configuration...")
        self.config = self._load_config()
        if self.config is None:
            self.logger.error("Failed to reload configuration. Previous configuration might still be in use if not overwritten.")
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
        Gets the list of corral models from the configured corral config file (TOML format).
        Returns an empty list if corral is not configured or config file cannot be read.
        """
        main_config = self.get_config()
        llama_swap_config_path = main_config.get("corral", {}).get("config_file")

        if not llama_swap_config_path:
            self.logger.warning("Warning: 'corral.config_file' not specified in config.yaml.")
            return []

        if not os.path.exists(llama_swap_config_path):
            self.logger.error("Error: corral configuration file %s not found.", llama_swap_config_path)
            return []

        # Check if tomli is available
        if tomli is None:
            self.logger.error("Error: tomli library not available for parsing TOML files.")
            return []

        try:
            # Read the file as text first to handle Jinja2 templates
            with open(llama_swap_config_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            # Check if the file contains Jinja2 templates
            if jinja2 is not None and ('{{' in raw_content or '{%' in raw_content):
                # Process Jinja2 templates
                try:
                    template = jinja2.Template(raw_content)
                    processed_content = template.render()
                    # Parse the processed content as TOML
                    llama_swap_config_data = tomli.loads(processed_content)
                except jinja2.TemplateError as e:
                    self.logger.error("Error processing Jinja2 templates in %s: %s", llama_swap_config_path, e)
                    return []
            else:
                # Parse directly as TOML
                llama_swap_config_data = tomli.loads(raw_content)

        except FileNotFoundError:
            self.logger.error("Error: corral configuration file %s not found during read.", llama_swap_config_path)
            return []
        except tomli.TOMLDecodeError as e:
            self.logger.error("Error parsing TOML in %s: %s", llama_swap_config_path, e)
            return []
        except Exception as e:
            self.logger.error("An unexpected error occurred while loading %s: %s", llama_swap_config_path, e)
            return []

        # If data was loaded successfully, process it
        if llama_swap_config_data is None:
            self.logger.warning("Warning: %s is empty or invalid.", llama_swap_config_path)
            return [] # Return empty list for empty/invalid config

        # The 'models' key in corral TOML config is a dictionary of model configurations.
        # We need to extract the keys (model aliases) from this dictionary.
        models_dict = llama_swap_config_data.get("models", {})
        if not isinstance(models_dict, dict):
             self.logger.warning("Warning: 'models' key in %s is not a dictionary.", llama_swap_config_path)
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
                    self.logger.warning("Ollama API at %s returned status 503 (Attempt %d/%d). Retrying in %ds...", url, attempt + 1, max_retries, retry_delay_seconds)
                else:
                    self.logger.error("Ollama API request failed. URL: %s, Status: %s, Response: %s", url, r.status_code, r.text[:100])
                    return [] # Non-retryable error or final attempt failed
            except (RequestException, ConnectionError, TimeoutError) as e:
                self.logger.error("Could not connect to Ollama server at %s (Attempt %d/%d): %s", url, attempt + 1, max_retries, e)
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay_seconds)
            else:
                self.logger.error("Failed to fetch Ollama models from %s after %d attempts.", url, max_retries)
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
        self.logger.info("Opened %s", self.config_file_path)
    except Exception as e:
        self.logger.error("Failed to open %s: %s", self.config_file_path, e)

# For direct testing or use as a singleton instance
config_manager = ConfigManager()

if __name__ == '__main__':
    # Example usage:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Current config file path: %s", CONFIG_FILE)
    if not os.path.exists(CONFIG_FILE):
        logger.info("Config file %s does not exist. Please create it.", CONFIG_FILE)
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
        logger.info("Created dummy config file at %s", CONFIG_FILE)
        config_manager.reload_config()


    logger.info("Llama Server Host/Port: %s", config_manager.get_llama_host_port())
    logger.info("Llama Models: %s", config_manager.get_llama_models())
    logger.info("MCP Servers: %s", config_manager.get_mcp_servers())
    
    # Test reloading (e.g., after manually editing the config.yaml)
    # input("Edit config.yaml and press Enter to reload and see changes...")
    # config_manager.reload_config()
    # logger.info("Reloaded MCP Servers: %s", config_manager.get_mcp_servers())
