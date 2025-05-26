

Install
OpenMP_ROOT=$(brew --prefix)/opt/libomp FORCE_CMAKE="1" CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DGGML_METAL=on" pip install --upgrade --verbose --force-reinstall --no-cache-dir --prefer-binary llama-cpp-python


 To add the OpenAI params to tools, use:
 http://localhost:9999/v1

 TODO:
 - Run the open-webui server `WEBUI_AUTH=False open-webui server`
 - Switch to using llama-cpp process directly
 - More MCPs:
    - https://github.com/pashpashpash/mcp-taskmanager
    - https://github.com/pashpashpash/mcp-webresearch
    - https://github.com/Saik0s/mcp-browser-use
    - https://github.com/upstash/context7
 - Fix the config to not load the MCP and others when pushing to llama

 To run as a service:
 - edit the sigma_pikachu.plist for the correct paths
 - cp ./sigma_pikachu.plist ~/Library/LaunchAgents
 - launchctl load ~/Library/LaunchAgents/sigma_pikachu.plist   
 - launchctl print gui/501/com.example.SigmaPikachu
 - .. to ensure it's running

 To benchmark the system:
 - run python tools/benchmark_threads.py

 ## Build
 - Ollama
 brew upgrade ollama && cp /opt/homebrew/opt/ollama/bin/ollama ./sigma_pikachu/bin/


