

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

