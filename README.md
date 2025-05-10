

Install
OpenMP_ROOT=$(brew --prefix)/opt/libomp FORCE_CMAKE="1" CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DGGML_METAL=on" pip install --upgrade --verbose --force-reinstall --no-cache-dir --prefer-binary llama-cpp-python


 To add the OpenAI params to tools, use:
 http://localhost:9999/v1
