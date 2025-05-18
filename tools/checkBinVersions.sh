echo "llama-server"
DYLD_LIBRARY_PATH=~/src/sigma_pikachu/sigma_pikachu/lib: ./sigma_pikachu/bin/llama-server --version
echo "llama-swap"
DYLD_LIBRARY_PATH=~/src/sigma_pikachu/sigma_pikachu/lib: ./sigma_pikachu/bin/llama-swap --version
echo "ollama"
DYLD_LIBRARY_PATH=~/src/sigma_pikachu/sigma_pikachu/lib: ./sigma_pikachu/bin/ollama --version
