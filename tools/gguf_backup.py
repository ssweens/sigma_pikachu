import os
import sys
import psutil
import platform
from gguf import GGUFReader

# ---------------------------------------------
# 🧠 Utility functions
# ---------------------------------------------

def resolve_field(fields, key):
    val = fields.get(key)
    if val and hasattr(val, "data") and len(val.data) > 0:
        return val.data[0]
    return None

def parse_gguf_metadata(path):
    reader = GGUFReader(path)
    return reader.fields

def print_all_metadata(fields):
    print("\n📋 Full GGUF Metadata:\n")
    for key, value in sorted(fields.items()):
        val = value.data[0] if hasattr(value, "data") else value
        print(f"{key}: {val}")

def guess_model_size_from_name(name):
    name = name.lower()
    for size in ["7b", "13b", "14b", "30b", "65b"]:
        if size in name:
            return size.upper()
    raise ValueError("Unable to infer model size from model name.")

def map_quantization(qval):
    if isinstance(qval, str):
        qval = qval.upper()
        known = ["Q2_K", "Q4_0", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]
        for known_q in known:
            if known_q in qval:
                return known_q
    return "Q4_K"  # fallback

def get_unified_memory_gb():
    return round(psutil.virtual_memory().total / (1024**3))

def get_apple_gpu_memory_gb():
    if platform.system() != "Darwin":
        return None
    try:
        import subprocess
        result = subprocess.run(
            ["/usr/sbin/system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True
        )
        lines = result.stdout.splitlines()
        for line in lines:
            if "Unified Memory" in line:
                value = line.split(":")[-1].strip().upper().replace("GB", "").strip()
                return int(value)
    except Exception:
        pass
    return None

# ---------------------------------------------
# 📏 Estimation logic
# ---------------------------------------------

model_max_layers = {
    "7B": 32,
    "13B": 40,
    "14B": 40,  # assumed similar to 13B
    "30B": 60,
    "65B": 80,
}

layer_costs = {
    "7B":   {"Q2_K": 0.2, "Q4_0": 0.3, "Q4_K": 0.35, "Q5_K": 0.4, "Q6_K": 0.45, "Q8_0": 0.6},
    "13B":  {"Q2_K": 0.35, "Q4_0": 0.5, "Q4_K": 0.55, "Q5_K": 0.6, "Q6_K": 0.65, "Q8_0": 0.9},
    "14B":  {"Q2_K": 0.5, "Q4_0": 0.7, "Q4_K": 0.75, "Q5_K": 0.8, "Q6_K": 0.9, "Q8_0": 1.2},
    "30B":  {"Q2_K": 0.6, "Q4_0": 0.9, "Q4_K": 1.0, "Q5_K": 1.1, "Q6_K": 1.2, "Q8_0": 1.5},
    "65B":  {"Q2_K": 0.8, "Q4_0": 1.2, "Q4_K": 1.4, "Q5_K": 1.6, "Q6_K": 1.8, "Q8_0": 2.2},
}

def estimate_gpu_layers(model_size, gpu_mem_gb, quant):
    if model_size not in layer_costs or quant not in layer_costs[model_size]:
        raise ValueError(f"Unsupported model size {model_size} or quantization {quant}")
    layer_cost = layer_costs[model_size][quant]
    max_layers = model_max_layers[model_size]
    max_fit = int(gpu_mem_gb // layer_cost)
    return min(max_fit, max_layers)

def recommend_llama_cpp_params(model_size, quant, n_gpu_layers, n_threads):
    return {
        "n_gpu_layers": n_gpu_layers,
        "n_threads": n_threads,
        "n_batch": 512,
        "n_ctx": 4096,
        "rope_freq_base": 10000 if model_size in ["7B", "13B"] else 50000
    }

# ---------------------------------------------
# 🚀 Main entry
# ---------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python gguf_analyzer.py /path/to/model.gguf [--show-meta]")
        return

    model_path = sys.argv[1]
    show_metadata = "--show-meta" in sys.argv

    if not os.path.isfile(model_path):
        print("❌ Model file not found.")
        return

    fields = parse_gguf_metadata(model_path)

    name = resolve_field(fields, "general.name")
    if not isinstance(name, str) or len(name.strip()) == 0:
        name = os.path.basename(model_path)

    quant_raw = resolve_field(fields, "quantization.version") or resolve_field(fields, "tensor_data_layout") or "unknown"

    print(f"Model name: {name}")
    print(f"Raw quantization info: {quant_raw}")
    if quant_raw == "unknown":
        print("⚠️  Warning: Quantization format could not be determined from metadata.")

    try:
        model_size = guess_model_size_from_name(name)
        quant = map_quantization(quant_raw)
    except ValueError as e:
        print(f"Error: {e}")
        return

    unified_mem = get_unified_memory_gb()
    gpu_mem = get_apple_gpu_memory_gb() or unified_mem
    total_cores = psutil.cpu_count(logical=False)

    print(f"\nDetected unified memory: {unified_mem} GB")
    print(f"Estimated usable GPU memory: {gpu_mem} GB")
    print(f"Model size: {model_size}, Quantization: {quant}")
    print(f"Physical CPU cores: {total_cores}")

    try:
        n_gpu_layers = estimate_gpu_layers(model_size, gpu_mem, quant)
        params = recommend_llama_cpp_params(model_size, quant, n_gpu_layers, total_cores)

        print(f"\n✅ Recommended llama.cpp launch parameters:\n")
        for key, val in params.items():
            print(f"--{key.replace('_', '-')} {val}")

    except ValueError as e:
        print(f"Estimation error: {e}")

    if show_metadata:
        print_all_metadata(fields)

if __name__ == "__main__":
    main()
