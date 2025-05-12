import sys
import os
import struct
import mmap
import psutil
import platform
import multiprocessing
from pathlib import Path
import gguf

GGUF_MAGIC = b'GGUF'
GGUF_VERSION_SUPPORTED = {1, 2, 3}

# --- GGUF reader ---
class GGUFReader:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.meta = {}
        self._parse()

    def _parse(self):
        with open(self.file_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            magic = mm.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError("Not a GGUF file")

            version = struct.unpack('<I', mm.read(4))[0]
            if version not in GGUF_VERSION_SUPPORTED:
                raise ValueError(f"Unsupported GGUF version: {version}")

            mm.seek(0)
            self.mm = mm

    def get_metadata(self):
        content = self.mm[:]
        self.meta = {}
        for tag in [b'general.name', b'general.quantization_version', b'general.architecture', b'tokenizer.ggml.model']:
            idx = content.find(tag)
            if idx != -1:
                end = content.find(b'\x00', idx)
                value_start = end + 1
                value_end = content.find(b'\x00', value_start)
                self.meta[tag.decode()] = content[value_start:value_end].decode(errors='ignore')
        return self.meta

# --- Heuristics ---
def guess_model_size_from_name(name):
    name = name.lower()
    if '7b' in name:
        return '7B'
    elif '8b' in name:
        return '8B'
    elif '13b' in name:
        return '13B'
    elif '14b' in name:
        return '14B'
    elif '30b' in name:
        return '30B'
    elif '34b' in name:
        return '34B'
    elif '65b' in name:
        return '65B'
    else:
        return None

def detect_all_quants_from_name(name):
    name = name.upper()
    known_quants = ["Q2_K", "Q3_K", "Q3_K_S", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]
    return [qt for qt in known_quants if qt in name]

def guess_quant_type_from_name(name):
    all_quants = detect_all_quants_from_name(name)
    return all_quants[0] if all_quants else "unknown"

def estimate_n_gpu_layers(model_size, quant_type):
    base_layers = {
        '7B': 30,
        '8B': 28,
        '13B': 20,
        '14B': 18,
        '30B': 10,
        '34B': 8,
        '65B': 4
    }
    q_scale = {
        'Q2_K': 1.4,
        'Q3_K': 1.3,
        'Q3_K_S': 1.2,
        'Q4_0': 1.0,
        'Q4_K_S': 1.0,
        'Q4_K_M': 0.9,
        'Q5_K_M': 0.8,
        'Q6_K': 0.6,
        'Q8_0': 0.4,
        'F16': 0.2
    }
    if model_size in base_layers and quant_type in q_scale:
        return int(base_layers[model_size] * q_scale[quant_type])
    return None

def recommend_threading_options():
    logical_cores = multiprocessing.cpu_count()
    physical_cores = psutil.cpu_count(logical=False)
    cpu_info = platform.processor()
    mac_info = platform.mac_ver()[0] if sys.platform == "darwin" else None

    recommended_threads = min(logical_cores, 8)
    batch_size = 512 if logical_cores > 8 else 256

    return {
        "logical_cores": logical_cores,
        "physical_cores": physical_cores,
        "recommended_threads": recommended_threads,
        "recommended_batch_size": batch_size,
        "cpu_info": cpu_info,
        "mac_info": mac_info
    }

def estimate_ctx_size(model_size):
    ctx_defaults = {
        '7B': 4096,
        '8B': 4096,
        '13B': 4096,
        '14B': 4096,
        '30B': 3276,
        '34B': 2048,
        '65B': 2048
    }
    return ctx_defaults.get(model_size, 2048)



def main():
    import argparse

    parser = argparse.ArgumentParser(description="GGUF Analyzer")
    parser.add_argument("model_path", help="Path to GGUF model file")
    parser.add_argument("--show-meta", action="store_true", help="Show all GGUF metadata")
    args = parser.parse_args()

    if args.show_meta:
        reader = gguf.GGUFReader(args.model_path)
        print("\nGGUF Metadata:")
        for k, v in reader.metadata.items():
            print(f"{k}: {v}")
        print()
        return

    model_path = args.model_path
    reader = GGUFReader(model_path)
    meta = reader.get_metadata()

    name = Path(model_path).name
    print(f"Model name: {name}")

    model_size = guess_model_size_from_name(name)
    quant_types = detect_all_quants_from_name(name)

    if not quant_types:
        print("Raw quantization info: unknown")
        quant_type = "unknown"
    else:
        print(f"Detected quantization types: {', '.join(quant_types)}")
        quant_type = quant_types[0]

    if len(quant_types) > 1:
        print("\u26a0\ufe0f  Warning: Multiple quant types detected. Hybrid quantization may affect GPU layer estimation.")
        print("   Using the first detected type for estimation only. Please adjust manually if needed.")

    if not model_size:
        print("Error: Unable to infer model size from model name.")
        sys.exit(1)

    n_gpu_layers = estimate_n_gpu_layers(model_size, quant_type)
    if n_gpu_layers is None:
        print("Warning: Could not estimate --n-gpu-layers for this model.")
    else:
        print(f"Recommended --n-gpu-layers: {n_gpu_layers}")

    threading = recommend_threading_options()
    ctx_size = estimate_ctx_size(model_size)

    print("\nThreading Recommendations:")
    for k, v in threading.items():
        print(f"{k}: {v}")

    print("\nSuggested llama.cpp flags:")
    print(f"--n-gpu-layers {n_gpu_layers if n_gpu_layers else 0} \\")
    print(f"--threads {threading['recommended_threads']} \\")
    print(f"--batch_size {threading['recommended_batch_size']} \\")
    print(f"--ctx-size {ctx_size}")

if __name__ == '__main__':
    main()
