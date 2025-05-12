#!/usr/bin/env python3
import os
import sys
import psutil
import platform
import multiprocessing
import argparse
from pathlib import Path
import numpy as np # For handling numpy types returned by gguf
# Assuming GGMLArchitectureType exists or we handle it gracefully
from gguf import GGUFReader, GGMLQuantizationType, GGUFValueType 
try:
    from gguf import GGMLArchitectureType
except ImportError:
    GGMLArchitectureType = None # Placeholder if not available

# ---------------------------------------------
# 🧠 Utility functions
# ---------------------------------------------

def resolve_field_value(field_value_obj):
    """Helper to extract the actual value from a GGUF field's value object."""
    if field_value_obj is None:
        return None

    val = None
    # Newer gguf.py library might store parts directly or in a list
    if hasattr(field_value_obj, 'parts') and field_value_obj.parts:
        val = field_value_obj.parts[0]
    # Older or different structure might use .data
    elif hasattr(field_value_obj, 'data'):
        if isinstance(field_value_obj.data, list) and len(field_value_obj.data) > 0:
            val = field_value_obj.data[0]
        else:
            val = field_value_obj.data # Could be a direct value
    # If the object itself is a simple type (less likely for complex fields but possible)
    elif isinstance(field_value_obj, (str, int, float, bool, bytes)):
        val = field_value_obj
    
    if isinstance(val, bytes):
        try:
            return val.decode('utf-8')
        except UnicodeDecodeError:
            return val # Return raw bytes if not decodable
    return val

def resolve_field(fields, key_name):
    """Resolves a GGUF field by name and returns its processed value."""
    if not fields:
        return None
    
    field_value_obj = fields.get(key_name) # GGUFReader.fields keys are strings
    if field_value_obj is None:
        # Try case-insensitive match for common keys if direct match fails
        for k_str, v_obj in fields.items(): # k_str is the string key
            if k_str.lower() == key_name.lower():
                field_value_obj = v_obj
                break
        if field_value_obj is None:
            return None # Key not found
            
    value = resolve_field_value(field_value_obj) # field_value_obj is a GGUFValue instance

    # Handle numpy array of single element first
    if isinstance(value, np.ndarray) and value.size == 1:
        try:
            value = value.item() # Converts to Python scalar
        except ValueError: # If .item() fails for some reason on a size 1 array
            pass # Keep original value and let subsequent checks handle it or fail

    # If value is a list of length 1, extract the element, as enums should be single values.
    if isinstance(value, list) and len(value) == 1:
        value = value[0]

    # Convert numpy scalar types (like memmap of a single value, int32, uint32) to Python int/float etc.
    # This should catch cases where value was an np.generic scalar from the start,
    # or became one after the list/ndarray unwrap if the element itself was np.generic.
    if isinstance(value, np.generic):
        try:
            value = value.item() # Converts to Python scalar
        except AttributeError: # .item() might not exist for all np types if it's already scalar-like
            pass


    # Handle specific GGUF enums if the value is an integer.
    # GGUFValue instances have a .type attribute (GGUFValueType enum).
    # Enums in GGUF are often stored as UINT32.
    # Debug prints removed.
    # if key_name in ["general.architecture", "general.quantization_version", "general.file_type"]:
    #     # Temporarily resolve field_value_obj again just for this print, to avoid altering 'value' if it was already processed
    #     initial_val_for_debug = resolve_field_value(fields.get(key_name))
    #     if isinstance(initial_val_for_debug, np.ndarray) and initial_val_for_debug.size == 1:
    #         initial_val_for_debug = initial_val_for_debug.item()
    #     elif isinstance(initial_val_for_debug, list) and len(initial_val_for_debug) == 1:
    #         initial_val_for_debug = initial_val_for_debug[0]
    #
    #     print(f"DEBUG: key={key_name}, initial_value_from_gguf_lib_for_key='{initial_val_for_debug}' (type: {type(initial_val_for_debug)}), current_value_for_enum_check='{value}' (type: {type(value)})")
    #     if hasattr(field_value_obj, 'type'):
    #         print(f"DEBUG: key={key_name}, field_value_obj.type={field_value_obj.type}")
    #     if key_name == "general.architecture":
    #         print(f"DEBUG: GGMLArchitectureType available: {GGMLArchitectureType is not None}")

    if isinstance(value, int) and hasattr(field_value_obj, 'type') and field_value_obj.type == GGUFValueType.UINT32:
        if key_name == "general.architecture" and GGMLArchitectureType:
            try:
                return GGMLArchitectureType(value).name
            except (ValueError, TypeError):
                return value # Return raw int if mapping fails
        elif key_name == "general.quantization_version" or key_name == "general.file_type":
            try:
                return GGMLQuantizationType(value).name
            except (ValueError, TypeError):
                return value # Return raw int if mapping fails
    
    # If 'value' itself is already an instance of GGMLQuantizationType or GGMLArchitectureType
    # (this can happen if resolve_field_value returns such an object if gguf.py is newer
    # or if the GGUFValue.parts[0] was already an enum instance)
    # (this can happen if resolve_field_value returns such an object directly if gguf.py evolves)
    if GGMLArchitectureType and isinstance(value, GGMLArchitectureType):
        return value.name
    if isinstance(value, GGMLQuantizationType):
        return value.name

    return value


def parse_gguf_metadata(model_path):
    """Parses GGUF metadata using GGUFReader."""
    try:
        reader = GGUFReader(model_path, 'r')
        return reader.fields
    except Exception as e:
        print(f"❌ Error reading GGUF file metadata: {e}")
        return None

def print_all_metadata(fields):
    """Prints all metadata from GGUF fields."""
    if not fields:
        print("No metadata to display.")
        return
    print("\n📋 Full GGUF Metadata:\n")
    # Sort by the key string itself (item[0])
    for key_name_str, value_obj in sorted(fields.items(), key=lambda item: item[0]):
        # key_name_str is already the string name of the key
        # Resolve value, trying different attributes
        val_resolved = "N/A"
        if hasattr(value_obj, 'parts') and value_obj.parts:
            val_resolved = value_obj.parts[0]
        elif hasattr(value_obj, 'data'):
            if isinstance(value_obj.data, list) and value_obj.data:
                val_resolved = value_obj.data[0]
            else:
                val_resolved = value_obj.data
        
        # Attempt to decode if bytes
        if isinstance(val_resolved, bytes):
            try:
                val_resolved = val_resolved.decode('utf-8', errors='replace')
            except: # Keep as bytes if decode fails
                pass
        
        print(f"{key_name_str}: {val_resolved}")


def guess_model_size_from_name(name):
    """Guesses model size (e.g., 7B, 13B) from the model name string."""
    name = name.lower()
    # Ordered from larger to smaller and more specific to less specific
    # to avoid "7b" matching "17b" or "70b" if those existed.
    sizes = {
        "65b": "65B", "70b": "70B", # Added 70B
        "34b": "34B", "30b": "30B", "40b": "40B", # Added 40B
        "22b": "22B", # Added 22B (e.g. Mixtral 8x22B)
        "14b": "14B", "13b": "13B",
        "8b": "8B", "7b": "7B",
        "3b": "3B", "1.5b": "1.5B", "0.5b": "0.5B" # Smaller models
    }
    for key, val in sizes.items():
        if key in name:
            return val
    return None

def map_quantization_from_meta(fields):
    """Maps GGUF quantization type from metadata to a known string."""
    if not fields: return None
    
    # Try general.quantization_version first (newer GGUF versions)
    quant_version_enum = resolve_field(fields, "general.quantization_version")
    if quant_version_enum is not None:
        try:
            # GGMLQuantizationType is an enum, get its name
            if isinstance(quant_version_enum, GGMLQuantizationType):
                 return quant_version_enum.name
            # If it's already a string from resolve_field (due to direct value)
            if isinstance(quant_version_enum, str):
                return quant_version_enum.upper()
            # If it's an int, try to map it
            if isinstance(quant_version_enum, int):
                return GGMLQuantizationType(quant_version_enum).name
        except ValueError:
            pass # Enum value not recognized

    # Fallback to tensor_data_layout or other fields if general.quantization_version is not present/useful
    # This part is tricky as "tensor_data_layout" might not directly map to a quant string like "Q4_K_M"
    # It often indicates the highest quantization present, e.g., "Q8_0" or "Q4_K"
    # For simplicity, we'll rely on general.quantization_version for now.
    # If more advanced fallback is needed, it would require deeper inspection of tensor types.
    
    # Try to get it from general.file_type (older way, often an int)
    file_type_val = resolve_field(fields, "general.file_type")
    if file_type_val is not None and isinstance(file_type_val, int):
        try:
            return GGMLQuantizationType(file_type_val).name
        except ValueError:
            pass
            
    return None


def guess_quant_type_from_name(name):
    """Guesses quantization type from the model filename as a fallback."""
    name = name.upper()
    # Ordered by typical preference, commonality, and length for better matching
    known_quants = [
        "Q4_K_M", "IQ4_NL", "IQ4_XS", "IQ3_S", "IQ3_M", "IQ3_XXS", "IQ2_S", "IQ2_M", "IQ2_XS", "IQ2_XXS", "IQ1_S", "IQ1_M", # Common IQ quants
        "Q5_K_M", "Q6_K", "Q4_K_S", "Q3_K_M", "Q3_K_L", "Q3_K_S",
        "Q8_0", "Q5_K", "Q4_K", "Q3_K", "Q2_K", "Q4_0", "Q5_0",
        "F16", "BF16", "FP16", "F32", "FP32"
    ]
    for qt in known_quants:
        if qt in name:
            return qt
    return None

def get_system_info():
    """Gathers system memory and CPU information."""
    info = {}
    info["unified_memory_gb"] = round(psutil.virtual_memory().total / (1024**3))
    info["apple_gpu_memory_gb"] = None
    if platform.system() == "Darwin":
        try:
            result = os.popen("/usr/sbin/system_profiler SPHardwareDataType").read()
            for line in result.splitlines():
                if "Chip" in line and "Memory" in line: # e.g. "Chip: Apple M2 Pro, Memory: 16 GB"
                    mem_str = line.split("Memory:")[-1].strip().upper().replace("GB", "").strip()
                    info["apple_gpu_memory_gb"] = int(mem_str)
                    break
                elif "Unified Memory" in line: # Fallback for older system_profiler output
                     value = line.split(":")[-1].strip().upper().replace("GB", "").strip()
                     info["apple_gpu_memory_gb"] = int(value)
                     break
        except Exception:
            pass # Could not get Apple GPU memory

    info["logical_cores"] = multiprocessing.cpu_count()
    info["physical_cores"] = psutil.cpu_count(logical=False)
    info["cpu_info"] = platform.processor()
    info["mac_ver"] = platform.mac_ver()[0] if sys.platform == "darwin" else None
    
    if info["apple_gpu_memory_gb"] is None and platform.system() == "Darwin":
        # This typically means system_profiler parsing failed or didn't find a clear "Memory" line associated with "Chip".
        # unified_memory_gb is already the total system RAM, which is what's used in unified memory architectures.
        pass # No change needed here, unified_memory_gb will be used.
            
    return info

# ---------------------------------------------
# 📏 Estimation logic
# ---------------------------------------------

# Max layers for common model sizes (approximate, can vary by specific model architecture)
MODEL_MAX_LAYERS = {
    "0.5B": 24, "1.5B": 24, "3B": 32,
    "7B": 32, "8B": 32, "13B": 40, "14B": 40, # Llama-like
    "22B": 48, # e.g. for Mixtral layers per expert or similar sized models
    "30B": 60, "34B": 48, # CodeLlama 34B has 48
    "40B": 60,
    "65B": 80, "70B": 80  # Llama-like
}

# Estimated VRAM cost per layer in GB (highly approximate, for Metal on Apple Silicon)
# These are rough estimates and can vary significantly.
# Based on gguf_backup.py and extended.
LAYER_COSTS_GB_PER_LAYER = {
    "7B":   {"IQ2_XXS": 0.09, "Q2_K": 0.18, "IQ3_XXS": 0.12, "Q3_K_S": 0.22, "Q4_0": 0.28, "Q4_K_M": 0.30, "Q5_0":0.33, "Q5_K_M": 0.35, "Q6_K": 0.40, "Q8_0": 0.55, "F16": 0.9},
    "8B":   {"IQ2_XXS": 0.10, "Q2_K": 0.20, "IQ3_XXS": 0.14, "Q3_K_S": 0.24, "Q4_0": 0.30, "Q4_K_M": 0.33, "Q5_0":0.36, "Q5_K_M": 0.38, "Q6_K": 0.45, "Q8_0": 0.60, "F16": 1.0},
    "13B":  {"IQ2_XXS": 0.15, "Q2_K": 0.30, "IQ3_XXS": 0.20, "Q3_K_S": 0.38, "Q4_0": 0.48, "Q4_K_M": 0.52, "Q5_0":0.57, "Q5_K_M": 0.60, "Q6_K": 0.70, "Q8_0": 0.85, "F16": 1.5},
    "14B":  {"IQ2_XXS": 0.16, "Q2_K": 0.32, "IQ3_XXS": 0.22, "Q3_K_S": 0.40, "Q4_0": 0.50, "Q4_K_M": 0.55, "Q5_0":0.60, "Q5_K_M": 0.63, "Q6_K": 0.75, "Q8_0": 0.90, "F16": 1.6},
    "30B":  {"IQ2_XXS": 0.28, "Q2_K": 0.55, "IQ3_XXS": 0.38, "Q3_K_S": 0.70, "Q4_0": 0.85, "Q4_K_M": 0.95, "Q5_0":1.00, "Q5_K_M": 1.05, "Q6_K": 1.15, "Q8_0": 1.40, "F16": 2.8},
    "34B":  {"IQ2_XXS": 0.30, "Q2_K": 0.60, "IQ3_XXS": 0.42, "Q3_K_S": 0.75, "Q4_0": 0.90, "Q4_K_M": 1.00, "Q5_0":1.05, "Q5_K_M": 1.10, "Q6_K": 1.25, "Q8_0": 1.50, "F16": 3.0},
    "70B":  {"IQ2_XXS": 0.50, "Q2_K": 0.75, "IQ3_XXS": 0.65, "Q3_K_S": 1.10, "Q4_0": 1.30, "Q4_K_M": 1.45, "Q5_0":1.55, "Q5_K_M": 1.65, "Q6_K": 1.90, "Q8_0": 2.40, "F16": 5.0},
    # Fallback for sizes not explicitly listed (e.g. 65B, using 70B values)
    "65B":  {"IQ2_XXS": 0.50, "Q2_K": 0.75, "IQ3_XXS": 0.65, "Q3_K_S": 1.10, "Q4_0": 1.30, "Q4_K_M": 1.45, "Q5_0":1.55, "Q5_K_M": 1.65, "Q6_K": 1.90, "Q8_0": 2.40, "F16": 5.0},
}
# Add aliases for some quants if needed, e.g. Q4_K could map to Q4_K_M if Q4_K_M is the primary one listed
for size_params in LAYER_COSTS_GB_PER_LAYER.values():
    if "Q4_K_M" in size_params: size_params.setdefault("Q4_K", size_params["Q4_K_M"])
    if "Q5_K_M" in size_params: size_params.setdefault("Q5_K", size_params["Q5_K_M"])
    if "FP16" in size_params: size_params.setdefault("F16", size_params["FP16"]) # Common alias


def estimate_gpu_layers(model_size, quant_type, gpu_mem_gb, system_ram_gb):
    """Estimates optimal n_gpu_layers based on model, quant, and available VRAM."""
    if model_size not in LAYER_COSTS_GB_PER_LAYER or model_size not in MODEL_MAX_LAYERS:
        print(f"⚠️ Warning: Model size {model_size} not in cost/layer database. Cannot estimate GPU layers accurately.")
        return 0 # Default to 0 if no data
    if quant_type not in LAYER_COSTS_GB_PER_LAYER[model_size]:
        # Try to find a close quant if exact match fails
        available_quants = LAYER_COSTS_GB_PER_LAYER[model_size]
        fallback_quant = None
        if quant_type and "K_M" in quant_type and quant_type.replace("_M","") in available_quants:
            fallback_quant = quant_type.replace("_M","")
        elif quant_type and "K_S" in quant_type and quant_type.replace("_S","") in available_quants:
            fallback_quant = quant_type.replace("_S","")
        
        if fallback_quant:
            print(f"⚠️ Warning: Quantization {quant_type} not in cost database for {model_size}. Using {fallback_quant} as fallback.")
            quant_type = fallback_quant
        else:
            print(f"⚠️ Warning: Quantization {quant_type} not in cost database for {model_size}. Cannot estimate GPU layers accurately.")
            return 0


    layer_cost_gb = LAYER_COSTS_GB_PER_LAYER[model_size].get(quant_type)
    if layer_cost_gb is None or layer_cost_gb == 0: # Ensure layer_cost is not zero to prevent division error
        print(f"⚠️ Warning: Layer cost for {model_size} {quant_type} is zero or undefined. Cannot estimate GPU layers.")
        return 0

    # Reserve some VRAM for OS/other apps (e.g., 1-2GB, or more for smaller total VRAM)
    # Also consider context size, KV cache. This is a rough estimate.
    # For Apple Silicon, unified memory is shared, so be more conservative.
    # Let's reserve ~15-25% of GPU memory or at least 2GB.
    reserved_mem_gb = max(2.0, gpu_mem_gb * 0.20)
    
    # If system RAM is very low (e.g. <= 8GB), and it's unified memory, be even more cautious
    if platform.system() == "Darwin" and system_ram_gb <= 8:
        reserved_mem_gb = max(reserved_mem_gb, gpu_mem_gb * 0.35, 2.5)
    elif platform.system() == "Darwin" and system_ram_gb <= 16:
         reserved_mem_gb = max(reserved_mem_gb, gpu_mem_gb * 0.25, 2.0)


    usable_gpu_mem_gb = gpu_mem_gb - reserved_mem_gb
    if usable_gpu_mem_gb <= 0:
        return 0 # Not enough memory to offload anything

    max_fit_layers = int(usable_gpu_mem_gb // layer_cost_gb)
    model_total_layers = MODEL_MAX_LAYERS[model_size]
    
    # Don't offload all layers if it leaves too little RAM for other things, especially on unified memory.
    # Heuristic: try to keep at least 1 layer on CPU if many layers, or if it's close to max.
    estimated_layers = min(max_fit_layers, model_total_layers)
    
    if estimated_layers == model_total_layers and model_total_layers > 10 : # If all layers fit
        # Keep a few layers on CPU for very large models or if it's tight, to be safe
        # This is a heuristic, might need adjustment.
        if gpu_mem_gb / layer_cost_gb < model_total_layers * 1.1: # If it's a tight fit
             return max(0, estimated_layers - 2) # Keep 2 on CPU
        return estimated_layers # Offload all
    
    return max(0, estimated_layers)


def estimate_ctx_size(model_size_str):
    """Estimates a reasonable default context size based on model size."""
    if not model_size_str: return 2048
    # Based on common model capabilities
    ctx_map = {
        "0.5B": 4096, "1.5B": 4096, "3B": 4096,
        "7B": 4096, "8B": 8192, # Llama 3 8B supports 8k
        "13B": 4096, "14B": 4096, # Llama 2 13B/14B often 4k
        "30B": 4096, "34B": 4096, # CodeLlama 34B can go higher, but 4k is safe
        "65B": 4096, "70B": 8192  # Llama 3 70B supports 8k
    }
    # A simple heuristic: smaller number in string = smaller model
    val = float(model_size_str.replace('B',''))
    if val <= 3: return ctx_map.get(model_size_str, 4096)
    if val <= 8: return ctx_map.get(model_size_str, 8192 if "8B" in model_size_str else 4096)
    if val <= 34: return ctx_map.get(model_size_str, 4096)
    return ctx_map.get(model_size_str, 8192 if "70B" in model_size_str else 4096)


def recommend_llama_cpp_params(model_size, quant_type, sys_info, fields, arch_str=None):
    """Recommends llama.cpp parameters."""
    # Use the pre-determined gpu_mem_for_estimation from sys_info
    gpu_mem_to_use = sys_info.get("gpu_mem_for_estimation", sys_info["unified_memory_gb"])
    n_gpu_layers = estimate_gpu_layers(model_size, quant_type, gpu_mem_to_use, sys_info["unified_memory_gb"])

    # Threading
    # For Apple Silicon, physical cores are usually best for n_threads
    # For Intel/AMD with SMT, logical_cores / 2 (physical cores) or slightly more can be good.
    # Max 8 threads often a good balance unless CPU is very powerful.
    if sys_info["mac_ver"] and "Apple" in sys_info["cpu_info"]: # Apple Silicon
        n_threads = sys_info["physical_cores"]
    else: # Intel/AMD
        n_threads = max(4, min(sys_info["physical_cores"], sys_info["logical_cores"] // 2 if sys_info["logical_cores"] > sys_info["physical_cores"] else sys_info["logical_cores"], 8))
    
    n_batch = 512 # Default, can be tuned

    # Context size
    n_ctx = estimate_ctx_size(model_size)
    # Check metadata for context length if available
    # Use the passed arch_str
    if arch_str and arch_str != "Unknown" and not arch_str.startswith("Unknown ("):
        ctx_len_key = f"{arch_str}.context_length" # e.g. llama.context_length
        meta_ctx = resolve_field(fields, ctx_len_key)
        if meta_ctx and isinstance(meta_ctx, int) and meta_ctx > 0:
            n_ctx = min(n_ctx, meta_ctx) # Use model's stated max if smaller than our guess

    # Rope scaling / frequency base
    rope_freq_base = 10000 # Default
    rope_freq_scale = None # Typically not set unless specified

    if arch_str and arch_str != "Unknown" and not arch_str.startswith("Unknown ("):
        # Try to get rope parameters from metadata first
        meta_rope_base = resolve_field(fields, f"{arch_str}.rope.freq_base")
        meta_rope_scale = resolve_field(fields, f"{arch_str}.rope.scaling.factor") # Common key for scale
        # meta_rope_type = resolve_field(fields, f"{arch_str}.rope.scaling.type")


        if meta_rope_base and isinstance(meta_rope_base, (int, float)) and meta_rope_base > 0:
            rope_freq_base = int(meta_rope_base)
        # Fallback heuristics if metadata not present or not useful
        elif arch_str == "llama":
            model_name_str = resolve_field(fields, "general.name") or ""
            if "llama3" in model_name_str.lower() or "llama-3" in model_name_str.lower():
                rope_freq_base = 500000
            else: # Llama 1/2 default
                rope_freq_base = 10000
        elif arch_str == "qwen2": # Qwen2 models often use 1M
             rope_freq_base = 1000000
        
        if meta_rope_scale and isinstance(meta_rope_scale, (int, float)) and meta_rope_scale > 0:
            rope_freq_scale = float(meta_rope_scale)

    params_to_return = {
        "n_gpu_layers": n_gpu_layers,
        "n_threads": n_threads,
        "n_batch": n_batch,
        "n_ctx": n_ctx,
        "rope_freq_base": rope_freq_base
    }
    if rope_freq_scale is not None:
        params_to_return["rope_freq_scale"] = rope_freq_scale
        
    return params_to_return

# ---------------------------------------------
# 🚀 Main entry
# ---------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GGUF Model Analyzer & llama.cpp Parameter Recommender")
    parser.add_argument("model_path", help="Path to GGUF model file")
    parser.add_argument("--show-meta", action="store_true", help="Show all GGUF metadata from the model file")
    args = parser.parse_args()

    model_filepath = Path(args.model_path)
    if not model_filepath.is_file():
        print(f"❌ Error: Model file not found at {model_filepath}")
        sys.exit(1)

    print(f"🔍 Analyzing GGUF model: {model_filepath.name}\n")

    fields = parse_gguf_metadata(str(model_filepath))
    if fields is None:
        sys.exit(1) # Error already printed in parse_gguf_metadata

    if args.show_meta:
        print_all_metadata(fields)
        sys.exit(0)

    # --- Gather Model Info ---
    model_name_meta = resolve_field(fields, "general.name")
    used_model_name_source = "metadata"
    if model_name_meta and isinstance(model_name_meta, str) and len(model_name_meta.strip()) > 0:
        model_name_to_use = model_name_meta
    else:
        model_name_to_use = model_filepath.name
        used_model_name_source = "filename"
    
    print(f"🧠 Model Name: {model_name_to_use} (from {used_model_name_source})")

    # Architecture
    arch_meta_val = resolve_field(fields, "general.architecture")
    arch_str = "Unknown"
    arch_source = "" # Initialize

    if isinstance(arch_meta_val, str):
        arch_str = arch_meta_val
        arch_source = "metadata"
    elif isinstance(arch_meta_val, int):
        arch_str = f"Unknown (enum value: {arch_meta_val})"
        arch_source = "metadata (enum mapping failed or GGMLArchitectureType unavailable)"
    elif arch_meta_val is not None: # Value found, but not str or int after resolve_field
        arch_str = f"Unknown (unhandled metadata value: {arch_meta_val})"
        arch_source = "metadata (unexpected value type)"
    else: # arch_meta_val is None
        # arch_str remains "Unknown"
        arch_source = "not found in metadata"
    print(f"   Architecture: {arch_str} (from {arch_source})")

    # Model Size
    model_size = guess_model_size_from_name(model_name_to_use)
    model_size_source = f"guessed from name ('{model_name_to_use}')"
    
    if not model_size and arch_str != "Unknown" and not arch_str.startswith("Unknown ("):
        # Try to derive from metadata if name guess fails and arch is known
        param_count_key = f"{arch_str}.parameter_count"
        block_count_key = f"{arch_str}.block_count"
        
        param_count = resolve_field(fields, param_count_key)
        block_count = resolve_field(fields, block_count_key)

        if param_count and isinstance(param_count, (int, float)) and param_count > 1000000:
            param_b = param_count / 1_000_000_000
            if param_b < 1: model_size = f"{round(param_b*1000)}M"
            else: model_size = f"{round(param_b)}B" if param_b >=1 else f"{param_b:.1f}B"
            model_size_source = f"derived from metadata ({param_count:,} params)"
        elif block_count and isinstance(block_count, int):
            # Rough heuristic based on block count
            if block_count >= 80: model_size = "70B"
            elif block_count >= 60: model_size = "30B"
            elif block_count >= 40: model_size = "13B"
            elif block_count >= 30: model_size = "7B"
            if model_size:
                model_size_source = f"derived from metadata ({block_count} blocks)"
    
    if model_size:
        print(f"   Model Size: ~{model_size} ({model_size_source})")
    else:
        print(f"   ⚠️ Warning: Could not determine model size. Parameter recommendations may be inaccurate. (Name used for guess: '{model_name_to_use}')")

    # Quantization
    quant_from_meta_str = None
    raw_meta_quant_value = None # To store the actual value from metadata if not a string
    quant_source_detail = ""

    q_ver = resolve_field(fields, "general.quantization_version")
    if isinstance(q_ver, str):
        quant_from_meta_str = q_ver
        quant_source_detail = "metadata ('general.quantization_version')"
    # Store the raw value from general.quantization_version if it exists
    if q_ver is not None:
        raw_meta_quant_value = q_ver

    if not quant_from_meta_str: # If q_ver didn't yield a string
        q_ft = resolve_field(fields, "general.file_type")
        if isinstance(q_ft, str):
            quant_from_meta_str = q_ft
            quant_source_detail = "metadata ('general.file_type')"
        
        # If q_ver was None (key not found) and q_ft has a value (even if not str),
        # raw_meta_quant_value should reflect q_ft.
        # If q_ver had a non-string value, raw_meta_quant_value already holds it.
        if q_ver is None and q_ft is not None:
            raw_meta_quant_value = q_ft
        # If both q_ver and q_ft were None, raw_meta_quant_value remains None.

    quant_type_filename = guess_quant_type_from_name(model_filepath.name)
    quant_type_to_use = None
    
    if quant_from_meta_str:
        quant_type_to_use = quant_from_meta_str
        print(f"   Quantization: {quant_type_to_use} (from {quant_source_detail})")
    elif quant_type_filename:
        quant_type_to_use = quant_type_filename
        print(f"   Quantization: {quant_type_to_use} (from guessed from filename)")
        if raw_meta_quant_value is not None and not isinstance(raw_meta_quant_value, str):
             print(f"      (Metadata field for quantization was: {raw_meta_quant_value})")
        elif raw_meta_quant_value is None and not quant_from_meta_str : # Only if metadata keys were truly not found
             print(f"      (Relevant metadata fields for quantization not found or did not yield a string)")
    else:
        print("   ⚠️ Warning: Could not determine quantization type. Parameter recommendations may be inaccurate.")
        if raw_meta_quant_value is not None and not isinstance(raw_meta_quant_value, str):
             print(f"      (Metadata field for quantization was: {raw_meta_quant_value})")
        elif raw_meta_quant_value is None: # Both metadata attempts yielded None
             print(f"      (Relevant metadata fields for quantization not found)")
    
    # Store for later use
    model_info = {
        "name": model_name_to_use,
        "architecture": arch_str,
        "size": model_size,
        "quantization": quant_type_to_use # This must be updated with the final quant_type_to_use
    }

    # --- Gather System Info ---
    print("\n💻 System Information:")
    sys_info = get_system_info()
    print(f"   Total System Memory (RAM): {sys_info['unified_memory_gb']} GB")
    
    # Determine GPU memory for estimation and store it in sys_info
    sys_info["gpu_mem_for_estimation"] = sys_info["unified_memory_gb"] # Default
    if platform.system() == "Darwin":
        if sys_info["apple_gpu_memory_gb"]:
            print(f"   Apple Specific GPU Memory (from system_profiler): {sys_info['apple_gpu_memory_gb']} GB")
            sys_info["gpu_mem_for_estimation"] = sys_info["apple_gpu_memory_gb"]
            print(f"      (Using {sys_info['gpu_mem_for_estimation']}GB for GPU layer estimation)")
        else:
            print(f"   Apple Specific GPU Memory (from system_profiler): Not detected.")
            # sys_info["gpu_mem_for_estimation"] is already unified_memory_gb by default
            print(f"      (Using total system memory {sys_info['gpu_mem_for_estimation']}GB for GPU layer estimation on Apple Silicon)")
    # Placeholder for discrete GPU memory detection on other OS (e.g., nvidia-smi) could be added here
    # else: 
    #     print(f"   System Memory (RAM): {sys_info['unified_memory_gb']} GB (using this for GPU layer estimation, assuming shared or no discrete GPU info)")


    print(f"   CPU: {sys_info['cpu_info']} ({sys_info['physical_cores']} physical, {sys_info['logical_cores']} logical cores)")
    if sys_info["mac_ver"]:
        print(f"   macOS Version: {sys_info['mac_ver']}")

    # --- Recommendations ---
    # Use model_info dictionary which holds the determined values
    if model_info["size"] and model_info["quantization"]:
        print("\n⚙️ Recommended llama.cpp Parameters:")
        params = recommend_llama_cpp_params(model_info["size"], model_info["quantization"], sys_info, fields, model_info["architecture"])
        for key, val in params.items():
            cli_key = key.replace('_', '-')
            print(f"   --{cli_key} {val}")
        
        # Additional advice based on n_gpu_layers
        gpu_mem_available = sys_info["gpu_mem_for_estimation"]
        # Use model_info["size"] for MODEL_MAX_LAYERS check
        current_model_max_layers = MODEL_MAX_LAYERS.get(model_info["size"])

        if params["n_gpu_layers"] == 0 and gpu_mem_available > 2 : 
            print("\n   💡 Tip: n_gpu_layers is 0.")
            if platform.system() == "Darwin":
                 print("      Ensure Metal is enabled in llama.cpp for Apple Silicon Macs.")
            print("      If GPU/System memory is low, this might be correct. Otherwise, check model/quant type or VRAM availability.")
        elif params["n_gpu_layers"] > 0 and current_model_max_layers and params["n_gpu_layers"] < current_model_max_layers:
             print("\n   💡 Tip: Not all layers are offloaded. This is common if VRAM is limited. Performance will be mixed CPU/GPU.")
        elif params["n_gpu_layers"] > 0 and current_model_max_layers and params["n_gpu_layers"] >= current_model_max_layers:
             print("\n   💡 Tip: All (or most) layers offloaded to GPU. Expect good performance if your GPU can handle it!")
        elif params["n_gpu_layers"] > 0 and not current_model_max_layers: 
             print(f"\n   💡 Tip: {params['n_gpu_layers']} layers offloaded. Max layers for ~{model_info['size']} models unknown to this script, so full offload status is unclear.")

    else:
        print("\n⚠️ Could not provide full parameter recommendations due to missing model size or quantization information.")

    print("\nDisclaimer: These are estimates. Actual performance and optimal settings may vary.")

if __name__ == "__main__":
    main()
