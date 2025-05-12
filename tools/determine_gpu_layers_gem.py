#!/usr/bin/env python3

import argparse
import subprocess
import re
import math
from pathlib import Path

try:
    from gguf import GGUFReader, GGUFValueType
except ImportError:
    print("Error: The 'gguf' library is not installed. Please install it with 'pip install gguf'")
    exit(1)

# --- Constants and Configuration ---
# Typical sizes in bytes for different data types
# These are estimates and can vary slightly based on specific GGUF quantization details
BYTES_PER_F16 = 2
BYTES_PER_F32 = 4
# For KV cache, typically F16 is used
BYTES_PER_KV_ELEMENT = BYTES_PER_F16

# --- Helper Functions ---

def get_macos_vram_mb():
    """
    Gets the total VRAM in MB for Metal-supported GPUs on macOS.
    For Apple Silicon, this will be the total system unified memory.
    Returns None if VRAM cannot be determined.
    """
    try:
        sp_displays_process = subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True, check=True)
        displays_output = sp_displays_process.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running system_profiler SPDisplaysDataType: {e}")
        return None

    # Check for Apple Silicon (M1, M2, M3, etc.)
    apple_silicon_match = re.search(r"Chipset Model: Apple M\d", displays_output)
    if apple_silicon_match:
        print("Apple Silicon detected. Using total system unified memory as VRAM.")
        try:
            sysctl_process = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True, check=True)
            memsize_output = sysctl_process.stdout
            # Output is like "hw.memsize: 34359738368"
            memsize_match = re.search(r"hw.memsize:\s*(\d+)", memsize_output)
            if memsize_match:
                total_bytes = int(memsize_match.group(1))
                return total_bytes / (1024 * 1024) # Convert bytes to MB
            else:
                print("Warning: Could not parse hw.memsize output for Apple Silicon.")
                return None
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error running sysctl hw.memsize: {e}")
            return None
    else:
        # Logic for Intel Macs with discrete GPUs or older integrated GPUs
        print("Non-Apple Silicon or older Mac detected. Looking for explicit VRAM.")
        gpu_blocks = re.split(r'Hardware:', displays_output)[1:] # Split by "Hardware:" to get individual GPU blocks
        
        for block in gpu_blocks:
            if "Metal Support:" in block or "Metal Family:" in block: # Check for Metal support
                # Try to find VRAM (Total) or VRAM (Dynamic, Max)
                vram_match = re.search(r"VRAM \(Total\):\s*(\d+)\s*([GMK]B)", block)
                if not vram_match:
                    vram_match = re.search(r"VRAM \(Dynamic, Max\):\s*(\d+)\s*([GMK]B)", block)
                if not vram_match: # Fallback for some integrated GPUs that might just list VRAM
                     vram_match = re.search(r"VRAM:\s*(\d+)\s*([GMK]B)", block)

                if vram_match:
                    vram_value = int(vram_match.group(1))
                    vram_unit = vram_match.group(2).upper()

                    if vram_unit == "GB":
                        return vram_value * 1024
                    elif vram_unit == "MB":
                        return vram_value
                    elif vram_unit == "KB":
                        return vram_value / 1024
                    # Should not happen with typical outputs
                    print(f"Warning: Matched VRAM but unknown unit: {vram_unit}")
                    return None 
        
    print("Warning: Could not determine VRAM for a Metal-supported GPU through system_profiler SPDisplaysDataType.")
    return None

def get_tensor_size_in_bytes(tensor_info):
    """
    Estimates the size of a tensor in bytes from its GGUF tensor info.
    """
    # Robustly check if shape is None, an empty list/tuple, or a NumPy array with size 0
    is_empty_shape = False
    if tensor_info.shape is None:
        is_empty_shape = True
    elif hasattr(tensor_info.shape, 'size'): # Check if it's a NumPy array
        if tensor_info.shape.size == 0:
            is_empty_shape = True
    elif hasattr(tensor_info.shape, '__len__'): # Check if it's a list or tuple
        if len(tensor_info.shape) == 0:
            is_empty_shape = True
    else: # Unknown shape type, assume problematic
        is_empty_shape = True 
        print(f"Warning: Tensor '{tensor_info.name}' has an unusual shape type: {type(tensor_info.shape)}. Assuming size 0.")

    if is_empty_shape:
        return 0 # Skip tensors with no shape or an empty shape

    num_elements = math.prod(tensor_info.shape)
    
    # GGUF Tensor Types mapping to rough byte sizes
    # This is a simplified mapping. For precise sizes, one would need detailed knowledge
    # of each GGUF_TYPE's block size and internal structure for quantized types.
    # For our purpose (estimating layers), this should be sufficient.
    # Ref: ggml-common.h in llama.cpp for ggml_type_size and ggml_blck_size
    
    # Common unquantized types
    # Accessing enum members by string name to avoid potential AttributeError with direct access
    # tensor_info.tensor_type is an instance of GGUFValueType enum
    
    current_type_name = tensor_info.tensor_type.name.upper()

    if current_type_name == 'F32':
        return num_elements * BYTES_PER_F32
    if current_type_name == 'F16':
        return num_elements * BYTES_PER_F16
    if current_type_name == 'BF16': # Similar to F16
        return num_elements * BYTES_PER_F16

    # Common quantized types (approximations)
    # These are rough estimates. The actual size depends on block sizes (e.g., QK_K)
    # For example, Q4_0 means 4 bits per weight.
    # For block-quantized types, there's also metadata per block.
    # We'll use the nominal bits per weight.
    
    # type_name = tensor_info.tensor_type.name.upper() # Already got this as current_type_name

    if "Q8_0" in current_type_name: # 8 bits per weight
        return num_elements * 1 
    if "Q6_K" in current_type_name: # ~6 bits per weight
        return num_elements * (6 / 8)
    if "Q5_0" in current_type_name or "Q5_1" in current_type_name or "Q5_K" in current_type_name: # ~5 bits
        return num_elements * (5 / 8)
    if "Q4_0" in current_type_name or "Q4_1" in current_type_name or "Q4_K" in current_type_name: # ~4 bits
        return num_elements * (4 / 8)
    if "Q3_K" in current_type_name: # ~3 bits
        return num_elements * (3 / 8)
    if "Q2_K" in current_type_name: # ~2 bits
        return num_elements * (2 / 8)
    
    # If unknown, assume F16 as a fallback for estimation, or warn
    # print(f"Warning: Unknown tensor type {current_type_name} for {tensor_info.name}. Estimating as F16.")
    # For layer estimation, it's better to be slightly conservative if unsure.
    # However, most model weights will be one of the above.
    # If a type is truly exotic and large, this might underestimate.
    # Let's assume F16 if not matched, as it's a common non-quantized type.
    return num_elements * BYTES_PER_F16


def get_gguf_model_info(gguf_path):
    """
    Parses a GGUF file and extracts model information.
    Returns a dictionary with info like total_layers, layer_vram_mb, hidden_size, etc.
    """
    reader = GGUFReader(gguf_path, 'r')
    model_info = {
        "total_layers": 0,
        "layer_tensors_vram_mb_per_layer": 0, # VRAM for one layer's weights
        "hidden_size": 0, # For KV cache calculation (e.g., llama.embedding_length)
        "default_n_ctx": 2048, # A common default
        "num_attention_heads": 0,
        "num_key_value_heads": 0,
        "architecture": None,
        "tensor_info_by_layer": {} # tensor_name: size_in_bytes
    }

    arch_field = next((field for field in reader.fields.values() if field.name.endswith('.architecture')), None)
    if arch_field and arch_field.parts and len(arch_field.parts) > arch_field.data[0]:
        try:
            # Explicitly convert to bytes before decoding to handle memmap objects robustly
            model_info["architecture"] = bytes(arch_field.parts[arch_field.data[0]]).decode()
        except Exception as e:
            print(f"Warning: Could not decode architecture string: {e}")
            # Fallback or leave as None

    n_layers_field = next((field for field in reader.fields.values() if field.name.endswith('.block_count')), None)
    if n_layers_field and n_layers_field.parts and len(n_layers_field.parts) > n_layers_field.data[0]:
        raw_val = n_layers_field.parts[n_layers_field.data[0]]
        try:
            if hasattr(raw_val, 'item'): # Check if it's a NumPy-like scalar array
                model_info["total_layers"] = int(raw_val.item())
            elif isinstance(raw_val, (bytes, bytearray, memoryview)): # Check if it's bytes that need decoding
                model_info["total_layers"] = int(bytes(raw_val).decode())
            else: # Assume it's already a Python int or compatible
                model_info["total_layers"] = int(raw_val)
        except Exception as e:
            print(f"Warning: Could not parse total_layers (raw: '{raw_val}'): {e}")


    hidden_size_field = next((field for field in reader.fields.values() if field.name.endswith('.embedding_length')), None)
    if hidden_size_field and hidden_size_field.parts and len(hidden_size_field.parts) > hidden_size_field.data[0]:
        raw_val = hidden_size_field.parts[hidden_size_field.data[0]]
        try:
            if hasattr(raw_val, 'item'):
                model_info["hidden_size"] = int(raw_val.item())
            elif isinstance(raw_val, (bytes, bytearray, memoryview)):
                model_info["hidden_size"] = int(bytes(raw_val).decode())
            else:
                model_info["hidden_size"] = int(raw_val)
        except Exception as e:
            print(f"Warning: Could not parse hidden_size (raw: '{raw_val}'): {e}")

    
    ctx_len_field = next((field for field in reader.fields.values() if field.name.endswith('.context_length')), None)
    if ctx_len_field and ctx_len_field.parts and len(ctx_len_field.parts) > ctx_len_field.data[0]:
        raw_val = ctx_len_field.parts[ctx_len_field.data[0]]
        try:
            if hasattr(raw_val, 'item'):
                model_info["default_n_ctx"] = int(raw_val.item())
            elif isinstance(raw_val, (bytes, bytearray, memoryview)):
                model_info["default_n_ctx"] = int(bytes(raw_val).decode())
            else:
                model_info["default_n_ctx"] = int(raw_val)
        except Exception as e:
            print(f"Warning: Could not parse default_n_ctx (raw: '{raw_val}'): {e}")


    attn_heads_field = next((field for field in reader.fields.values() if field.name.endswith('.attention.head_count')), None)
    if attn_heads_field and attn_heads_field.parts and len(attn_heads_field.parts) > attn_heads_field.data[0]:
        raw_val = attn_heads_field.parts[attn_heads_field.data[0]]
        try:
            if hasattr(raw_val, 'item'):
                model_info["num_attention_heads"] = int(raw_val.item())
            elif isinstance(raw_val, (bytes, bytearray, memoryview)):
                model_info["num_attention_heads"] = int(bytes(raw_val).decode())
            else:
                model_info["num_attention_heads"] = int(raw_val)
        except Exception as e:
            print(f"Warning: Could not parse num_attention_heads (raw: '{raw_val}'): {e}")


    kv_heads_field = next((field for field in reader.fields.values() if field.name.endswith('.attention.head_count_kv')), None)
    if kv_heads_field and kv_heads_field.parts and len(kv_heads_field.parts) > kv_heads_field.data[0]:
        raw_val = kv_heads_field.parts[kv_heads_field.data[0]]
        try:
            if hasattr(raw_val, 'item'):
                model_info["num_key_value_heads"] = int(raw_val.item())
            elif isinstance(raw_val, (bytes, bytearray, memoryview)):
                model_info["num_key_value_heads"] = int(bytes(raw_val).decode())
            else:
                model_info["num_key_value_heads"] = int(raw_val)
        except Exception as e:
            print(f"Warning: Could not parse num_key_value_heads (raw: '{raw_val}'): {e}")
    elif model_info["num_attention_heads"] > 0 :
         model_info["num_key_value_heads"] = model_info["num_attention_heads"]


    # Estimate VRAM per layer by summing tensor sizes for a typical layer (e.g., layer 0 or 1)
    # This assumes layers are mostly homogenous in size.
    # We look for tensors like "blk.0.attn_norm.weight", "blk.0.ffn_gate.weight" etc.
    
    # More robust: sum all tensors and divide by number of layers for an average
    # But some tensors are not per-layer (tok_embeddings, output_norm, output weights)
    # Let's try to identify per-layer tensors by "blk.<N>." pattern
    
    total_block_tensor_vram_bytes = 0
    layer_tensor_counts = {} # To count how many tensors per block index

    for tensor in reader.tensors:
        tensor_name = tensor.name
        tensor_vram_bytes = get_tensor_size_in_bytes(tensor)

        match = re.match(r"blk\.(\d+)\.(.+)", tensor_name) # Common pattern for LLaMA-like models
        if not match and model_info["architecture"] == "gemma": # Gemma uses "block.<N>"
             match = re.match(r"block\.(\d+)\.(.+)", tensor_name)

        if match:
            layer_idx = int(match.group(1))
            if layer_idx not in model_info["tensor_info_by_layer"]:
                model_info["tensor_info_by_layer"][layer_idx] = {}
            
            model_info["tensor_info_by_layer"][layer_idx][tensor_name] = tensor_vram_bytes
            total_block_tensor_vram_bytes += tensor_vram_bytes
            layer_tensor_counts[layer_idx] = layer_tensor_counts.get(layer_idx, 0) + 1


    if model_info["total_layers"] > 0 and total_block_tensor_vram_bytes > 0 :
        # Average VRAM for tensors within blocks
        model_info["layer_tensors_vram_mb_per_layer"] = (total_block_tensor_vram_bytes / model_info["total_layers"]) / (1024 * 1024)
    elif model_info["tensor_info_by_layer"].get(0): # Fallback: use layer 0 if averaging failed
        layer_0_vram = sum(model_info["tensor_info_by_layer"][0].values())
        model_info["layer_tensors_vram_mb_per_layer"] = layer_0_vram / (1024*1024)
    else:
        print("Warning: Could not determine average VRAM per layer from GGUF tensors.")
        # This is a critical value. If we can't estimate it, we can't proceed well.
        # User might need to provide this, or we make a very rough guess.
        # For now, we'll let it be 0 and the calculation will likely suggest 0 layers.
        pass


    if model_info["hidden_size"] == 0: # Fallback if not directly available
        # Try to infer from a common tensor like 'tok_embeddings.weight'
        tok_emb_tensor = next((t for t in reader.tensors if t.name.endswith('token_embd.weight') or t.name.endswith('tok_embeddings.weight')), None)
        if tok_emb_tensor and len(tok_emb_tensor.shape) == 2:
            model_info["hidden_size"] = tok_emb_tensor.shape[1] # Usually [vocab_size, hidden_size]
            print(f"Inferred hidden_size: {model_info['hidden_size']} from token embeddings")
        else:
            print("Warning: Could not determine hidden_size from GGUF.")


    if model_info["num_key_value_heads"] == 0 and model_info["hidden_size"] > 0:
        # Try to infer from attention K/V weight tensor shapes if possible, e.g., blk.0.attn_k.weight
        # Shape is often [hidden_size, num_kv_heads * head_dim] or similar
        # This is more complex and model-specific. For now, rely on metadata or default.
        print("Warning: Could not determine num_key_value_heads. KV cache estimation might be less accurate.")


    return model_info


def estimate_kv_cache_vram_mb(n_ctx, n_gpu_layers, hidden_size, num_key_value_heads, bytes_per_kv_element=BYTES_PER_KV_ELEMENT):
    """
    Estimates VRAM for the KV cache in MB.
    Formula based on llama.cpp `struct ggml_context_params gf` and `ggml_new_tensor_2d` for k/v cache.
    Each layer in cache:
      K-cache: n_ctx * (hidden_size / num_attention_heads * num_key_value_heads) * bytes_per_kv_element
      V-cache: n_ctx * (hidden_size / num_attention_heads * num_key_value_heads) * bytes_per_kv_element
      (More accurately, it's n_ctx * head_dim * num_key_value_heads for K, and same for V)
      And head_dim = hidden_size / num_attention_heads
    
    Simplified: For each layer, K and V caches are (n_ctx * hidden_size_of_kv_projection)
    The hidden_size_of_kv_projection is effectively (hidden_size / num_attn_heads) * num_kv_heads.
    If num_kv_heads is not available or zero, assume it's same as num_attn_heads (no GQA/MQA),
    then it simplifies to n_ctx * hidden_size.
    """
    if n_gpu_layers == 0 or hidden_size == 0 : # num_key_value_heads can be 0 if not found
        return 0

    # This is a common formulation: n_ctx * (embedding_length_per_kv_head * n_head_kv)
    # where embedding_length_per_kv_head is often hidden_size / num_attention_heads
    # and n_head_kv is num_key_value_heads
    # So, total elements per K or V tensor per layer: n_ctx * (hidden_size / num_attn_heads) * num_kv_heads
    # If num_key_value_heads is not reliably found, this becomes tricky.
    # A simpler, often used approximation if head counts are uncertain:
    # Each K or V tensor per layer is roughly n_ctx * hidden_size elements (if no GQA/MQA, or as an upper bound)
    # Let's use a more direct approach if possible:
    # llama.cpp: ggml_new_tensor_2d(ctx, type, n_embd_gqa, n_ctx)
    # where n_embd_gqa = (n_embd / n_head) * n_head_kv
    # So, elements = n_ctx * ( (hidden_size / num_attn_heads) * num_kv_heads )
    
    # If num_attention_heads is missing, we can't calculate head_dim.
    # Fallback: assume kv_cache_per_layer_per_tok_bytes is roughly hidden_size * bytes_per_kv_element
    # This is an oversimplification if GQA/MQA is used.
    
    # Let's assume hidden_size is the primary factor if head counts are problematic.
    # The GGUF metadata for llama.cpp often includes `llama.embedding_length` (hidden_size)
    # and `llama.attention.head_count_kv`.
    # The size of each K or V tensor for a layer is: n_ctx * (embedding_length / head_count * head_count_kv)
    # If head_count (num_attention_heads) is not available, this is hard.
    # Let's use a common approximation: each K and V cache tensor for a layer is n_ctx * hidden_size elements.
    # This is an overestimate if GQA/MQA is used and head_count_kv < head_count.
    # It's safer to overestimate slightly than underestimate for fitting.

    # A more direct formula from llama.cpp context:
    # memory_k = n_layer * n_ctx * n_embd_gqa * ggml_type_sizef(type_k)
    # memory_v = n_layer * n_ctx * n_embd_gqa * ggml_type_sizef(type_v)
    # n_embd_gqa = (hidden_size / num_attention_heads) * num_key_value_heads
    # If num_attention_heads is 0, this breaks.
    
    # Let's use a simplified but generally safe estimate:
    # For each layer, K and V caches together are: 2 * n_ctx * hidden_size * bytes_per_kv_element
    # This assumes the "width" of the KV cache per token is effectively `hidden_size`.
    # This is true if no GQA/MQA (num_kv_heads == num_attn_heads).
    # If GQA/MQA, then num_kv_heads < num_attn_heads, and the actual KV cache is smaller.
    # Using hidden_size directly is a conservative (larger) estimate if GQA details are missing.

    if num_key_value_heads > 0 and hidden_size > 0 : # Prefer this if available
        # This is a common way to express the size of K or V cache per layer
        # It's effectively n_ctx * (size of all KV head projections)
        # For many models, this is n_ctx * hidden_size (if no GQA/MQA) or n_ctx * (num_kv_heads * head_dim)
        # A common approximation for total KV cache for all layers:
        # n_layers * n_ctx * hidden_size * 2 (for K and V) * bytes_per_element
        # So for one layer:
        kv_bytes_per_layer = n_ctx * hidden_size * 2 * bytes_per_kv_element
    else: # Fallback if KV head count is unknown
        print("Warning: num_key_value_heads not found or zero. KV cache estimation will be less precise (likely overestimate).")
        kv_bytes_per_layer = n_ctx * hidden_size * 2 * bytes_per_kv_element
        
    total_kv_cache_bytes = n_gpu_layers * kv_bytes_per_layer
    return total_kv_cache_bytes / (1024 * 1024)


# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(description="Estimate optimal n_gpu_layers for llama.cpp on macOS.")
    parser.add_argument("gguf_model_path", type=str, help="Path to the GGUF model file.")
    parser.add_argument("--n-ctx", type=int, default=None, help="Context size (e.g., 2048, 4096). If None, tries to use model default or 2048.")
    parser.add_argument("--n-batch", type=int, default=512, help="Batch size for llama.cpp (affects some scratch buffer sizes, minor impact on layer offload).")
    parser.add_argument("--safety-margin-mb", type=int, default=512, help="Safety VRAM margin in MB for OS and other apps.")
    parser.add_argument("--llama-overhead-mb", type=int, default=768, help="Estimated base VRAM overhead for llama.cpp (Metal backend, context, scratch buffers) in MB, excluding model layers and KV cache.")
    
    args = parser.parse_args()

    if not Path(args.gguf_model_path).is_file():
        print(f"Error: GGUF model file not found at {args.gguf_model_path}")
        return

    print("--- System and Model Analysis ---")
    total_vram_mb = get_macos_vram_mb()
    if total_vram_mb is None:
        print("Could not determine system VRAM. Exiting.")
        return
    print(f"Total System VRAM: {total_vram_mb:.2f} MB")

    model_info = get_gguf_model_info(args.gguf_model_path)
    if model_info["total_layers"] == 0 or model_info["layer_tensors_vram_mb_per_layer"] == 0 or model_info["hidden_size"] == 0:
        print("Critical model information missing from GGUF or could not be estimated. Cannot proceed.")
        print(f"  GGUF Info: Total Layers: {model_info['total_layers']}, VRAM/Layer: {model_info['layer_tensors_vram_mb_per_layer']:.2f} MB, Hidden Size: {model_info['hidden_size']}")
        return
        
    print(f"GGUF Model: {Path(args.gguf_model_path).name}")
    print(f"  Architecture: {model_info['architecture']}")
    print(f"  Total Layers: {model_info['total_layers']}")
    print(f"  Est. VRAM per Layer (weights only): {model_info['layer_tensors_vram_mb_per_layer']:.2f} MB")
    print(f"  Hidden Size (embedding_length): {model_info['hidden_size']}")
    print(f"  Default Context Length: {model_info['default_n_ctx']}")
    print(f"  Attention Heads: {model_info['num_attention_heads']}")
    print(f"  KV Heads: {model_info['num_key_value_heads']}")


    n_ctx = args.n_ctx if args.n_ctx is not None else model_info["default_n_ctx"]
    print(f"Using Context Size (n_ctx): {n_ctx}")
    print(f"Using Batch Size (n_batch): {args.n_batch} (primarily for info, minor impact on this script's VRAM calc)")
    print(f"Safety VRAM Margin: {args.safety_margin_mb} MB")
    print(f"Llama.cpp Base Overhead (est.): {args.llama_overhead_mb} MB")

    available_vram_for_model = total_vram_mb - args.safety_margin_mb - args.llama_overhead_mb
    print(f"Available VRAM for Model Layers & KV Cache: {available_vram_for_model:.2f} MB")

    if available_vram_for_model <= 0:
        print("Not enough VRAM available after safety margin and base overhead for any layers.")
        print(f"Recommended n_gpu_layers: 0")
        return

    print("\n--- Estimating n_gpu_layers ---")
    best_n_gpu_layers = 0
    for current_n_gpu_layers in range(1, model_info["total_layers"] + 1):
        model_layers_vram = current_n_gpu_layers * model_info["layer_tensors_vram_mb_per_layer"]
        
        kv_cache_vram = estimate_kv_cache_vram_mb(
            n_ctx=n_ctx,
            n_gpu_layers=current_n_gpu_layers,
            hidden_size=model_info["hidden_size"],
            num_key_value_heads=model_info["num_key_value_heads"] # Pass this along
            # bytes_per_kv_element is default
        )
        
        total_needed_vram = model_layers_vram + kv_cache_vram
        
        print(f"  Trying {current_n_gpu_layers} layers: Model VRAM {model_layers_vram:.2f} MB + KV Cache VRAM {kv_cache_vram:.2f} MB = Total {total_needed_vram:.2f} MB")

        if total_needed_vram <= available_vram_for_model:
            best_n_gpu_layers = current_n_gpu_layers
        else:
            print(f"    Exceeds available VRAM ({available_vram_for_model:.2f} MB). Stopping.")
            break
            
    print("\n--- Recommendation ---")
    print(f"Recommended n_gpu_layers: {best_n_gpu_layers}")

    if best_n_gpu_layers == model_info["total_layers"]:
        print("All model layers can potentially be offloaded to GPU.")
    elif best_n_gpu_layers == 0:
        print("Consider reducing n_ctx, safety_margin_mb, or llama_overhead_mb if you expect some layers to fit.")
    
    # Suggestion for n_ctx if it was constrained
    # This is a more complex calculation: if best_n_gpu_layers > 0,
    # how much n_ctx could we afford with *those* layers?
    if best_n_gpu_layers > 0:
        remaining_vram_for_kv_only = available_vram_for_model - (best_n_gpu_layers * model_info["layer_tensors_vram_mb_per_layer"])
        if remaining_vram_for_kv_only > 0 :
            # Estimate max n_ctx for the *determined* best_n_gpu_layers
            # kv_bytes_per_layer_per_ctx_token = hidden_size * 2 * BYTES_PER_KV_ELEMENT (simplified)
            # More accurately: ( (hidden_size / num_attn_heads) * num_kv_heads ) * 2 * BYTES_PER_KV_ELEMENT if heads known
            bytes_per_kv_token_total_layers = 0
            if model_info["hidden_size"] > 0: # and model_info["num_key_value_heads"] > 0: (handle if kv_heads is 0)
                 # Simplified:
                 bytes_per_kv_token_total_layers = best_n_gpu_layers * model_info["hidden_size"] * 2 * BYTES_PER_KV_ELEMENT

            if bytes_per_kv_token_total_layers > 0:
                max_n_ctx_for_kv = (remaining_vram_for_kv_only * 1024 * 1024) / bytes_per_kv_token_total_layers
                max_n_ctx_for_kv = int(max_n_ctx_for_kv // 64 * 64) # Often aligned to 64 or similar
                if args.n_ctx is None or max_n_ctx_for_kv < n_ctx :
                     print(f"With {best_n_gpu_layers} layers, you could potentially support n_ctx up to around {max_n_ctx_for_kv} (if KV cache is the limiter).")
                elif max_n_ctx_for_kv < n_ctx:
                     print(f"Warning: Your chosen n_ctx ({n_ctx}) with {best_n_gpu_layers} layers might exceed available VRAM for KV cache. Consider n_ctx around {max_n_ctx_for_kv}.")


    print("\nNote: This is an estimation. Actual VRAM usage can vary. Test with llama.cpp.")
    print("You might need to adjust --safety-margin-mb or --llama-overhead-mb based on observed behavior.")
    print("Ensure 'gguf' library is installed: pip install gguf")

if __name__ == "__main__":
    main()
