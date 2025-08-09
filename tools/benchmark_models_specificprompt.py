import subprocess
#import matplotlib.pyplot as plt
import pathlib
import pprint
import re
import time
from pathlib import Path
from typing import List
import requests
import sys
import threading
import os
import psutil # Added for VRAM/memory monitoring attempt
from openai import OpenAI
import psutil

N_PREDICT = 1024

# --- OpenAI API Configuration ---
OPENAI_API_BASE_URL = "http://192.168.1.55:9999/v1"  # Example: "https://api.openai.com/v1" or your local API endpoint
OPENAI_API_KEY = "sk-your-api-key-here"  # Replace with your actual API key

MODELS_DIR = "/Volumes/DabbleFiles/Models" # !!! IMPORTANT: Set your models directory !!!
MODELS_TO_TEST = [] #sorted(set(extract_model_paths_from_config())) # if not MODELS_TO_TEST else MODELS_TO_TEST

os.environ["HF_HOME"] = MODELS_DIR + "/huggingface"


# List of local models to test (e.g., Llama-2-7B-Chat.Q4_K_M.gguf)
# Make sure these models exist in your MODELS_DIR
MODELS_TO_TEST = [
   # "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
    #"qwen3/Qwen3-30B-A3B-Q4_K_M.gguf",
    #"open-thoughts_OpenThinker3-7B-Q4_K_M.gguf",
    #"qwen3/Qwen3-4B-Q4_K_M.gguf",
    # "qwen3/Qwen3-8B-UD-Q6_K_XL.gguf",
    # "qwen3/Qwen3-30B-A3B-UD-Q4_K_XL.gguf",
    # "DeepSeek-R1-Distill-Llama-8B-F16.gguf",
    # "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    # "nvidia_OpenCodeReasoning-Nemotron-14B-Q4_K_M.gguf",
    # "Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_M.gguf",
    # Add more models here
]

for model in pathlib.Path(MODELS_DIR).rglob("*.gguf"):
    relative_path = model.relative_to(MODELS_DIR)
    MODELS_TO_TEST.append(str(relative_path))



MODELS_TO_TEST = [
#     'TheDrummer_Big-Alice-28B-v1-Q4_K_M.gguf',
# 'DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf',
# 'Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf',
#'Dolphin-Mistral-24B-Venice-Edition.i1-Q4_K_M.gguf',


# 'DeepSeek-V2-Lite-Chat.Q4_K_M.gguf',
# 'DeepSeek-V2-Lite.Q4_K_M.gguf',
# 'DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf',
# 'deepseek-v2-lite-q6_k.gguf',
# #'DeepSeek-V2-Lite-Chat-Uncensored-Unbiased-Reasoner.Q4_K_M.gguf',
# 'DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf',


#'gemma3-12b-claude-3.7-sonnet-reasoning-distilled.Q8_0.gguf',
# 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
# 'Dolphin-Mistral-24B-Venice-Edition.i1-Q4_K_M.gguf',
# 'Llama-3.2-4X3B-MOE-Ultra-Instruct-10B-D_AU-Q4_k_m.gguf',
# 'Llama-3.2-3B-Instruct-Q4_K_M.gguf',
# 'Tesslate_Tessa-Rust-T1-7B-Q6_K.gguf',
# 'open-thoughts_OpenThinker3-7B-Q4_K_M.gguf',
# 'DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf',
#  'L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-Q4_k_m.gguf',
#  'Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf',


# 'deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-Q6_K_L.gguf',
#  'Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf',
#  'Mistral-Nemo-Instruct-2407-Q4_K_M.gguf',
#  'Llama-3.2-3B-Instruct-f16.gguf',
#  'nvidia_Nemotron-Research-Reasoning-Qwen-1.5B-bf16.gguf',
#  'Phi-4-reasoning-plus-Q4_K_M.gguf',
#  'mistralai_Devstral-Small-2505-Q4_K_M.gguf',
#  'Bootes-Qwen3_Coder-Reasoning.Q4_K_M.gguf',
#  'meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf',
#  'DeepSeek-R1-Distill-Llama-8B-F16.gguf',
#  'nvidia_OpenCodeReasoning-Nemotron-14B-Q4_K_M.gguf',
#  'Mixtral-8x7B-v0.1.i1-Q4_K_S.gguf',
#  'Seed-Coder-8B-Reasoning-UD-Q4_K_XL.gguf',
#  'Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf',
#  'Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_M.gguf',
#  'SmolLM2-135M-Instruct-Q4_K_M.gguf',
#  'Codestral-22B-v0.1-Q6_K.gguf',
#  'Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf',
#  'gemma3/mmproj-BF16.gguf',
#  'gemma3/gemma-3-27b-it-Q4_K_M.gguf',
#  'gemma3/gemma-3-12b-it-Q6_K.gguf',
#  'gemma3/google_gemma-3-27b-it-qat-Q4_0.gguf',
#  'qwen3/Qwen_Qwen3-0.6B-bf16.gguf',
#  'qwen3/Qwen3-14B-Q6_K.gguf',
#  'qwen3/Qwen3-4B-Q6_K.gguf',
#  'qwen3/Qwen3-14B-Q4_K_M.gguf',
#  'qwen3/Qwen3-8B-UD-Q6_K_XL.gguf',
#  'qwen3/Qwen3-30B-A3B-UD-Q4_K_XL.gguf',
#  'qwen3/Qwen3-0.6B-Q4_K_M.gguf',
#  'qwen3/Qwen3-4B-Q4_K_M.gguf',
#  'qwen3/Qwen3-30B-A3B-Q4_K_S.gguf',
#  'qwen3/Qwen3-32B-Q4_K_M.gguf',
#  'qwen3/Qwen3-1.7B-BF16.gguf',
#  'qwen3/Qwen3-30B-A3B-Q4_K_M.gguf',
#  'vision/Devstral-Small-2505-Q4_K_M.gguf']
'Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf'
]

# List of OpenAI-compatible models to test
API_MODELS_TO_TEST = [
    #"qwen3:30b-a3b",
    #"qwen3:4b",
    #"qwen3:1.7b",
    # "gpt-3.5-turbo",
    # "gpt-4",
    # "llama3", # Example for local OpenAI-compatible server like Ollama
    "Qwen/Qwen3-Coder-30B-A3B-Instruct"
]

# List of MLX models to test (requires mlx_lm package)
MLX_MODELS_TO_TEST = [
    # #"mlx-community/Dolphin3.0-Llama3.1-8B-4bit",
    # #"mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit-AWQ",
    # "mlx-community/Qwen3-30B-A3B-4bit",  # 150 s
    # "mlx-community/OpenThinker3-7B-4bit", # 119s
    # #"mlx-community/Qwen2.5-3B-Instruct-4bit",
    # #"mlx-community/Qwen2.5-7B-Instruct-4bit",
    # # "mlx-community/gemma-3n-E4B-it-bf16",  # Not working in mlx_lm atm
    # #"mlx-community/QwQ-32B-4bit",  # Too big 
    # #"mlx-community/Qwen3-14B-4bit-DWQ-053125",  # 260 s... thinking??
    # "mlx-community/Kimi-VL-A3B-Thinking-4bit",  # 68s!
    # #"mlx-community/QwQ-DeepSeek-R1-SkyT1-Flash-Lightest-32B-mlx-4Bit",  # Too big
    # "mlx-community/AceReason-Nemotron-14B-4bit",  # 263 s
    # "mlx-community/gemma-3-27b-it-qat-4bit", # Super slow, 400+ seconds
    # # "mlx-community/Mistral-Nemo-Instruct-2407-4bit",  # 149.116224527359 s
    # "mlx-community/Qwen3-14B-4bit-AWQ",  #255.52090203762054 s thinking, 
    # #"mlx-community/Qwen3-30B-A3B-4bit"
    # # "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx",  #55.79968845844269 s
    # # "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",  # 90.80711543560028 s
    # # "mlx-community/Llama-3.2-3B-Instruct-4bit",  # 47.81926929950714 s
    # # "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",  #76.93342363834381 s
    # # Add more MLX models here
    #"mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ"
    #"mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
    "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-dwq-v2"
]

BENCHMARK_PROMPT = open('tools/sample_reddit_prompt.md', 'r').read()

pprint.pprint(MODELS_TO_TEST)
pprint.pprint(API_MODELS_TO_TEST)

def bytes2human(n):
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if abs(n) >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%sB" % n

# --- VRAM Monitoring Components ---

def get_current_vram_gb(pid=None):
    """
    Attempts to get current process memory usage (USS) in GB using psutil.
    NOTE: This measures system RAM (USS), not dedicated GPU VRAM.
          On Apple Silicon, this is part of the unified memory, but it's not
          a direct measure of GPU-specific allocation in the same way as
          nvidia-smi would report for discrete GPUs.

    If `pid` is provided, it attempts to get memory for that specific process.
    Returns memory usage in GB, or 0.0 if not measurable or an error occurs.
    """
    if pid is None:
        print("[MEMORY_MONITOR_DEBUG] No PID provided.")
        return 0.0

    try:
        if not psutil.pid_exists(pid):
            #print(f"[MEMORY_MONITOR_DEBUG] Process {pid} does not exist.")
            return 0.0
        
        p = psutil.Process(pid)
        # Ensure the process object is still valid (e.g., hasn't exited between pid_exists and now)
        if not p.is_running():
            #print(f"[MEMORY_MONITOR_DEBUG] Process {pid} is no longer running.")
            return 0.0
            
        #mem_info = p.memory_full_info() # USS (Unique Set Size) is often a good measure of actual RAM used by a process.
        #print(mem_info)
        #print(f"[MEMORY_MONITOR_DEBUG] rss: {mem_info.rss / (1024 ** 3)} GB, vms: {mem_info.vms / (1024 ** 3)} GB, uss: {getattr(mem_info, 'uss', 'N/A') / (1024 ** 3)} GB")
        v_mem_info = psutil.virtual_memory()
        #for k, v in v_mem_info._asdict().items():
        #    print(f"[MEMORY_MONITOR_DEBUG] {k}: {v / (1024 ** 3)} GB")

        #print(p.memory_info())
        #rss_gb = mem_info.rss / (1024 ** 3)
        wired_gb = v_mem_info.wired / (1024 ** 3)
        print(f"[MEMORY_MONITOR_DEBUG] Process {pid} - Wired Memory: {wired_gb:.3f} GB")
        return wired_gb
    except psutil.NoSuchProcess:
        #print(f"[MEMORY_MONITOR_DEBUG] Process {pid} not found (NoSuchProcess).")
        return 0.0
    except psutil.AccessDenied:
        print(f"[MEMORY_MONITOR_DEBUG] Access denied for process {pid} - normal if only once.")
        return 0.0
    except Exception as e:
        print(f"[MEMORY_MONITOR_DEBUG] Error getting memory for process {pid}: {e}")
        return 0.0

class VRAMMonitor:
    def __init__(self, pid_to_monitor, sample_interval_seconds=15.0):
        self.pid_to_monitor = pid_to_monitor
        self.sample_interval_seconds = sample_interval_seconds
        self.peak_vram_gb = 0.0
        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_started = False
        self.error_message = None

    def _is_process_running(self):
        if self.pid_to_monitor is None:
            return False
        try:
            os.kill(self.pid_to_monitor, 0)  # Check if process exists
        except OSError:
            return False
        return True

    def _monitor_loop(self):
        self.monitoring_started = True
        try:
            while not self._stop_event.is_set():
                if not self._is_process_running():
                    break  # Process ended

                # Try to get VRAM. Use system-wide as a fallback if per-PID is hard.
                # The effectiveness of this depends entirely on get_current_vram_gb()

                # Do a wait cycle to start to allow prior process memory to settle
                self._stop_event.wait(self.sample_interval_seconds)
                current_vram = get_current_vram_gb(self.pid_to_monitor)
                
                if current_vram > self.peak_vram_gb:
                    self.peak_vram_gb = current_vram
        except Exception as e:
            self.error_message = f"Error in VRAM monitor loop: {e}"
            # print(self.error_message) # For debugging

    def start(self):
        if self.pid_to_monitor is None:
            self.error_message = "VRAM Monitor: No PID to monitor."
            # print(self.error_message)
            return
        if not self._monitor_thread.is_alive():
            self._stop_event.clear()
            self._monitor_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=self.sample_interval_seconds * 5 + 1)
        # print(f"[VRAM_MONITOR_DEBUG] VRAM Monitor stopped. Peak VRAM for PID {self.pid_to_monitor} (or system): {self.peak_vram_gb:.3f} GB")
        if self.error_message:
            # print(f"[VRAM_MONITOR_DEBUG] {self.error_message}")
            pass
        return self.peak_vram_gb

# --- End of VRAM Monitoring Components ---




def extract_model_paths_from_config() -> List[str]:
    """Grep the corral.yaml config for uncommented filenames ending in .gguf and check if they exist."""
    config_path = Path.home() / ".config" / "sigma_pikachu" / "corral.yaml"
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    model_paths = []
    try:
        with open(config_path, 'r') as file:
            for line in file:
                line = line.strip()
                # Ignore comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Find lines ending with .gguf (possibly quoted)
                if ".gguf" in line:
                    # Remove inline comments
                    line = line.split("#", 1)[0].strip()
                    # Extract filename (handles quotes and possible YAML formatting)
                    match = re.search(r'([\'"]?)([^\'"\s]+\.gguf)\1', line)
                    if match:
                        model_file = match.group(2)
                        if Path(model_file).exists():
                            model_paths.append(model_file)
                        else:
                            print(f"✗ Model file not found: {model_file}")
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        sys.exit(1)

    return model_paths


# Defining the command template
cmd_parts = [
    "llama-cli",
    "--seed 147369852",
    "--threads 6",
    "--model {model}",
    "--ctx_size 32768",
    "--prio 3",
    "-fa",
    "--cache-type-k q8_0 --cache-type-v q8_0",
    "--n_gpu_layers 999",
    "--top_k 40",
    "--min_p 0.01",
    "--temp 0.6",
    "-no-cnv",
    "--no-display-prompt",
    "--file tools/sample_reddit_prompt.md",
    "--simple-io",
    "--single-turn",
    "--jinja",
    #"--no-mmap",  # --mlock blows it up when used in tandem on some big boys
    "--reasoning-budget 0",  # No reasoning budget for this benchmark, turn off thinking
    "--predict {}".format(N_PREDICT),  #keep same as MLX and others
    #"--verbose"
]
cmd = " ".join(cmd_parts)
# --model-draft /Users/ssweens/models/qwen3/Qwen3-0.6B-Q4_K_M.gguf \
# --draft-max 16 \
# --draft-min 1"

# Defining the number of runs for each evaluation
n_api_runs = 3  # APIs probably need more runs due to network variability
n_local_runs = 1  # Local models can be faster, so fewer runs are sufficient
n_mlx_runs = 2  # MLX models similar to local models

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE_URL)

# --- Function to benchmark OpenAI-compatible API models ---
def benchmark_openai_model(client, api_model_name, prompt_content, n_predict=N_PREDICT):
    prompt_metrics_batch = []
    generation_metrics_batch = []
    total_time_batch = []

    print(f"Running API benchmark for {api_model_name} ...")

    print("Running a quick warm-up run to ensure the model is loaded...")
    api_url = f"{OPENAI_API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    warmup_payload = {
        "model": api_model_name,
        "messages": [{"role": "user", "content": "Tell me a joke."}],
        "max_tokens": 128
    }
    warmup_response = requests.post(api_url, headers=headers, json=warmup_payload)
    if warmup_response.status_code != 200:
        raise Exception(f"Warm-up API call failed: {warmup_response.status_code} {warmup_response.text}")
    warmup_response = warmup_response.json()
    print(f"Warm-up run completed for {api_model_name}.")
    print(f"Starting benchmark runs for {api_model_name} with prompt length {len(prompt_content)}...")

    for run in range(n_api_runs):
        start_time = time.time()
        try:
            
            api_url = f"{OPENAI_API_BASE_URL}/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": api_model_name,
                "messages": [{"role": "user", "content": prompt_content}],
                "max_tokens": n_predict,
                "temperature": 0.5,
                "stream_options": {"include_usage": True},
                "cache_prompt": False
            }
            response = requests.post(api_url, headers=headers, json=payload)
            if response.status_code != 200:
                raise Exception(f"API call failed: {response.status_code} {response.text}")
            response = response.json()
            end_time = time.time()
            total_time = end_time - start_time

            print(f"API call completed for {api_model_name} (run {run+1}/{n_api_runs}).")
            print(f"Response: {response['timings']}")
            prompt_tokens = response['usage']['prompt_tokens']
            completion_tokens = response['usage']['completion_tokens']
            prompt_time = response['timings']['prompt_ms']/1000.0
            completion_time = response['timings']['predicted_ms']/1000.0
            prompt_tps = response['timings']['prompt_per_second']
            completion_tps = response['timings']['predicted_per_second']

            ## for output only
            total_tokens_per_second = 0
            if completion_tokens > 0 and total_time > 0:
                total_tokens_per_second = (prompt_tokens + completion_tokens) / total_time

            prompt_metrics = {
                'time': prompt_time,
                'tokens': prompt_tokens,
                'tps': prompt_tps
            }
            generation_metrics = {
                'time': completion_time,
                'tokens': completion_tokens,
                'tps': completion_tps
            }

            print(f"\t {api_model_name} API | run {run+1}/{n_api_runs} | Total Time: {round(total_time, 2)}s | Prompt Tokens: {prompt_tokens} | Completion Tokens: {completion_tokens} | Tokens/Sec: {round(total_tokens_per_second, 2)}")

            prompt_metrics_batch.append(prompt_metrics)
            generation_metrics_batch.append(generation_metrics)
            total_time_batch.append(total_time)

        except Exception as e:
            print(f"API call failed for {api_model_name} (run {run+1}/{n_api_runs}): {e}")
            continue

    prompt_avg = {
        'time': sum(d['time'] for d in prompt_metrics_batch) / len(prompt_metrics_batch) if prompt_metrics_batch else 0.0,
        'tokens': sum(d['tokens'] for d in prompt_metrics_batch) / len(prompt_metrics_batch) if prompt_metrics_batch else 0.0,
        'tps': sum(d['tps'] for d in prompt_metrics_batch) / len(prompt_metrics_batch) if prompt_metrics_batch else 0.0,
    }
    generation_avg = {
        'time': sum(d['time'] for d in generation_metrics_batch) / len(generation_metrics_batch) if generation_metrics_batch else 0.0,
        'tokens': sum(d['tokens'] for d in generation_metrics_batch) / len(generation_metrics_batch) if generation_metrics_batch else 0.0,
        'tps': sum(d['tps'] for d in generation_metrics_batch) / len(generation_metrics_batch) if generation_metrics_batch else 0.0,
    }
    total_avg = {
        'time': sum(total_time_batch) / len(total_time_batch) if total_time_batch else 0.0,
    }

    return {
        'prompt': prompt_avg,
        'generation': generation_avg,
        'total': total_avg
    }


# --- Function to benchmark MLX models ---
def benchmark_mlx_model(model_name, prompt_content, n_predict=N_PREDICT):
    """Benchmark MLX model using mlx_lm.generate command line"""
    prompt_metrics_batch = []
    generation_metrics_batch = []
    total_time_batch = []
    peak_vram_batch = []
    
    print(f"Running MLX benchmark for {model_name} ...")

    # Ensure the MLX model is downloaded and cached
    subprocess.run(f"huggingface-cli download {model_name}", shell=True, env=os.environ)
    
    for run in range(n_mlx_runs):
        start_time = time.time()
        
        try:
            # Build the MLX command - read prompt from the existing file to avoid shell quoting issues
            mlx_cmd = f"python -m mlx_lm generate --model {model_name} --prompt \"$(cat tools/sample_reddit_prompt.md)\" --system-prompt \"\\no_think\" --max-tokens {n_predict} --temp 0.5 --kv-bits 8 --chat-template-config '{{\"enable_thinking\":false}}'"
            print(f"Executing: {mlx_cmd}")
            result = subprocess.run(mlx_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            output = result.stdout.decode()
            lines = output.strip().split('\n')
            
            end_time = time.time()
            
            prompt_tokens = 0
            generation_tokens = 0
            generation_tps = 0
            
            for line in lines:
                if line.startswith("Prompt:"):
                    # Extract prompt tokens
                    prompt_match = re.search(r"Prompt:\s+(\d+)\s+tokens,\s+([\.\d]+)\s+tokens-per-sec", line)
                    if prompt_match:
                        prompt_tokens = int(prompt_match.group(1))
                        prompt_tps = float(prompt_match.group(2))
                elif line.startswith("Generation:"):
                    # Extract generation tokens and tokens per second
                    gen_match = re.search(r"Generation:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec", line)
                    if gen_match:
                        generation_tokens = int(gen_match.group(1))
                        generation_tps = float(gen_match.group(2))
                elif line.startswith("Peak memory:"):
                    # Extract peak memory usage
                    peak_mem_match = re.search(r"Peak memory:\s+([\.\d]+)\s+GB", line)
                    if peak_mem_match:
                        peak_vram_usage_gb = float(peak_mem_match.group(1))
                    else:
                        peak_vram_usage_gb = 0.0
            # Debug output for the first run to see what we're parsing
            if run == 0:
                print("Debug - Last few lines of output:")
                for line in lines[-8:]:
                    print(f"  '{line}'")
                print(f"Debug - Found {prompt_tokens} prompt tokens, {prompt_tps} tps, {generation_tokens} generation tokens, {generation_tps} tps")
           
            total_time = end_time - start_time
            print(f"\t {model_name} MLX | run {run+1}/{n_mlx_runs} | total: {total_time} s | prompt: {prompt_tokens} tokens in {prompt_tokens/prompt_tps} s ({prompt_tps} tps) | generation: {generation_tokens} tokens in {generation_tokens/generation_tps} s ({generation_tps} tps)")

            prompt_metrics = {
                'time': prompt_tokens/prompt_tps,  
                'tokens': prompt_tokens,
                'tps': prompt_tps
            }
            generation_metrics = {
                'time': generation_tokens/generation_tps,
                'tokens': generation_tokens,
                'tps': generation_tps
            }

            prompt_metrics_batch.append(prompt_metrics)
            generation_metrics_batch.append(generation_metrics)
            total_time_batch.append(total_time)
            peak_vram_batch.append(peak_vram_usage_gb)

        except Exception as e:
            print(f"MLX generation failed for {model_name} (run {run+1}/{n_mlx_runs}): {e}")
            continue

    prompt_avg = {
        'time': sum(d['time'] for d in prompt_metrics_batch) / len(prompt_metrics_batch) if prompt_metrics_batch else 0.0,
        'tokens': sum(d['tokens'] for d in prompt_metrics_batch) / len(prompt_metrics_batch) if prompt_metrics_batch else 0.0,
        'tps': sum(d['tps'] for d in prompt_metrics_batch) / len(prompt_metrics_batch) if prompt_metrics_batch else 0.0,
    }
    generation_avg = {
        'time': sum(d['time'] for d in generation_metrics_batch) / len(generation_metrics_batch) if generation_metrics_batch else 0.0,
        'tokens': sum(d['tokens'] for d in generation_metrics_batch) / len(generation_metrics_batch) if generation_metrics_batch else 0.0,
        'tps': sum(d['tps'] for d in generation_metrics_batch) / len(generation_metrics_batch) if generation_metrics_batch else 0.0,
    }
    total_avg = {
        'time': sum(total_time_batch) / len(total_time_batch) if total_time_batch else 0.0,
        'vram_gb_max': sum(peak_vram_batch) / len(peak_vram_batch) if peak_vram_batch else 0.0,
    }

    return {
        'prompt': prompt_avg,
        'generation': generation_avg,
        'total': total_avg
    }


# --- Function to benchmark local models ---
def benchmark_local_model(model_path, n_predict=N_PREDICT):
    """Benchmark local model using llama-cli"""
    prompt_metrics_batch = []
    generation_metrics_batch = []
    total_time_batch = []
    peak_vram_batch = []

    peak_vram_usage_gb = 0.0  # Initialize peak VRAM tracking
    
    print(f"Running local benchmark for {model_path} ...")
    
    for run in range(n_local_runs):
        full_cmd = cmd.format(model=model_path)
        print(f"Executing: {full_cmd}")
        
        process = None
        vram_monitor = None
        peak_vram_usage_gb = 0.0
        decoded_output = ""
        process_return_code = -1 # Default to an error-like code

        try:
            # Using shell=True can be a security risk if full_cmd contains untrusted input.
            # Consider passing full_cmd as a list of arguments if possible, e.g., cmd_list = shlex.split(full_cmd)
            # and then shell=False. For now, retaining shell=True as in original.
            process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                text=True,
                errors='ignore'  # For decoding stdout/stderr
            )

            # Start VRAM monitoring in parallel
            if process.pid:
                # print(f"[DEBUG] Monitoring VRAM for PID: {process.pid}")
                vram_monitor = VRAMMonitor(pid_to_monitor=process.pid)
                vram_monitor.start()
            else:
                # print("[DEBUG] Could not get PID to start VRAM monitor.")
                pass # VRAM monitoring won't start

            # .communicate() waits for the process to complete and reads all output.
            stdout_str, _ = process.communicate() # stderr is merged into stdout_str
            decoded_output = stdout_str
            process_return_code = process.returncode

            # print(f"[DEBUG] Subprocess finished with return code: {process_return_code}")

        except FileNotFoundError:
            command_name = str(full_cmd).split()[0] if isinstance(full_cmd, str) else "N/A"
            decoded_output = f"Error: Command not found ('{command_name}'). Check command and system PATH."
            process_return_code = -2 # Custom code for file not found
        except subprocess.TimeoutExpired:
            decoded_output = f"Error: Command '{str(full_cmd)}' timed out."
            process_return_code = -3 # Custom code for timeout
            if process:
                process.kill()
                # Try to get any remaining output
                try:
                    remaining_output, _ = process.communicate(timeout=1)
                    decoded_output += remaining_output
                except:
                    pass # Ignore errors during cleanup after timeout
        except Exception as e:
            decoded_output = f"An unexpected error occurred while running/monitoring the command: {type(e).__name__} - {e}"
            process_return_code = -4 # Custom code for other exceptions
            if process and process.poll() is None:
                process.kill()
                try:
                    remaining_output, _ = process.communicate(timeout=1)
                    decoded_output += remaining_output
                except:
                    pass
        finally:
            if vram_monitor:
                # print("[DEBUG] Stopping VRAM monitor...")
                peak_vram_usage_gb = vram_monitor.stop()
                # print(f"[DEBUG] Peak VRAM recorded: {peak_vram_usage_gb:.3f} GB")
            
            if process and process.poll() is None: # If process somehow still running
                # print(f"[DEBUG] Process {process.pid} still running in finally, attempting to terminate.")
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # print(f"[DEBUG] Process {process.pid} did not terminate gracefully, killing.")
                    process.kill()
                    process.wait()
        
        # Assign to 'output' as expected by subsequent original code
        output = decoded_output
        
        # Now you have 'output' (the subprocess's stdout/stderr) and 
        # 'peak_vram_usage_gb' (the monitored peak VRAM, 0.0 if placeholder is used or error).
        # You also have 'process_return_code'.
        # You can log or use peak_vram_usage_gb as needed.
        # For example:
        # print(f"Command output: {output}")
        # print(f"Peak VRAM Usage (GB): {peak_vram_usage_gb:.3f}")
        # print(f"Process Return Code: {process_return_code}")

        # if run == 0:
        #     print("Debug - Last few lines of output:")
        #     for line in output.split("\n")[-20:]:
        #         print(f"  '{line}'")
        #     #print(f"Debug - Found {prompt_tokens} prompt tokens, {prompt_tps} tps, {generation_tokens} generation tokens, {generation_tps} tps")


        # Extracting the token time, evaluation time, and prompt evaluation time using regular expressions
        # Use the eval time's ms per token value, not the sampling time
        prompt_eval_time_match = re.search(r"llama_perf_context_print: prompt eval time =\s+(\d+.\d+) ms\s+/\s+(\d+) tokens\s+\(\s+(\d+\.\d+)\s+ms per token,\s+(\d+\.\d+)\s+tokens per second\)", output)
        gen_eval_time_match = re.search(r"llama_perf_context_print:\s+eval time =\s+(\d+\.\d+) ms\s+/\s+(\d+) runs\s+\(\s+(\d+\.\d+)\s+ms per token,\s+(\d+\.\d+)\s+tokens per second\)", output)
        total_time_match = re.search(r"llama_perf_context_print:\s+total time =\s+(\d+\.\d+) ms", output)
        
        prompt_metrics = {
            'time': float(prompt_eval_time_match.group(1))/1000.0 if prompt_eval_time_match else 0.0,
            'tokens': int(prompt_eval_time_match.group(2)) if prompt_eval_time_match else 0.0,
            'tps': float(prompt_eval_time_match.group(4)) if prompt_eval_time_match else 0.0
        }

        generation_metrics = {
            'time': float(gen_eval_time_match.group(1))/1000.0 if gen_eval_time_match else 0.0,
            'tokens': int(gen_eval_time_match.group(2)) if gen_eval_time_match else 0.0,
            'tps': float(gen_eval_time_match.group(4)) if gen_eval_time_match else 0.0
        }

        total_time = float(total_time_match.group(1))/1000.0 if total_time_match else 0.0

        print(f"\t {model_path} model | run {run+1}/{n_local_runs} | total: {total_time} s | prompt: {prompt_metrics['tokens']} tokens in {prompt_metrics['time']} s ({prompt_metrics['tps']} tps) | generation: {generation_metrics['tokens']} tokens in {generation_metrics['time']} s ({generation_metrics['tps']} tps) | peak VRAM: {peak_vram_usage_gb:.3f} GB")

        prompt_metrics_batch.append(prompt_metrics)
        generation_metrics_batch.append(generation_metrics)
        total_time_batch.append(total_time)
        peak_vram_batch.append(peak_vram_usage_gb)
        
        # Track peak VRAM across all runs (note: peak_vram_usage_gb from the current run's monitor is already the maximum for this run)

    # Calculate averages for each metric in the batch
    prompt_avg = {
        'time': sum(d['time'] for d in prompt_metrics_batch) / len(prompt_metrics_batch) if prompt_metrics_batch else 0.0,
        'tokens': sum(d['tokens'] for d in prompt_metrics_batch) / len(prompt_metrics_batch) if prompt_metrics_batch else 0.0,
        'tps': sum(d['tps'] for d in prompt_metrics_batch) / len(prompt_metrics_batch) if prompt_metrics_batch else 0.0,
    }
    generation_avg = {
        'time': sum(d['time'] for d in generation_metrics_batch) / len(generation_metrics_batch) if generation_metrics_batch else 0.0,
        'tokens': sum(d['tokens'] for d in generation_metrics_batch) / len(generation_metrics_batch) if generation_metrics_batch else 0.0,
        'tps': sum(d['tps'] for d in generation_metrics_batch) / len(generation_metrics_batch) if generation_metrics_batch else 0.0,
    }
    total_avg = {
        'time': sum(total_time_batch) / len(total_time_batch) if total_time_batch else 0.0,
        'vram_gb_max': sum(peak_vram_batch) / len(peak_vram_batch) if peak_vram_batch else 0.0,
    }
    
    return {
        'prompt': prompt_avg,
        'generation': generation_avg,
        'total': total_avg
    }

local_model_metrics = {}
mlx_model_metrics = {}
api_model_metrics = {}

# --- Benchmark MLX Models ---
print("\n--- Benchmarking MLX Models ---")
for mlx_model in MLX_MODELS_TO_TEST:
    mlx_metrics = benchmark_mlx_model(mlx_model, BENCHMARK_PROMPT, n_predict=1024)
    if mlx_metrics:
        mlx_model_metrics[mlx_model] = mlx_metrics

# --- Benchmark Local Models ---
print("\n--- Benchmarking Local Models ---")
for model in MODELS_TO_TEST:
    local_metrics = benchmark_local_model("/users/ssweens/models/" + model)
    if local_metrics:
        local_model_metrics[model] = local_metrics

# --- Benchmark API Models ---
print("\n--- Benchmarking API Models ---")
for api_model in API_MODELS_TO_TEST:
    api_metrics = benchmark_openai_model(client, api_model, BENCHMARK_PROMPT, n_predict=1024)
    if api_metrics:
        api_model_metrics[api_model] = api_metrics

# --- Print Summary Results ---
print("\n--- Benchmark Summary ---")
print("\nLocal Models:")
for model, metrics in local_model_metrics.items():
    print(f"  Model: {model}")

    print(f"    Prompt - Time: {metrics['prompt']['time']} s, Tokens: {metrics['prompt']['tokens']}, TPS: {metrics['prompt']['tps']}")
    print(f"    Generation - Time: {metrics['generation']['time']} s, Tokens: {metrics['generation']['tokens']}, TPS: {metrics['generation']['tps']}")
    print(f"    Total - Time: {metrics['total']['time']} s")
    print(f"    Total - TPS: {(metrics['prompt']['tokens'] + metrics['generation']['tokens']) / metrics['total']['time']} tps")
    print(f"    Total - Peak VRAM: {metrics['total']['vram_gb_max']} GB")

print("\nMLX Models:")
for model, metrics in mlx_model_metrics.items():
    print(f"  Model: {model}")
    print(f"    Prompt - Time: {metrics['prompt']['time']} s, Tokens: {metrics['prompt']['tokens']}, TPS: {metrics['prompt']['tps']}")
    print(f"    Generation - Time: {metrics['generation']['time']} s, Tokens: {metrics['generation']['tokens']}, TPS: {metrics['generation']['tps']}")
    print(f"    Total - Time: {metrics['total']['time']} s")
    print(f"    Total - TPS: {(metrics['prompt']['tokens'] + metrics['generation']['tokens']) / metrics['total']['time']} tps")
    print(f"    Total - Peak VRAM: {metrics['total']['vram_gb_max']} GB")

print("\nAPI Models:")
for model, metrics in api_model_metrics.items():
    print(f"  Model: {model}")
    print(f"    Prompt - Time: {metrics['prompt']['time']} s, Tokens: {metrics['prompt']['tokens']}, TPS: {metrics['prompt']['tps']}")
    print(f"    Generation - Time: {metrics['generation']['time']} s, Tokens: {metrics['generation']['tokens']}, TPS: {metrics['generation']['tps']}")
    print(f"    Total - Time: {metrics['total']['time']} s")
    print(f"    Total - TPS: {(metrics['prompt']['tokens'] + metrics['generation']['tokens']) / metrics['total']['time']} tps")
    print(f"    Total - Peak VRAM: N/A (API models do not report VRAM)")



# The plotting code is commented out as per original script and not directly modified for API results.
# If plotting for API results is desired, it would need separate implementation.
# fig, axs = plt.subplots(1, 3, figsize=(12, 4))
# ... (plotting code remains commented)
# plt.show()
