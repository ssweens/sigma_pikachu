import subprocess
import matplotlib.pyplot as plt
import re
import time
from pathlib import Path
from typing import List
import sys
from openai import OpenAI

N_PREDICT = 1024

# --- OpenAI API Configuration ---
OPENAI_API_BASE_URL = "http://localhost:9999/v1"  # Example: "https://api.openai.com/v1" or your local API endpoint
OPENAI_API_KEY = "sk-your-api-key-here"  # Replace with your actual API key

MODELS_TO_TEST = [] #sorted(set(extract_model_paths_from_config())) # if not MODELS_TO_TEST else MODELS_TO_TEST

# List of local models to test (e.g., Llama-2-7B-Chat.Q4_K_M.gguf)
# Make sure these models exist in your MODELS_DIR
MODELS_TO_TEST = [
   # "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
    "qwen3/Qwen3-30B-A3B-Q4_K_M.gguf",
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

# List of OpenAI-compatible models to test
API_MODELS_TO_TEST = [
    #"qwen3:30b-a3b",
    #"qwen3:4b",
    #"qwen3:1.7b",
    # "gpt-3.5-turbo",
    # "gpt-4",
    # "llama3", # Example for local OpenAI-compatible server like Ollama
]

# List of MLX models to test (requires mlx_lm package)
MLX_MODELS_TO_TEST = [
    #"mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit-AWQ",
    "mlx-community/Qwen3-30B-A3B-4bit"
    #"mlx-community/OpenThinker3-7B-4bit"
    #"mlx-community/Qwen2.5-3B-Instruct-4bit",
    #"mlx-community/Qwen2.5-7B-Instruct-4bit",
    # "mlx-community/Llama-3.2-3B-Instruct-4bit",
    # Add more MLX models here
]

BENCHMARK_PROMPT = open('tools/sample_reddit_prompt.md', 'r').read()

MODELS_DIR = "/Users/ssweens/models" # !!! IMPORTANT: Set your models directory !!!




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
n_local_runs = 2  # Local models can be faster, so fewer runs are sufficient
n_mlx_runs = 2  # MLX models similar to local models

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE_URL)

# --- Function to benchmark OpenAI-compatible API models ---
def benchmark_openai_model(client, api_model_name, prompt_content, n_predict=N_PREDICT):
    # total_request_times = []
    # prompt_tokens_list = []
    # completion_tokens_list = []
    # tokens_per_second_list = []
    prompt_metrics_batch = []
    generation_metrics_batch = []
    total_time_batch = []

    print(f"Running API benchmark for {api_model_name} ...")

    print("Running a quick warm-up run to ensure the model is loaded...")
    response = client.chat.completions.create(
                model=api_model_name,
                messages=[{"role": "user", "content": "Tell me a joke."}],
                max_tokens=128
            )
    print(f"Warm-up run completed for {api_model_name}.")
    print(f"Starting benchmark runs for {api_model_name} with prompt length {len(prompt_content)}...")

    for run in range(n_api_runs):
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=api_model_name,
                messages=[{"role": "user", "content": prompt_content}],
                max_tokens=n_predict,
                temperature=0.5, # Align with local model temp
            )
            end_time = time.time()
            total_time = end_time - start_time

            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            ## for output only
            total_tokens_per_second = 0
            if completion_tokens > 0 and total_time > 0:
                total_tokens_per_second = (prompt_tokens + completion_tokens) / total_time

            prompt_metrics = {
                #'time': total_time,
                'tokens': prompt_tokens,
                #'tps': prompt_tokens / total_time if total_time > 0 else 0.0
            }
            generation_metrics = {
                #'time': total_time,
                'tokens': completion_tokens,
                #'tps': completion_tokens / total_time if total_time > 0 else 0.0
            }

            #print(f"\t {api_model_name} API | run {run+1}/{n_api_runs} | Total Time: {round(total_request_time, 2)}s | Tokens/Sec: {round(tokens_per_second, 2)} | Prompt Tokens: {prompt_tokens} | Completion Tokens: {completion_tokens}")
            print(f"\t {api_model_name} API | run {run+1}/{n_api_runs} | Total Time: {round(total_time, 2)}s | Prompt Tokens: {prompt_tokens} | Completion Tokens: {completion_tokens} | Tokens/Sec: {round(total_tokens_per_second, 2)}")

            # total_request_times.append(total_request_time)
            # prompt_tokens_list.append(prompt_tokens)
            # completion_tokens_list.append(completion_tokens)
            # tokens_per_second_list.append(tokens_per_second)

            prompt_metrics_batch.append(prompt_metrics)
            generation_metrics_batch.append(generation_metrics)
            total_time_batch.append(total_time)

        except Exception as e:
            print(f"API call failed for {api_model_name} (run {run+1}/{n_api_runs}): {e}")
            # Append None or 0 for failed runs to maintain list length
            # total_request_times.append(0)
            # prompt_tokens_list.append(0)
            # completion_tokens_list.append(0)
            # tokens_per_second_list.append(0)
            continue
    
    # Calculate averages for API model
    # avg_total_request_time = sum(total_request_times) / n_api_runs
    # avg_prompt_tokens = sum(prompt_tokens_list) / n_api_runs
    # avg_completion_tokens = sum(completion_tokens_list) / n_api_runs
    # avg_tokens_per_second = sum(tokens_per_second_list) / n_api_runs

    prompt_avg = {
        'tokens': sum(d['tokens'] for d in prompt_metrics_batch) / len(prompt_metrics_batch) if prompt_metrics_batch else 0.0,
    }
    generation_avg = {
        'tokens': sum(d['tokens'] for d in generation_metrics_batch) / len(generation_metrics_batch) if generation_metrics_batch else 0.0,
    }
    total_avg = {
        'time': sum(total_time_batch) / len(total_time_batch) if total_time_batch else 0.0,
    }

    # return {
    #     'avg_total_request_time': avg_total_request_time,
    #     'avg_prompt_tokens': avg_prompt_tokens,
    #     'avg_completion_tokens': avg_completion_tokens,
    #     'avg_tokens_per_second': avg_tokens_per_second
    # }

    return {
        'prompt': prompt_avg,
        'generation': generation_avg,
        'total': total_avg
    }


# --- Function to benchmark MLX models ---
def benchmark_mlx_model(model_name, prompt_content, n_predict=N_PREDICT):
    """Benchmark MLX model using mlx_lm.generate command line"""
    # token_times = []
    # eval_times = []
    # prompt_eval_times = []
    # tokens_per_second_list = []
    # total_times = []

    prompt_metrics_batch = []
    generation_metrics_batch = []
    total_time_batch = []
    
    print(f"Running MLX benchmark for {model_name} ...")
    
    for run in range(n_mlx_runs):
        start_time = time.time()
        
        try:
            # Build the MLX command - read prompt from the existing file to avoid shell quoting issues
            mlx_cmd = f"python -m mlx_lm generate --model {model_name} --prompt \"$(cat tools/sample_reddit_prompt.md)\" --max-tokens {n_predict} --temp 0.5 --kv-bits 8"
            
            result = subprocess.run(mlx_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            output = result.stdout.decode()
            lines = output.strip().split('\n')
            
            end_time = time.time()
            
            # Extract tokens generated and timing from MLX output
            # MLX outputs timing info like:
            # Prompt: 9 tokens, 7.060 tokens-per-sec
            # Generation: 50 tokens, 49.190 tokens-per-sec
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
            
            # Debug output for the first run to see what we're parsing
            # if run == 0:
            #     print("Debug - Last few lines of output:")
            #     for line in lines[-8:]:
            #         print(f"  '{line}'")
            #     print(f"Debug - Found {prompt_tokens} prompt tokens, {prompt_tps} tps, {generation_tokens} generation tokens, {generation_tps} tps")
           
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
    
    print(f"Running local benchmark for {model_path} ...")
    
    for run in range(n_local_runs):
        full_cmd = cmd.format(model=model_path)
        print(f"Executing: {full_cmd}")
        
        try:
            result = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            output = result.stdout.decode(errors='ignore')  # Handle UTF-8 decoding errors gracefully

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

            print(f"\t {model_path} model | run {run+1}/{n_local_runs} | total: {total_time} s | prompt: {prompt_metrics['tokens']} tokens in {prompt_metrics['time']} s ({prompt_metrics['tps']} tps) | generation: {generation_metrics['tokens']} tokens in {generation_metrics['time']} s ({generation_metrics['tps']} tps)")

            prompt_metrics_batch.append(prompt_metrics)
            generation_metrics_batch.append(generation_metrics)
            total_time_batch.append(total_time)            
        except Exception as e:
            print(f"Local model execution failed for {model_path} (run {run+1}/{n_local_runs}): {e}")
            continue

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
    }

    return {
        'prompt': prompt_avg,
        'generation': generation_avg,
        'total': total_avg
    }

local_model_metrics = {}
mlx_model_metrics = {}
api_model_metrics = {}

# --- Benchmark Local Models ---
print("\n--- Benchmarking Local Models ---")
for model in MODELS_TO_TEST:
    local_metrics = benchmark_local_model("/users/ssweens/models/" + model)
    if local_metrics:
        local_model_metrics[model] = local_metrics

# --- Benchmark MLX Models ---
print("\n--- Benchmarking MLX Models ---")
for mlx_model in MLX_MODELS_TO_TEST:
    mlx_metrics = benchmark_mlx_model(mlx_model, BENCHMARK_PROMPT, n_predict=1024)
    if mlx_metrics:
        mlx_model_metrics[mlx_model] = mlx_metrics

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

print("\nMLX Models:")
for model, metrics in mlx_model_metrics.items():
    print(f"  Model: {model}")
    print(f"    Prompt - Time: {metrics['prompt']['time']} s, Tokens: {metrics['prompt']['tokens']}, TPS: {metrics['prompt']['tps']}")
    print(f"    Generation - Time: {metrics['generation']['time']} s, Tokens: {metrics['generation']['tokens']}, TPS: {metrics['generation']['tps']}")
    print(f"    Total - Time: {metrics['total']['time']} s")
    print(f"    Total - TPS: {(metrics['prompt']['tokens'] + metrics['generation']['tokens']) / metrics['total']['time']} tps")

print("\nAPI Models:")
for model, metrics in api_model_metrics.items():
    print(f"  Model: {model}")
    print(f"    Prompt - Tokens: {metrics['prompt']['tokens']}")
    print(f"    Generation - Tokens: {metrics['generation']['tokens']}")
    print(f"    Total - Time: {metrics['total']['time']} s")
    print(f"    Total - TPS: {(metrics['prompt']['tokens'] + metrics['generation']['tokens']) / metrics['total']['time']} tps")




# The plotting code is commented out as per original script and not directly modified for API results.
# If plotting for API results is desired, it would need separate implementation.
# fig, axs = plt.subplots(1, 3, figsize=(12, 4))
# ... (plotting code remains commented)
# plt.show()
