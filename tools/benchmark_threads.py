import subprocess
import matplotlib.pyplot as plt
import re

# Defining the command template
cmd = "llama-cli \
 --seed 147369852 \
 --threads {threads} \
 --n_predict 128 \
 --model /Users/ssweens/models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
 --top_k 40 \
 --top_p 0.9 \
 --temp 0.5 \
 --repeat_last_n 64 \
 --repeat_penalty 1.1 \
 -p \"Write a funny joke:\" \
 --ignore-eos \
 -no-cnv"

# Defining the range of threads to loop over
min_threads = 1 
max_threads = 10
step = 1

# Defining the number of runs for each thread cmd evaluation
n_runs = 3 

# Initializing the lists to store the results
threads_list = []
avg_token_time = []
max_token_time = []
min_token_time = []
token_time_list = []
eval_time_list = []
prompt_eval_time_list = []

for threads in range(min_threads, max_threads + 1, step):

    print(f"Running with {threads} threads...")

    avg_token_time = []
    eval_times = []
    prompt_eval_times = []

    for run in range(n_runs):
        #print(cmd.format(threads=threads))
        result = subprocess.run(cmd.format(threads=threads), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        output = result.stdout.decode()

        # Extracting the token time, evaluation time, and prompt evaluation time using regular expressions
        token_time = float(re.search(r"\s+(\d+\.\d+) ms per token", output).group(1))
        eval_time = float(re.search(r"llama_perf_context_print:\s+eval time =\s+(\d+\.\d+) ms", output).group(1))
        prompt_eval_time = float(re.search(r"llama_perf_context_print: prompt eval time =\s+(\d+\.\d+) ms", output).group(1))

        print(f"\t {threads} threads | run {run+1}/{n_runs} | current token time {round(token_time, 2)} ms - eval time {round(eval_time, 2)} ms - prompt eval time {round(prompt_eval_time, 2)} ms")

        avg_token_time.append(token_time)
        eval_times.append(eval_time)
        prompt_eval_times.append(prompt_eval_time)

    # Get the average token time, evaluation time, and prompt evaluation time for the current number of threads
    min_token_time.append(min(avg_token_time))
    max_token_time.append(max(avg_token_time))
    avg_token_time = sum(avg_token_time) / len(avg_token_time)
    
    avg_eval_time  = sum(eval_times) / len(eval_times)
    avg_prompt_eval_time = sum(prompt_eval_times) / len(prompt_eval_times)

    token_time_list.append(avg_token_time)
    eval_time_list.append(avg_eval_time)
    prompt_eval_time_list.append(avg_prompt_eval_time)
    threads_list.append(threads)


# Plot the results
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot token time vs number of threads
axs[0].plot(threads_list, token_time_list)
axs[0].plot(threads_list, min_token_time, label='min token time', color='lightgreen', linewidth=0.75)
axs[0].plot(threads_list, max_token_time, label='max token time', color='lightcoral', linewidth=0.75)
axs[0].fill_between(threads_list, min_token_time, max_token_time, alpha=0.2, color='lightblue')
axs[0].set_xlabel("Number of threads")
axs[0].set_ylabel("Token time (ms)")
axs[0].set_title("Token time vs Number of threads")
axs[0].legend()
axs[0].grid(color='lightgray', linestyle='--', linewidth=0.5)

# Plot evaluation time vs number of threads
axs[1].plot(threads_list, eval_time_list)
axs[1].set_xlabel("Number of threads")
axs[1].set_ylabel("Evaluation time (ms)")
axs[1].set_title("Evaluation time vs Number of threads")
axs[1].grid(color='lightgray', linestyle='--', linewidth=0.5)

# Plot evaluation time vs number of threads
axs[2].plot(threads_list, prompt_eval_time_list)
axs[2].set_xlabel("Number of threads")
axs[2].set_ylabel("Prompt evaluation time (ms)")
axs[2].set_title("Prompt evaluation time vs Number of threads")
axs[2].grid(color='lightgray', linestyle='--', linewidth=0.5)

plt.show()
