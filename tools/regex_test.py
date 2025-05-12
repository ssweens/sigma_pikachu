import re
from tools.gguf_analyzer_combined import GGUF_FILENAME_REGEX

regex = re.compile(GGUF_FILENAME_REGEX, re.IGNORECASE)
samples = [
    "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    "mellum-4b-sft-python.Q8_0.gguf",
    "Qwen3-8B-Q8_0.gguf",
    "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
    "qwen2.5-coder-3b-q8_0.gguf",
    "Qwen3-14B-Q4_K_M.gguf",
    "gemma3-12b-claude-3.7-sonnet-reasoning-distilled.Q8_0.gguf",
    "qwen2.5-coder-7b-instruct-q8_0.gguf",
    "Qwen3-30B-A3B-UD-Q4_K_XL.gguf",
    "Goekdeniz-Guelmez_Josiefied-Qwen3-8B-abliterated-v1-Q8_0.gguf",
    "Qwen3-4B-Q8_0.gguf"
]

for f in samples:
    m = regex.match(f)
    print(f"{f} => Match: {bool(m)}")
