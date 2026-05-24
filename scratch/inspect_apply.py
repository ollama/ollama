with open(r"C:\Users\rr\Desktop\Ollama\ollama-for-amd\apply_COMPLETE.ps1", 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if "GGML_HIP_TURBOQUANT" in line:
        start = max(0, idx - 10)
        end = min(len(lines), idx + 10)
        print(f"--- Occurrence at line {idx+1} ---")
        for i in range(start, end):
            print(f"{i+1}: {lines[i]}", end="")
