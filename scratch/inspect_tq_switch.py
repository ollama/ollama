with open(r"C:\Users\rr\Desktop\Ollama\ollama-for-amd\apply_COMPLETE.ps1", 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if "EnableTurboQuant" in line:
        start = max(0, idx - 5)
        end = min(len(lines), idx + 10)
        print(f"--- Line {idx+1} ---")
        for i in range(start, end):
            print(f"{i+1}: {lines[i]}", end="")
