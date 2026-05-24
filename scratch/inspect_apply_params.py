with open(r"C:\Users\rr\Desktop\Ollama\ollama-for-amd\apply_COMPLETE.ps1", 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

for idx, line in enumerate(lines[:60]):
    print(f"{idx+1}: {line}", end="")
