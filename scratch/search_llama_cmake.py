with open(r"C:\Users\rr\Desktop\Ollama\ollama-for-amd\CMakeLists.txt", 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if "llama" in line.lower():
        print(f"{idx+1}: {line}", end="")
