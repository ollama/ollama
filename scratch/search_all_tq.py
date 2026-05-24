import os

root_dir = r"C:\Users\rr\Desktop\Ollama\ollama-for-amd"
query = "turboquant"

for r, d, files in os.walk(root_dir):
    for f in files:
        if f.endswith(('.go', '.cpp', '.h', '.c', '.cu', '.txt', '.ps1', '.json', '.cmake', '.txt')):
            path = os.path.join(r, f)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    if query in content.lower():
                        print(f"Found in: {os.path.relpath(path, root_dir)}")
            except Exception as e:
                pass
