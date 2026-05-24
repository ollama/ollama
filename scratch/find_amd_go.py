import os

root_dir = r"C:\Users\rr\Desktop\Ollama\ollama-for-amd"

for r, d, files in os.walk(root_dir):
    for f in files:
        if f.endswith('.go') and 'amd' in f.lower():
            print(os.path.relpath(os.path.join(r, f), root_dir))
