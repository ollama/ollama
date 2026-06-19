import urllib.request
import json
import re

url = "https://huggingface.co/api/models/bartowski/Llama-3.2-1B-Instruct-GGUF/tree/main"
req = urllib.request.Request(url)
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())
    
for f in data:
    if f["path"].endswith(".gguf"):
        # extract quant
        match = re.search(r'-([A-Za-z0-9_.]+)\.gguf$', f["path"])
        quant = match.group(1) if match else "unknown"
        print(f"File: {f['path']}, Size: {f['size']/1e9:.2f} GB, Quant: {quant}")
