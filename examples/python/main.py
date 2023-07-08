import http.client
import json
import os
import sys

if len(sys.argv) < 2:
    print("Usage: python main.py <model file>")
    sys.exit(1)

conn = http.client.HTTPConnection('localhost', 11434)

headers = { 'Content-Type': 'application/json' }

# generate text from the model
conn.request("POST", "/api/generate", json.dumps({
    'model': os.path.join(os.getcwd(), sys.argv[1]),
    'prompt': 'write me a short story',
    'stream': True
}), headers)

response = conn.getresponse()

def parse_generate(data):
    for event in data.decode('utf-8').split("\n"):
        if not event:
            continue
        yield event

if response.status == 200:
    for chunk in response:
        for event in parse_generate(chunk):
            print(json.loads(event)['response'], end="", flush=True)
