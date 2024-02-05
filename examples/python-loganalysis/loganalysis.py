import sys
import re
import requests
import json

# prelines and postlines represent the number of lines of context to include in the output around the error
prelines = 10
postlines = 10

def find_errors_in_log_file():
  if len(sys.argv) < 2:
    print("Usage: python loganalysis.py <filename>")
    return

  log_file_path = sys.argv[1]
  with open(log_file_path, 'r') as log_file:
    log_lines = log_file.readlines()

  error_logs = []
  for i, line in enumerate(log_lines):
      if "error" in line.lower():
          start_index = max(0, i - prelines)
          end_index = min(len(log_lines), i + postlines + 1)
          error_logs.extend(log_lines[start_index:end_index])

  return error_logs

error_logs = find_errors_in_log_file()

data = {
  "prompt": "\n".join(error_logs), 
  "model": "mattw/loganalyzer"
}

response = requests.post("http://localhost:11434/api/generate", json=data, stream=True)
for line in response.iter_lines():
  if line:
    json_data = json.loads(line)
    if json_data['done'] == False:
      print(json_data['response'], end='', flush=True)

