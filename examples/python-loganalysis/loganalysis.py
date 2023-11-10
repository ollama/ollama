import sys
import re
import requests
import json

prelines = 10
postlines = 10

def find_errors_in_log_file():
  if len(sys.argv) < 2:
    print("Usage: python loganalysis.py <filename>")
    return

  log_file_path = sys.argv[1]
  with open(log_file_path, 'r') as log_file:
    log_lines = log_file.readlines()

  error_lines = []
  for i, line in enumerate(log_lines):
    if re.search('error', line, re.IGNORECASE):
      error_lines.append(i)

  error_logs = []
  for error_line in error_lines:
    start_index = max(0, error_line - prelines)
    end_index = min(len(log_lines), error_line + postlines)
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







