# Log Analysis example

![loganalyzer 2023-11-10 08_53_29](https://github.com/ollama/ollama/assets/633681/ad30f1fc-321f-4953-8914-e30e24db9921)

This example shows one possible way to create a log file analyzer. It uses the model **mattw/loganalyzer** which is based on **codebooga**, a 34b parameter model.

To use it, run:

`python loganalysis.py <logfile>`

You can try this with the `logtest.logfile` file included in this directory.

## Running the Example

1. Ensure you have the `mattw/loganalyzer` model installed:

   ```bash
   ollama pull mattw/loganalyzer
   ```

2. Install the Python Requirements.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run the example:

   ```bash
   python loganalysis.py logtest.logfile
   ```

## Review the code

The first part of this example is a Modelfile that takes `codebooga` and applies a new System Prompt:

```plaintext
SYSTEM """
You are a log file analyzer. You will receive a set of lines from a log file for some software application, find the errors and other interesting aspects of the logs, and explain them so a new user can understand what they mean. If there are any steps they can do to resolve them, list the steps in your answer.
"""
```

This model is available at https://ollama.com/mattw/loganalyzer. You can customize it and add to your own namespace using the command `ollama create <namespace/modelname> -f <path-to-modelfile>` then `ollama push <namespace/modelname>`.

Then loganalysis.py scans all the lines in the given log file and searches for the word 'error'. When the word is found, the 10 lines before and after are set as the prompt for a call to the Generate API.

```python
data = {
  "prompt": "\n".join(error_logs),
  "model": "mattw/loganalyzer"
}
```

Finally, the streamed output is parsed and the response field in the output is printed to the line.

```python
response = requests.post("http://localhost:11434/api/generate", json=data, stream=True)
for line in response.iter_lines():
  if line:
    json_data = json.loads(line)
    if json_data['done'] == False:
      print(json_data['response'], end='')

```

## Next Steps

There is a lot more that can be done here. This is a simple way to detect errors, looking for the word error. Perhaps it would be interesting to find anomalous activity in the logs. It could be interesting to create embeddings for each line and compare them, looking for similar lines. Or look into applying Levenshtein Distance algorithms to find similar lines to help identify the anomalous lines.

Try different models and different prompts to analyze the data. You could consider adding retrieval augmented generation (RAG) to this to help understand newer log formats.
