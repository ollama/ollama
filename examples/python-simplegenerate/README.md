# Simple Generate Example

This is a simple example using the **Generate** endpoint.

## Running the Example

1. Ensure you have the `stablelm-zephyr` model installed:

   ```bash
   ollama pull stablelm-zephyr
   ```

2. Install the Python Requirements.

   ```bash
   pip install -r requirements.txt
   ```

3. Run the example:

   ```bash
   python client.py
   ```

## Review the Code

The **main** function simply asks for input, then passes that to the generate function. The output from generate is then passed back to generate on the next run.

The **generate** function uses `requests.post` to call `/api/generate`, passing the model, prompt, and context. The `generate` endpoint returns a stream of JSON blobs that are then iterated through, looking for the response values. That is then printed out. The final JSON object includes the full context of the conversation so far, and that is the return value from the function.
