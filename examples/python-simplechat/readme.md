# Simple Chat Example

The **chat** endpoint is one of two ways to generate text from an LLM with Ollama, and is introduced in version 0.1.14. At a high level, you provide the endpoint an array of objects with a role and content specified. Then with each output and prompt, you add more of those role/content objects, which builds up the history.

## Running the Example

1. Ensure you have the `llama3.2` model installed:

   ```bash
   ollama pull llama3.2
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

You can see in the **chat** function that actually calling the endpoint is done simply with:

```python
r = requests.post(
  "http://0.0.0.0:11434/api/chat",
  json={"model": model, "messages": messages, "stream": True},
)
```

With the **generate** endpoint, you need to provide a `prompt`. But with **chat**, you provide `messages`. And the resulting stream of responses includes a `message` object with a `content` field.

The final JSON object doesn't provide the full content, so you will need to build the content yourself.

In the **main** function, we collect `user_input` and add it as a message to our messages and that is passed to the chat function. When the LLM is done responding the output is added as another message.

## Next Steps

In this example, all generations are kept. You might want to experiment with summarizing everything older than 10 conversations to enable longer history with less context being used.
