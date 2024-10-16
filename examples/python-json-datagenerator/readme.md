# JSON Output Example

![llmjson 2023-11-10 15_31_31](https://github.com/ollama/ollama/assets/633681/e599d986-9b4a-4118-81a4-4cfe7e22da25)

There are two python scripts in this example. `randomaddresses.py` generates random addresses from different countries. `predefinedschema.py` sets a template for the model to fill in.

## Running the Example

1. Ensure you have the `llama3.2` model installed:

   ```bash
   ollama pull llama3.2
   ```

2. Install the Python Requirements.

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Random Addresses example:

   ```bash
   python randomaddresses.py
   ```

4. Run the Predefined Schema example:

   ```bash
   python predefinedschema.py
   ```

## Review the Code

Both programs are basically the same, with a different prompt for each, demonstrating two different ideas. The key part of getting JSON out of a model is to state in the prompt or system prompt that it should respond using JSON, and specifying the `format` as `json` in the data body.

```python
prompt = f"generate one realistically believable sample data set of a persons first name, last name, address in {country}, and  phone number. Do not use common names. Respond using JSON. Key names should with no backslashes, values should use plain ascii with no special characters."

data = {
    "prompt": prompt,
    "model": model,
    "format": "json",
    "stream": False,
    "options": {"temperature": 2.5, "top_p": 0.99, "top_k": 100},
}
```

When running `randomaddresses.py` you will see that the schema changes and adapts to the chosen country.

In `predefinedschema.py`, a template has been specified in the prompt as well. It's been defined as JSON and then dumped into the prompt string to make it easier to work with.

Both examples turn streaming off so that we end up with the completed JSON all at once. We need to convert the `response.text` to JSON so that when we output it as a string we can set the indent spacing to make the output easy to read.

```python
response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
json_data = json.loads(response.text)

print(json.dumps(json.loads(json_data["response"]), indent=2))
```
