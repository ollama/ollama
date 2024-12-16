# Code Iterate Example

This example is a based on the simple chat example, and it is used for code generation. In particular, it provides a little bit of extra functionality to execute the generated code and capture any error messages for feedback to the code generation model. It performs the following loop:

- take chat instructions and generate code
- execute generated code
- if execution fails, provide the error message to the LLM as a user message

This is a very simplistic piece of software, it simply saves the generated code to a temp file and uses subprocess to invoke the python interpreter. If this program is started in a conda environment, it attempts to run the generated python program in that same environment. No attempt is made to create a conda environment or install any python packages.

Finally, under reasonable exit conditions, this program appends the log of messages to chatlog.md

This example is best suited for generating small complete stand alone python programs that can be run and tested in the currently active conda environment. Once the generated python program is functional, you can integrate it into more complex software.

## Running the Example

1. Ensure you have the `qwen2.5-coder:7b` model installed:

   ```bash
   ollama pull qwen2.5-coder:7b
   ```

2. Install the Python Requirements.

   ```bash
   pip install -r requirements.txt
   ```

3. Run the example:

   ```bash
   python codeiterate.py
   ```

## Review the Code

You can see in the **chat** function that actually calling the endpoint is done simply with:

```python
r = requests.post(
  "http://0.0.0.0:11434/api/chat",
  json={"model": model, "messages": messages, "stream": True},
)
```

With the **chat** endpoint, you provide `messages`. The resulting stream of responses includes a `message` object with a `content` field.

The final JSON object doesn't provide the full content, so you will need to build the content yourself.

In the **main** function, we collect `user_input` and add it as a message to our messages and that is passed to the chat function. When the LLM is done responding the output is added as another message.

If the environment variables **CONDA_EXE** and **CONDA_DEFAULT_ENV** are present, then the generated program is executed using the following command:
   ```bash
    conda run -n <your_current_conda_environment> python <generated_program>
   ```

## Next Steps

In this example, all generations are kept. You might want to experiment with summarizing everything older than 10 conversations to enable longer history with less context being used.
