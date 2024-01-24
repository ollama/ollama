# Example Modelfile - Tweetwriter

This simple examples shows what you can do without any code, simply relying on a Modelfile. The file has two instructions:

1. FROM - The From instructions defines the parent model to use for this one. If you choose a model from the library, you can enter just the model name. For all other models, you need to specify the namespace as well. You could also use a local file. Just include the relative path to the converted, quantized model weights file. To learn more about creating that file, see the `import.md` file in the docs folder of this repository.
2. SYSTEM - This defines the system prompt for the model and overrides the system prompt from the parent model.

## Running the Example

1. Create the model:

   ```bash
   ollama create tweetwriter
   ```

2. Enter a topic to generate a tweet about.
3. Show the Modelfile in the REPL.

   ```bash
   /show modelfile
   ```

   Notice that the FROM and SYSTEM match what was in the file. But there is also a TEMPLATE and PARAMETER. These are inherited from the parent model.