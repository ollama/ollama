# JSON Output Example

New in version 0.1.9 is support for JSON output. There are two python scripts in this example. `randomaddresses.py` generates random addresses from different countries. `predefinedschema.py` sets a template for the model to fill in.

## Review the Code

Both programs are basically the same, with a different prompt for each, demonstrating two different ideas. The key part of getting JSON out of a model is to state in the prompt or system prompt that it should respond using JSON, and specifying the `format` as `json` in the data body.

When running `randomaddresses.py` you will see that the schema changes and adapts to the chosen country.

In `predefinedschema.py`, a template has been specified in the prompt as well. It's been defined as JSON and then dumped into the prompt string to make it easier to work with.

Both examples turn streaming off so that we end up with the completed JSON all at once. We need to convert the `response.text` to JSON so that when we output it as a string we can set the indent spacing to make the output attractive.
