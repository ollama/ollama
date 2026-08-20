# Function calling

![function calling 2023-11-16 16_12_58](https://github.com/ollama/ollama/assets/633681/a0acc247-9746-45ab-b325-b65dfbbee4fb)

One of the features added to some models is 'function calling'. It's a bit of a confusing name. It's understandable if you think that means the model can call functions, but that's not what it means. Function calling simply means that the output of the model is formatted in JSON, using a preconfigured schema, and uses the expected types. Then your code can use the output of the model and call functions with it. Using the JSON format in Ollama, you can use any model for function calling. 

The two examples provided can extract information out of the provided texts. The first example uses the first couple of chapters from War and Peace by Lev Nikolayevich Tolstoy, and extracts the names and titles of the characters introduced in the story. The second example uses a more complicated schema to pull out addresses and event information from a series of emails.

## Running the examples

1. Clone this repo and navigate to the `examples/typescript-functioncalling` directory.
2. Install the dependencies with `npm install`.
3. Review the `wp.txt` file.
4. Run `tsx extractwp.ts`.
5. Review the `info.txt` file.
6. Run `tsx extractemail.ts`.

## Review the Code

Both examples do roughly the same thing with different source material. They both use the same system prompt, which tells the model to expect some instructions and a schema. Then we inject the schema into the prompt and generate an answer.

The first example, `extractwp.ts`, outputs the resulting JSON to the console, listing the characters introduced at the start of War and Peace. The second example, `extractemail.ts`, is a bit more complicated, extracting two different types of information: addresses and events. It outputs the results to a JSON blob, then the addresses are handed off to one function called `reportAddresses` and the events are handed off to another function called `reportEvents`.

Notice that both examples are using the model from Intel called `neural-chat`. This is not a model tuned for function calling, yet it performs very well at this task.

## Next Steps

Try exporting some of your real emails to the input file and seeing how well the model does. Try pointing the first example at other books. You could even have it cycle through all the sections and maybe add up the number of times any character is seen throughout the book, determining the most important characters. You can also try out different models.
