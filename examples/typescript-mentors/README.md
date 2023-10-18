# Ask the Mentors

This example demonstrates how one would create a set of 'mentors' you can have a conversation with. The mentors are generated using the `character-generator.ts` file. This will use **Stable Beluga 70b** to create a bio and list of verbal ticks and common phrases used by each person. Then `mentors.ts` will take a question, and choose three of the 'mentors' and start a conversation with them. Occasionally, they will talk to each other, and other times they will just deliver a set of monologues. It's fun to see what they do and say.

## Usage

```bash
ts-node ./character-generator.ts "Lorne Greene"
```

This will create `lornegreene/Modelfile`. Now you can create a model with this command:

```bash
ollama create lornegreene -f lornegreene/Modelfile
```

If you want to add your own mentors, you will have to update the code to look at your namespace instead of **mattw**. Also set the list of mentors to include yours.

```bash
ts-node ./mentors.ts "What is a Jackalope?"
```
