# Simple Chat Example

The **chat** endpoint, available as of v0.1.14, is one of two ways to generate text from an LLM with Ollama. At a high level, you provide the endpoint an array of message objects with a role and content specified. Then with each output and prompt, you add more messages, which builds up the history.

## Run the Example

`npm start`

## Review the Code

You can see in the **chat** function that is actually calling the endpoint is simply done with:

```typescript
const body = {
  model: model,
  messages: messages
}

const response = await fetch("http://localhost:11434/api/chat", {
  method: "POST",
  body: JSON.stringify(body)
})
```

With the **generate** endpoint, you need to provide a `prompt`. But with **chat**, you provide `messages`. And the resulting stream of responses includes a `message` object with a `content` field.

The final JSON object doesn't provide the full content, so you will need to build the content yourself. In this example, **chat** takes the full array of messages and outputs the resulting message from this call of the chat endpoint.

In the **askQuestion** function, we collect `user_input` and add it as a message to our messages, and that is passed to the chat function. When the LLM is done responding, the output is added as another message to the messages array.

At the end, you will see a printout of all the messages.

## Next Steps

In this example, all generations are kept. You might want to experiment with summarizing everything older than 10 conversations to enable longer history with less context being used.
