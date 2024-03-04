# OpenAI compatibility

> **Note:** OpenAI compatibility is experimental and is subject to major adjustments including breaking changes. For fully-featured access to the Ollama API, see the Ollama [Python library](https://github.com/ollama/ollama-python), [JavaScript library](https://github.com/ollama/ollama-js) and [REST API](https://github.com/jmorganca/ollama/blob/main/docs/api.md).

Ollama provides experimental compatibility with parts of the [OpenAI API](https://platform.openai.com/docs/api-reference) to help connect existing applications to Ollama.

## Usage

### OpenAI Python library

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Say this is a test',
        }
    ],
    model='llama2',
)

embeddings = client.embeddings.create(
  model="llama2",
  input=["The food was delicious and the waiter...","Say this is a test"]
)
```

### OpenAI JavaScript library

```javascript
import OpenAI from 'openai'

const openai = new OpenAI({
  baseURL: 'http://localhost:11434/v1/',

  // required but ignored
  apiKey: 'ollama',
})

const chatCompletion = await openai.chat.completions.create({
  messages: [{ role: 'user', content: 'Say this is a test' }],
  model: 'llama2',
})

const embedding = await openai.embeddings.create({
  model: "llama2",
  input: "The quick brown fox jumped over the lazy dog",
});

console.log(embedding);
```

### `curl`

```
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama2",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello!"
            }
        ]
    }'
```

```
curl https://api.openai.com/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The food was delicious and the waiter...",
    "model": "llama2"
  }'
```

## Endpoints

### `/v1/chat/completions`

#### Supported features

- [x] Chat completions
- [x] Streaming
- [x] JSON mode
- [x] Reproducible outputs
- [ ] Vision
- [ ] Function calling
- [ ] Logprobs

#### Supported request fields

- [x] `model`
- [x] `messages`
  - [x] Text `content`
  - [ ] Array of `content` parts
- [x] `frequency_penalty`
- [x] `presence_penalty`
- [x] `response_format`
- [x] `seed`
- [x] `stop`
- [x] `stream`
- [x] `temperature`
- [x] `top_p`
- [x] `max_tokens`
- [ ] `logit_bias`
- [ ] `tools`
- [ ] `tool_choice`
- [ ] `user`
- [ ] `n`

#### Notes

- Setting `seed` will always set `temperature` to `0`
- `finish_reason` will always be `stop`
- `usage.prompt_tokens` will be 0 for completions where prompt evaluation is cached

## Models

Before using a model, pull it locally `ollama pull`:

```shell
ollama pull llama2
```

### Default model names

For tooling that relies on default OpenAI model names such as `gpt-3.5-turbo`, use `ollama cp` to copy an existing model name to a temporary name:

```
ollama cp llama2 gpt-3.5-turbo
```

Afterwards, this new model name can be specified the `model` field:

```shell
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": "Hello!"
            }
        ]
    }'
```

## Endpoints

### `/v1/embeddings`

#### Supported request fields

- [x] `model`
- [x] `input`
- [ ] `encoding_format`

#### Notes

- Setting `seed` will always set `temperature` to `0`
- `usage.prompt_tokens` will be 0

## Models

Before using a model, pull it locally `ollama pull`:

```shell
ollama pull llama2
```

