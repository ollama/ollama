# OpenAI Compatibility

> [!NOTE]
> OpenAI compatibility is experimental and is subject to major adjustments including breaking changes. For fully-featured access to the Ollama API, see the Ollama [Python library](https://github.com/ollama/ollama-python), [JavaScript library](https://github.com/ollama/ollama-js) and [REST API](https://github.com/ollama/ollama/blob/main/docs/api.md).

Ollama provides experimental compatibility with parts of the [OpenAI API](https://platform.openai.com/docs/api-reference) to help connect existing applications to Ollama.

## Usage

### OpenAI Python Library

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
    model='llama3.2',
)

completion = client.completions.create(
    model="llama3.2",
    prompt="Say this is a test",
)

list_completion = client.models.list()

model = client.models.retrieve("llama3.2")

embeddings = client.embeddings.create(
    model="all-minilm",
    input=["why is the sky blue?", "why is the grass green?"],
)
```

#### Multimodal Support

```python
response = client.chat.completions.create(
    model="llava",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,..."
                    }
                },
            ],
        }
    ],
    max_tokens=300,
)
```

#### Structured Outputs

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Define the schema for the response
class FriendInfo(BaseModel):
    name: str
    age: int 
    is_available: bool

class FriendList(BaseModel):
    friends: list[FriendInfo]

try:
    completion = client.beta.chat.completions.parse(
        temperature=0,
        model="llama3.1:8b",
        messages=[
            {"role": "user", "content": "I have two friends. The first is Ollama 22 years old busy saving the world, and the second is Alonso 23 years old and wants to hang out. Return a list of friends in JSON format"}
        ],
        response_format=FriendList,
    )

    friends_response = completion.choices[0].message
    if friends_response.parsed:
        print(friends_response.parsed)
    elif friends_response.refusal:
        print(friends_response.refusal)
except Exception as e:
    print(f"Error: {e}")
```

### OpenAI JavaScript Library

```javascript
import OpenAI from 'openai'

const openai = new OpenAI({
  baseURL: 'http://localhost:11434/v1/',

  // required but ignored
  apiKey: 'ollama',
})

async function main() {
  const chatCompletion = await openai.chat.completions.create({
    messages: [
      {
        role: 'user',
        content: 'Say this is a test',
      },
    ],
    model: 'llama3.2',
  })

  const completion = await openai.completions.create({
    model: 'llama3.2',
    prompt: 'Say this is a test',
  })

  const models = await openai.models.list()

  const model = await openai.models.retrieve('llama3.2')

  const embeddings = await openai.embeddings.create({
    model: 'all-minilm',
    input: ['why is the sky blue?', 'why is the grass green?'],
  })
}

main()
```

## Supported Endpoints

Ollama provides compatibility with the following OpenAI API endpoints:

- `/v1/chat/completions` - Generate a chat completion
- `/v1/completions` - Generate a completion
- `/v1/embeddings` - Generate embeddings
- `/v1/models` - List available models
- `/v1/models/{model}` - Retrieve a model

## Limitations

- Not all OpenAI API features are supported
- Some parameters may be ignored or have different behavior
- Error messages and response formats may differ from the OpenAI API

## Further Reading

For more detailed information about OpenAI compatibility, including advanced usage and examples, please refer to the [full OpenAI compatibility documentation](https://github.com/ollama/ollama/blob/main/docs/openai.md).