# Turbo

> ⚠️ Turbo is preview

Ollama’s [Turbo](https://ollama.com/turbo) is a new way to run open-source models with acceleration from datacenter-grade hardware.

Currently, the following models are available in Turbo:

- `gpt-oss:20b`
- `gpt-oss:120b`

## Get started

### Ollama for macOS & Windows

Download Ollama

- Select a model such as `gpt-oss:20b` or `gpt-oss:120b`
- Click on **Turbo**. You’ll be prompted to create an account or sign in

### Ollama’s CLI

- [Sign up](https://ollama.com/signup) for an Ollama account
- Add your Ollama key [to ollama.com](https://ollama.com/settings/keys).

  On macOS and Linux:

  ```shell
  cat ~/.ollama/id_ed25519.pub
  ```

  On Windows:

  ```
  type "%USERPROFILE%\.ollama\id_ed25519.pub"
  ```

- Then run a model setting `OLLAMA_HOST` to `ollama.com`:
  ```shell
  OLLAMA_HOST=ollama.com ollama run gpt-oss:120b
  ```

### Ollama’s Python library

- Download Ollama's [Python library](https://github.com/ollama/ollama-python)
- [Sign up](https://ollama.com/signup) for an Ollama account
- Create an API key by visiting https://ollama.com/settings/keys

```python
from ollama import Client

client = Client(
    host="https://ollama.com",
    headers={'Authorization': '<api key>'}
)

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)
```

### Ollama’s JavaScript library

- Download Ollama's [JavaScript library](https://github.com/ollama/ollama-js)
- [Sign up](https://ollama.com/signup) for an Ollama account
- Create an API key by visiting https://ollama.com/settings/keys

```typescript
import { Ollama } from 'ollama';

const ollama = new Ollama({
  host: 'https://ollama.com',
  headers: {
	  Authorization: "Bearer <api key>"
  }
});

const response = await ollama.chat({
  model: 'gpt-oss:120b',
  messages: [{ role: 'user', content: 'Explain quantum computing' }],
  stream: true
});

for await (const part of response) {
    process.stdout.write(part.message.content)
}
```

### Community integrations

Turbo mode is also compatible with several community integrations.

#### Open WebUI

- Go to **settings** → **Admin settings** → **Connections**
- Under **Ollama API,** click **+**
- For the **URL** put `https://ollama.com`
- For the **API key,** create an API key on https://ollama.com/settings/keys and add it.
- Click **Save**

Now, if you navigate to the model selector, Turbo models should be available under **External**.
