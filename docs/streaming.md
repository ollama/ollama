# Streaming responses in the Ollama Client API

## JavaScript / TypeScript / Deno

```javascript
const pull = async () => {
  const request = await fetch("http://localhost:11434/api/pull", {
    method: "POST",
    body: JSON.stringify({ name: "llama2:7b-q5_0" }),
  });

  const reader = await request.body?.pipeThrough(new TextDecoderStream());
  if (!reader) throw new Error("No reader");
  for await (const chunk of reader) {
    const out = JSON.parse(chunk);
    if (out.status.startsWith("downloading")) {
      console.log(`${out.status} - ${(out.completed / out.total) * 100}%`);
    }
  }
}

pull();
```

## Python

```python
import requests
import json
response = requests.post("http://localhost:11434/api/pull", json={"name": "llama2:7b-q5_0"}, stream=True)
for data in response.iter_lines():
  out = json.loads(data)
  if "completed" in out:
    print(out["completed"] / out["total"] * 100)
```
