<p align="center">
  <a href="https://ollama.com">
    <img src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7" alt="ollama" width="200"/>
  </a>
</p>

# Ollama

Comienza a construir con modelos abiertos.

> 🌐 [English](../README.md)

## Descargar

### macOS

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

o [descargar manualmente](https://ollama.com/download/Ollama.dmg)

### Windows

```shell
irm https://ollama.com/install.ps1 | iex
```

o [descargar manualmente](https://ollama.com/download/OllamaSetup.exe)

### Linux

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

[Instrucciones de instalación manual](https://docs.ollama.com/linux#manual-install)

### Docker

La [imagen oficial de Docker de Ollama](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` está disponible en Docker Hub.

### Bibliotecas

- [ollama-python](https://github.com/ollama/ollama-python)
- [ollama-js](https://github.com/ollama/ollama-js)

### Comunidad

- [Discord](https://discord.gg/ollama)
- [𝕏 (Twitter)](https://x.com/ollama)
- [Reddit](https://reddit.com/r/ollama)

## Primeros pasos

```
ollama
```

Se te pedirá ejecutar un modelo o conectar Ollama a tus agentes o aplicaciones existentes como `claude`, `codex`, `openclaw` y más.

### Programación

Para iniciar una integración específica:

```
ollama launch claude
```

Las integraciones soportadas incluyen [Claude Code](https://docs.ollama.com/integrations/claude-code), [Codex](https://docs.ollama.com/integrations/codex), [Droid](https://docs.ollama.com/integrations/droid) y [OpenCode](https://docs.ollama.com/integrations/opencode).

### Asistente de IA

Usa [OpenClaw](https://docs.ollama.com/integrations/openclaw) para convertir Ollama en un asistente personal de IA a través de WhatsApp, Telegram, Slack, Discord y más:

```
ollama launch openclaw
```

### Chatear con un modelo

Ejecuta y chatea con [Gemma 3](https://ollama.com/library/gemma3):

```
ollama run gemma3
```

Consulta [ollama.com/library](https://ollama.com/library) para la lista completa.

Consulta la [guía de inicio rápido](https://docs.ollama.com/quickstart) para más detalles.

## API REST

Ollama tiene una API REST para ejecutar y gestionar modelos.

```
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{
    "role": "user",
    "content": "¿Por qué el cielo es azul?"
  }],
  "stream": false
}'
```

Consulta la [documentación de la API](https://docs.ollama.com/api) para todos los endpoints.

### Python

```
pip install ollama
```

```python
from ollama import chat

response = chat(model='gemma3', messages=[
  {
    'role': 'user',
    'content': '¿Por qué el cielo es azul?',
  },
])
print(response.message.content)
```

### JavaScript

```
npm i ollama
```

```javascript
import ollama from "ollama";

const response = await ollama.chat({
  model: "gemma3",
  messages: [{ role: "user", content: "¿Por qué el cielo es azul?" }],
});
console.log(response.message.content);
```

## Backends soportados

- [llama.cpp](https://github.com/ggml-org/llama.cpp) proyecto fundado por Georgi Gerganov.

## Documentación

- [Referencia del CLI](https://docs.ollama.com/cli)
- [Referencia de la API REST](https://docs.ollama.com/api)
- [Importar modelos](https://docs.ollama.com/import)
- [Referencia de Modelfile](https://docs.ollama.com/modelfile)
- [Compilar desde el código fuente](https://github.com/ollama/ollama/blob/main/docs/development.md)
