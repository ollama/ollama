# n8n Workflows for Ollama

This directory contains n8n workflow templates for integrating with Ollama. [n8n](https://n8n.io/) is an open-source workflow automation tool that allows you to connect Ollama with other applications and services.

## Prerequisites

1. **Ollama** running locally (default: `http://localhost:11434`)
2. **n8n** installed and running ([Installation Guide](https://docs.n8n.io/hosting/installation/))

## Available Workflows

### 1. Basic Chat Workflow (`ollama-chat-workflow.json`)

A simple workflow that demonstrates how to send chat messages to Ollama and receive responses.

**Features:**
- Manual trigger for testing
- Send chat messages to Ollama's `/api/chat` endpoint
- Extract and format the response

**Usage:**
1. Import the workflow into n8n
2. Configure the model name (default: `llama3.2`)
3. Run the workflow manually or connect to other triggers

### 2. Webhook Workflow (`ollama-webhook-workflow.json`)

A more advanced workflow that exposes Ollama through a webhook API with multiple actions.

**Features:**
- Webhook trigger for external integrations
- Route requests to different Ollama endpoints based on action
- Support for chat, generate, and list models

**Supported Actions:**
- `chat` - Send chat messages with system prompts
- `generate` - Generate text completions
- `list` - List available models

**Example Request:**
```bash
curl -X POST http://localhost:5678/webhook/ollama-webhook \
  -H "Content-Type: application/json" \
  -d '{
    "action": "chat",
    "model": "llama3.2",
    "message": "What is the capital of France?",
    "system_prompt": "You are a helpful geography assistant."
  }'
```

## Importing Workflows

1. Open n8n in your browser
2. Click on the menu (☰) and select "Import from File"
3. Select the desired workflow JSON file
4. Click "Import"
5. Activate the workflow

## Configuration

### Ollama Server URL

By default, the workflows connect to `http://localhost:11434`. To change the Ollama server URL:

#### Method 1: Edit in n8n UI

1. Open the workflow in n8n
2. Click on the HTTP Request node (e.g., "Ollama Chat API" or "Ollama Generate API")
3. Find the **URL** field
4. Change `http://localhost:11434` to your Ollama server address
5. Click **Save** to apply changes

#### Method 2: Edit the JSON file directly

Before importing, edit the workflow JSON file:

1. Open the `.json` file in a text editor
2. Find all occurrences of `http://localhost:11434`
3. Replace with your Ollama server URL (e.g., `http://192.168.1.100:11434` or `http://ollama.example.com:11434`)
4. Save the file and import into n8n

**Example URLs:**
- Local: `http://localhost:11434`
- LAN server: `http://192.168.1.100:11434`
- Docker: `http://host.docker.internal:11434`
- Remote: `https://ollama.example.com:11434`

#### Verify Connection

Test your Ollama server is reachable:

```bash
curl http://YOUR_OLLAMA_URL/api/version
```

Expected response:
```json
{"version": "0.5.1"}
```

### Model Selection

You can change the default model by:

1. Modifying the JSON body in the HTTP Request node
2. Or passing the `model` parameter in your request

### Available Models

Run `ollama list` to see installed models, or use the List Models endpoint:

```bash
curl http://localhost:11434/api/tags
```

## Extending Workflows

These workflows can be extended to:

- **Add memory**: Store conversation history in a database
- **Connect to messaging platforms**: Telegram, Slack, Discord
- **Process documents**: PDF, Word, text files
- **Chain with other AI services**: OpenAI, Claude, etc.
- **Add RAG capabilities**: Combine with vector databases

## Troubleshooting

### Connection Refused

If you get a "connection refused" error:

1. Ensure Ollama is running (`ollama serve`)
2. Check that the URL is correct
3. Verify firewall settings

### Timeout Errors

For large models or long responses:

1. Increase the timeout in the HTTP Request node (Options → Timeout)
2. Default is set to 120 seconds (120000 ms)

### Model Not Found

If the model is not found:

1. Pull the model first: `ollama pull llama3.2`
2. Verify the model name is correct

## Related Resources

- [Ollama API Documentation](../../docs/api.md)
- [n8n Documentation](https://docs.n8n.io/)
- [Ollama Model Library](https://ollama.com/library)

## Contributing

Feel free to contribute additional workflow templates or improvements to existing ones.
