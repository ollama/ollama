# HuggingFace Prompt Renderer MCP Server

Model Context Protocol (MCP) server for rendering conversation messages into
model-specific prompt strings using HuggingFace tokenizer chat templates.

## Requirements

- [uv](https://docs.astral.sh/uv/) - Fast Python package installer

## Usage

### MCP Server Mode

Run the MCP server over stdio for use with MCP clients:

```bash
uv run cmd/prompt-rendering/server.py --mcp
```

Add to your MCP client configuration (e.g., for Claude Desktop):

```json
{
  "mcpServers": {
    "huggingface-prompt-renderer": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "<path-to-ollama-repo>",
        "cmd/prompt-rendering/server.py",
        "--mcp"
      ]
    }
  }
}
```

### FastAPI Server Mode

Start a FastAPI server for manual HTTP testing:

```bash
# Start on default port 8000
uv run cmd/prompt-rendering/server.py --host 0.0.0.0 --port 8000

# Start on custom port
uv run cmd/prompt-rendering/server.py --host 0.0.0.0 --port 9000
```

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/generate-prompt` | Generate prompt from messages |
| GET | `/health` | Health check |

### Test with curl

```bash
# Basic user message
curl -X POST http://localhost:8000/generate-prompt \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# With tools
curl -X POST http://localhost:8000/generate-prompt \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the weather?"}
    ],
    "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather",
        "parameters": {
          "type": "object",
          "required": ["location"],
          "properties": {
            "location": {"type": "string", "description": "The city"}
          }
        }
      }
    }]
  }'

# With tool calls
curl -X POST http://localhost:8000/generate-prompt \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the weather in SF?"},
      {
        "role": "assistant",
        "tool_calls": [{
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {"location": "San Francisco"}
          }
        }]
      },
      {"role": "tool", "content": "{\"temperature\": 68}", "tool_call_id": "call_1"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}}
        }
      }
    }]
  }'
```

## Supported Message Formats

The server supports multiple message formats:

| Format | Description |
|--------|-------------|
| OpenAI | Standard `role`, `content`, `tool_calls`, `tool_call_id` |
| OLMo | Adds `functions` and `function_calls` fields |
| DeepSeek | Tool call arguments must be JSON strings |

## Tool Support

| Setting | Description |
|---------|-------------|
| `inject_tools_as_functions=true` | Injects tools into system message as `functions` key (OLMo-style) |
| `inject_tools_as_functions=false` | Passes tools separately to `apply_chat_template` (standard transformers) |

## Models

The server uses HuggingFace's `transformers` library and supports any model
with a chat template. Default: `Qwen/Qwen3-Coder-480B-A35B-Instruct`

## Dependencies

The script uses PEP 723 inline dependency metadata. When run with `uv`,
dependencies are automatically installed into an isolated environment:

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `transformers` - HuggingFace tokenizer
- `jinja2` - Template engine
- `mcp` - Model Context Protocol
