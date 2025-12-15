# MCP (Model Context Protocol) Integration

Ollama supports the Model Context Protocol (MCP), enabling language models to execute tools and interact with external systems autonomously.

> **Status**: Experimental

## Quick Start

### CLI Usage

```bash
# Run with filesystem tools restricted to a directory
ollama run qwen2.5:7b --tools /path/to/directory

# In a git repository, both filesystem AND git tools auto-enable
ollama run qwen2.5:7b --tools /path/to/git-repo

# Example interaction
>>> List all files in the directory
# Model will automatically execute filesystem:list_directory tool

>>> Show the git status
# Model will automatically execute git:status tool (if in a git repo)
```

### API Usage

```bash
curl -X POST http://localhost:11434/api/chat \
  -d '{
    "model": "qwen2.5:7b",
    "messages": [{"role": "user", "content": "List the files"}],
    "mcp_servers": [{
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/safe/path"]
    }]
  }'
```

## How It Works

1. **Model generates tool call** in JSON format
2. **Parser detects** the tool call during streaming
3. **MCP server executes** the tool via JSON-RPC over stdio
4. **Results returned** to model context
5. **Model continues** generation with tool results
6. **Loop repeats** for multi-step tasks (up to 15 rounds)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Public API (mcp.go)                          │
│  GetMCPServersForTools()  - Get servers for --tools flag        │
│  GetMCPManager()          - Get manager for explicit configs    │
│  GetMCPManagerForPath()   - Get manager for tools path          │
│  ListMCPServers()         - List available server definitions   │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────┐                 ┌─────────────────────┐
│   MCPDefinitions    │                 │  MCPSessionManager  │
│  (mcp_definitions)  │                 │   (mcp_sessions)    │
│                     │                 │                     │
│  Static config of   │                 │  Runtime sessions   │
│  available servers  │                 │  with connections   │
└─────────────────────┘                 └─────────────────────┘
                                                  │
                                                  ▼
                                        ┌─────────────────────┐
                                        │     MCPManager      │
                                        │   (mcp_manager)     │
                                        │                     │
                                        │  Multi-client mgmt  │
                                        │  Tool execution     │
                                        └─────────────────────┘
                                                  │
                                                  ▼
                                        ┌─────────────────────┐
                                        │      MCPClient      │
                                        │    (mcp_client)     │
                                        │                     │
                                        │  Single JSON-RPC    │
                                        │  connection         │
                                        └─────────────────────┘
```

## Auto-Enable Configuration

MCP servers can declare when they should automatically enable with the `--tools` flag.

### Auto-Enable Modes

| Mode | Description |
|------|-------------|
| `never` | Server must be explicitly configured via API (default) |
| `always` | Server enables whenever `--tools` is used |
| `with_path` | Server enables when `--tools` has a path argument |
| `if_match` | Server enables if conditions in `enable_if` match |

### Conditional Enabling (if_match)

The `enable_if` object supports these conditions:

| Condition | Description |
|-----------|-------------|
| `file_exists` | Check if a file/directory exists in the tools path |
| `env_set` | Check if an environment variable is set (non-empty) |

### Example Configuration

Create `~/.ollama/mcp-servers.json`:

```json
{
  "servers": [
    {
      "name": "filesystem",
      "description": "File system operations",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem"],
      "requires_path": true,
      "auto_enable": "with_path"
    },
    {
      "name": "git",
      "description": "Git repository operations",
      "command": "npx",
      "args": ["-y", "@cyanheads/git-mcp-server"],
      "requires_path": true,
      "auto_enable": "if_match",
      "enable_if": {
        "file_exists": ".git"
      }
    },
    {
      "name": "postgres",
      "description": "PostgreSQL database operations",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "auto_enable": "if_match",
      "enable_if": {
        "env_set": "POSTGRES_CONNECTION"
      }
    },
    {
      "name": "python",
      "description": "Python code execution",
      "command": "python",
      "args": ["-m", "mcp_server_python"],
      "auto_enable": "never"
    }
  ]
}
```

With this configuration:
- **filesystem** enables for any `--tools /path`
- **git** enables only if `/path/.git` exists
- **postgres** enables only if `POSTGRES_CONNECTION` env var is set
- **python** never auto-enables (must use API explicitly)

## API Reference

### Chat Endpoint with MCP

**Endpoint:** `POST /api/chat`

**Request:**
```json
{
  "model": "qwen2.5:7b",
  "messages": [{"role": "user", "content": "Your prompt"}],
  "mcp_servers": [
    {
      "name": "server-name",
      "command": "executable",
      "args": ["arg1", "arg2"],
      "env": {"KEY": "value"}
    }
  ],
  "max_tool_rounds": 10,
  "tool_timeout": 30000
}
```

**MCP Server Configuration:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique identifier for the server |
| `command` | string | Executable to run |
| `args` | []string | Command-line arguments |
| `env` | map | Environment variables |

### Server Definition Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique server identifier |
| `description` | string | Human-readable description |
| `command` | string | Executable to run (npx, python, etc.) |
| `args` | []string | Command-line arguments |
| `env` | map | Environment variables |
| `requires_path` | bool | Whether server needs a path argument |
| `path_arg_index` | int | Where to insert path in args (-1 = append) |
| `capabilities` | []string | List of capability tags |
| `auto_enable` | string | Auto-enable mode (never/always/with_path/if_match) |
| `enable_if` | object | Conditions for if_match mode |

## Security

### Implemented Safeguards

- **Process isolation**: MCP servers run in separate process groups
- **Path restrictions**: Filesystem access limited to specified directories
- **Environment filtering**: Allowlist-based, sensitive variables removed
- **Command validation**: Dangerous commands (shells, sudo, rm) blocked
- **Argument sanitization**: Shell injection prevention
- **Timeouts**: 30-second default with graceful shutdown

### Blocked Commands

Shells (`bash`, `sh`, `zsh`), privilege escalation (`sudo`, `su`), destructive commands (`rm`, `dd`), and network tools (`curl`, `wget`, `nc`) are blocked by default.

## Creating MCP Servers

MCP servers communicate via JSON-RPC 2.0 over stdin/stdout and must implement three methods:

### Required Methods

1. **`initialize`** - Returns server capabilities
2. **`tools/list`** - Returns available tools with schemas
3. **`tools/call`** - Executes a tool and returns results

### Minimal Python Example

```python
#!/usr/bin/env python3
import json
import sys

def handle_request(request):
    method = request.get("method")
    request_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": request_id,
            "result": {
                "protocolVersion": "0.1.0",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "my-server", "version": "1.0.0"}
            }
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": request_id,
            "result": {
                "tools": [{
                    "name": "hello",
                    "description": "Say hello",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name to greet"}
                        },
                        "required": ["name"]
                    }
                }]
            }
        }

    elif method == "tools/call":
        name = request["params"]["arguments"].get("name", "World")
        return {
            "jsonrpc": "2.0", "id": request_id,
            "result": {
                "content": [{"type": "text", "text": f"Hello, {name}!"}]
            }
        }

if __name__ == "__main__":
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        request = json.loads(line)
        response = handle_request(request)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
```

### Testing Your Server

```bash
# Test initialize
echo '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}' | python3 my_server.py

# Test tools/list
echo '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":2}' | python3 my_server.py

# Test tools/call
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"hello","arguments":{"name":"Alice"}},"id":3}' | python3 my_server.py
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_DEBUG=INFO` | Enable debug logging |
| `OLLAMA_MCP_TIMEOUT` | Tool execution timeout (ms) |
| `OLLAMA_MCP_SERVERS` | JSON config for MCP servers (overrides file) |
| `OLLAMA_MCP_DISABLE=1` | Disable MCP validation on startup |

## Supported Models

MCP works best with models that support tool calling:
- Qwen 2.5 / Qwen 3 series
- Llama 3.1+ with tool support
- Other models with JSON tool call output

## Limitations

- **Transport**: stdio only (no HTTP/WebSocket)
- **Protocol**: MCP 1.0
- **Concurrency**: Max 10 parallel MCP servers
- **Platform**: Linux/macOS (Windows untested)

## Troubleshooting

**"Tool not found"**
- Verify MCP server initialized correctly
- Check tool name includes namespace prefix

**"MCP server failed to initialize"**
- Check command path exists
- Verify Python/Node environment
- Test server manually with JSON input

**"No MCP servers matched for --tools context"**
- Check auto_enable settings in config
- Verify path exists and conditions match

**"Access denied"**
- Path outside allowed directories
- Security policy violation

**Debug mode:**
```bash
OLLAMA_DEBUG=INFO ollama serve
```

## Resources

- [MCP Specification](https://modelcontextprotocol.io/docs)
- [Official MCP Servers](https://github.com/modelcontextprotocol/servers)
