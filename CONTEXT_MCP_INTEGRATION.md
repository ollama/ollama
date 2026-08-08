# Context MCP Integration

This repository now includes integration with Context7 for fetching up-to-date library documentation into chat responses.

## Overview

The system supports two modes:
1. **Cursor IDE** - Direct MCP integration for development
2. **Gateway Web Chat** - REST API integration for end users

## Configuration

### Backend (Gateway)

Add to your `.env` file:

```env
# Library Documentation Context (Optional - Context7 integration)
DOCS_CONTEXT_ENABLED=false
DOCS_CONTEXT_MAX_CHARS=4000
DOCS_CONTEXT_CACHE_TTL=86400
# Optional: Get API key from https://context7.com for higher rate limits
# CONTEXT7_API_KEY=
```

**Environment Variables:**

- `DOCS_CONTEXT_ENABLED` - Master switch (default: `false`)
- `DOCS_CONTEXT_MAX_CHARS` - Maximum characters to fetch (default: `4000` - tuned for small models)
- `DOCS_CONTEXT_CACHE_TTL` - Redis cache duration in seconds (default: `86400` = 24 hours)
- `CONTEXT7_API_KEY` - Optional API key for higher rate limits (get from https://context7.com)

### Cursor IDE

Context7 MCP is configured in `~/.cursor/mcp.json`. To add an API key:

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"],
      "env": {
        "CONTEXT7_API_KEY": "your-key-here"
      }
    }
  }
}
```

## How It Works

### Gateway (Web Chat)

When `DOCS_CONTEXT_ENABLED=true`, the system:

1. **Analyzes each message** using heuristics to determine if library docs are needed
2. **Skips fetching** when:
   - GitHub repo context is active (repo questions take priority)
   - Message is too short or not library-related
3. **Fetches docs** when the message mentions:
   - Library/framework names (fastapi, react, next.js, ollama, etc.)
   - API-related keywords (how to, api, sdk, migrate, version, etc.)
   - Explicit requests ("use docs", "check docs", "official docs")
4. **Caches results** in Redis for 24 hours to minimize external API calls
5. **Appends docs** to the system prompt (not stored in message history)
6. **Fails open** - chat works normally if doc fetching fails

### Cursor IDE

When working in Cursor:

1. The agent has access to Context7 tools: `resolve-library-id` and `query-docs`
2. Use explicit phrases: "use context7", "check docs", "fetch docs"
3. The agent follows rules in `.cursor/rules/documentation-lookup.md`
4. Context7 is preferred for **external library APIs**, codebase search for **this repo's code**

## Usage Examples

### Web Chat (Gateway)

```
User: How do I use Ollama's streaming API in FastAPI?
→ System fetches Context7 docs about Ollama + FastAPI streaming
→ LLM responds with up-to-date code examples

User: Explain how our gateway session management works
→ No docs fetch (repo question, uses GitHub context if linked)
→ LLM responds from codebase knowledge
```

### Cursor IDE

```
User: How do I use Next.js 15 server actions? use context7
→ Agent calls resolve-library-id → query-docs
→ Returns current Next.js 15 documentation

User: Refactor the Chat component
→ Agent uses codebase search (local code, no Context7 needed)
```

## Performance & Efficiency

**Minimal overhead:**
- **0 extra API calls** when disabled (default)
- **0 extra calls** for normal chat or repo-linked sessions  
- **~2 HTTP calls max** per unique question (then cached)
- **Fail-open design** - API errors don't break chat
- **Token-capped** - critical for small models like llama3.2:1b

**Cache strategy:**
- Docs cached in Redis for 24 hours
- Cache key based on message hash
- Same efficient pattern as GitHub repo context

## Priority Order

The system uses context in this order:

1. **GitHub Repository Context** (if linked) - Your own code
2. **Library Documentation** (if relevant) - External APIs
3. **Model Knowledge** (always) - General programming

## Offline Behavior

- **Context7 API requires network** (not fully offline)
- **Cached docs work offline** after first fetch
- **Chat still works** if Context7 is unavailable (fail-open)
- For true offline docs, consider adding local documentation to GitHub repo context

## Rate Limits

**Without API key:**
- 60 requests/hour from Context7

**With API key:**
- 5,000 requests/hour from Context7
- Get key at: https://context7.com

**In practice:**
- Redis caching means repeated questions use 0 external requests
- Most users won't hit rate limits even without a key

## Troubleshooting

**Docs not appearing in responses:**
1. Check `DOCS_CONTEXT_ENABLED=true` in `.env`
2. Restart the gateway: `docker compose restart api`
3. Try explicit trigger: "How to use X? use docs"
4. Check Redis is connected: `curl http://localhost:8080/api/health`

**Rate limit errors:**
1. Get a free API key from https://context7.com
2. Add `CONTEXT7_API_KEY` to `.env`
3. Restart: `docker compose restart api`

**Docs for wrong library:**
1. Be more specific in your question
2. Include version numbers when relevant
3. Mention the exact library name

## Architecture Notes

- **No MCP in Docker** - Gateway uses Context7 REST API directly
- **No tool-calling loop** - Single-turn injection into system prompt
- **No message history pollution** - Docs in system prompt only, not stored
- **Same pattern as GitHub context** - Familiar, proven design
- **Small diff** - <150 lines added to `main.py`

## Documentation

- **Backend logic**: `api-gateway/main.py` (lines ~807-948, ~1298-1310)
- **Cursor rule**: `.cursor/rules/documentation-lookup.md`
- **Cursor MCP config**: `~/.cursor/mcp.json`
- **Environment variables**: `.env`
