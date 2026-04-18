---
name: server-api-instructions
description: "Use when: building REST API handlers, working with server/ package, implementing HTTP endpoints, or adding OpenAI-compatible API features"
applyTo: "server/**"
---

# Server & API Package Instructions

## Overview
The `server/` package implements Ollama's REST API using the Gin web framework. It provides:
- HTTP request handling
- WebSocket streaming
- Middleware (logging, CORS, auth)
- OpenAI-compatible API layer

## Key Files

- `server.go` - Main server setup and route registration
- `routes_*.go` - Endpoint implementations
- `middleware/` - HTTP middleware
- `openai/` - OpenAI API compatibility layer
- `types.go` - Request/response types

## API Handler Pattern

```go
// Good: Clear handler with structured responses
func handleGenerate(c *gin.Context) {
    var req GenerateRequest
    if err := c.BindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }
    
    resp, err := service.Generate(c.Request.Context(), req)
    if err != nil {
        c.JSON(500, gin.H{"error": "generation failed"})
        return
    }
    
    c.JSON(200, resp)
}
```

## Streaming Responses

For large outputs, use streaming instead of buffering:

```go
// Good: Streaming token-by-token
c.Header("Transfer-Encoding", "chunked")
for token := range generateStream(ctx, req) {
    data, _ := json.Marshal(token)
    c.Writer.WriteString(string(data) + "\n")
    c.Writer.Flush()
}

// Avoid: Buffering entire response
var responses []Response
for token := range stream {
    responses = append(responses, token)
}
c.JSON(200, responses)
```

## WebSocket Support

- Use `github.com/gorilla/websocket` for WS upgrades
- Handle disconnections gracefully
- Send heartbeats for long-running operations
- Clean up resources on close

## Middleware Guidelines

```go
// Middleware pattern: func(c *gin.Context)
func loggingMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        c.Next()
        duration := time.Since(start)
        log.Printf("%s %s took %v", c.Request.Method, c.Request.URL, duration)
    }
}

// Register: router.Use(loggingMiddleware())
```

## OpenAI Compatibility

- Endpoints must match `/v1/*` pattern
- Response format must match OpenAI's format
- Support streaming with `stream: true` parameter
- Include `model` field in all responses

Example endpoint: `/v1/chat/completions`
- Request format: `{"model": "...", "messages": [...], "stream": false}`
- Response format: `{"id": "...", "object": "chat.completion", "choices": [...]}`

## Error Handling in HTTP

```go
// Structured errors:
- 400: Bad request (validation error)
- 404: Model not found
- 500: Internal server error
- 503: Service unavailable (model loading)

// Always include error message:
c.JSON(code, gin.H{
    "error": "descriptive message",
})
```

## Context Propagation

```go
// Always pass context through the request:
ctx := c.Request.Context()
result, err := service.DoSomething(ctx, ...)

// Support cancellation:
ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
defer cancel()
```

## Testing HTTP Handlers

```go
func TestHandleGenerate(t *testing.T) {
    router := setupTestRouter()
    req := httptest.NewRequest("POST", "/v1/generate", 
        bytes.NewBufferString(`{"model":"llama"}`))
    w := httptest.NewRecorder()
    
    router.ServeHTTP(w, req)
    
    assert.Equal(t, 200, w.Code)
    var resp Response
    require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
}
```

## Performance Considerations

- Use connection pooling for backends
- Implement request timeouts
- Stream large responses
- Cache model metadata
- Use goroutine pools for CPU-bound work
