# Integrating MCP (Model Context Protocol) into Ollama

## 1. Overview

This document outlines the plan to integrate the **Model Context Protocol (MCP)** into the Ollama codebase. The goal is to enable any MCP-compatible model running on Ollama to interact with external tools and data sources through user-configured MCP servers.

This will make Ollama a powerful **MCP client**, acting as a bridge between the local model and the broader ecosystem of MCP-compliant tools.

### Key Concepts

*   **MCP Server:** An external service (e.g., a weather API, a GitHub integration) that exposes tools and resources to an LLM according to the MCP standard.
*   **Ollama as MCP Client:** Ollama will be responsible for:
    1.  Discovering available tools from user-configured MCP servers.
    2.  Informing the model about these tools via the prompt.
    3.  Detecting when the model wants to call a tool.
    4.  Making a network request to the appropriate MCP server.
    5.  Returning the tool's output to the model so it can formulate a final response.

---

## 2. Implementation Plan

The implementation is broken down into three phases. Each phase includes a checklist to track progress.

### Phase 1: Configuration Loading

**Goal:** Make Ollama aware of the user's MCP tool servers by reading a configuration file.

#### ✅ Task Checklist: Phase 1

-   [ ] Create a new package `mcp/` for all MCP-related code.
-   [ ] In `mcp/config.go`, define Go structs to represent the `mcp.json` file format.
-   [ ] In `mcp/config.go`, implement the `LoadConfig()` function.
-   [ ] In `server/routes.go`, call `mcp.LoadConfig()` at startup and store the loaded configuration.

#### Step 1.1: Define the Configuration File Structure

The configuration will live in `~/.ollama/mcp.json`. We will support a simple list of named servers.

**File: `~/.ollama/mcp.json` (Example)**
```json
{
  "servers": [
    {
      "name": "local_weather_service",
      "url": "http://localhost:8000/mcp"
    },
    {
      "name": "github_issue_tracker",
      "url": "https://api.github.com/mcp/v1"
    }
  ]
}
```

#### Step 1.2: Implement the Configuration Loader

Create a new package `mcp` to house our logic.

**File: `mcp/config.go`**
```go
package mcp

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type Server struct {
	Name string `json:"name"`
	URL  string `json:"url"`
}

type Config struct {
	Servers []Server `json:"servers"`
}

// LoadConfig reads and parses the MCP configuration file from the user's home directory.
func LoadConfig() (*Config, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	configPath := filepath.Join(home, ".ollama", "mcp.json")

	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		// Return an empty config if the file doesn't exist, this is not an error.
		return &Config{}, nil
	}

	file, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}

	var config Config
	if err := json.Unmarshal(file, &config); err != nil {
		return nil, err
	}

	return &config, nil
}
```

#### Step 1.3: Integrate into the Server

Now, let's load this configuration when the Ollama server starts.

**File: `server/routes.go` (Additions)**
```go
// Import the new mcp package
import (
	"github.com/ollama/ollama/mcp"
)

// Global variable to hold the loaded MCP config
var mcpConfig *mcp.Config

// In the init() function or a suitable startup location within routes.go:
func init() {
	var err error
	mcpConfig, err = mcp.LoadConfig()
	if err != nil {
		// Log the error but don't prevent startup.
		// Users may have a malformed config file.
		log.Printf("Error loading MCP config: %v", err)
	}
}
```

---

### Phase 2: Tool Discovery and Prompt Injection

**Goal:** Fetch the list of available tools from all configured MCP servers and make them available to the model's prompt template.

#### ✅ Task Checklist: Phase 2

-   [ ] In `mcp/client.go`, create a `FetchTools()` function to get tool definitions from a server.
-   [ ] In `server/routes.go`, iterate through the configured servers and fetch all tools at the beginning of a chat request.
-   [ ] Implement a translation function to convert MCP tool definitions into Ollama's `api.Tool` struct.
-   [ ] Pass the combined list of tools to the `chatPrompt()` function.

#### Step 2.1: Implement the MCP Client

This client will be responsible for all network communication with MCP servers.

**File: `mcp/client.go`**
```go
package mcp

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/ollama/ollama/api"
)

// FetchTools retrieves the list of available tools from a single MCP server.
func FetchTools(server Server) ([]api.Tool, error) {
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(server.URL)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var mcpTools []MCPTool // MCPTool is a struct representing the MCP tool definition format.
	if err := json.NewDecoder(resp.Body).Decode(&mcpTools); err != nil {
		return nil, err
	}

	// We need to translate from MCPTool to api.Tool
	return translateTools(mcpTools), nil
}

// MCPTool represents the tool definition format as specified by MCP.
// This may need to be adjusted based on the final MCP standard.
type MCPTool struct {
	// ... fields for MCP tool definition (e.g., Name, Description, Parameters)
}

// translateTools converts a slice of MCP tools to Ollama's internal tool format.
func translateTools(mcpTools []MCPTool) []api.Tool {
	var ollamaTools []api.Tool
	for _, mcpTool := range mcpTools {
		// Translation logic goes here.
		// For example:
		ollamaTool := api.Tool{
			Function: api.ToolFunction{
				Name:        mcpTool.Name,
				Description: mcpTool.Description,
				// ... translate parameters
			},
		}
		ollamaTools = append(ollamaTools, ollamaTool)
	}
	return ollamaTools
}
```

#### Step 2.2: Integrate into the Chat Request Flow

In `server/routes.go`, we'll fetch these tools before building the prompt.

**File: `server/routes.go` (Inside the chat handler function)**
```go
// ... inside the chat handler, before calling chatPrompt ...

var allTools []api.Tool

// Add tools from the request body first
if req.Tools != nil {
    allTools = append(allTools, req.Tools...)
}

// Fetch and add tools from MCP servers
if mcpConfig != nil {
    for _, server := range mcpConfig.Servers {
        mcpTools, err := mcp.FetchTools(server)
        if err != nil {
            log.Printf("Error fetching tools from MCP server %s: %v", server.Name, err)
            continue // Don't block the request if a server is down
        }
        allTools = append(allTools, mcpTools...)
    }
}

// Now, pass `allTools` to the chatPrompt function
prompt, images, err := chatPrompt(c.Request.Context(), m, r.Tokenize, opts, msgs, allTools, req.Think)
```

---

### Phase 3: Tool Call, Execution, and Response

**Goal:** Detect when the model wants to call an MCP tool, execute it via an HTTP request, and feed the result back to the model.

#### ✅ Task Checklist: Phase 3

-   [ ] In `server/routes.go`, after receiving a response from the model, check if the output contains a valid MCP tool call (JSON).
-   [ ] If an MCP tool call is found, parse it.
-   [ ] In `mcp/client.go`, implement an `ExecuteTool()` function that sends a `POST` request to the correct MCP server.
-   [ ] In `server/routes.go`, call `mcp.ExecuteTool()` and get the result.
-   [ ] Format the result into a new `api.Message` with `role: "tool"`.
-   [ ] Append this message to the conversation history and send it back to the model for a final response.

#### Step 3.1: Implement Tool Execution in the Client

**File: `mcp/client.go` (Additions)**
```go
// ExecuteTool sends a tool call to the appropriate MCP server and returns the result.
func ExecuteTool(toolName string, arguments map[string]interface{}, config *Config) (string, error) {
	server, err := findServerForTool(toolName, config)
	if err != nil {
		return "", err
	}

	// Construct the MCP POST request body
	requestBody, err := json.Marshal(map[string]interface{}{
		"tool_name":  toolName,
		"parameters": arguments,
	})
	if err != nil {
		return "", err
	}

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Post(server.URL, "application/json", bytes.NewBuffer(requestBody))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	return string(body), nil
}

// findServerForTool determines which configured server provides a given tool.
// This might require an initial tool discovery and caching step. For now, we can
// re-fetch on each call, but this should be optimized later.
func findServerForTool(toolName string, config *Config) (*Server, error) {
    // This logic needs to be implemented. It will involve checking which server
    // advertised the tool during the discovery phase.
    // For now, we can return a placeholder.
    if len(config.Servers) > 0 {
        return &config.Servers[0], nil // Placeholder
    }
    return nil, fmt.Errorf("no MCP servers configured")
}
```

#### Step 3.2: Integrate into the Server Response Flow

This is the most complex part. We need to modify the chat handler in `server/routes.go` to create a loop for tool calls.

**File: `server/routes.go` (Conceptual changes in the chat handler)**
```go
// ... after receiving a response `resp` from the model ...

// 1. Attempt to parse for MCP tool calls
var toolCalls []api.ToolCall
err := json.Unmarshal([]byte(resp.Message.Content), &toolCalls)

if err == nil && len(toolCalls) > 0 {
    // This is an MCP tool call.

    // 2. Append the assistant's tool call message to history
    msgs = append(msgs, resp.Message)

    // 3. Execute each tool call and append the results
    for _, toolCall := range toolCalls {
        result, err := mcp.ExecuteTool(toolCall.Function.Name, toolCall.Function.Arguments, mcpConfig)
        if err != nil {
            // Handle error, maybe return an error message to the user
            // or feed the error back to the model.
            result = fmt.Sprintf("Error executing tool %s: %v", toolCall.Function.Name, err)
        }

        // 4. Create the tool result message
        toolMessage := api.Message{
            Role:    "tool",
            Content: result,
            // We may need a ToolCallID here depending on the MCP spec
        }
        msgs = append(msgs, toolMessage)
    }

    // 5. Call the model AGAIN with the updated history
    // This will generate the final response to the user.
    // The logic for calling the model needs to be refactored into a reusable function.
    finalResp, err := callModelFunction(c.Request.Context(), m, r.Tokenize, opts, msgs, allTools, req.Think)
    if err != nil {
        // Handle error
    }

    // Stream the final response back to the user
    streamResponse(c, finalResp)

} else {
    // This is a regular message, not a tool call.
    // Stream the response back to the user as usual.
    streamResponse(c, resp)
}
```

This change will likely require refactoring the existing chat handler to better support this new, multi-step conversational flow.

---
## 3. Next Steps & Considerations

*   **Error Handling:** The plan needs robust error handling for cases like invalid config files, unreachable servers, or failed tool executions.
*   **Security:** Since Ollama will be making network requests on behalf of the user, we must be careful. Timeouts are a good start, but we should also consider security implications like preventing access to local network resources unless explicitly allowed.
*   **Optimization:** The current plan fetches tool definitions on every request. This should be cached to improve performance.
*   **Streaming:** The plan above simplifies the process by waiting for the full tool call JSON. A more advanced implementation would parse the streaming output from the model to detect tool calls as they are being generated.
*   **User Experience:** We need to provide clear feedback to the user when a tool is being called (e.g., a "thinking" indicator in the UI).

This document provides a solid foundation for integrating MCP into Ollama. By following these steps, we can add a powerful new capability to the platform.
