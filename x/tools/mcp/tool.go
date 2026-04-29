package mcp

import (
	"fmt"
	"log/slog"
	"sync"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/tools"
)

// Tool wraps an MCP server tool to implement the tools.Tool interface.
// This allows MCP tools to be registered in the standard tool registry
// alongside native tools like bash and websearch.
type Tool struct {
	client      *Client
	toolName    string
	description string
	schema      api.ToolFunction
}

// Ensure Tool implements tools.Tool interface
var _ tools.Tool = (*Tool)(nil)

// Name returns the tool's unique identifier, namespaced by server name.
func (t *Tool) Name() string {
	return fmt.Sprintf("%s:%s", t.client.Name(), t.toolName)
}

// Description returns a human-readable description.
func (t *Tool) Description() string {
	return t.description
}

// Schema returns the tool's parameter schema for the LLM.
func (t *Tool) Schema() api.ToolFunction {
	return t.schema
}

// Execute runs the tool with the given arguments.
func (t *Tool) Execute(args map[string]any) (string, error) {
	slog.Debug("Executing MCP tool", "server", t.client.Name(), "tool", t.toolName)
	result, err := t.client.CallTool(t.toolName, args)
	if err != nil {
		slog.Debug("MCP tool execution failed", "server", t.client.Name(), "tool", t.toolName, "error", err)
		return "", err
	}
	slog.Debug("MCP tool executed", "server", t.client.Name(), "tool", t.toolName, "result_length", len(result))
	return result, nil
}

// ServerConfig defines configuration for an MCP server.
type ServerConfig struct {
	Name    string            `json:"name"`
	Command string            `json:"command"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
}

// Manager manages multiple MCP server connections.
type Manager struct {
	mu      sync.RWMutex
	clients map[string]*Client
}

// NewManager creates a new MCP manager.
func NewManager() *Manager {
	return &Manager{
		clients: make(map[string]*Client),
	}
}

// RegisterServer connects to an MCP server and registers all its tools
// with the provided registry. Tools are namespaced as "servername:toolname".
func (m *Manager) RegisterServer(registry *tools.Registry, config ServerConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if already registered
	if _, exists := m.clients[config.Name]; exists {
		return fmt.Errorf("MCP server '%s' already registered", config.Name)
	}

	// Create and initialize client
	client := NewClient(config.Name, config.Command, config.Args, config.Env)
	if err := client.Initialize(); err != nil {
		client.Close()
		return fmt.Errorf("failed to initialize MCP server '%s': %w", config.Name, err)
	}

	// Register each tool from the server
	serverTools := client.Tools()
	for _, toolInfo := range serverTools {
		mcpTool := &Tool{
			client:      client,
			toolName:    toolInfo.Name,
			description: toolInfo.Description,
			schema:      convertSchema(config.Name, toolInfo),
		}
		registry.Register(mcpTool)
		slog.Debug("Registered MCP tool", "server", config.Name, "tool", toolInfo.Name)
	}

	m.clients[config.Name] = client
	slog.Debug("MCP server registered", "name", config.Name, "tools", len(serverTools))
	return nil
}

// Close shuts down all MCP servers.
func (m *Manager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var lastErr error
	for name, client := range m.clients {
		if err := client.Close(); err != nil {
			slog.Warn("Error closing MCP client", "name", name, "error", err)
			lastErr = err
		}
	}
	m.clients = make(map[string]*Client)
	return lastErr
}

// convertSchema converts MCP tool schema to ollama's ToolFunction format.
func convertSchema(serverName string, info ToolInfo) api.ToolFunction {
	schema := api.ToolFunction{
		Name:        fmt.Sprintf("%s:%s", serverName, info.Name),
		Description: info.Description,
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Properties: api.NewToolPropertiesMap(),
			Required:   []string{},
		},
	}

	// Convert properties
	if props, ok := info.InputSchema["properties"].(map[string]any); ok {
		for propName, propDef := range props {
			if propDefMap, ok := propDef.(map[string]any); ok {
				prop := api.ToolProperty{
					Description: getStringFromMap(propDefMap, "description"),
				}

				if propType, ok := propDefMap["type"].(string); ok {
					prop.Type = api.PropertyType{propType}
				}

				if items, ok := propDefMap["items"]; ok {
					prop.Items = items
				}

				if enumVal, ok := propDefMap["enum"].([]any); ok {
					prop.Enum = enumVal
				}

				schema.Parameters.Properties.Set(propName, prop)
			}
		}
	}

	// Convert required fields
	if required, ok := info.InputSchema["required"].([]any); ok {
		for _, req := range required {
			if reqStr, ok := req.(string); ok {
				schema.Parameters.Required = append(schema.Parameters.Required, reqStr)
			}
		}
	}

	return schema
}

func getStringFromMap(m map[string]any, key string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return ""
}
