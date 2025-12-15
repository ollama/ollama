package server

import (
	"fmt"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/api"
)

// MCPCodeAPI provides context injection for MCP tools
type MCPCodeAPI struct {
	manager *MCPManager
}

// NewMCPCodeAPI creates a new MCP code API
func NewMCPCodeAPI(manager *MCPManager) *MCPCodeAPI {
	return &MCPCodeAPI{
		manager: manager,
	}
}

// GenerateMinimalContext returns essential runtime context for tool usage.
// Tool schemas are already provided via the template's TypeScript rendering,
// so we only need to add runtime-specific info like working directories.
func (m *MCPCodeAPI) GenerateMinimalContext(configs []api.MCPServerConfig) string {
	slog.Debug("GenerateMinimalContext called", "configs_count", len(configs))

	var context strings.Builder

	// Add filesystem working directory if applicable
	for _, config := range configs {
		if workingDir := m.extractFilesystemPath(config); workingDir != "" {
			context.WriteString(fmt.Sprintf(`
Filesystem working directory: %s
All filesystem tool paths must be within this directory.
`, workingDir))
		}
	}

	result := context.String()
	if result != "" {
		slog.Debug("Generated MCP context", "length", len(result))
	}
	return result
}

// extractFilesystemPath extracts the working directory from filesystem server config
func (m *MCPCodeAPI) extractFilesystemPath(config api.MCPServerConfig) string {
	isFilesystem := strings.Contains(config.Command, "filesystem") ||
		(len(config.Args) > 0 && strings.Contains(strings.Join(config.Args, " "), "filesystem"))

	if isFilesystem && len(config.Args) > 0 {
		// Path is typically the last argument
		return config.Args[len(config.Args)-1]
	}
	return ""
}

// InjectContextIntoMessages adds runtime context to the message stream
func (m *MCPCodeAPI) InjectContextIntoMessages(messages []api.Message, configs []api.MCPServerConfig) []api.Message {
	context := m.GenerateMinimalContext(configs)
	if context == "" {
		return messages
	}

	// Check if there's already a system message
	if len(messages) > 0 && messages[0].Role == "system" {
		// Append to existing system message
		messages[0].Content += context
	} else {
		// Create new system message
		systemMsg := api.Message{
			Role:    "system",
			Content: context,
		}
		messages = append([]api.Message{systemMsg}, messages...)
	}

	return messages
}
