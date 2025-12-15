package server

import (
	"fmt"
	"log/slog"
	"strings"
	
	"github.com/ollama/ollama/api"
)

// MCPCodeAPI provides a code-like interface for MCP tools
type MCPCodeAPI struct {
	manager *MCPManager
}

// NewMCPCodeAPI creates a new MCP code API
func NewMCPCodeAPI(manager *MCPManager) *MCPCodeAPI {
	return &MCPCodeAPI{
		manager: manager,
	}
}

// GenerateMinimalContext returns essential context for tool usage
func (m *MCPCodeAPI) GenerateMinimalContext(configs []api.MCPServerConfig) string {
	slog.Debug("GenerateMinimalContext called", "configs_count", len(configs))
	if len(configs) == 0 {
		slog.Debug("No MCP configs provided, returning empty context")
		return ""
	}

	var context strings.Builder
	context.WriteString("\n=== MCP Tool Context ===\n")

	for _, config := range configs {
		slog.Debug("Processing MCP config", "command", config.Command, "args", config.Args)
		// Check if this is a filesystem server (command or first arg contains filesystem)
		isFilesystem := strings.Contains(config.Command, "filesystem") ||
			(len(config.Args) > 0 && strings.Contains(config.Args[0], "filesystem"))

		if isFilesystem && len(config.Args) > 1 {
			// Extract working directory from filesystem server
			workingDir := config.Args[1]
			slog.Debug("Adding filesystem context", "working_dir", workingDir)
			context.WriteString(fmt.Sprintf(`
Filesystem tools are available with these constraints:
- Working directory: %s
- All file operations must use paths within this directory
- Example usage:
  - List files: "List all files in %s"
  - Read file: "Read %s/filename.txt"
  - Create file: "Create %s/newfile.txt with content"
- Paths outside %s will be rejected

When working with files, ALWAYS use the full path starting with %s
`, workingDir, workingDir, workingDir, workingDir, workingDir, workingDir))
		}
		// Add other server types as needed
	}
	
	context.WriteString("\n")
	result := context.String()
	slog.Debug("Generated MCP context", "length", len(result))
	return result
}

// GenerateProgressiveContext returns context based on what tools are being used
func (m *MCPCodeAPI) GenerateProgressiveContext(toolNames []string) string {
	var context strings.Builder
	
	// Group tools by server
	serverTools := make(map[string][]string)
	for _, toolName := range toolNames {
		if clientName, exists := m.manager.GetToolClient(toolName); exists {
			serverTools[clientName] = append(serverTools[clientName], toolName)
		}
	}
	
	// Generate context for each server's tools
	for serverName, tools := range serverTools {
		context.WriteString(fmt.Sprintf("\n%s tools being used:\n", serverName))
		for _, tool := range tools {
			// Get tool definition from manager
			if toolDef := m.manager.GetToolDefinition(serverName, tool); toolDef != nil {
				context.WriteString(fmt.Sprintf("- %s: %s\n", tool, toolDef.Function.Description))
			}
		}
	}
	
	return context.String()
}

// InjectContextIntoMessages intelligently injects context into the message stream
func (m *MCPCodeAPI) InjectContextIntoMessages(messages []api.Message, configs []api.MCPServerConfig) []api.Message {
	// Generate minimal context
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

// ExtractWorkingDirectory extracts the working directory from MCP server args
func ExtractWorkingDirectory(config api.MCPServerConfig) string {
	if strings.Contains(config.Command, "filesystem") && len(config.Args) > 1 {
		return config.Args[1]
	}
	return ""
}

// GenerateToolCallExample generates an example of how to call a specific tool
func (m *MCPCodeAPI) GenerateToolCallExample(serverName, toolName string) string {
	workingDir := ""
	
	// Get working directory if filesystem
	if serverName == "filesystem" {
		if clients := m.manager.GetServerNames(); len(clients) > 0 {
			// This is a simplified approach - in production we'd properly track server configs
			workingDir = "/home/velvetm/Desktop/mcp-test-files" // Would be extracted from actual config
		}
	}
	
	// Generate appropriate example based on tool
	switch toolName {
	case "list_directory":
		return fmt.Sprintf(`"List all files in %s"`, workingDir)
	case "read_file":
		return fmt.Sprintf(`"Read the file %s/example.txt"`, workingDir)
	case "write_file":
		return fmt.Sprintf(`"Create a file at %s/output.txt with content 'Hello World'"`, workingDir)
	case "create_directory":
		return fmt.Sprintf(`"Create a directory called %s/newdir"`, workingDir)
	default:
		return fmt.Sprintf(`"Use the %s tool"`, toolName)
	}
}