// Package server provides MCP (Model Context Protocol) integration for Ollama.
//
// MCP Architecture:
//
//	┌─────────────────────────────────────────────────────────────────┐
//	│                    Public API (this file)                       │
//	│  GetMCPServersForTools()  - Get servers for --tools flag        │
//	│  GetMCPManager()          - Get manager for explicit configs    │
//	│  GetMCPManagerForPath()   - Get manager for tools path          │
//	│  ListMCPServers()         - List available server definitions   │
//	└─────────────────────────────────────────────────────────────────┘
//	                              │
//	          ┌───────────────────┴───────────────────┐
//	          ▼                                       ▼
//	┌─────────────────────┐                 ┌─────────────────────┐
//	│   MCPDefinitions    │                 │  MCPSessionManager  │
//	│  (mcp_definitions)  │                 │   (mcp_sessions)    │
//	│                     │                 │                     │
//	│  Static config of   │                 │  Runtime sessions   │
//	│  available servers  │                 │  with connections   │
//	└─────────────────────┘                 └─────────────────────┘
//	                                                  │
//	                                                  ▼
//	                                        ┌─────────────────────┐
//	                                        │     MCPManager      │
//	                                        │   (mcp_manager)     │
//	                                        │                     │
//	                                        │  Multi-client mgmt  │
//	                                        │  Tool execution     │
//	                                        └─────────────────────┘
//	                                                  │
//	                                                  ▼
//	                                        ┌─────────────────────┐
//	                                        │      MCPClient      │
//	                                        │    (mcp_client)     │
//	                                        │                     │
//	                                        │  Single JSON-RPC    │
//	                                        │  connection         │
//	                                        └─────────────────────┘

package server

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/api"
)

// ============================================================================
// Public API - Clean interface for external code
// ============================================================================

// GetMCPServersForTools returns the MCP server configs that should be enabled
// for the given tools spec. It handles path normalization:
//   - "." or "true" → current working directory
//   - "~/path" → expands to home directory
//   - relative paths → resolved to absolute paths
//
// Returns the server configs and the resolved absolute path.
// On error, still returns the resolved path so callers can implement fallback.
// This is used by the --tools CLI flag.
func GetMCPServersForTools(toolsSpec string) ([]api.MCPServerConfig, string, error) {
	// Normalize the tools path first (needed even for fallback on error)
	toolsPath := toolsSpec
	if toolsSpec == "." || toolsSpec == "true" {
		if cwd, err := os.Getwd(); err == nil {
			toolsPath = cwd
		}
	}

	// Expand tilde to home directory
	if strings.HasPrefix(toolsPath, "~") {
		if home := os.Getenv("HOME"); home != "" {
			toolsPath = filepath.Join(home, toolsPath[1:])
		}
	}

	// Resolve to absolute path
	if absPath, err := filepath.Abs(toolsPath); err == nil {
		toolsPath = absPath
	}

	// Load definitions
	defs, err := LoadMCPDefinitions()
	if err != nil {
		return nil, toolsPath, err
	}

	ctx := AutoEnableContext{ToolsPath: toolsPath}
	return defs.GetAutoEnableServers(ctx), toolsPath, nil
}

// GetMCPManager returns an MCP manager for the given session and configs.
// If a session with matching configs already exists, it will be reused.
func GetMCPManager(sessionID string, configs []api.MCPServerConfig) (*MCPManager, error) {
	return GetMCPSessionManager().GetOrCreateManager(sessionID, configs)
}

// GetMCPManagerForPath returns an MCP manager for servers that auto-enable
// for the given tools path. Used by CLI: `ollama run model --tools /path`
func GetMCPManagerForPath(model string, toolsPath string) (*MCPManager, error) {
	return GetMCPSessionManager().GetManagerForToolsPath(model, toolsPath)
}

// ListMCPServers returns information about all available MCP server definitions.
func ListMCPServers() ([]MCPServerInfo, error) {
	defs, err := LoadMCPDefinitions()
	if err != nil {
		return nil, err
	}
	return defs.ListServers(), nil
}
