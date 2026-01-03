package server

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/api"
)

// AutoEnableMode determines when a server auto-enables with --tools
type AutoEnableMode string

const (
	// AutoEnableNever means the server must be explicitly configured (default)
	AutoEnableNever AutoEnableMode = "never"
	// AutoEnableAlways means the server enables whenever --tools is used
	AutoEnableAlways AutoEnableMode = "always"
	// AutoEnableWithPath means the server enables when --tools has a path
	AutoEnableWithPath AutoEnableMode = "with_path"
	// AutoEnableIfMatch means the server enables if EnableIf condition matches
	AutoEnableIfMatch AutoEnableMode = "if_match"
)

// EnableCondition specifies conditions for AutoEnableIfMatch mode
type EnableCondition struct {
	// FileExists checks if a specific file exists in the tools path
	FileExists string `json:"file_exists,omitempty"`
	// EnvSet checks if an environment variable is set (non-empty)
	EnvSet string `json:"env_set,omitempty"`
}

// AutoEnableContext provides context for auto-enable decisions
type AutoEnableContext struct {
	// ToolsPath is the path from --tools flag (may be empty)
	ToolsPath string
	// Env contains environment variables (optional, falls back to os.Getenv)
	Env map[string]string
}

// MCPDefinitions holds available MCP server definitions loaded from configuration.
// This is the static configuration of what servers CAN be used.
type MCPDefinitions struct {
	Servers map[string]MCPServerDefinition `json:"servers"`
}

// MCPServerDefinition defines an available MCP server type
type MCPServerDefinition struct {
	Name         string            `json:"name"`
	Description  string            `json:"description"`
	Command      string            `json:"command"`
	Args         []string          `json:"args,omitempty"`
	RequiresPath bool              `json:"requires_path,omitempty"`
	PathArgIndex int               `json:"path_arg_index,omitempty"`
	Env          map[string]string `json:"env,omitempty"`
	Capabilities []string          `json:"capabilities,omitempty"`

	// AutoEnable determines when this server auto-enables with --tools
	// Default is "never" (must be explicitly configured via API)
	AutoEnable AutoEnableMode `json:"auto_enable,omitempty"`

	// EnableIf specifies conditions for AutoEnableIfMatch mode
	EnableIf EnableCondition `json:"enable_if,omitempty"`
}

// MCPServerInfo provides information about an available MCP server
type MCPServerInfo struct {
	Name         string         `json:"name"`
	Description  string         `json:"description"`
	RequiresPath bool           `json:"requires_path"`
	Capabilities []string       `json:"capabilities,omitempty"`
	AutoEnable   AutoEnableMode `json:"auto_enable,omitempty"`
}

// DefaultMCPServers returns minimal built-in MCP server definitions
// Full examples are provided in examples/mcp-servers.json
func DefaultMCPServers() map[string]MCPServerDefinition {
	// Only include filesystem by default - it requires only npx which is commonly available
	// Users can add more servers via ~/.ollama/mcp-servers.json
	return map[string]MCPServerDefinition{
		"filesystem": {
			Name:         "filesystem",
			Description:  "File system operations with path-based access control",
			Command:      "npx",
			Args:         []string{"-y", "@modelcontextprotocol/server-filesystem"},
			RequiresPath: true,
			PathArgIndex: -1,
			Capabilities: []string{"read", "write", "list", "search"},
			AutoEnable:   AutoEnableWithPath, // Enable when --tools has a path
		},
	}
}

// LoadMCPDefinitions loads MCP server definitions from configuration files.
// Priority order: user config (~/.ollama) > system config (/etc/ollama) > defaults
func LoadMCPDefinitions() (*MCPDefinitions, error) {
	defs := &MCPDefinitions{
		Servers: DefaultMCPServers(),
	}

	// Load from user config if exists
	configPaths := []string{
		filepath.Join(os.Getenv("HOME"), ".ollama", "mcp-servers.json"),
		"/etc/ollama/mcp-servers.json",
		"./mcp-servers.json",
	}

	for _, path := range configPaths {
		if err := defs.LoadFromFile(path); err == nil {
			slog.Debug("Loaded MCP definitions", "path", path)
			break
		}
	}

	// Load from environment variable if set
	if mcpConfig := os.Getenv("OLLAMA_MCP_SERVERS"); mcpConfig != "" {
		if err := defs.LoadFromJSON([]byte(mcpConfig)); err != nil {
			return nil, fmt.Errorf("failed to parse OLLAMA_MCP_SERVERS: %w", err)
		}
	}

	return defs, nil
}

// LoadFromFile loads MCP server definitions from a JSON file
func (d *MCPDefinitions) LoadFromFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return d.LoadFromJSON(data)
}

// LoadFromJSON loads MCP server definitions from JSON data
func (d *MCPDefinitions) LoadFromJSON(data []byte) error {
	var config struct {
		Servers []MCPServerDefinition `json:"servers"`
	}

	if err := json.Unmarshal(data, &config); err != nil {
		return err
	}

	for _, server := range config.Servers {
		d.Servers[server.Name] = server
	}

	return nil
}

// ListServers returns information about all available MCP servers
func (d *MCPDefinitions) ListServers() []MCPServerInfo {
	var servers []MCPServerInfo
	for _, def := range d.Servers {
		servers = append(servers, MCPServerInfo{
			Name:         def.Name,
			Description:  def.Description,
			RequiresPath: def.RequiresPath,
			Capabilities: def.Capabilities,
			AutoEnable:   def.AutoEnable,
		})
	}
	return servers
}

// GetAutoEnableServers returns servers that should auto-enable for the given context.
// This method checks each server's AutoEnable mode and EnableIf conditions.
func (d *MCPDefinitions) GetAutoEnableServers(ctx AutoEnableContext) []api.MCPServerConfig {
	var configs []api.MCPServerConfig

	for _, def := range d.Servers {
		if !d.shouldAutoEnable(def, ctx) {
			continue
		}

		config, err := d.buildConfigForAutoEnable(def, ctx)
		if err != nil {
			slog.Warn("Failed to build config for auto-enable server",
				"name", def.Name, "error", err)
			continue
		}

		configs = append(configs, config)
	}

	return configs
}

// shouldAutoEnable checks if a server should auto-enable for the given context
func (d *MCPDefinitions) shouldAutoEnable(def MCPServerDefinition, ctx AutoEnableContext) bool {
	switch def.AutoEnable {
	case AutoEnableNever, "":
		return false

	case AutoEnableAlways:
		return true

	case AutoEnableWithPath:
		return ctx.ToolsPath != ""

	case AutoEnableIfMatch:
		return d.checkEnableCondition(def.EnableIf, ctx)

	default:
		return false
	}
}

// checkEnableCondition evaluates an EnableCondition against the context
func (d *MCPDefinitions) checkEnableCondition(cond EnableCondition, ctx AutoEnableContext) bool {
	// All specified conditions must match (AND logic)

	if cond.FileExists != "" {
		checkPath := filepath.Join(ctx.ToolsPath, cond.FileExists)
		if _, err := os.Stat(checkPath); err != nil {
			return false
		}
	}

	if cond.EnvSet != "" {
		// Check context env first, fall back to os.Getenv
		val := ""
		if ctx.Env != nil {
			val = ctx.Env[cond.EnvSet]
		}
		if val == "" {
			val = os.Getenv(cond.EnvSet)
		}
		if val == "" {
			return false
		}
	}

	return true
}

// buildConfigForAutoEnable creates an MCPServerConfig for auto-enabled servers
func (d *MCPDefinitions) buildConfigForAutoEnable(def MCPServerDefinition, ctx AutoEnableContext) (api.MCPServerConfig, error) {
	// Resolve the command using the command resolver
	resolvedCommand := DefaultCommandResolver.ResolveForEnvironment(def.Command)

	config := api.MCPServerConfig{
		Name:    def.Name,
		Command: resolvedCommand,
		Args:    append([]string{}, def.Args...), // Copy args
		Env:     make(map[string]string),
	}

	// Copy environment variables
	for k, v := range def.Env {
		config.Env[k] = v
	}

	// Add path if required
	if def.RequiresPath {
		if ctx.ToolsPath == "" {
			return config, fmt.Errorf("server '%s' requires a path but none provided", def.Name)
		}

		// Validate path exists
		if _, err := os.Stat(ctx.ToolsPath); err != nil {
			return config, fmt.Errorf("invalid path for server '%s': %w", def.Name, err)
		}

		// Add path to args at specified position
		if def.PathArgIndex < 0 {
			config.Args = append(config.Args, ctx.ToolsPath)
		} else if def.PathArgIndex <= len(config.Args) {
			config.Args = append(config.Args[:def.PathArgIndex],
				append([]string{ctx.ToolsPath}, config.Args[def.PathArgIndex:]...)...)
		} else {
			// PathArgIndex out of bounds, append to end
			config.Args = append(config.Args, ctx.ToolsPath)
		}
	}

	return config, nil
}
