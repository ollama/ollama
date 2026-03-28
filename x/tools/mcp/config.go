package mcp

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/x/tools"
)

// Config holds MCP server configurations loaded from file.
type Config struct {
	Servers []ServerConfig `json:"servers"`
}

// blockedCommands are commands that cannot be used as MCP servers.
// These are shells and interpreters that could be used to bypass security.
// Runtime security for tool execution is handled by the approval system.
var blockedCommands = []string{
	// Shells - these could execute arbitrary commands
	"sh", "bash", "zsh", "fish", "csh", "ksh", "dash",
	"cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh",
}

// blockedMetacharacters prevent shell injection in server arguments.
var blockedMetacharacters = []string{
	";", "|", "&", "$(", "`", ">", "<", ">>", "<<",
	"||", "&&", "\n", "\r",
}

// LoadConfig loads MCP server configurations from standard locations.
// Priority: ~/.ollama/mcp-servers.json > /etc/ollama/mcp-servers.json
func LoadConfig() (*Config, error) {
	paths := []string{
		filepath.Join(os.Getenv("HOME"), ".ollama", "mcp-servers.json"),
		"/etc/ollama/mcp-servers.json",
	}

	for _, path := range paths {
		config, err := loadConfigFromFile(path)
		if err == nil {
			slog.Debug("Loaded MCP config", "path", path, "servers", len(config.Servers))
			return config, nil
		}
		if !os.IsNotExist(err) {
			slog.Warn("Error loading MCP config", "path", path, "error", err)
		}
	}

	// Return empty config if no file found
	return &Config{Servers: []ServerConfig{}}, nil
}

// loadConfigFromFile loads MCP configuration from a specific file.
func loadConfigFromFile(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	return &config, nil
}

// validateServerConfig checks if an MCP server configuration is safe.
func validateServerConfig(config ServerConfig) error {
	// Check command against blocklist
	cmdLower := strings.ToLower(config.Command)
	for _, blocked := range blockedCommands {
		if cmdLower == blocked || strings.HasSuffix(cmdLower, "/"+blocked) {
			return fmt.Errorf("command '%s' is blocked: shells cannot be MCP servers", config.Command)
		}
	}

	// Check args for shell metacharacters
	for _, arg := range config.Args {
		for _, meta := range blockedMetacharacters {
			if strings.Contains(arg, meta) {
				return fmt.Errorf("argument contains blocked character '%s'", meta)
			}
		}
	}

	return nil
}

// RegisterFromConfig loads MCP config and registers all servers.
func RegisterFromConfig(registry *tools.Registry, manager *Manager) error {
	config, err := LoadConfig()
	if err != nil {
		return err
	}

	for _, serverConfig := range config.Servers {
		// Validate security
		if err := validateServerConfig(serverConfig); err != nil {
			slog.Warn("Skipping MCP server due to security validation",
				"name", serverConfig.Name, "error", err)
			continue
		}

		// Register server
		if err := manager.RegisterServer(registry, serverConfig); err != nil {
			slog.Warn("Failed to register MCP server",
				"name", serverConfig.Name, "error", err)
			continue
		}
	}

	return nil
}
