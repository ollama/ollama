package mcp

import (
	"testing"

	"github.com/ollama/ollama/x/tools"
)

func TestSecurityValidation(t *testing.T) {
	tests := []struct {
		name    string
		config  ServerConfig
		wantErr bool
	}{
		{
			name: "valid npx command",
			config: ServerConfig{
				Name:    "test",
				Command: "npx",
				Args:    []string{"-y", "@modelcontextprotocol/server-filesystem"},
			},
			wantErr: false,
		},
		{
			name: "valid python command",
			config: ServerConfig{
				Name:    "test",
				Command: "python",
				Args:    []string{"-m", "mcp_server"},
			},
			wantErr: false,
		},
		{
			name: "blocked bash command",
			config: ServerConfig{
				Name:    "test",
				Command: "bash",
				Args:    []string{"-c", "echo hello"},
			},
			wantErr: true,
		},
		{
			name: "blocked sh command",
			config: ServerConfig{
				Name:    "test",
				Command: "sh",
				Args:    []string{"-c", "echo hello"},
			},
			wantErr: true,
		},
		{
			name: "shell metacharacter in args",
			config: ServerConfig{
				Name:    "test",
				Command: "npx",
				Args:    []string{"-y", "package; rm -rf /"},
			},
			wantErr: true,
		},
		{
			name: "pipe in args",
			config: ServerConfig{
				Name:    "test",
				Command: "npx",
				Args:    []string{"-y", "package | cat"},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateServerConfig(tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateServerConfig() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestToolName(t *testing.T) {
	client := NewClient("filesystem", "npx", nil, nil)
	tool := &Tool{
		client:   client,
		toolName: "read_file",
	}

	expected := "filesystem:read_file"
	if got := tool.Name(); got != expected {
		t.Errorf("Tool.Name() = %v, want %v", got, expected)
	}
}

func TestManagerBasics(t *testing.T) {
	manager := NewManager()
	_ = tools.NewRegistry() // Would be used with real MCP server

	// Can't actually test registration without a real MCP server
	// But we can test the manager structure
	if manager.clients == nil {
		t.Error("Manager.clients should be initialized")
	}

	// Close should work on empty manager
	if err := manager.Close(); err != nil {
		t.Errorf("Close() on empty manager should not error: %v", err)
	}
}
