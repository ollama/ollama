package server

// =============================================================================
// MCP Integration Tests
// =============================================================================
//
// This file contains tests for the MCP (Model Context Protocol) implementation.
//
// Test Categories:
//
// 1. Client Tests (TestMCPClient*)
//    - Client initialization and lifecycle
//    - Environment variable filtering
//    - Timeout handling
//
// 2. Security Tests (TestDangerous*, TestShellInjection*, TestSecure*)
//    - Command blocklist validation
//    - Shell metacharacter detection
//    - Credential filtering
//
// 3. Manager Tests (TestMCPManager*, TestToolResult*, TestParallel*)
//    - Server registration
//    - Tool caching
//    - Parallel execution
//
// 4. Auto-Enable Tests (TestAutoEnable*)
//    - Mode: never, always, with_path, if_match
//    - Conditions: file_exists, env_set
//
// Run all MCP tests:
//   go test -v ./server/... -run "TestMCP|TestSecure|TestShell|TestTool|TestDanger|TestParallel|TestAutoEnable"
//
// =============================================================================

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
)

// TestMCPClientInitialization tests the MCP client initialization
func TestMCPClientInitialization(t *testing.T) {
	client := NewMCPClient("test", "echo", []string{"test"}, nil)

	require.Equal(t, "test", client.name)
	require.Equal(t, "echo", client.command)
	require.False(t, client.initialized, "Client should not be initialized on creation")
}

// TestSecureEnvironmentFiltering tests environment variable filtering
func TestSecureEnvironmentFiltering(t *testing.T) {
	// Set some test environment variables
	os.Setenv("TEST_SAFE_VAR", "safe_value")
	os.Setenv("AWS_SECRET_ACCESS_KEY", "secret_key")
	os.Setenv("PATH", "/usr/local/bin:/usr/bin:/bin:/root/bin")
	defer os.Unsetenv("TEST_SAFE_VAR")
	defer os.Unsetenv("AWS_SECRET_ACCESS_KEY")

	client := NewMCPClient("test", "echo", []string{}, nil)
	env := client.buildSecureEnvironment()

	// Check that sensitive variables are filtered out
	for _, e := range env {
		require.False(t, strings.HasPrefix(e, "AWS_SECRET_ACCESS_KEY="),
			"Sensitive AWS_SECRET_ACCESS_KEY should be filtered out")
		require.False(t, strings.Contains(e, "/root/bin"),
			"Dangerous PATH component /root/bin should be filtered out")
	}

	// Check that PATH is present but sanitized
	hasPath := false
	for _, e := range env {
		if strings.HasPrefix(e, "PATH=") {
			hasPath = true
			require.NotContains(t, e, "/root",
				"PATH should not contain /root directories")
		}
	}
	require.True(t, hasPath, "PATH should be present in environment")
}

// TestMCPManagerAddServer tests adding MCP servers to the manager
func TestMCPManagerAddServer(t *testing.T) {
	manager := NewMCPManager(5)

	// Test adding a valid server config
	config := api.MCPServerConfig{
		Name:    "test_server",
		Command: "python",
		Args:    []string{"-m", "test_module"},
		Env:     map[string]string{"TEST": "value"},
	}

	// This will fail in test environment but validates the validation logic
	err := manager.AddServer(config)
	if err != nil {
		require.Contains(t, err.Error(), "failed to initialize",
			"Expected initialization failure in test environment")
	}

	// Test invalid server names
	invalidConfigs := []api.MCPServerConfig{
		{Name: "", Command: "python"},                       // Empty name
		{Name: strings.Repeat("a", 101), Command: "python"}, // Too long
		{Name: "test/server", Command: "python"},            // Invalid characters
	}

	for _, cfg := range invalidConfigs {
		err := manager.validateServerConfig(cfg)
		require.Error(t, err, "Should reject invalid config: %+v", cfg)
	}
}

// TestDangerousCommandValidation tests rejection of dangerous commands
func TestDangerousCommandValidation(t *testing.T) {
	manager := NewMCPManager(5)

	dangerousConfigs := []api.MCPServerConfig{
		{Name: "test1", Command: "bash"},
		{Name: "test2", Command: "/bin/sh"},
		{Name: "test3", Command: "sudo"},
		{Name: "test4", Command: "rm"},
		{Name: "test5", Command: "curl"},
		{Name: "test6", Command: "eval"},
	}

	for _, cfg := range dangerousConfigs {
		err := manager.validateServerConfig(cfg)
		require.Error(t, err, "Should reject dangerous command: %s", cfg.Command)
		require.Contains(t, err.Error(), "not allowed for security",
			"Expected security error for command %s", cfg.Command)
	}

	// Test that safe commands are allowed
	safeConfigs := []api.MCPServerConfig{
		{Name: "test1", Command: "python"},
		{Name: "test2", Command: "node"},
		{Name: "test3", Command: "/usr/bin/python3"},
	}

	for _, cfg := range safeConfigs {
		err := manager.validateServerConfig(cfg)
		require.NoError(t, err, "Should allow safe command %s", cfg.Command)
	}
}

// TestShellInjectionPrevention tests prevention of shell injection
func TestShellInjectionPrevention(t *testing.T) {
	manager := NewMCPManager(5)

	// Test arguments with shell metacharacters
	injectionConfigs := []api.MCPServerConfig{
		{
			Name:    "test1",
			Command: "python",
			Args:    []string{"; rm -rf /"},
		},
		{
			Name:    "test2",
			Command: "python",
			Args:    []string{"test", "| cat /etc/passwd"},
		},
		{
			Name:    "test3",
			Command: "python",
			Args:    []string{"$(whoami)"},
		},
		{
			Name:    "test4",
			Command: "python",
			Args:    []string{"`id`"},
		},
	}

	for _, cfg := range injectionConfigs {
		err := manager.validateServerConfig(cfg)
		require.Error(t, err, "Should reject shell injection attempt in args: %v", cfg.Args)
		require.Contains(t, err.Error(), "shell metacharacters",
			"Expected shell metacharacter error")
	}
}


// TestParallelToolExecution tests parallel execution of tools
func TestParallelToolExecution(t *testing.T) {
	manager := NewMCPManager(5)

	// Create test tool calls
	toolCalls := []api.ToolCall{
		{
			Function: api.ToolCallFunction{
				Name:      "tool1",
				Arguments: map[string]interface{}{"test": "1"},
			},
		},
		{
			Function: api.ToolCallFunction{
				Name:      "tool2",
				Arguments: map[string]interface{}{"test": "2"},
			},
		},
		{
			Function: api.ToolCallFunction{
				Name:      "tool3",
				Arguments: map[string]interface{}{"test": "3"},
			},
		},
	}

	// Execute in parallel (will fail but tests the mechanism)
	results := manager.ExecuteToolsParallel(toolCalls)

	require.Len(t, results, len(toolCalls))

	// All should have errors since no MCP servers are connected
	for i, result := range results {
		require.Error(t, result.Error, "Expected error for tool call %d", i)
	}
}


// TestMCPClientTimeout tests timeout handling for tool execution
func TestMCPClientTimeout(t *testing.T) {
	client := NewMCPClient("test", "sleep", []string{"60"}, nil)

	// Create a context with very short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// Try to call with timeout (will fail but tests the mechanism)
	req := mcpCallToolRequest{
		Name:      "test_tool",
		Arguments: map[string]interface{}{},
	}

	var resp mcpCallToolResponse
	err := client.callWithContext(ctx, "tools/call", req, &resp)

	// Should timeout or fail
	require.Error(t, err, "Expected timeout or error")
}

// TestEnvironmentVariableValidation tests validation of environment variables
func TestEnvironmentVariableValidation(t *testing.T) {
	manager := NewMCPManager(5)

	// Test invalid environment variable names
	invalidEnvConfigs := []api.MCPServerConfig{
		{
			Name:    "test1",
			Command: "python",
			Env:     map[string]string{"VAR=BAD": "value"},
		},
		{
			Name:    "test2",
			Command: "python",
			Env:     map[string]string{"VAR;CMD": "value"},
		},
		{
			Name:    "test3",
			Command: "python",
			Env:     map[string]string{"VAR|PIPE": "value"},
		},
	}

	for _, cfg := range invalidEnvConfigs {
		err := manager.validateServerConfig(cfg)
		require.Error(t, err, "Should reject invalid environment variable names: %v", cfg.Env)
	}

	// Test valid environment variables
	validConfig := api.MCPServerConfig{
		Name:    "test",
		Command: "python",
		Env: map[string]string{
			"PYTHONPATH": "/usr/lib/python3",
			"MY_VAR":     "value",
			"TEST_123":   "test",
		},
	}

	err := manager.validateServerConfig(validConfig)
	require.NoError(t, err, "Should allow valid environment variables")
}

// BenchmarkToolExecution benchmarks tool execution performance
func BenchmarkToolExecution(b *testing.B) {
	manager := NewMCPManager(10)

	toolCall := api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      "test_tool",
			Arguments: map[string]interface{}{"param": "value"},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = manager.ExecuteTool(toolCall)
	}
}

// BenchmarkParallelToolExecution benchmarks parallel tool execution
func BenchmarkParallelToolExecution(b *testing.B) {
	manager := NewMCPManager(10)

	toolCalls := make([]api.ToolCall, 10)
	for i := range toolCalls {
		toolCalls[i] = api.ToolCall{
			Function: api.ToolCallFunction{
				Name:      fmt.Sprintf("tool_%d", i),
				Arguments: map[string]interface{}{"param": i},
			},
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = manager.ExecuteToolsParallel(toolCalls)
	}
}

// =============================================================================
// Auto-Enable Unit Tests
// =============================================================================

// TestAutoEnableMode_Never verifies servers with auto_enable:"never" don't auto-enable
func TestAutoEnableMode_Never(t *testing.T) {
	defs := &MCPDefinitions{
		Servers: map[string]MCPServerDefinition{
			"never_server": {
				Name:       "never_server",
				Command:    "python",
				AutoEnable: AutoEnableNever,
			},
			"empty_mode": {
				Name:    "empty_mode",
				Command: "python",
				// AutoEnable not set - defaults to never
			},
		},
	}

	ctx := AutoEnableContext{ToolsPath: "/some/path"}
	servers := defs.GetAutoEnableServers(ctx)

	require.Empty(t, servers, "Expected 0 auto-enabled servers for 'never' mode")
}

// TestAutoEnableMode_Always verifies servers with auto_enable:"always" always enable
func TestAutoEnableMode_Always(t *testing.T) {
	defs := &MCPDefinitions{
		Servers: map[string]MCPServerDefinition{
			"always_server": {
				Name:       "always_server",
				Command:    "python",
				Args:       []string{"-m", "server"},
				AutoEnable: AutoEnableAlways,
			},
		},
	}

	// Should enable even with empty path
	ctx := AutoEnableContext{ToolsPath: ""}
	servers := defs.GetAutoEnableServers(ctx)

	require.Len(t, servers, 1)
	require.Equal(t, "always_server", servers[0].Name)

	// Should also enable with path
	ctx = AutoEnableContext{ToolsPath: "/tmp"}
	servers = defs.GetAutoEnableServers(ctx)

	require.Len(t, servers, 1)
}

// TestAutoEnableMode_WithPath verifies servers with auto_enable:"with_path" enable only when path is provided
func TestAutoEnableMode_WithPath(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "mcp-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	defs := &MCPDefinitions{
		Servers: map[string]MCPServerDefinition{
			"path_server": {
				Name:         "path_server",
				Command:      "python",
				Args:         []string{"-m", "server"},
				RequiresPath: true,
				PathArgIndex: -1,
				AutoEnable:   AutoEnableWithPath,
			},
		},
	}

	// Should NOT enable without path
	ctx := AutoEnableContext{ToolsPath: ""}
	servers := defs.GetAutoEnableServers(ctx)
	require.Empty(t, servers, "Expected 0 servers without path")

	// Should enable with valid path
	ctx = AutoEnableContext{ToolsPath: tmpDir}
	servers = defs.GetAutoEnableServers(ctx)
	require.Len(t, servers, 1)

	// Verify path was appended to args
	expectedArgs := []string{"-m", "server", tmpDir}
	require.Equal(t, expectedArgs, servers[0].Args)
}

// TestAutoEnableMode_IfMatch_FileExists verifies file_exists condition
func TestAutoEnableMode_IfMatch_FileExists(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "mcp-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	// Create .git directory to simulate git repo
	gitDir := filepath.Join(tmpDir, ".git")
	require.NoError(t, os.Mkdir(gitDir, 0755))

	defs := &MCPDefinitions{
		Servers: map[string]MCPServerDefinition{
			"git_server": {
				Name:         "git_server",
				Command:      "python",
				Args:         []string{"-m", "git_server"},
				RequiresPath: true,
				PathArgIndex: -1,
				AutoEnable:   AutoEnableIfMatch,
				EnableIf:     EnableCondition{FileExists: ".git"},
			},
		},
	}

	// Should enable when .git exists
	ctx := AutoEnableContext{ToolsPath: tmpDir}
	servers := defs.GetAutoEnableServers(ctx)
	require.Len(t, servers, 1, "Expected 1 server when .git exists")

	// Should NOT enable in directory without .git
	noGitDir, err := os.MkdirTemp("", "mcp-test-nogit-*")
	require.NoError(t, err)
	defer os.RemoveAll(noGitDir)

	ctx = AutoEnableContext{ToolsPath: noGitDir}
	servers = defs.GetAutoEnableServers(ctx)
	require.Empty(t, servers, "Expected 0 servers without .git")
}

// TestAutoEnableMode_IfMatch_EnvSet verifies env_set condition
func TestAutoEnableMode_IfMatch_EnvSet(t *testing.T) {
	defs := &MCPDefinitions{
		Servers: map[string]MCPServerDefinition{
			"env_server": {
				Name:       "env_server",
				Command:    "python",
				AutoEnable: AutoEnableIfMatch,
				EnableIf:   EnableCondition{EnvSet: "MCP_TEST_VAR"},
			},
		},
	}

	// Test with env in context
	ctx := AutoEnableContext{
		ToolsPath: "",
		Env:       map[string]string{"MCP_TEST_VAR": "some_value"},
	}
	servers := defs.GetAutoEnableServers(ctx)
	require.Len(t, servers, 1, "Expected 1 server when env is set in context")

	// Test with env NOT set
	ctx = AutoEnableContext{
		ToolsPath: "",
		Env:       map[string]string{},
	}
	servers = defs.GetAutoEnableServers(ctx)
	require.Empty(t, servers, "Expected 0 servers when env not set")

	// Test with os.Getenv fallback
	os.Setenv("MCP_TEST_VAR_FALLBACK", "fallback_value")
	defer os.Unsetenv("MCP_TEST_VAR_FALLBACK")

	defsFallback := &MCPDefinitions{
		Servers: map[string]MCPServerDefinition{
			"env_server": {
				Name:       "env_server",
				Command:    "python",
				AutoEnable: AutoEnableIfMatch,
				EnableIf:   EnableCondition{EnvSet: "MCP_TEST_VAR_FALLBACK"},
			},
		},
	}
	ctx = AutoEnableContext{ToolsPath: "", Env: nil}
	servers = defsFallback.GetAutoEnableServers(ctx)
	require.Len(t, servers, 1, "Expected 1 server with os.Getenv fallback")
}

// TestAutoEnableMode_IfMatch_CombinedConditions verifies AND logic for conditions
func TestAutoEnableMode_IfMatch_CombinedConditions(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "mcp-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	markerFile := filepath.Join(tmpDir, ".marker")
	require.NoError(t, os.WriteFile(markerFile, []byte("test"), 0644))

	defs := &MCPDefinitions{
		Servers: map[string]MCPServerDefinition{
			"combined_server": {
				Name:         "combined_server",
				Command:      "python",
				RequiresPath: true,
				PathArgIndex: -1,
				AutoEnable:   AutoEnableIfMatch,
				EnableIf: EnableCondition{
					FileExists: ".marker",
					EnvSet:     "MCP_COMBINED_TEST",
				},
			},
		},
	}

	// Should NOT enable when only file exists
	ctx := AutoEnableContext{
		ToolsPath: tmpDir,
		Env:       map[string]string{},
	}
	servers := defs.GetAutoEnableServers(ctx)
	require.Empty(t, servers, "Expected 0 servers when only file condition matches")

	// Should NOT enable when only env is set
	ctx = AutoEnableContext{
		ToolsPath: "/nonexistent",
		Env:       map[string]string{"MCP_COMBINED_TEST": "value"},
	}
	servers = defs.GetAutoEnableServers(ctx)
	require.Empty(t, servers, "Expected 0 servers when only env condition matches")

	// Should enable when BOTH conditions match
	ctx = AutoEnableContext{
		ToolsPath: tmpDir,
		Env:       map[string]string{"MCP_COMBINED_TEST": "value"},
	}
	servers = defs.GetAutoEnableServers(ctx)
	require.Len(t, servers, 1, "Expected 1 server when both conditions match")
}

// TestGetAutoEnableServers_MultipleServers verifies multiple servers can auto-enable
func TestGetAutoEnableServers_MultipleServers(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "mcp-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	// Create .git directory
	require.NoError(t, os.Mkdir(filepath.Join(tmpDir, ".git"), 0755))

	defs := &MCPDefinitions{
		Servers: map[string]MCPServerDefinition{
			"filesystem": {
				Name:         "filesystem",
				Command:      "npx",
				Args:         []string{"-y", "@mcp/server-filesystem"},
				RequiresPath: true,
				PathArgIndex: -1,
				AutoEnable:   AutoEnableWithPath,
			},
			"git": {
				Name:         "git",
				Command:      "python",
				Args:         []string{"-m", "mcp_git"},
				RequiresPath: true,
				PathArgIndex: -1,
				AutoEnable:   AutoEnableIfMatch,
				EnableIf:     EnableCondition{FileExists: ".git"},
			},
			"never_server": {
				Name:       "never_server",
				Command:    "python",
				AutoEnable: AutoEnableNever,
			},
		},
	}

	ctx := AutoEnableContext{ToolsPath: tmpDir}
	servers := defs.GetAutoEnableServers(ctx)

	require.Len(t, servers, 2, "Expected 2 auto-enabled servers")

	// Verify both filesystem and git are enabled
	names := make(map[string]bool)
	for _, s := range servers {
		names[s.Name] = true
	}

	require.True(t, names["filesystem"], "Expected 'filesystem' server to be auto-enabled")
	require.True(t, names["git"], "Expected 'git' server to be auto-enabled")
	require.False(t, names["never_server"], "'never_server' should NOT be auto-enabled")
}

// TestBuildConfigForAutoEnable_PathArgIndex verifies path insertion at different positions
func TestBuildConfigForAutoEnable_PathArgIndex(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "mcp-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	testCases := []struct {
		name         string
		args         []string
		pathArgIndex int
		expected     []string
	}{
		{
			name:         "append at end (index -1)",
			args:         []string{"arg1", "arg2"},
			pathArgIndex: -1,
			expected:     []string{"arg1", "arg2", tmpDir},
		},
		{
			name:         "insert at beginning (index 0)",
			args:         []string{"arg1", "arg2"},
			pathArgIndex: 0,
			expected:     []string{tmpDir, "arg1", "arg2"},
		},
		{
			name:         "insert in middle (index 1)",
			args:         []string{"arg1", "arg2"},
			pathArgIndex: 1,
			expected:     []string{"arg1", tmpDir, "arg2"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defs := &MCPDefinitions{
				Servers: map[string]MCPServerDefinition{
					"test": {
						Name:         "test",
						Command:      "python",
						Args:         tc.args,
						RequiresPath: true,
						PathArgIndex: tc.pathArgIndex,
						AutoEnable:   AutoEnableWithPath,
					},
				},
			}

			ctx := AutoEnableContext{ToolsPath: tmpDir}
			servers := defs.GetAutoEnableServers(ctx)

			require.Len(t, servers, 1)
			require.Equal(t, tc.expected, servers[0].Args)
		})
	}
}

// TestBuildConfigForAutoEnable_InvalidPath verifies error handling for invalid paths
func TestBuildConfigForAutoEnable_InvalidPath(t *testing.T) {
	defs := &MCPDefinitions{
		Servers: map[string]MCPServerDefinition{
			"path_server": {
				Name:         "path_server",
				Command:      "python",
				RequiresPath: true,
				AutoEnable:   AutoEnableWithPath,
			},
		},
	}

	// Should fail with non-existent path
	ctx := AutoEnableContext{ToolsPath: "/definitely/not/a/real/path/12345"}
	servers := defs.GetAutoEnableServers(ctx)

	// Server should be skipped due to invalid path
	require.Empty(t, servers, "Expected 0 servers with invalid path")
}

// TestBuildConfigForAutoEnable_EnvCopy verifies environment variables are copied
func TestBuildConfigForAutoEnable_EnvCopy(t *testing.T) {
	defs := &MCPDefinitions{
		Servers: map[string]MCPServerDefinition{
			"env_server": {
				Name:       "env_server",
				Command:    "python",
				AutoEnable: AutoEnableAlways,
				Env: map[string]string{
					"VAR1": "value1",
					"VAR2": "value2",
				},
			},
		},
	}

	ctx := AutoEnableContext{}
	servers := defs.GetAutoEnableServers(ctx)

	require.Len(t, servers, 1)
	require.Equal(t, "value1", servers[0].Env["VAR1"])
	require.Equal(t, "value2", servers[0].Env["VAR2"])

	// Verify original wasn't mutated by modifying copy
	servers[0].Env["VAR1"] = "modified"
	original := defs.Servers["env_server"]
	require.Equal(t, "value1", original.Env["VAR1"], "Original server definition was mutated")
}

// TestEnableCondition_EmptyConditions verifies empty conditions always match
func TestEnableCondition_EmptyConditions(t *testing.T) {
	defs := &MCPDefinitions{
		Servers: map[string]MCPServerDefinition{
			"empty_cond": {
				Name:       "empty_cond",
				Command:    "python",
				AutoEnable: AutoEnableIfMatch,
				EnableIf:   EnableCondition{}, // Empty - should always match
			},
		},
	}

	ctx := AutoEnableContext{ToolsPath: ""}
	servers := defs.GetAutoEnableServers(ctx)

	require.Len(t, servers, 1, "Expected 1 server with empty conditions")
}
