//go:build integration

package mcp

import (
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/x/tools"
)

// TestFilesystemIntegration tests the MCP client with a real filesystem server.
// Run with: go test ./x/tools/mcp/... -tags=integration -run TestFilesystemIntegration -v
func TestFilesystemIntegration(t *testing.T) {
	// Check if npx is available
	if _, err := exec.LookPath("npx"); err != nil {
		t.Skip("npx not available, skipping integration test")
	}

	// Create test directory and file
	testDir := "/tmp/mcp-integration-test"
	testFile := testDir + "/hello.txt"
	testContent := "Hello from MCP integration test!"

	if err := os.MkdirAll(testDir, 0755); err != nil {
		t.Fatalf("Failed to create test directory: %v", err)
	}
	defer os.RemoveAll(testDir)

	if err := os.WriteFile(testFile, []byte(testContent), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Create registry and manager
	registry := tools.NewRegistry()
	manager := NewManager()
	defer manager.Close()

	// Register filesystem MCP server
	config := ServerConfig{
		Name:    "filesystem",
		Command: "npx",
		Args:    []string{"-y", "@modelcontextprotocol/server-filesystem", testDir},
	}

	t.Log("Registering MCP filesystem server...")
	if err := manager.RegisterServer(registry, config); err != nil {
		t.Fatalf("Failed to register MCP server: %v", err)
	}

	// Check that tools were registered
	toolNames := registry.Names()
	t.Logf("Registered tools: %v", toolNames)

	if len(toolNames) == 0 {
		t.Fatal("No tools registered from MCP server")
	}

	// Find the read_file tool
	var readFileTool tools.Tool
	for _, name := range toolNames {
		if strings.HasSuffix(name, ":read_file") {
			tool, _ := registry.Get(name)
			readFileTool = tool
			break
		}
	}

	if readFileTool == nil {
		t.Fatal("read_file tool not found in registry")
	}

	t.Logf("Found tool: %s", readFileTool.Name())
	t.Logf("Description: %s", readFileTool.Description())

	// Execute the read_file tool
	t.Log("Executing read_file tool...")
	result, err := readFileTool.Execute(map[string]any{
		"path": testFile,
	})

	if err != nil {
		t.Fatalf("Failed to execute read_file: %v", err)
	}

	t.Logf("Result: %s", result)

	if !strings.Contains(result, testContent) {
		t.Errorf("Expected result to contain %q, got %q", testContent, result)
	}

	t.Log("Integration test passed!")
}

// TestServerLifecycle tests server startup and shutdown.
func TestServerLifecycle(t *testing.T) {
	if _, err := exec.LookPath("npx"); err != nil {
		t.Skip("npx not available, skipping integration test")
	}

	testDir := "/tmp/mcp-lifecycle-test"
	os.MkdirAll(testDir, 0755)
	defer os.RemoveAll(testDir)

	manager := NewManager()
	registry := tools.NewRegistry()

	config := ServerConfig{
		Name:    "lifecycle-test",
		Command: "npx",
		Args:    []string{"-y", "@modelcontextprotocol/server-filesystem", testDir},
	}

	// Register
	t.Log("Starting MCP server...")
	start := time.Now()
	if err := manager.RegisterServer(registry, config); err != nil {
		t.Fatalf("Failed to register: %v", err)
	}
	t.Logf("Server started in %v", time.Since(start))

	// Verify tools
	if registry.Count() == 0 {
		t.Error("No tools registered")
	}

	// Close
	t.Log("Closing MCP server...")
	start = time.Now()
	if err := manager.Close(); err != nil {
		t.Errorf("Failed to close: %v", err)
	}
	t.Logf("Server closed in %v", time.Since(start))
}
