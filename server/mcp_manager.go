package server

import (
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/ollama/ollama/api"
)

// MCPManager manages multiple MCP server connections and provides tool execution services
type MCPManager struct {
	mu          sync.RWMutex
	clients     map[string]*MCPClient
	toolRouting map[string]string // tool name -> client name mapping
	maxClients  int
}

// MCPServerConfig is imported from api package

// ToolResult represents the result of a tool execution
type ToolResult struct {
	Content string
	Error   error
}

// ExecutionPlan represents the execution strategy for a set of tool calls
type ExecutionPlan struct {
	RequiresSequential bool
	Groups             [][]int // Groups of tool indices that can run in parallel
	Reason             string  // Explanation of why this plan was chosen
}

// NewMCPManager creates a new MCP manager
func NewMCPManager(maxClients int) *MCPManager {
	return &MCPManager{
		clients:     make(map[string]*MCPClient),
		toolRouting: make(map[string]string),
		maxClients:  maxClients,
	}
}

// AddServer adds a new MCP server to the manager
func (m *MCPManager) AddServer(config api.MCPServerConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.clients) >= m.maxClients {
		return fmt.Errorf("maximum number of MCP servers reached (%d)", m.maxClients)
	}

	if _, exists := m.clients[config.Name]; exists {
		return fmt.Errorf("MCP server '%s' already exists", config.Name)
	}

	// Validate server configuration for security
	if err := m.validateServerConfig(config); err != nil {
		return fmt.Errorf("invalid MCP server configuration: %w", err)
	}

	// Create and initialize the MCP client
	client := NewMCPClient(config.Name, config.Command, config.Args, config.Env)
	
	if err := client.Initialize(); err != nil {
		client.Close()
		return fmt.Errorf("failed to initialize MCP server '%s': %w", config.Name, err)
	}

	// Discover tools
	tools, err := client.ListTools()
	if err != nil {
		client.Close()
		return fmt.Errorf("failed to list tools from MCP server '%s': %w", config.Name, err)
	}

	// Update tool routing
	for _, tool := range tools {
		m.toolRouting[tool.Function.Name] = config.Name
	}

	m.clients[config.Name] = client

	slog.Info("MCP server added", "name", config.Name, "tools", len(tools))
	return nil
}

// RemoveServer removes an MCP server from the manager
func (m *MCPManager) RemoveServer(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	client, exists := m.clients[name]
	if !exists {
		return fmt.Errorf("MCP server '%s' not found", name)
	}

	// Remove tool routing entries
	for toolName, clientName := range m.toolRouting {
		if clientName == name {
			delete(m.toolRouting, toolName)
		}
	}

	// Close the client
	if err := client.Close(); err != nil {
		slog.Warn("Error closing MCP client", "name", name, "error", err)
	}

	delete(m.clients, name)

	slog.Info("MCP server removed", "name", name)
	return nil
}

// GetAllTools returns all available tools from all MCP servers
func (m *MCPManager) GetAllTools() []api.Tool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var allTools []api.Tool
	
	for _, client := range m.clients {
		tools, err := client.ListTools()
		if err != nil {
			slog.Warn("Failed to get tools from MCP client", "name", client.name, "error", err)
			continue
		}
		allTools = append(allTools, tools...)
	}

	return allTools
}

// ExecuteTool executes a single tool call
func (m *MCPManager) ExecuteTool(toolCall api.ToolCall) ToolResult {
	toolName := toolCall.Function.Name

	m.mu.RLock()
	clientName, exists := m.toolRouting[toolName]
	if !exists {
		m.mu.RUnlock()
		return ToolResult{Error: fmt.Errorf("tool '%s' not found", toolName)}
	}

	client, exists := m.clients[clientName]
	if !exists {
		m.mu.RUnlock()
		return ToolResult{Error: fmt.Errorf("MCP client '%s' not found", clientName)}
	}
	m.mu.RUnlock()

	// Convert arguments to map[string]interface{}
	args := make(map[string]interface{})
	for k, v := range toolCall.Function.Arguments {
		args[k] = v
	}

	// Execute the tool
	content, err := client.CallTool(toolName, args)
	if err != nil {
		slog.Debug("MCP tool execution failed", "tool", toolName, "client", clientName)
	} else {
		slog.Debug("MCP tool executed", "tool", toolName, "client", clientName, "result_length", len(content))
	}
	return ToolResult{
		Content: content,
		Error:   err,
	}
}

// AnalyzeExecutionPlan analyzes tool calls to determine optimal execution strategy
func (m *MCPManager) AnalyzeExecutionPlan(toolCalls []api.ToolCall) ExecutionPlan {
	if len(toolCalls) <= 1 {
		return ExecutionPlan{
			RequiresSequential: false,
			Groups:             [][]int{{0}},
			Reason:             "Single tool call",
		}
	}

	// Analyze tool patterns for dependencies
	hasWriteOperations := false
	hasReadOperations := false
	fileTargets := make(map[string][]int) // Track which tools operate on which files
	
	for i, toolCall := range toolCalls {
		toolName := toolCall.Function.Name
		args := toolCall.Function.Arguments
		
		// Check for file operations
		if strings.Contains(toolName, "write") || strings.Contains(toolName, "create") ||
		   strings.Contains(toolName, "edit") || strings.Contains(toolName, "append") {
			hasWriteOperations = true
			
			// Try to extract file path from arguments
			if pathArg, exists := args["path"]; exists {
				if path, ok := pathArg.(string); ok {
					fileTargets[path] = append(fileTargets[path], i)
				}
			} else if fileArg, exists := args["file"]; exists {
				if file, ok := fileArg.(string); ok {
					fileTargets[file] = append(fileTargets[file], i)
				}
			}
		}
		
		if strings.Contains(toolName, "read") || strings.Contains(toolName, "list") ||
		   strings.Contains(toolName, "get") {
			hasReadOperations = true
			
			// Try to extract file path from arguments
			if pathArg, exists := args["path"]; exists {
				if path, ok := pathArg.(string); ok {
					fileTargets[path] = append(fileTargets[path], i)
				}
			} else if fileArg, exists := args["file"]; exists {
				if file, ok := fileArg.(string); ok {
					fileTargets[file] = append(fileTargets[file], i)
				}
			}
		}
	}
	
	// Determine if sequential execution is needed
	requiresSequential := false
	reason := "Can execute in parallel"
	
	// Check for file operation dependencies
	if hasWriteOperations && hasReadOperations {
		requiresSequential = true
		reason = "Mixed read and write operations detected"
	}
	
	// Check for operations on the same file
	for file, indices := range fileTargets {
		if len(indices) > 1 {
			requiresSequential = true
			reason = fmt.Sprintf("Multiple operations on the same file: %s", file)
			break
		}
	}
	
	// Check for explicit ordering patterns in tool names
	for i := 0; i < len(toolCalls)-1; i++ {
		curr := toolCalls[i].Function.Name
		next := toolCalls[i+1].Function.Name
		
		// Common patterns that suggest ordering
		if (strings.Contains(curr, "create") && strings.Contains(next, "read")) ||
		   (strings.Contains(curr, "write") && strings.Contains(next, "read")) ||
		   (strings.Contains(curr, "1") && strings.Contains(next, "2")) ||
		   (strings.Contains(curr, "first") && strings.Contains(next, "second")) ||
		   (strings.Contains(curr, "init") && strings.Contains(next, "use")) {
			requiresSequential = true
			reason = "Tool names suggest sequential dependency"
			break
		}
	}
	
	// Build execution groups
	var groups [][]int
	if requiresSequential {
		// Each tool in its own group for sequential execution
		for i := range toolCalls {
			groups = append(groups, []int{i})
		}
	} else {
		// All tools in one group for parallel execution
		group := make([]int, len(toolCalls))
		for i := range toolCalls {
			group[i] = i
		}
		groups = [][]int{group}
	}
	
	plan := ExecutionPlan{
		RequiresSequential: requiresSequential,
		Groups:             groups,
		Reason:             reason,
	}
	
	slog.Debug("Execution plan analyzed",
		"sequential", requiresSequential,
		"reason", reason,
		"tool_count", len(toolCalls))
	
	return plan
}

// ExecuteWithPlan executes tool calls according to the execution plan
func (m *MCPManager) ExecuteWithPlan(toolCalls []api.ToolCall, plan ExecutionPlan) []ToolResult {
	results := make([]ToolResult, len(toolCalls))
	
	for _, group := range plan.Groups {
		if len(group) == 1 {
			// Single tool, execute directly
			idx := group[0]
			results[idx] = m.ExecuteTool(toolCalls[idx])
		} else {
			// Multiple tools in group, execute in parallel
			var wg sync.WaitGroup
			for _, idx := range group {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					results[i] = m.ExecuteTool(toolCalls[i])
				}(idx)
			}
			wg.Wait()
		}
	}
	
	return results
}

// ExecuteToolsParallel executes multiple tool calls in parallel
func (m *MCPManager) ExecuteToolsParallel(toolCalls []api.ToolCall) []ToolResult {
	if len(toolCalls) == 0 {
		return nil
	}

	results := make([]ToolResult, len(toolCalls))
	
	// For single tool call, execute directly
	if len(toolCalls) == 1 {
		results[0] = m.ExecuteTool(toolCalls[0])
		return results
	}

	// Execute multiple tools in parallel
	var wg sync.WaitGroup
	for i, toolCall := range toolCalls {
		wg.Add(1)
		go func(index int, tc api.ToolCall) {
			defer wg.Done()
			results[index] = m.ExecuteTool(tc)
		}(i, toolCall)
	}

	wg.Wait()
	return results
}

// ExecuteToolsSequential executes multiple tool calls sequentially
func (m *MCPManager) ExecuteToolsSequential(toolCalls []api.ToolCall) []ToolResult {
	results := make([]ToolResult, len(toolCalls))
	
	for i, toolCall := range toolCalls {
		results[i] = m.ExecuteTool(toolCall)
		
		// Stop on first error if desired
		if results[i].Error != nil {
			slog.Warn("Tool execution failed", "tool", toolCall.Function.Name, "error", results[i].Error)
		}
	}

	return results
}

// GetToolClient returns the client name for a given tool
func (m *MCPManager) GetToolClient(toolName string) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	clientName, exists := m.toolRouting[toolName]
	return clientName, exists
}

// GetServerNames returns a list of all registered MCP server names
func (m *MCPManager) GetServerNames() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	names := make([]string, 0, len(m.clients))
	for name := range m.clients {
		names = append(names, name)
	}
	
	return names
}

// GetToolDefinition returns the definition for a specific tool
func (m *MCPManager) GetToolDefinition(serverName, toolName string) *api.Tool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	client, exists := m.clients[serverName]
	if !exists {
		return nil
	}
	
	// Get tools from the client
	tools := client.GetTools()
	for _, tool := range tools {
		if tool.Function.Name == toolName {
			return &tool
		}
	}
	
	return nil
}

// Close shuts down all MCP clients
func (m *MCPManager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var errs []string

	for name, client := range m.clients {
		if err := client.Close(); err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", name, err))
		}
	}

	// Clear all data
	m.clients = make(map[string]*MCPClient)
	m.toolRouting = make(map[string]string)

	if len(errs) > 0 {
		return fmt.Errorf("errors closing MCP clients: %s", strings.Join(errs, "; "))
	}

	return nil
}

// Shutdown is an alias for Close for consistency with registry
func (m *MCPManager) Shutdown() error {
	slog.Info("Shutting down MCP manager", "clients", len(m.clients))
	return m.Close()
}

// validateServerConfig validates MCP server configuration for security
func (m *MCPManager) validateServerConfig(config api.MCPServerConfig) error {
	// Validate name
	if config.Name == "" {
		return fmt.Errorf("server name cannot be empty")
	}
	if len(config.Name) > 100 {
		return fmt.Errorf("server name too long (max 100 characters)")
	}
	if strings.ContainsAny(config.Name, "/\\:*?\"<>|") {
		return fmt.Errorf("server name contains invalid characters")
	}

	// Validate command
	if config.Command == "" {
		return fmt.Errorf("command cannot be empty")
	}
	
	// Get security configuration
	securityConfig := GetSecurityConfig()
	
	// Check if command is allowed by security policy
	if !securityConfig.IsCommandAllowed(config.Command) {
		return fmt.Errorf("command '%s' is not allowed for security reasons", config.Command)
	}
	
	// Validate command path (must be absolute or in PATH)
	if strings.Contains(config.Command, "..") {
		return fmt.Errorf("command path cannot contain '..'")
	}
	
	// Validate arguments
	for _, arg := range config.Args {
		if strings.Contains(arg, "..") || strings.HasPrefix(arg, "-") && len(arg) > 50 {
			return fmt.Errorf("suspicious argument detected: %s", arg)
		}
		// Check for shell injection attempts using security config
		if securityConfig.HasShellMetacharacters(arg) {
			return fmt.Errorf("argument contains shell metacharacters: %s", arg)
		}
	}
	
	// Validate environment variables
	for key := range config.Env {
		if securityConfig.HasShellMetacharacters(key) {
			return fmt.Errorf("environment variable name contains invalid characters: %s", key)
		}
	}
	
	return nil
}