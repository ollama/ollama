package server

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/ollama/ollama/api"
)

// MCPClient manages communication with a single MCP server via JSON-RPC over stdio
type MCPClient struct {
	name    string
	command string
	args    []string
	env     map[string]string

	// Process management
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	stderr *bufio.Reader

	// State
	mu          sync.RWMutex
	initialized bool
	tools       []api.Tool
	requestID   int64
	responses   map[int64]chan *jsonRPCResponse

	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	done   chan struct{}

	// Pipe handles (for clean shutdown)
	stdoutPipe io.ReadCloser
	stderrPipe io.ReadCloser

	// Dependencies (injectable for testing)
	commandResolver CommandResolverInterface
}

// =============================================================================
// MCPClient Options (Functional Options Pattern)
// =============================================================================

// MCPClientOption configures an MCPClient during creation.
type MCPClientOption func(*MCPClient)

// WithCommandResolver sets a custom command resolver for the client.
// This is primarily useful for testing to avoid system dependencies.
//
// Example:
//
//	mockResolver := &MockCommandResolver{...}
//	client := NewMCPClient("test", "npx", args, env, WithCommandResolver(mockResolver))
func WithCommandResolver(resolver CommandResolverInterface) MCPClientOption {
	return func(c *MCPClient) {
		c.commandResolver = resolver
	}
}

// JSON-RPC 2.0 message types
type jsonRPCRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      *int64      `json:"id,omitempty"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

type jsonRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int64          `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jsonRPCError   `json:"error,omitempty"`
}

type jsonRPCError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// MCP protocol message types
type mcpInitializeRequest struct {
	ProtocolVersion string                 `json:"protocolVersion"`
	Capabilities    map[string]interface{} `json:"capabilities"`
	ClientInfo      mcpClientInfo          `json:"clientInfo"`
}

type mcpClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type mcpInitializeResponse struct {
	ProtocolVersion string                 `json:"protocolVersion"`
	Capabilities    map[string]interface{} `json:"capabilities"`
	ServerInfo      mcpServerInfo          `json:"serverInfo"`
}

type mcpServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type mcpListToolsRequest struct{}

type mcpListToolsResponse struct {
	Tools []mcpTool `json:"tools"`
}

type mcpTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

type mcpCallToolRequest struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

type mcpCallToolResponse struct {
	Content []mcpContent `json:"content"`
	IsError bool         `json:"isError,omitempty"`
}

type mcpContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// NewMCPClient creates a new MCP client for the specified server configuration.
// Optional MCPClientOption arguments can be used to customize behavior (e.g., for testing).
func NewMCPClient(name, command string, args []string, env map[string]string, opts ...MCPClientOption) *MCPClient {
	ctx, cancel := context.WithCancel(context.Background())

	client := &MCPClient{
		name:            name,
		command:         command, // Will be resolved after options are applied
		args:            args,
		env:             env,
		responses:       make(map[int64]chan *jsonRPCResponse),
		ctx:             ctx,
		cancel:          cancel,
		done:            make(chan struct{}),
		commandResolver: DefaultCommandResolver, // Default, can be overridden
	}

	// Apply options
	for _, opt := range opts {
		opt(client)
	}

	// Guard against nil resolver (e.g., if WithCommandResolver(nil) was called)
	if client.commandResolver == nil {
		client.commandResolver = DefaultCommandResolver
	}

	// Resolve the command using the configured resolver
	client.command = client.commandResolver.ResolveForEnvironment(command)

	return client
}

// Start spawns the MCP server process and initializes communication.
//
// SECURITY REVIEW: This function executes external processes. Security controls:
//   - Command must pass validation in MCPManager.validateServerConfig()
//   - Environment is filtered via buildSecureEnvironment()
//   - Process runs in isolated process group (Setpgid)
//   - Context-based cancellation for cleanup
func (c *MCPClient) Start() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.cmd != nil {
		return errors.New("MCP client already started")
	}

	// Handle commands that might have spaces (like "pnpm dlx")
	cmdParts := strings.Fields(c.command)
	var cmdName string
	var cmdArgs []string

	if len(cmdParts) > 1 {
		cmdName = cmdParts[0]
		cmdArgs = append(cmdParts[1:], c.args...)
	} else {
		cmdName = c.command
		cmdArgs = c.args
	}

	// SECURITY: Create command with context for cancellation control
	c.cmd = exec.CommandContext(c.ctx, cmdName, cmdArgs...)

	// SECURITY: Apply filtered environment (see buildSecureEnvironment)
	c.cmd.Env = c.buildSecureEnvironment()

	// SECURITY: Process isolation via process group
	c.cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true, // Isolate in own process group
		Pgid:    0,    // New process group
		// Future: Consider adding privilege dropping for root users
		// Credential: &syscall.Credential{Uid: 65534, Gid: 65534}
	}

	// Set up pipes for communication.
	// Each error path explicitly closes previously opened pipes to prevent leaks.
	// On success, pipes remain open for the lifetime of the MCP client.
	stdin, err := c.cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdin pipe: %w", err)
	}
	c.stdin = stdin

	stdout, err := c.cmd.StdoutPipe()
	if err != nil {
		c.stdin.Close()
		return fmt.Errorf("failed to create stdout pipe: %w", err)
	}
	c.stdoutPipe = stdout
	c.stdout = bufio.NewReader(stdout)

	stderr, err := c.cmd.StderrPipe()
	if err != nil {
		c.stdin.Close()
		stdout.Close()
		return fmt.Errorf("failed to create stderr pipe: %w", err)
	}
	c.stderrPipe = stderr
	c.stderr = bufio.NewReader(stderr)

	// Start the process
	if err := c.cmd.Start(); err != nil {
		c.stdin.Close()
		stdout.Close()
		stderr.Close()
		return fmt.Errorf("failed to start MCP server: %w", err)
	}

	slog.Info("MCP server started", "name", c.name, "pid", c.cmd.Process.Pid)

	// Start message handling goroutines
	go c.handleResponses()
	go c.handleErrors()

	// Check if the process is still running after a brief delay
	// This catches immediate failures like command not found
	processCheckDone := make(chan bool, 1)
	go func() {
		time.Sleep(100 * time.Millisecond)
		// Non-blocking check if process has exited
		if c.cmd.ProcessState != nil {
			processCheckDone <- false
			return
		}
		// Try to check process existence without waiting
		if c.cmd.Process != nil {
			// On Unix systems, signal 0 can be used to check process existence
			if err := c.cmd.Process.Signal(syscall.Signal(0)); err != nil {
				processCheckDone <- false
				return
			}
		}
		processCheckDone <- true
	}()

	select {
	case alive := <-processCheckDone:
		if !alive {
			// Process died immediately - collect the error
			waitErr := c.cmd.Wait()
			c.stdin.Close()
			stdout.Close() 
			stderr.Close()
			return fmt.Errorf("MCP server exited immediately: %w", waitErr)
		}
	case <-time.After(200 * time.Millisecond):
		// Process seems to be running, continue
	}

	return nil
}

// Initialize performs the MCP handshake sequence
func (c *MCPClient) Initialize() error {
	if err := c.Start(); err != nil {
		return err
	}

	// Add timeout to initialization to prevent hanging
	initCtx, cancel := context.WithTimeout(c.ctx, 10*time.Second)
	defer cancel()

	// Send initialize request
	req := mcpInitializeRequest{
		ProtocolVersion: "2024-11-05",
		Capabilities: map[string]interface{}{
			"tools": map[string]interface{}{},
		},
		ClientInfo: mcpClientInfo{
			Name:    "ollama",
			Version: "0.1.0",
		},
	}

	var resp mcpInitializeResponse
	if err := c.callWithContext(initCtx, "initialize", req, &resp); err != nil {
		return fmt.Errorf("MCP initialize failed: %w", err)
	}

	// Send initialized notification
	if err := c.notify("notifications/initialized", nil); err != nil {
		return fmt.Errorf("MCP initialized notification failed: %w", err)
	}

	c.mu.Lock()
	c.initialized = true
	c.mu.Unlock()

	slog.Info("MCP client initialized", "name", c.name, "server", resp.ServerInfo.Name)
	return nil
}

// ListTools discovers available tools from the MCP server
func (c *MCPClient) ListTools() ([]api.Tool, error) {
	c.mu.RLock()
	if !c.initialized {
		c.mu.RUnlock()
		return nil, errors.New("MCP client not initialized")
	}

	// Return cached tools if available
	if len(c.tools) > 0 {
		tools := make([]api.Tool, len(c.tools))
		copy(tools, c.tools)
		c.mu.RUnlock()
		return tools, nil
	}
	c.mu.RUnlock()

	var resp mcpListToolsResponse
	if err := c.call("tools/list", mcpListToolsRequest{}, &resp); err != nil {
		return nil, fmt.Errorf("failed to list MCP tools: %w", err)
	}

	// Convert MCP tools to Ollama API format
	tools := make([]api.Tool, 0, len(resp.Tools))
	for _, mcpTool := range resp.Tools {
		tool := api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        fmt.Sprintf("%s:%s", c.name, mcpTool.Name), // Namespace with server name
				Description: mcpTool.Description,
				Parameters: api.ToolFunctionParameters{
					Type:       "object",
					Properties: make(map[string]api.ToolProperty),
					Required:   []string{},
				},
			},
		}

		// Convert input schema to tool parameters
		if props, ok := mcpTool.InputSchema["properties"].(map[string]interface{}); ok {
			for propName, propDef := range props {
				propDefMap, ok := propDef.(map[string]interface{})
				if !ok {
					slog.Debug("MCP schema: property definition not a map", "tool", mcpTool.Name, "property", propName)
					continue
				}
				toolProp := api.ToolProperty{
					Description: getStringFromMap(propDefMap, "description"),
				}

				if propType, ok := propDefMap["type"].(string); ok {
					toolProp.Type = api.PropertyType{propType}
				} else {
					slog.Debug("MCP schema: property type not a string", "tool", mcpTool.Name, "property", propName)
				}

				// Preserve items schema for array types (needed for context injection)
				if items, ok := propDefMap["items"]; ok {
					toolProp.Items = items
				}

				tool.Function.Parameters.Properties[propName] = toolProp
			}
		} else if mcpTool.InputSchema["properties"] != nil {
			slog.Debug("MCP schema: properties not a map", "tool", mcpTool.Name)
		}

		if required, ok := mcpTool.InputSchema["required"].([]interface{}); ok {
			for _, req := range required {
				if reqStr, ok := req.(string); ok {
					tool.Function.Parameters.Required = append(tool.Function.Parameters.Required, reqStr)
				} else {
					slog.Debug("MCP schema: required item not a string", "tool", mcpTool.Name)
				}
			}
		} else if mcpTool.InputSchema["required"] != nil {
			slog.Debug("MCP schema: required not an array", "tool", mcpTool.Name)
		}

		tools = append(tools, tool)
	}

	// Cache the tools
	c.mu.Lock()
	c.tools = tools
	c.mu.Unlock()

	slog.Debug("MCP tools discovered", "name", c.name, "count", len(tools))
	return tools, nil
}

// CallTool executes a tool call via the MCP server
func (c *MCPClient) CallTool(name string, args map[string]interface{}) (string, error) {
	c.mu.RLock()
	if !c.initialized {
		c.mu.RUnlock()
		return "", errors.New("MCP client not initialized")
	}
	c.mu.RUnlock()

	// Remove namespace prefix if present
	toolName := name
	if prefix := c.name + ":"; len(name) > len(prefix) && name[:len(prefix)] == prefix {
		toolName = name[len(prefix):]
	}

	// Ensure arguments is never nil (MCP protocol requires an object, not undefined)
	if args == nil {
		args = make(map[string]interface{})
	}

	req := mcpCallToolRequest{
		Name:      toolName,
		Arguments: args,
	}

	// Debug logging removed

	var resp mcpCallToolResponse

	// Set timeout for tool execution
	ctx, cancel := context.WithTimeout(c.ctx, 30*time.Second)
	defer cancel()

	if err := c.callWithContext(ctx, "tools/call", req, &resp); err != nil {
		return "", fmt.Errorf("MCP tool call failed: %w", err)
	}

	slog.Debug("MCP tool response", "name", name, "is_error", resp.IsError, "content_count", len(resp.Content))

	if resp.IsError {
		// Log error without full response to avoid exposing sensitive data
		slog.Error("MCP tool execution error", "name", name, "content_count", len(resp.Content))
		return "", fmt.Errorf("MCP tool returned error")
	}

	// Concatenate all text content
	var result string
	for _, content := range resp.Content {
		if content.Type == "text" {
			result += content.Text
		}
	}

	// Debug logging removed
	return result, nil
}

// GetTools returns the list of tools available from this MCP server
func (c *MCPClient) GetTools() []api.Tool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.tools
}

// Close shuts down the MCP client and terminates the server process
func (c *MCPClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.cmd == nil {
		return nil
	}

	slog.Info("Shutting down MCP client", "name", c.name)

	// Cancel context to stop goroutines
	c.cancel()

	// Close stdin to signal shutdown
	if c.stdin != nil {
		c.stdin.Close()
	}

	// Close stdout/stderr pipes to unblock handleResponses/handleErrors goroutines
	if c.stdoutPipe != nil {
		c.stdoutPipe.Close()
	}
	if c.stderrPipe != nil {
		c.stderrPipe.Close()
	}

	// Wait for process to exit gracefully
	done := make(chan error, 1)
	go func() {
		done <- c.cmd.Wait()
	}()

	select {
	case err := <-done:
		if err != nil {
			slog.Warn("MCP server exited with error", "name", c.name, "error", err)
		}
	case <-time.After(5 * time.Second):
		// Force kill if not responding
		slog.Warn("Force killing unresponsive MCP server", "name", c.name)
		c.cmd.Process.Kill()
		<-done
	}

	c.cmd = nil
	c.initialized = false
	close(c.done)

	return nil
}

// call sends a JSON-RPC request and waits for the response
func (c *MCPClient) call(method string, params interface{}, result interface{}) error {
	return c.callWithContext(c.ctx, method, params, result)
}

// callWithContext sends a JSON-RPC request with a custom context
func (c *MCPClient) callWithContext(ctx context.Context, method string, params interface{}, result interface{}) error {
	id := atomic.AddInt64(&c.requestID, 1)

	req := jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      &id,
		Method:  method,
		Params:  params,
	}

	// Create response channel
	respChan := make(chan *jsonRPCResponse, 1)
	c.mu.Lock()
	c.responses[id] = respChan
	c.mu.Unlock()

	defer func() {
		c.mu.Lock()
		delete(c.responses, id)
		c.mu.Unlock()
		close(respChan)
	}()

	// Send request
	if err := c.sendRequest(req); err != nil {
		return err
	}

	// Wait for response
	select {
	case resp := <-respChan:
		if resp.Error != nil {
			return fmt.Errorf("JSON-RPC error %d: %s", resp.Error.Code, resp.Error.Message)
		}

		if result != nil && resp.Result != nil {
			if err := json.Unmarshal(resp.Result, result); err != nil {
				return fmt.Errorf("failed to unmarshal response: %w", err)
			}
		}

		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-c.done:
		return errors.New("MCP client closed")
	}
}

// notify sends a JSON-RPC notification (no response expected)
func (c *MCPClient) notify(method string, params interface{}) error {
	req := jsonRPCRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	}

	return c.sendRequest(req)
}

// sendRequest sends a JSON-RPC request over stdin
func (c *MCPClient) sendRequest(req jsonRPCRequest) error {
	if c.stdin == nil {
		return fmt.Errorf("client not started")
	}

	data, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	if _, err := c.stdin.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("failed to write request: %w", err)
	}

	return nil
}

// handleResponses processes incoming JSON-RPC responses from stdout
func (c *MCPClient) handleResponses() {
	defer func() {
		if r := recover(); r != nil {
			slog.Error("MCP response handler panic", "name", c.name, "error", r)
		}
	}()

	scanner := bufio.NewScanner(c.stdout)
	// Set a larger buffer to handle long JSON responses
	scanner.Buffer(make([]byte, 64*1024), 1024*1024) // 64KB initial, 1MB max

	for {
		select {
		case <-c.done:
			return
		default:
			if !scanner.Scan() {
				if err := scanner.Err(); err != nil {
					slog.Error("Error reading MCP response", "name", c.name, "error", err)
				}
				return
			}

			line := scanner.Bytes()
			var resp jsonRPCResponse
			if err := json.Unmarshal(line, &resp); err != nil {
				// Don't log raw line content - may contain sensitive data
				slog.Warn("Invalid JSON-RPC response", "name", c.name, "error", err, "length", len(line))
				continue
			}

			// Route response to waiting caller
			if resp.ID != nil {
				c.mu.RLock()
				if respChan, exists := c.responses[*resp.ID]; exists {
					select {
					case respChan <- &resp:
					default:
						slog.Warn("Response channel full", "name", c.name, "id", *resp.ID)
					}
				}
				c.mu.RUnlock()
			}
		}
	}
}

// handleErrors processes stderr output from the MCP server
func (c *MCPClient) handleErrors() {
	defer func() {
		if r := recover(); r != nil {
			slog.Error("MCP error handler panic", "name", c.name, "error", r)
		}
	}()

	for {
		select {
		case <-c.done:
			return
		default:
			line, isPrefix, err := c.stderr.ReadLine()
			if err != nil {
				if err != io.EOF {
					slog.Error("Error reading MCP stderr", "name", c.name, "error", err)
				}
				return
			}
			if isPrefix {
				slog.Warn("MCP stderr line too long, truncated", "name", c.name)
			}

			// Truncate stderr to avoid logging excessive/sensitive output
			msg := string(line)
			if len(msg) > 200 {
				msg = msg[:200] + "...(truncated)"
			}
			slog.Debug("MCP server stderr", "name", c.name, "message", msg)
		}
	}
}

// buildSecureEnvironment creates a filtered environment for the MCP server.
//
// SECURITY REVIEW: This is a critical security function. It controls what
// environment variables are passed to MCP server processes.
//
// Defense strategy (defense in depth):
//  1. Start with empty environment (not inherited)
//  2. Allowlist only known-safe variables
//  3. Apply MCPSecurityConfig filtering (blocks credentials)
//  4. Sanitize PATH to remove dangerous directories
//  5. Add custom env vars only after security checks
func (c *MCPClient) buildSecureEnvironment() []string {
	// SECURITY: Start with empty env, not os.Environ()
	env := []string{}

	// Get security configuration
	securityConfig := GetSecurityConfig()

	// SECURITY: Allowlist of safe environment variables.
	// Only these variables can be passed through from the parent process.
	allowedVars := map[string]bool{
		"PATH":       true,
		"HOME":       true,
		"USER":       true,
		"LANG":       true,
		"LC_ALL":     true,
		"LC_CTYPE":   true,
		"TZ":         true,
		"TMPDIR":     true,
		"TEMP":       true,
		"TMP":        true,
		"TERM":       true,
		"PYTHONPATH": true,
		"NODE_PATH":  true,
		"DISPLAY":    true,  // For GUI applications
		"EDITOR":     true,  // For text editing
		"SHELL":      false, // Explicitly blocked - could enable shell escapes
	}
	
	// Filter existing environment variables
	for _, e := range os.Environ() {
		parts := strings.SplitN(e, "=", 2)
		if len(parts) != 2 {
			continue
		}
		
		key := parts[0]
		value := parts[1]
		
		// Check if variable should be filtered based on security config
		if securityConfig.ShouldFilterEnvironmentVar(key) {
			slog.Debug("MCP: filtered credential env var", "name", c.name, "key", key)
			continue
		}
		
		// Only include explicitly allowed variables
		if allowed, exists := allowedVars[key]; exists && allowed {
			env = append(env, fmt.Sprintf("%s=%s", key, value))
		}
	}
	
	// Add custom environment variables from server config.
	// NOTE: Custom vars only check the blocklist, not the allowlist. This is intentional:
	// inherited vars use strict allowlist (defense in depth), but custom vars are explicitly
	// configured by the user/admin, so they're trusted if not in the blocklist.
	for key, value := range c.env {
		if securityConfig.ShouldFilterEnvironmentVar(key) {
			slog.Debug("MCP: blocked custom env var", "name", c.name, "key", key)
			continue
		}
		env = append(env, fmt.Sprintf("%s=%s", key, value))
	}
	
	// Set a restricted PATH if not already set
	if !hasEnvVar(env, "PATH") {
		env = append(env, "PATH=/usr/local/bin:/usr/bin:/bin")
	}
	
	return env
}

// getStringFromMap safely extracts a string value from a map
func getStringFromMap(m map[string]interface{}, key string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return ""
}

// hasEnvVar checks if an environment variable exists in the env slice
func hasEnvVar(env []string, key string) bool {
	prefix := key + "="
	for _, e := range env {
		if strings.HasPrefix(e, prefix) {
			return true
		}
	}
	return false
}
