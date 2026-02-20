// Package mcp provides MCP (Model Context Protocol) client support.
// MCP enables ollama to connect to external tool servers using JSON-RPC over stdio.
package mcp

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
	"time"
)

// Client manages communication with a single MCP server via JSON-RPC over stdio.
type Client struct {
	name    string
	command string
	args    []string
	env     map[string]string

	// Process management
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	stderr *bufio.Reader

	// Pipe handles for clean shutdown
	stdoutPipe io.ReadCloser
	stderrPipe io.ReadCloser

	// State
	mu          sync.RWMutex
	initialized bool
	tools       []ToolInfo
	requestID   int64
	responses   map[int64]chan *jsonRPCResponse

	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	done   chan struct{}
}

// ToolInfo holds information about a tool from an MCP server.
type ToolInfo struct {
	Name        string
	Description string
	InputSchema map[string]any
}

// JSON-RPC 2.0 message types
type jsonRPCRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      *int64 `json:"id,omitempty"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type jsonRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int64          `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jsonRPCError   `json:"error,omitempty"`
}

type jsonRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

// MCP protocol types
type mcpInitializeRequest struct {
	ProtocolVersion string         `json:"protocolVersion"`
	Capabilities    map[string]any `json:"capabilities"`
	ClientInfo      mcpClientInfo  `json:"clientInfo"`
}

type mcpClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type mcpInitializeResponse struct {
	ProtocolVersion string         `json:"protocolVersion"`
	Capabilities    map[string]any `json:"capabilities"`
	ServerInfo      mcpServerInfo  `json:"serverInfo"`
}

type mcpServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type mcpListToolsResponse struct {
	Tools []mcpTool `json:"tools"`
}

type mcpTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"inputSchema"`
}

type mcpCallToolRequest struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type mcpCallToolResponse struct {
	Content []mcpContent `json:"content"`
	IsError bool         `json:"isError,omitempty"`
}

type mcpContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// NewClient creates a new MCP client for the specified server.
func NewClient(name, command string, args []string, env map[string]string) *Client {
	ctx, cancel := context.WithCancel(context.Background())
	return &Client{
		name:      name,
		command:   command,
		args:      args,
		env:       env,
		responses: make(map[int64]chan *jsonRPCResponse),
		ctx:       ctx,
		cancel:    cancel,
		done:      make(chan struct{}),
	}
}

// Name returns the server name.
func (c *Client) Name() string {
	return c.name
}

// Start spawns the MCP server process.
func (c *Client) Start() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.cmd != nil {
		return errors.New("MCP client already started")
	}

	// Handle commands with spaces (like "pnpm dlx")
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

	c.cmd = exec.CommandContext(c.ctx, cmdName, cmdArgs...)
	c.cmd.Env = c.buildEnvironment()

	// Set up pipes
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

	if err := c.cmd.Start(); err != nil {
		c.stdin.Close()
		stdout.Close()
		stderr.Close()
		return fmt.Errorf("failed to start MCP server: %w", err)
	}

	slog.Debug("MCP server started", "name", c.name, "pid", c.cmd.Process.Pid)

	// Start message handlers
	go c.handleResponses()
	go c.handleErrors()

	return nil
}

// Initialize performs the MCP handshake and discovers tools.
func (c *Client) Initialize() error {
	if err := c.Start(); err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(c.ctx, 10*time.Second)
	defer cancel()

	// Send initialize request
	req := mcpInitializeRequest{
		ProtocolVersion: "2024-11-05",
		Capabilities:    map[string]any{"tools": map[string]any{}},
		ClientInfo:      mcpClientInfo{Name: "ollama", Version: "0.1.0"},
	}

	var resp mcpInitializeResponse
	if err := c.callWithContext(ctx, "initialize", req, &resp); err != nil {
		return fmt.Errorf("MCP initialize failed: %w", err)
	}

	// Send initialized notification
	if err := c.notify("notifications/initialized", nil); err != nil {
		return fmt.Errorf("MCP initialized notification failed: %w", err)
	}

	// Discover tools
	var toolsResp mcpListToolsResponse
	if err := c.callWithContext(ctx, "tools/list", struct{}{}, &toolsResp); err != nil {
		return fmt.Errorf("failed to list MCP tools: %w", err)
	}

	// Store tool info
	c.mu.Lock()
	c.initialized = true
	c.tools = make([]ToolInfo, len(toolsResp.Tools))
	for i, t := range toolsResp.Tools {
		c.tools[i] = ToolInfo{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		}
	}
	c.mu.Unlock()

	slog.Debug("MCP client initialized", "name", c.name, "server", resp.ServerInfo.Name, "tools", len(c.tools))
	return nil
}

// Tools returns the discovered tools.
func (c *Client) Tools() []ToolInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()
	tools := make([]ToolInfo, len(c.tools))
	copy(tools, c.tools)
	return tools
}

// CallTool executes a tool on the MCP server.
func (c *Client) CallTool(name string, args map[string]any) (string, error) {
	c.mu.RLock()
	if !c.initialized {
		c.mu.RUnlock()
		return "", errors.New("MCP client not initialized")
	}
	c.mu.RUnlock()

	if args == nil {
		args = make(map[string]any)
	}

	ctx, cancel := context.WithTimeout(c.ctx, 30*time.Second)
	defer cancel()

	req := mcpCallToolRequest{Name: name, Arguments: args}
	var resp mcpCallToolResponse

	if err := c.callWithContext(ctx, "tools/call", req, &resp); err != nil {
		return "", fmt.Errorf("MCP tool call failed: %w", err)
	}

	if resp.IsError {
		slog.Debug("MCP tool error", "name", name, "content_count", len(resp.Content))
		return "", fmt.Errorf("MCP tool returned error")
	}

	// Concatenate text content
	var result strings.Builder
	for _, content := range resp.Content {
		if content.Type == "text" {
			result.WriteString(content.Text)
		}
	}

	return result.String(), nil
}

// Close shuts down the MCP client.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.cmd == nil {
		return nil
	}

	slog.Debug("Shutting down MCP client", "name", c.name)

	c.cancel()

	// Close pipes to unblock goroutines
	if c.stdoutPipe != nil {
		c.stdoutPipe.Close()
	}
	if c.stderrPipe != nil {
		c.stderrPipe.Close()
	}
	if c.stdin != nil {
		c.stdin.Close()
	}

	// Wait for process
	done := make(chan error, 1)
	go func() { done <- c.cmd.Wait() }()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		slog.Warn("Force killing MCP server", "name", c.name)
		c.cmd.Process.Kill()
		<-done
	}

	c.cmd = nil
	c.initialized = false
	close(c.done)

	return nil
}

// callWithContext sends a JSON-RPC request and waits for response.
func (c *Client) callWithContext(ctx context.Context, method string, params, result any) error {
	id := atomic.AddInt64(&c.requestID, 1)

	req := jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      &id,
		Method:  method,
		Params:  params,
	}

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

	if err := c.sendRequest(req); err != nil {
		return err
	}

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

func (c *Client) notify(method string, params any) error {
	req := jsonRPCRequest{JSONRPC: "2.0", Method: method, Params: params}
	return c.sendRequest(req)
}

func (c *Client) sendRequest(req jsonRPCRequest) error {
	if c.stdin == nil {
		return errors.New("client not started")
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

func (c *Client) handleResponses() {
	defer func() {
		if r := recover(); r != nil {
			slog.Error("MCP response handler panic", "name", c.name, "error", r)
		}
	}()

	scanner := bufio.NewScanner(c.stdout)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	for {
		select {
		case <-c.done:
			return
		default:
			if !scanner.Scan() {
				if err := scanner.Err(); err != nil {
					slog.Debug("MCP response reader ended", "name", c.name, "error", err)
				}
				return
			}

			line := scanner.Bytes()
			var resp jsonRPCResponse
			if err := json.Unmarshal(line, &resp); err != nil {
				slog.Debug("Invalid JSON-RPC response", "name", c.name, "error", err)
				continue
			}

			if resp.ID != nil {
				c.mu.RLock()
				if respChan, exists := c.responses[*resp.ID]; exists {
					select {
					case respChan <- &resp:
					default:
					}
				}
				c.mu.RUnlock()
			}
		}
	}
}

func (c *Client) handleErrors() {
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
			line, _, err := c.stderr.ReadLine()
			if err != nil {
				if err != io.EOF {
					slog.Debug("MCP stderr reader ended", "name", c.name, "error", err)
				}
				return
			}
			// Truncate long messages
			msg := string(line)
			if len(msg) > 200 {
				msg = msg[:200] + "..."
			}
			slog.Debug("MCP server stderr", "name", c.name, "message", msg)
		}
	}
}

func (c *Client) buildEnvironment() []string {
	// Start with minimal safe environment
	env := []string{}

	allowed := map[string]bool{
		"PATH": true, "HOME": true, "USER": true,
		"LANG": true, "LC_ALL": true, "TZ": true,
		"TMPDIR": true, "TEMP": true, "TMP": true,
	}

	for _, e := range os.Environ() {
		parts := strings.SplitN(e, "=", 2)
		if len(parts) == 2 && allowed[parts[0]] {
			env = append(env, e)
		}
	}

	// Add custom env vars
	for k, v := range c.env {
		env = append(env, fmt.Sprintf("%s=%s", k, v))
	}

	if !hasEnvVar(env, "PATH") {
		env = append(env, "PATH=/usr/local/bin:/usr/bin:/bin")
	}

	return env
}

func hasEnvVar(env []string, key string) bool {
	prefix := key + "="
	for _, e := range env {
		if strings.HasPrefix(e, prefix) {
			return true
		}
	}
	return false
}
