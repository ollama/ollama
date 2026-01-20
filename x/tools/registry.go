// Package tools provides built-in tool implementations for the agent loop.
package tools

import (
	"fmt"
	"os"
	"sort"

	"github.com/ollama/ollama/api"
)

// Tool defines the interface for agent tools.
type Tool interface {
	// Name returns the tool's unique identifier.
	Name() string
	// Description returns a human-readable description of what the tool does.
	Description() string
	// Schema returns the tool's parameter schema for the LLM.
	Schema() api.ToolFunction
	// Execute runs the tool with the given arguments.
	Execute(args map[string]any) (string, error)
}

// Registry manages available tools.
type Registry struct {
	tools map[string]Tool
}

// NewRegistry creates a new tool registry.
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]Tool),
	}
}

// Register adds a tool to the registry.
func (r *Registry) Register(tool Tool) {
	r.tools[tool.Name()] = tool
}

// Unregister removes a tool from the registry by name.
func (r *Registry) Unregister(name string) {
	delete(r.tools, name)
}

// Has checks if a tool with the given name is registered.
func (r *Registry) Has(name string) bool {
	_, ok := r.tools[name]
	return ok
}

// RegisterBash adds the bash tool to the registry.
func (r *Registry) RegisterBash() {
	r.Register(&BashTool{})
}

// RegisterWebSearch adds the web search tool to the registry.
func (r *Registry) RegisterWebSearch() {
	r.Register(&WebSearchTool{})
}

// RegisterWebFetch adds the web fetch tool to the registry.
func (r *Registry) RegisterWebFetch() {
	r.Register(&WebFetchTool{})
}

// Get retrieves a tool by name.
func (r *Registry) Get(name string) (Tool, bool) {
	tool, ok := r.tools[name]
	return tool, ok
}

// Tools returns all registered tools in Ollama API format, sorted by name.
func (r *Registry) Tools() api.Tools {
	// Get sorted names for deterministic ordering
	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	sort.Strings(names)

	var tools api.Tools
	for _, name := range names {
		tool := r.tools[name]
		tools = append(tools, api.Tool{
			Type:     "function",
			Function: tool.Schema(),
		})
	}
	return tools
}

// Execute runs a tool call and returns the result.
func (r *Registry) Execute(call api.ToolCall) (string, error) {
	tool, ok := r.tools[call.Function.Name]
	if !ok {
		return "", fmt.Errorf("unknown tool: %s", call.Function.Name)
	}
	return tool.Execute(call.Function.Arguments.ToMap())
}

// Names returns the names of all registered tools, sorted alphabetically.
func (r *Registry) Names() []string {
	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Count returns the number of registered tools.
func (r *Registry) Count() int {
	return len(r.tools)
}

// DefaultRegistry creates a registry with all built-in tools.
// Tools can be disabled via environment variables:
// - OLLAMA_AGENT_DISABLE_WEBSEARCH=1 disables web_search
// - OLLAMA_AGENT_DISABLE_BASH=1 disables bash
func DefaultRegistry() *Registry {
	r := NewRegistry()
	// TODO(parthsareen): re-enable web search once it's ready for release
	// if os.Getenv("OLLAMA_AGENT_DISABLE_WEBSEARCH") == "" {
	// 	r.Register(&WebSearchTool{})
	// }
	if os.Getenv("OLLAMA_AGENT_DISABLE_BASH") == "" {
		r.Register(&BashTool{})
	}
	return r
}
