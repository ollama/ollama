package tools

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestRegistry_Register(t *testing.T) {
	r := NewRegistry()

	r.Register(&BashTool{})
	r.Register(&WebSearchTool{})

	if r.Count() != 2 {
		t.Errorf("expected 2 tools, got %d", r.Count())
	}

	names := r.Names()
	if len(names) != 2 {
		t.Errorf("expected 2 names, got %d", len(names))
	}
}

func TestRegistry_Get(t *testing.T) {
	r := NewRegistry()
	r.Register(&BashTool{})

	tool, ok := r.Get("bash")
	if !ok {
		t.Fatal("expected to find bash tool")
	}

	if tool.Name() != "bash" {
		t.Errorf("expected name 'bash', got '%s'", tool.Name())
	}

	_, ok = r.Get("nonexistent")
	if ok {
		t.Error("expected not to find nonexistent tool")
	}
}

func TestRegistry_Tools(t *testing.T) {
	r := NewRegistry()
	r.Register(&BashTool{})
	r.Register(&WebSearchTool{})

	tools := r.Tools()
	if len(tools) != 2 {
		t.Errorf("expected 2 tools, got %d", len(tools))
	}

	for _, tool := range tools {
		if tool.Type != "function" {
			t.Errorf("expected type 'function', got '%s'", tool.Type)
		}
	}
}

func TestRegistry_Execute(t *testing.T) {
	r := NewRegistry()
	r.Register(&BashTool{})

	// Test successful execution
	args := api.NewToolCallFunctionArguments()
	args.Set("command", "echo hello")
	result, err := r.Execute(api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      "bash",
			Arguments: args,
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "hello\n" {
		t.Errorf("expected 'hello\\n', got '%s'", result)
	}

	// Test unknown tool
	_, err = r.Execute(api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      "unknown",
			Arguments: api.NewToolCallFunctionArguments(),
		},
	})
	if err == nil {
		t.Error("expected error for unknown tool")
	}
}

func TestDefaultRegistry(t *testing.T) {
	r := DefaultRegistry()

	if r.Count() != 1 {
		t.Errorf("expected 1 tool in default registry, got %d", r.Count())
	}

	_, ok := r.Get("bash")
	if !ok {
		t.Error("expected bash tool in default registry")
	}
}

func TestDefaultRegistry_DisableWebsearch(t *testing.T) {
	t.Setenv("OLLAMA_AGENT_DISABLE_WEBSEARCH", "1")

	r := DefaultRegistry()

	if r.Count() != 1 {
		t.Errorf("expected 1 tool with websearch disabled, got %d", r.Count())
	}

	_, ok := r.Get("bash")
	if !ok {
		t.Error("expected bash tool in registry")
	}

	_, ok = r.Get("web_search")
	if ok {
		t.Error("expected web_search to be disabled")
	}
}

func TestDefaultRegistry_DisableBash(t *testing.T) {
	t.Setenv("OLLAMA_AGENT_DISABLE_BASH", "1")

	r := DefaultRegistry()

	if r.Count() != 0 {
		t.Errorf("expected 0 tools with bash disabled, got %d", r.Count())
	}
}

func TestDefaultRegistry_DisableBoth(t *testing.T) {
	t.Setenv("OLLAMA_AGENT_DISABLE_WEBSEARCH", "1")
	t.Setenv("OLLAMA_AGENT_DISABLE_BASH", "1")

	r := DefaultRegistry()

	if r.Count() != 0 {
		t.Errorf("expected 0 tools with both disabled, got %d", r.Count())
	}
}

func TestBashTool_Schema(t *testing.T) {
	tool := &BashTool{}

	schema := tool.Schema()
	if schema.Name != "bash" {
		t.Errorf("expected name 'bash', got '%s'", schema.Name)
	}

	if schema.Parameters.Type != "object" {
		t.Errorf("expected parameters type 'object', got '%s'", schema.Parameters.Type)
	}

	if _, ok := schema.Parameters.Properties.Get("command"); !ok {
		t.Error("expected 'command' property in schema")
	}
}

func TestWebSearchTool_Schema(t *testing.T) {
	tool := &WebSearchTool{}

	schema := tool.Schema()
	if schema.Name != "web_search" {
		t.Errorf("expected name 'web_search', got '%s'", schema.Name)
	}

	if schema.Parameters.Type != "object" {
		t.Errorf("expected parameters type 'object', got '%s'", schema.Parameters.Type)
	}

	if _, ok := schema.Parameters.Properties.Get("query"); !ok {
		t.Error("expected 'query' property in schema")
	}
}

func TestRegistry_Unregister(t *testing.T) {
	r := NewRegistry()
	r.Register(&BashTool{})

	if r.Count() != 1 {
		t.Errorf("expected 1 tool, got %d", r.Count())
	}

	r.Unregister("bash")

	if r.Count() != 0 {
		t.Errorf("expected 0 tools after unregister, got %d", r.Count())
	}

	_, ok := r.Get("bash")
	if ok {
		t.Error("expected bash tool to be removed")
	}
}

func TestRegistry_Has(t *testing.T) {
	r := NewRegistry()

	if r.Has("bash") {
		t.Error("expected Has to return false for unregistered tool")
	}

	r.Register(&BashTool{})

	if !r.Has("bash") {
		t.Error("expected Has to return true for registered tool")
	}
}

func TestRegistry_RegisterBash(t *testing.T) {
	r := NewRegistry()

	r.RegisterBash()

	if !r.Has("bash") {
		t.Error("expected bash tool to be registered")
	}
}
