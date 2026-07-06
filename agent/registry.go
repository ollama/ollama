package agent

import (
	"context"
	"fmt"
	"sort"

	"github.com/ollama/ollama/api"
)

type ToolContext struct {
	WorkingDir string
}

type ToolResult struct {
	Content    string
	WorkingDir string
}

type Tool interface {
	Name() string
	Description() string
	Schema() api.ToolFunction
	Execute(context.Context, ToolContext, map[string]any) (ToolResult, error)
}

type ApprovalRequired interface {
	RequiresApproval(map[string]any) bool
}

type Registry struct {
	tools map[string]Tool
}

func (r *Registry) Register(tool Tool) {
	if r == nil || tool == nil {
		return
	}
	if r.tools == nil {
		r.tools = make(map[string]Tool)
	}
	r.tools[tool.Name()] = tool
}

func (r *Registry) Get(name string) (Tool, bool) {
	if r == nil {
		return nil, false
	}
	tool, ok := r.tools[name]
	return tool, ok
}

func (r *Registry) Names() []string {
	if r == nil {
		return nil
	}
	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (r *Registry) Tools() api.Tools {
	names := r.Names()
	apiTools := make(api.Tools, 0, len(names))
	for _, name := range names {
		tool := r.tools[name]
		apiTools = append(apiTools, api.Tool{
			Type:     "function",
			Function: tool.Schema(),
		})
	}
	return apiTools
}

func (r *Registry) Execute(ctx context.Context, toolCtx ToolContext, call api.ToolCall) (ToolResult, error) {
	tool, ok := r.Get(call.Function.Name)
	if !ok {
		return ToolResult{}, fmt.Errorf("unknown tool: %s", call.Function.Name)
	}
	return tool.Execute(ctx, toolCtx, call.Function.Arguments.ToMap())
}

func ToolRequiresApproval(tool Tool, args map[string]any) bool {
	if tool == nil {
		return false
	}
	if t, ok := tool.(ApprovalRequired); ok {
		return t.RequiresApproval(args)
	}
	return false
}
