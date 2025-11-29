package main

import (
	"context"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// Test is a single evaluation test
type Test struct {
	Name    string
	Prompt  string
	System  string
	Tools   []api.Tool
	Think   bool
	Options map[string]any
	Check   func(response string, tools []api.ToolCall) bool
}

// Suite is a collection of tests
type Suite struct {
	Name  string
	Tests []Test
}

// Result holds test execution results
type Result struct {
	Name      string
	Passed    bool
	Error     error
	Duration  time.Duration
	Response  string
	Tools     []string
	ToolCalls []api.ToolCall
	Thinking  bool
}

// Run executes a test against a model
func Run(ctx context.Context, client *api.Client, model string, test Test) Result {
	result := Result{Name: test.Name}

	req := &api.ChatRequest{
		Model: model,
		Messages: []api.Message{
			{Role: "user", Content: test.Prompt},
		},
		Options: test.Options,
	}

	if test.System != "" {
		req.Messages = append([]api.Message{
			{Role: "system", Content: test.System},
		}, req.Messages...)
	}

	if len(test.Tools) > 0 {
		req.Tools = test.Tools
	}

	if test.Think {
		req.Think = &api.ThinkValue{Value: true}
	}

	var resp strings.Builder
	var toolCalls []api.ToolCall

	start := time.Now()
	err := client.Chat(ctx, req, func(r api.ChatResponse) error {
		resp.WriteString(r.Message.Content)
		if r.Message.Thinking != "" {
			result.Thinking = true
		}
		toolCalls = append(toolCalls, r.Message.ToolCalls...)
		return nil
	})
	result.Duration = time.Since(start)

	if err != nil {
		result.Error = err
		return result
	}

	result.Response = resp.String()
	result.Tools = uniqueToolNames(toolCalls)
	result.ToolCalls = toolCalls
	result.Passed = test.Check(result.Response, toolCalls)

	return result
}

func uniqueToolNames(calls []api.ToolCall) []string {
	seen := make(map[string]bool)
	var names []string
	for _, c := range calls {
		if !seen[c.Function.Name] {
			seen[c.Function.Name] = true
			names = append(names, c.Function.Name)
		}
	}
	return names
}

// Check functions for common test patterns

func HasResponse() func(string, []api.ToolCall) bool {
	return func(resp string, _ []api.ToolCall) bool {
		return strings.TrimSpace(resp) != ""
	}
}

func Contains(s string) func(string, []api.ToolCall) bool {
	return func(resp string, _ []api.ToolCall) bool {
		return strings.Contains(strings.ToLower(resp), strings.ToLower(s))
	}
}

func CallsTool(name string) func(string, []api.ToolCall) bool {
	return func(_ string, tools []api.ToolCall) bool {
		for _, t := range tools {
			if t.Function.Name == name {
				return true
			}
		}
		return false
	}
}

func NoTools() func(string, []api.ToolCall) bool {
	return func(_ string, tools []api.ToolCall) bool {
		return len(tools) == 0
	}
}

func MinTools(n int) func(string, []api.ToolCall) bool {
	return func(_ string, tools []api.ToolCall) bool {
		return len(tools) >= n
	}
}

func All(checks ...func(string, []api.ToolCall) bool) func(string, []api.ToolCall) bool {
	return func(resp string, tools []api.ToolCall) bool {
		for _, check := range checks {
			if !check(resp, tools) {
				return false
			}
		}
		return true
	}
}
