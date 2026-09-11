package chat

import (
	"context"
	"fmt"
	"regexp"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

type chatTestTool struct{}

type chatTestClient struct{}

type chatCaptureClient struct {
	requests []*api.ChatRequest
}

type chatToolLoopClient struct {
	calls      int
	toolRounds int
}

func (chatTestClient) Chat(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn(api.ChatResponse{
		Message: api.Message{Role: "assistant", Content: "ok"},
		Done:    true,
	})
}

func (c *chatCaptureClient) Chat(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	c.requests = append(c.requests, req)
	return fn(api.ChatResponse{
		Message: api.Message{Role: "assistant", Content: "ok"},
		Done:    true,
	})
}

func (c *chatToolLoopClient) Chat(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	c.calls++
	if c.calls > c.toolRounds {
		return fn(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "done"}, Done: true})
	}

	args := api.NewToolCallFunctionArguments()
	args.Set("value", "keep going")
	return fn(api.ChatResponse{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
		ID: fmt.Sprintf("call-%d", c.calls),
		Function: api.ToolCallFunction{
			Name:      "fake_tool",
			Arguments: args,
		},
	}}}})
}

func (chatTestTool) Name() string {
	return "fake_tool"
}

func (chatTestTool) Description() string {
	return "does test work"
}

func (chatTestTool) Schema() api.ToolFunction {
	return api.ToolFunction{
		Name:        "fake_tool",
		Description: "does test work",
		Parameters: api.ToolFunctionParameters{
			Type: "object",
		},
	}
}

func (chatTestTool) Execute(context.Context, coreagent.ToolContext, map[string]any) (coreagent.ToolResult, error) {
	return coreagent.ToolResult{Content: "ok"}, nil
}

func waitForRunDone(t *testing.T, events <-chan tea.Msg) chatRunDoneMsg {
	t.Helper()
	timeout := time.After(2 * time.Second)
	for {
		select {
		case msg, ok := <-events:
			if !ok {
				t.Fatal("events closed before run done")
			}
			if done, ok := msg.(chatRunDoneMsg); ok {
				return done
			}
		case <-timeout:
			t.Fatal("timed out waiting for run done")
		}
	}
}

func stripANSI(s string) string {
	re := regexp.MustCompile(`\x1b\[[0-9;:]*[A-Za-z]`)
	return re.ReplaceAllString(s, "")
}
