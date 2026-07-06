package chat

import (
	"context"
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

type chatTestCompactor struct {
	result   coreagent.CompactionResult
	err      error
	progress []int
	request  coreagent.CompactionRequest
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

func (c *chatTestCompactor) MaybeCompact(_ context.Context, req coreagent.CompactionRequest) (coreagent.CompactionResult, error) {
	c.request = req
	for _, tokens := range c.progress {
		if req.Progress != nil {
			req.Progress(coreagent.CompactionProgress{Tokens: tokens})
		}
	}
	return c.result, c.err
}

func nextChatMsg(t *testing.T, ch <-chan tea.Msg) tea.Msg {
	t.Helper()
	select {
	case msg, ok := <-ch:
		if !ok {
			t.Fatal("message channel closed")
		}
		return msg
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for chat message")
		return nil
	}
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
