package agent

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
)

type fakeClient struct {
	calls     int
	responses [][]api.ChatResponse
	requests  []*api.ChatRequest
	err       error
}

func (c *fakeClient) Chat(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
	c.requests = append(c.requests, req)
	if c.calls >= len(c.responses) {
		return nil
	}
	responses := c.responses[c.calls]
	c.calls++
	for _, response := range responses {
		if err := fn(response); err != nil {
			return err
		}
	}
	return c.err
}

type memoryStore struct {
	messages    []api.Message
	appendCalls int
	updateCalls int
}

func (s *memoryStore) EnsureChat(context.Context, string, string) error {
	return nil
}

func (s *memoryStore) AppendMessage(_ context.Context, _ string, msg api.Message, _ string) error {
	s.appendCalls++
	s.messages = append(s.messages, msg)
	return nil
}

func (s *memoryStore) UpdateLastMessage(_ context.Context, _ string, msg api.Message, _ string) error {
	s.updateCalls++
	if len(s.messages) == 0 {
		s.messages = append(s.messages, msg)
		return nil
	}
	s.messages[len(s.messages)-1] = msg
	return nil
}

type staticTool struct{}

type approvalTestTool struct {
	called *bool
}

type policyOnlyApprovalTool struct {
	name   string
	called *bool
}

type cwdTestTool struct{}

type wrappingApprovalHandler struct {
	inner         ApprovalHandler
	requiresCalls int
	approveCalls  int
}

func (staticTool) Name() string {
	return "echo_tool"
}

func (staticTool) Description() string {
	return "echoes a value"
}

func (staticTool) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("value", api.ToolProperty{Type: api.PropertyType{"string"}})
	return api.ToolFunction{
		Name:        "echo_tool",
		Description: "echoes a value",
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Properties: props,
		},
	}
}

func (staticTool) Execute(context.Context, ToolContext, map[string]any) (ToolResult, error) {
	return ToolResult{Content: "tool says hello"}, nil
}

func (t approvalTestTool) Name() string {
	return "approval_tool"
}

func (t approvalTestTool) Description() string {
	return "requires approval"
}

func (t approvalTestTool) Schema() api.ToolFunction {
	return api.ToolFunction{
		Name:        "approval_tool",
		Description: "requires approval",
		Parameters: api.ToolFunctionParameters{
			Type: "object",
		},
	}
}

func (t approvalTestTool) RequiresApproval(map[string]any) bool {
	return true
}

func (t approvalTestTool) Execute(context.Context, ToolContext, map[string]any) (ToolResult, error) {
	if t.called != nil {
		*t.called = true
	}
	return ToolResult{Content: "approved"}, nil
}

func (t policyOnlyApprovalTool) Name() string {
	return t.name
}

func (t policyOnlyApprovalTool) Description() string {
	return "does not self-declare approval"
}

func (t policyOnlyApprovalTool) Schema() api.ToolFunction {
	return api.ToolFunction{
		Name:        t.name,
		Description: "does not self-declare approval",
		Parameters: api.ToolFunctionParameters{
			Type: "object",
		},
	}
}

func (t policyOnlyApprovalTool) Execute(context.Context, ToolContext, map[string]any) (ToolResult, error) {
	if t.called != nil {
		*t.called = true
	}
	return ToolResult{Content: "ran"}, nil
}

func (h *wrappingApprovalHandler) RequiresApproval(ctx context.Context, tool Tool, req ApprovalRequest) bool {
	h.requiresCalls++
	return h.inner.RequiresApproval(ctx, tool, req)
}

func (h *wrappingApprovalHandler) Approve(ctx context.Context, req ApprovalRequest) (ApprovalResult, error) {
	h.approveCalls++
	return h.inner.Approve(ctx, req)
}

func (cwdTestTool) Name() string {
	return "cwd_tool"
}

func (cwdTestTool) Description() string {
	return "tests cwd state"
}

func (cwdTestTool) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("mode", api.ToolProperty{Type: api.PropertyType{"string"}})
	props.Set("path", api.ToolProperty{Type: api.PropertyType{"string"}})
	return api.ToolFunction{
		Name:        "cwd_tool",
		Description: "tests cwd state",
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Properties: props,
		},
	}
}

func (cwdTestTool) Execute(_ context.Context, toolCtx ToolContext, args map[string]any) (ToolResult, error) {
	switch args["mode"] {
	case "set":
		path, _ := args["path"].(string)
		return ToolResult{Content: "changed", WorkingDir: filepath.Join(toolCtx.WorkingDir, path)}, nil
	case "escape":
		return ToolResult{Content: "escaped", WorkingDir: filepath.Dir(toolCtx.WorkingDir)}, nil
	default:
		return ToolResult{Content: toolCtx.WorkingDir}, nil
	}
}

func TestSessionRunsToolLoop(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("value", "hello")

	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{
				{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-1",
					Function: api.ToolCallFunction{
						Name:      "echo_tool",
						Arguments: args,
					},
				}}}},
			},
			{
				{Message: api.Message{Role: "assistant", Content: "done"}},
			},
		},
	}

	registry := NewRegistry()
	registry.Register(staticTool{})
	store := &memoryStore{}

	session := &Session{
		Client: client,
		Store:  store,
		Tools:  registry,
	}

	result, err := session.Run(context.Background(), RunOptions{
		ChatID:      "chat-1",
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
		UseTools:    true,
	})
	if err != nil {
		t.Fatal(err)
	}

	if client.calls != 2 {
		t.Fatalf("client calls = %d, want 2", client.calls)
	}
	if len(result.Messages) != 4 {
		t.Fatalf("messages = %d, want 4", len(result.Messages))
	}
	if result.Messages[2].Role != "tool" || result.Messages[2].Content != "tool says hello" {
		t.Fatalf("tool message = %#v", result.Messages[2])
	}
	if len(client.requests[0].Tools) != 1 {
		t.Fatalf("first request tools = %d, want 1", len(client.requests[0].Tools))
	}
	if len(client.requests[1].Messages) != 3 {
		t.Fatalf("second request messages = %d, want 3", len(client.requests[1].Messages))
	}
	if len(store.messages) != 4 {
		t.Fatalf("stored messages = %d, want 4", len(store.messages))
	}
}

func TestSessionAddsSystemPromptOnlyToRequest(t *testing.T) {
	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{{Message: api.Message{Role: "assistant", Content: "done"}}},
		},
	}
	store := &memoryStore{}
	session := &Session{Client: client, Store: store}

	_, err := session.Run(context.Background(), RunOptions{
		ChatID:       "chat-1",
		Model:        "model",
		SystemPrompt: "available skills: go-code",
		NewMessages:  []api.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
	reqMessages := client.requests[0].Messages
	if len(reqMessages) != 2 || reqMessages[0].Role != "system" || reqMessages[0].Content != "available skills: go-code" {
		t.Fatalf("request messages = %#v", reqMessages)
	}
	if len(store.messages) != 2 {
		t.Fatalf("stored messages = %#v, want user and assistant only", store.messages)
	}
	for _, msg := range store.messages {
		if msg.Role == "system" && msg.Content == "available skills: go-code" {
			t.Fatalf("system prompt was persisted: %#v", store.messages)
		}
	}
}

func TestSessionBatchesStreamingPersistence(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("value", "hello")
	responses := make([]api.ChatResponse, 0, 100)
	var wantContent, wantThinking string
	for i := 0; i < 99; i++ {
		wantContent += "x"
		wantThinking += "t"
		responses = append(responses, api.ChatResponse{
			Message: api.Message{Role: "assistant", Content: "x", Thinking: "t"},
		})
	}
	toolCall := api.ToolCall{
		ID: "call-1",
		Function: api.ToolCallFunction{
			Name:      "echo_tool",
			Arguments: args,
		},
	}
	responses = append(responses, api.ChatResponse{
		Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{toolCall}},
	})

	store := &memoryStore{}
	session := &Session{
		Client: &fakeClient{responses: [][]api.ChatResponse{responses}},
		Store:  store,
	}

	result, err := session.Run(context.Background(), RunOptions{
		ChatID:      "chat-1",
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "stream"}},
	})
	if err != nil {
		t.Fatal(err)
	}

	writeCalls := store.appendCalls + store.updateCalls
	if maxWrites := len(responses)/streamPersistDeltaThreshold + 2; writeCalls > maxWrites {
		t.Fatalf("store writes = %d, want <= %d", writeCalls, maxWrites)
	}
	if len(store.messages) != 2 {
		t.Fatalf("stored messages = %#v", store.messages)
	}
	stored := store.messages[1]
	if stored.Content != wantContent || stored.Thinking != wantThinking || len(stored.ToolCalls) != 1 {
		t.Fatalf("stored assistant = %#v", stored)
	}
	if len(result.Messages) != 2 || result.Messages[1].Content != wantContent || result.Messages[1].Thinking != wantThinking || len(result.Messages[1].ToolCalls) != 1 {
		t.Fatalf("result messages = %#v", result.Messages)
	}
}

func TestSessionPersistsPartialStreamOnCancellation(t *testing.T) {
	store := &memoryStore{}
	session := &Session{
		Client: &fakeClient{
			responses: [][]api.ChatResponse{{
				{Message: api.Message{Role: "assistant", Content: "partial "}},
				{Message: api.Message{Role: "assistant", Content: "answer"}},
			}},
			err: context.Canceled,
		},
		Store: store,
	}

	result, err := session.Run(context.Background(), RunOptions{
		ChatID:      "chat-1",
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "cancel"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(store.messages) != 2 {
		t.Fatalf("stored messages = %#v", store.messages)
	}
	if store.messages[1].Content != "partial answer" {
		t.Fatalf("stored partial content = %q", store.messages[1].Content)
	}
	if len(result.Messages) != 2 || result.Messages[1].Content != "partial answer" {
		t.Fatalf("result messages = %#v", result.Messages)
	}
}

func TestSessionToolLoopHasNoDefaultRoundCap(t *testing.T) {
	responses := make([][]api.ChatResponse, 0, 26)
	for i := 0; i < 25; i++ {
		args := api.NewToolCallFunctionArguments()
		args.Set("value", "hello")
		responses = append(responses, []api.ChatResponse{{
			Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
				ID: "call-" + string(rune('a'+i)),
				Function: api.ToolCallFunction{
					Name:      "echo_tool",
					Arguments: args,
				},
			}}},
		}})
	}
	responses = append(responses, []api.ChatResponse{{
		Message: api.Message{Role: "assistant", Content: "done"},
	}})

	client := &fakeClient{responses: responses}
	registry := NewRegistry()
	registry.Register(staticTool{})

	session := &Session{
		Client: client,
		Tools:  registry,
	}

	if _, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "keep going"}},
		UseTools:    true,
	}); err != nil {
		t.Fatal(err)
	}

	if client.calls != 26 {
		t.Fatalf("client calls = %d, want 26", client.calls)
	}
}

func TestSessionPersistsToolWorkingDirWithinRun(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	setArgs := api.NewToolCallFunctionArguments()
	setArgs.Set("mode", "set")
	setArgs.Set("path", "sub")
	echoArgs := api.NewToolCallFunctionArguments()
	echoArgs.Set("mode", "echo")

	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{
				{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{
					{
						ID: "call-1",
						Function: api.ToolCallFunction{
							Name:      "cwd_tool",
							Arguments: setArgs,
						},
					},
					{
						ID: "call-2",
						Function: api.ToolCallFunction{
							Name:      "cwd_tool",
							Arguments: echoArgs,
						},
					},
				}}},
			},
			{
				{Message: api.Message{Role: "assistant", Content: "done"}},
			},
		},
	}
	registry := NewRegistry()
	registry.Register(cwdTestTool{})
	session := &Session{
		Client:     client,
		Tools:      registry,
		WorkingDir: root,
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use cwd"}},
		UseTools:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	want, err := filepath.EvalSymlinks(filepath.Join(root, "sub"))
	if err != nil {
		t.Fatal(err)
	}
	if session.WorkingDir != want {
		t.Fatalf("session cwd = %q, want %q", session.WorkingDir, want)
	}
	if result.WorkingDir != want {
		t.Fatalf("result cwd = %q, want %q", result.WorkingDir, want)
	}
	if result.Messages[2].Content != "changed" {
		t.Fatalf("cwd change tool content = %q, want unchanged output", result.Messages[2].Content)
	}
	if result.Messages[3].Content != want {
		t.Fatalf("second tool saw cwd %q, want %q", result.Messages[3].Content, want)
	}
}

func TestSessionAllowsToolWorkingDirOutsideInitialDir(t *testing.T) {
	root := t.TempDir()
	escapeArgs := api.NewToolCallFunctionArguments()
	escapeArgs.Set("mode", "escape")
	echoArgs := api.NewToolCallFunctionArguments()
	echoArgs.Set("mode", "echo")

	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{
				{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{
					{
						ID: "call-1",
						Function: api.ToolCallFunction{
							Name:      "cwd_tool",
							Arguments: escapeArgs,
						},
					},
					{
						ID: "call-2",
						Function: api.ToolCallFunction{
							Name:      "cwd_tool",
							Arguments: echoArgs,
						},
					},
				}}},
			},
			{
				{Message: api.Message{Role: "assistant", Content: "done"}},
			},
		},
	}
	registry := NewRegistry()
	registry.Register(cwdTestTool{})
	session := &Session{
		Client:     client,
		Tools:      registry,
		WorkingDir: root,
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use cwd"}},
		UseTools:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	want, err := filepath.EvalSymlinks(filepath.Dir(root))
	if err != nil {
		t.Fatal(err)
	}
	if session.WorkingDir != want {
		t.Fatalf("session cwd = %q, want %q", session.WorkingDir, want)
	}
	if result.Messages[2].Content != "escaped" {
		t.Fatalf("escape tool content = %q, want unchanged output", result.Messages[2].Content)
	}
	if result.Messages[3].Content != want {
		t.Fatalf("second tool saw cwd %q, want %q", result.Messages[3].Content, want)
	}
}

func TestSessionApprovalManagerDeniesWithoutPrompter(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	echoArgs := api.NewToolCallFunctionArguments()
	echoArgs.Set("value", "should not run")
	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{
				{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{
					{
						ID: "call-1",
						Function: api.ToolCallFunction{
							Name:      "approval_tool",
							Arguments: args,
						},
					},
					{
						ID: "call-2",
						Function: api.ToolCallFunction{
							Name:      "echo_tool",
							Arguments: echoArgs,
						},
					},
				}}},
			},
			{
				{Message: api.Message{Role: "assistant", Content: "done"}},
			},
		},
	}
	called := false
	registry := NewRegistry()
	registry.Register(approvalTestTool{called: &called})
	registry.Register(staticTool{})
	session := &Session{
		Client:   client,
		Tools:    registry,
		Approval: NewApprovalManager(ApprovalManagerOptions{}),
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
		UseTools:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if called {
		t.Fatal("tool executed despite denied approval")
	}
	if client.calls != 1 {
		t.Fatalf("client calls = %d, want 1 after denial", client.calls)
	}
	if len(result.Messages) != 3 {
		t.Fatalf("messages = %#v", result.Messages)
	}
	if result.Messages[2].Role != "tool" || result.Messages[2].ToolCallID != "call-1" {
		t.Fatalf("denial tool message = %#v", result.Messages[2])
	}
	if result.Messages[2].Content == "" || result.Messages[2].Content == "approved" || result.Messages[2].Content == "tool says hello" {
		t.Fatalf("tool denial content = %q", result.Messages[2].Content)
	}
}

func TestSessionConsultsWrappedApprovalRequirement(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("command", "pwd")
	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{
				{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-1",
					Function: api.ToolCallFunction{
						Name:      "bash",
						Arguments: args,
					},
				}}}},
			},
			{
				{Message: api.Message{Role: "assistant", Content: "done"}},
			},
		},
	}
	called := false
	registry := NewRegistry()
	registry.Register(policyOnlyApprovalTool{name: "bash", called: &called})
	approval := &wrappingApprovalHandler{inner: NewApprovalManager(ApprovalManagerOptions{})}
	session := &Session{
		Client:   client,
		Tools:    registry,
		Approval: approval,
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use bash"}},
		UseTools:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if approval.requiresCalls != 1 {
		t.Fatalf("requires calls = %d, want 1", approval.requiresCalls)
	}
	if approval.approveCalls != 1 {
		t.Fatalf("approve calls = %d, want 1", approval.approveCalls)
	}
	if called {
		t.Fatal("tool ran despite wrapped approval manager requiring approval")
	}
	if client.calls != 1 {
		t.Fatalf("client calls = %d, want 1 after denial", client.calls)
	}
	if len(result.Messages) != 3 || result.Messages[2].Role != "tool" {
		t.Fatalf("messages = %#v", result.Messages)
	}
}

func TestSessionAutoAllowApprovalExecutesApprovalTool(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{
				{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-1",
					Function: api.ToolCallFunction{
						Name:      "approval_tool",
						Arguments: args,
					},
				}}}},
			},
			{
				{Message: api.Message{Role: "assistant", Content: "done"}},
			},
		},
	}
	called := false
	registry := NewRegistry()
	registry.Register(approvalTestTool{called: &called})
	session := &Session{
		Client:   client,
		Tools:    registry,
		Approval: AutoAllowApproval{},
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
		UseTools:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("tool did not execute")
	}
	if result.Messages[2].Content != "approved" {
		t.Fatalf("tool content = %q, want approved", result.Messages[2].Content)
	}
}
