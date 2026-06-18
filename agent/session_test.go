package agent

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
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

type contextAwareStore struct {
	memoryStore
}

func (s *contextAwareStore) EnsureChat(ctx context.Context, id string, title string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return s.memoryStore.EnsureChat(ctx, id, title)
}

func (s *contextAwareStore) AppendMessage(ctx context.Context, chatID string, msg api.Message, model string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return s.memoryStore.AppendMessage(ctx, chatID, msg, model)
}

func (s *contextAwareStore) UpdateLastMessage(ctx context.Context, chatID string, msg api.Message, model string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return s.memoryStore.UpdateLastMessage(ctx, chatID, msg, model)
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

type largeTool struct{}

type cancelAfterToolCallClient struct {
	cancel context.CancelFunc
}

type recordingCompactor struct {
	requests []CompactionRequest
}

type oversizedCompactor struct {
	requests []CompactionRequest
}

type recordingEventSink struct {
	events []Event
}

func (s *recordingEventSink) Emit(event Event) error {
	s.events = append(s.events, event)
	return nil
}

func hasEventType(events []Event, eventType EventType) bool {
	for _, event := range events {
		if event.Type == eventType {
			return true
		}
	}
	return false
}

func hasEventWithTokens(events []Event, eventType EventType, tokens int) bool {
	for _, event := range events {
		if event.Type == eventType && event.Tokens == tokens {
			return true
		}
	}
	return false
}

func (c cancelAfterToolCallClient) Chat(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
	args := api.NewToolCallFunctionArguments()
	args.Set("value", "skip me")
	if err := fn(api.ChatResponse{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
		ID: "call-1",
		Function: api.ToolCallFunction{
			Name:      "echo_tool",
			Arguments: args,
		},
	}}}}); err != nil {
		return err
	}
	c.cancel()
	return context.Canceled
}

func (c *recordingCompactor) MaybeCompact(_ context.Context, req CompactionRequest) (CompactionResult, error) {
	c.requests = append(c.requests, req)
	result := CompactionResult{Messages: req.Messages, Due: true}
	if len(req.Messages) > 0 && req.Messages[len(req.Messages)-1].Role == "tool" {
		result.Messages = compactionSummaryMessages("tool result summarized")
		result.Compacted = true
		result.Summary = "tool result summarized"
	}
	return result, nil
}

func (c *oversizedCompactor) MaybeCompact(_ context.Context, req CompactionRequest) (CompactionResult, error) {
	c.requests = append(c.requests, req)
	summary := strings.Repeat("oversized summary ", 300)
	return CompactionResult{
		Messages:  compactionSummaryMessagesForTask(summary, req.ContinueTask),
		Compacted: true,
		Due:       true,
		Summary:   summary,
	}, nil
}

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

func (largeTool) Name() string {
	return "large_tool"
}

func (largeTool) Description() string {
	return "returns a large result"
}

func (largeTool) Schema() api.ToolFunction {
	return api.ToolFunction{
		Name:        "large_tool",
		Description: "returns a large result",
		Parameters: api.ToolFunctionParameters{
			Type: "object",
		},
	}
}

func (largeTool) Execute(context.Context, ToolContext, map[string]any) (ToolResult, error) {
	return ToolResult{Content: strings.Repeat("x", maxToolResultRunes+100)}, nil
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
	for range 99 {
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

func TestSessionRequestHistoryKeepsThinkingAndServerToolCallIDs(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("value", "hello")
	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{
				{Message: api.Message{Role: "assistant", Thinking: "private chain"}},
				{Message: api.Message{Role: "assistant", Content: "I'll check."}},
				{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "volatile-random-id",
					Function: api.ToolCallFunction{
						Name:      "echo_tool",
						Arguments: args,
					},
				}}}},
			},
			{{Message: api.Message{Role: "assistant", Content: "done"}}},
		},
	}
	store := &memoryStore{}
	registry := NewRegistry()
	registry.Register(staticTool{})
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
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(client.requests))
	}

	secondRequestMessages := client.requests[1].Messages
	if len(secondRequestMessages) != 3 {
		t.Fatalf("second request messages = %#v", secondRequestMessages)
	}
	assistant := secondRequestMessages[1]
	if assistant.Role != "assistant" {
		t.Fatalf("second request assistant = %#v", assistant)
	}
	if assistant.Thinking != "private chain" {
		t.Fatalf("assistant thinking = %q, want preserved", assistant.Thinking)
	}
	if len(assistant.ToolCalls) != 1 || assistant.ToolCalls[0].ID != "volatile-random-id" {
		t.Fatalf("assistant tool calls = %#v", assistant.ToolCalls)
	}
	tool := secondRequestMessages[2]
	if tool.Role != "tool" || tool.ToolCallID != "volatile-random-id" {
		t.Fatalf("tool result message = %#v", tool)
	}
	if len(result.Messages) < 3 || result.Messages[1].Thinking != "private chain" {
		t.Fatalf("visible result messages lost thinking: %#v", result.Messages)
	}
	if len(store.messages) < 3 || store.messages[1].Thinking != "private chain" {
		t.Fatalf("stored messages lost thinking: %#v", store.messages)
	}
	if store.messages[1].ToolCalls[0].ID != "volatile-random-id" || store.messages[2].ToolCallID != "volatile-random-id" {
		t.Fatalf("stored tool call IDs are not preserved/matched: %#v", store.messages)
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

func TestSessionTreatsHTTPContextCanceledStringAsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	client := &fakeClient{err: errors.New(`Post "http://127.0.0.1:11434/api/chat": context canceled`)}
	session := &Session{Client: client}

	result, err := session.Run(ctx, RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("Run returned error for canceled HTTP request: %v", err)
	}
	if result == nil {
		t.Fatal("Run returned nil result")
	}
	if len(result.Messages) != 1 || result.Messages[0].Content != "hello" {
		t.Fatalf("messages = %#v, want original user message only", result.Messages)
	}
}

func TestSessionCancellationAfterToolCallAppendsSkippedToolMessage(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	store := &contextAwareStore{}
	registry := NewRegistry()
	registry.Register(staticTool{})
	session := &Session{
		Client: cancelAfterToolCallClient{cancel: cancel},
		Store:  store,
		Tools:  registry,
	}

	result, err := session.Run(ctx, RunOptions{
		ChatID:      "chat-1",
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "cancel after tool"}},
		UseTools:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) != 3 {
		t.Fatalf("messages = %#v", result.Messages)
	}
	if len(store.messages) != len(result.Messages) {
		t.Fatalf("stored messages = %#v, want %d messages", store.messages, len(result.Messages))
	}
	if len(result.Messages[1].ToolCalls) != 1 {
		t.Fatalf("assistant tool calls = %#v", result.Messages[1])
	}
	if result.Messages[2].Role != "tool" || result.Messages[2].ToolCallID != "call-1" {
		t.Fatalf("skipped tool message = %#v", result.Messages[2])
	}
	if !strings.Contains(result.Messages[2].Content, "run was canceled") {
		t.Fatalf("skipped content = %q", result.Messages[2].Content)
	}
}

func TestSessionToolLoopAllowsRoundsUnderDefaultCap(t *testing.T) {
	responses := make([][]api.ChatResponse, 0, 26)
	for i := range 25 {
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

func TestSessionToolRoundLimitAppendsSkippedToolMessages(t *testing.T) {
	firstArgs := api.NewToolCallFunctionArguments()
	firstArgs.Set("value", "first")
	secondArgs := api.NewToolCallFunctionArguments()
	secondArgs.Set("value", "second")
	thirdArgs := api.NewToolCallFunctionArguments()
	thirdArgs.Set("value", "third")

	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{{
				Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-1",
					Function: api.ToolCallFunction{
						Name:      "echo_tool",
						Arguments: firstArgs,
					},
				}}},
			}},
			{{
				Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{
					{
						ID: "call-2",
						Function: api.ToolCallFunction{
							Name:      "echo_tool",
							Arguments: secondArgs,
						},
					},
					{
						ID: "call-3",
						Function: api.ToolCallFunction{
							Name:      "echo_tool",
							Arguments: thirdArgs,
						},
					},
				}},
			}},
		},
	}
	store := &memoryStore{}
	registry := NewRegistry()
	registry.Register(staticTool{})
	session := &Session{
		Client: client,
		Store:  store,
		Tools:  registry,
	}

	result, err := session.Run(context.Background(), RunOptions{
		ChatID:        "chat-1",
		Model:         "model",
		NewMessages:   []api.Message{{Role: "user", Content: "hit cap"}},
		UseTools:      true,
		MaxToolRounds: 1,
	})
	if err == nil || !strings.Contains(err.Error(), "tool round limit reached after 1 rounds") {
		t.Fatalf("error = %v, want tool-round limit", err)
	}
	if result == nil {
		t.Fatal("expected partial result with skipped tool messages")
	}
	if len(result.Messages) != 6 {
		t.Fatalf("messages = %#v", result.Messages)
	}
	if len(store.messages) != len(result.Messages) {
		t.Fatalf("stored messages = %#v, want %d messages", store.messages, len(result.Messages))
	}
	for i, wantID := range []string{"call-2", "call-3"} {
		msg := result.Messages[4+i]
		if msg.Role != "tool" || msg.ToolCallID != wantID {
			t.Fatalf("skipped tool %d = %#v", i, msg)
		}
		if !strings.Contains(msg.Content, "max tool-round limit of 1") {
			t.Fatalf("skipped content = %q", msg.Content)
		}
	}
}

func TestSessionToolLoopStopsAtDefaultRoundCap(t *testing.T) {
	responses := make([][]api.ChatResponse, 0, defaultMaxToolRounds+1)
	for range defaultMaxToolRounds + 1 {
		args := api.NewToolCallFunctionArguments()
		args.Set("value", "hello")
		responses = append(responses, []api.ChatResponse{{
			Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
				ID: "call",
				Function: api.ToolCallFunction{
					Name:      "echo_tool",
					Arguments: args,
				},
			}}},
		}})
	}

	client := &fakeClient{responses: responses}
	registry := NewRegistry()
	registry.Register(staticTool{})
	session := &Session{
		Client: client,
		Tools:  registry,
	}

	_, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "keep going"}},
		UseTools:    true,
	})
	if err == nil || !strings.Contains(err.Error(), "tool round limit reached after 100 rounds") {
		t.Fatalf("error = %v, want default tool round limit", err)
	}
	if client.calls != defaultMaxToolRounds+1 {
		t.Fatalf("client calls = %d, want %d", client.calls, defaultMaxToolRounds+1)
	}
}

func TestSessionToolLoopNegativeLimitIsUnlimited(t *testing.T) {
	responses := make([][]api.ChatResponse, 0, defaultMaxToolRounds+2)
	for range defaultMaxToolRounds + 1 {
		args := api.NewToolCallFunctionArguments()
		args.Set("value", "hello")
		responses = append(responses, []api.ChatResponse{{
			Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
				ID: "call",
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
		Model:         "model",
		NewMessages:   []api.Message{{Role: "user", Content: "keep going"}},
		UseTools:      true,
		MaxToolRounds: -1,
	}); err != nil {
		t.Fatal(err)
	}
	if client.calls != defaultMaxToolRounds+2 {
		t.Fatalf("client calls = %d, want %d", client.calls, defaultMaxToolRounds+2)
	}
}

func TestSessionTruncatesLargeToolResultsBeforeHistory(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{{
				Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-1",
					Function: api.ToolCallFunction{
						Name:      "large_tool",
						Arguments: args,
					},
				}}},
			}},
			{{Message: api.Message{Role: "assistant", Content: "done"}}},
		},
	}
	store := &memoryStore{}
	registry := NewRegistry()
	registry.Register(largeTool{})
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
	if len(result.Messages) < 3 {
		t.Fatalf("messages = %#v", result.Messages)
	}
	content := result.Messages[2].Content
	if !strings.Contains(content, "[tool output truncated: omitted 100 characters]") {
		t.Fatalf("tool content missing truncation marker: %q", content)
	}
	if strings.Count(content, "x") != maxToolResultRunes {
		t.Fatalf("truncated content x count = %d, want %d", strings.Count(content, "x"), maxToolResultRunes)
	}
	if store.messages[2].Content != content {
		t.Fatalf("stored tool content not truncated consistently")
	}
	if client.requests[1].Messages[2].Content != content {
		t.Fatalf("second model request did not use truncated tool content")
	}
}

func TestSessionCompactsAfterToolResultsBeforeContinuing(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("value", "hello")
	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{{
				Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-1",
					Function: api.ToolCallFunction{
						Name:      "echo_tool",
						Arguments: args,
					},
				}}},
			}},
			{{Message: api.Message{Role: "assistant", Content: "done after compact"}}},
		},
	}
	registry := NewRegistry()
	registry.Register(staticTool{})
	compactor := &recordingCompactor{}
	session := &Session{
		Client:    client,
		Tools:     registry,
		Compactor: compactor,
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
		UseTools:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if client.calls != 2 {
		t.Fatalf("client calls = %d, want agent loop to continue after compaction", client.calls)
	}
	if len(compactor.requests) == 0 {
		t.Fatal("compactor was not called")
	}
	firstCompaction := compactor.requests[0]
	if len(firstCompaction.Messages) == 0 || firstCompaction.Messages[len(firstCompaction.Messages)-1].Role != "tool" {
		t.Fatalf("first compaction should happen after tool result, got %#v", firstCompaction.Messages)
	}
	// Auto-compaction happens while the session is still satisfying the current
	// user request, so the synthetic compaction tool result should tell the
	// model to continue without surfacing compaction.
	if !firstCompaction.ContinueTask {
		t.Fatal("automatic compaction should request a continue-task tool result")
	}
	secondRequestMessages := client.requests[1].Messages
	if len(secondRequestMessages) == 0 || !strings.Contains(secondRequestMessages[len(secondRequestMessages)-1].Content, "tool result summarized") {
		t.Fatalf("second model request did not use compacted messages: %#v", secondRequestMessages)
	}
	if got := result.Messages[len(result.Messages)-1].Content; got != "done after compact" {
		t.Fatalf("final response = %q", got)
	}
}

func TestSessionStopsWhenCompactedHistoryStillExceedsContext(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("value", "hello")
	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{{
				Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-1",
					Function: api.ToolCallFunction{
						Name:      "echo_tool",
						Arguments: args,
					},
				}}},
			}},
			{{Message: api.Message{Role: "assistant", Content: "should not run"}}},
		},
	}
	registry := NewRegistry()
	registry.Register(staticTool{})
	events := &recordingEventSink{}
	compactor := &oversizedCompactor{}
	session := &Session{
		Client:    client,
		Events:    events,
		Tools:     registry,
		Compactor: compactor,
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
		UseTools:    true,
		Options:     map[string]any{"num_ctx": 512},
	})
	if err == nil {
		t.Fatal("expected post-compaction context error")
	}
	if !strings.Contains(err.Error(), "still too large after compaction") || !strings.Contains(err.Error(), "fresh request") {
		t.Fatalf("error = %q, want actionable post-compaction guidance", err.Error())
	}
	if result == nil {
		t.Fatal("expected partial result with compacted messages")
	}
	if client.calls != 1 || len(client.requests) != 1 {
		t.Fatalf("client calls = %d requests = %d, want no request after oversized compaction", client.calls, len(client.requests))
	}
	if len(compactor.requests) != 1 {
		t.Fatalf("compactor requests = %d, want 1", len(compactor.requests))
	}
	if !hasEventType(events.events, EventCompacted) {
		t.Fatalf("events missing compacted event: %#v", events.events)
	}
	if !hasEventType(events.events, EventError) {
		t.Fatalf("events missing post-compaction error: %#v", events.events)
	}
	if len(result.Messages) == 0 || !strings.Contains(result.Messages[len(result.Messages)-1].Content, "Conversation summary:") {
		t.Fatalf("result should retain compacted summary messages: %#v", result.Messages)
	}
}

func TestSessionContextCapsToolResultBeforeCompaction(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{{
				Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-1",
					Function: api.ToolCallFunction{
						Name:      "large_tool",
						Arguments: args,
					},
				}}},
			}},
			{{Message: api.Message{Role: "assistant", Content: "done"}}},
		},
	}
	registry := NewRegistry()
	registry.Register(largeTool{})
	session := &Session{
		Client: client,
		Tools:  registry,
		Compactor: NewSimpleCompactor(nil, nil, CompactionOptions{
			ContextWindowTokens: 100,
			Threshold:           0.8,
		}),
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
		UseTools:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	content := result.Messages[2].Content
	if !strings.Contains(content, "[tool output truncated: omitted ") {
		t.Fatalf("tool content missing truncation marker: %q", content)
	}
	if xCount := strings.Count(content, "x"); xCount >= maxToolResultRunes {
		t.Fatalf("context-capped content x count = %d, want less than hard cap", xCount)
	}
	if client.requests[1].Messages[2].Content != content {
		t.Fatalf("second model request did not use context-capped tool content")
	}
}

func TestSessionEmitsAutoCompactionActivityEvents(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{{
				Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-1",
					Function: api.ToolCallFunction{
						Name:      "large_tool",
						Arguments: args,
					},
				}}},
			}},
			{{Message: api.Message{Role: "assistant", Content: "summary"}, Metrics: api.Metrics{EvalCount: 7}}},
			{{Message: api.Message{Role: "assistant", Content: "done"}}},
		},
	}
	registry := NewRegistry()
	registry.Register(largeTool{})
	events := &recordingEventSink{}
	session := &Session{
		Client: client,
		Events: events,
		Tools:  registry,
		Compactor: NewSimpleCompactor(client, nil, CompactionOptions{
			ContextWindowTokens: 300,
			Threshold:           0.3,
		}),
	}

	if _, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
		UseTools:    true,
	}); err != nil {
		t.Fatal(err)
	}

	if !hasEventType(events.events, EventCompactionStarted) {
		t.Fatalf("events missing compaction start: %#v", events.events)
	}
	if !hasEventWithTokens(events.events, EventCompactionProgress, 7) {
		t.Fatalf("events missing compaction progress tokens: %#v", events.events)
	}
	if !hasEventType(events.events, EventCompacted) {
		t.Fatalf("events missing compacted event: %#v", events.events)
	}
}

func TestSessionTruncatesSeededToolMessagesBeforeHistory(t *testing.T) {
	largeContent := strings.Repeat("x", maxToolResultRunes+100)
	client := &fakeClient{
		responses: [][]api.ChatResponse{{
			{Message: api.Message{Role: "assistant", Content: "done"}},
		}},
	}
	store := &memoryStore{}
	session := &Session{
		Client: client,
		Store:  store,
	}

	result, err := session.Run(context.Background(), RunOptions{
		ChatID: "chat-1",
		Model:  "model",
		NewMessages: []api.Message{
			{Role: "user", Content: "use seeded tool"},
			{Role: "tool", ToolName: "skill", ToolCallID: "call-1", Content: largeContent},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) < 2 {
		t.Fatalf("messages = %#v", result.Messages)
	}
	content := result.Messages[1].Content
	if !strings.Contains(content, "[tool output truncated: omitted 100 characters]") {
		t.Fatalf("seeded tool content missing truncation marker: %q", content)
	}
	if store.messages[1].Content != content {
		t.Fatalf("stored seeded tool content not truncated consistently")
	}
	if client.requests[0].Messages[1].Content != content {
		t.Fatalf("model request did not use truncated seeded tool content")
	}
}

func TestSessionPreflightRejectsOversizedFirstRequest(t *testing.T) {
	client := &fakeClient{
		responses: [][]api.ChatResponse{{
			{Message: api.Message{Role: "assistant", Content: "should not run"}},
		}},
	}
	store := &memoryStore{}
	events := &recordingEventSink{}
	session := &Session{
		Client: client,
		Store:  store,
		Events: events,
		Compactor: NewSimpleCompactor(nil, nil, CompactionOptions{
			ContextWindowTokens: 128,
		}),
	}

	_, err := session.Run(context.Background(), RunOptions{
		ChatID:       "chat-1",
		Model:        "model",
		SystemPrompt: strings.Repeat("system skill description ", 200),
		NewMessages:  []api.Message{{Role: "user", Content: "hello"}},
	})
	if err == nil {
		t.Fatal("expected preflight context error")
	}
	if !strings.Contains(err.Error(), "/system off") || !strings.Contains(err.Error(), "remove installed skills") {
		t.Fatalf("error = %q, want actionable prompt guidance", err.Error())
	}
	if len(client.requests) != 0 {
		t.Fatalf("chat requests = %d, want none before preflight passes", len(client.requests))
	}
	if store.appendCalls != 0 {
		t.Fatalf("append calls = %d, want no persisted new messages", store.appendCalls)
	}
	if !hasEventType(events.events, EventError) {
		t.Fatalf("events missing error: %#v", events.events)
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
	store := &memoryStore{}
	session := &Session{
		Client:   client,
		Tools:    registry,
		Store:    store,
		Approval: NewApprovalManager(ApprovalManagerOptions{}),
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
	if called {
		t.Fatal("tool executed despite denied approval")
	}
	if client.calls != 1 {
		t.Fatalf("client calls = %d, want 1 after denial", client.calls)
	}
	if len(result.Messages) != 4 {
		t.Fatalf("messages = %#v", result.Messages)
	}
	if result.Messages[2].Role != "tool" || result.Messages[2].ToolCallID != "call-1" {
		t.Fatalf("denial tool message = %#v", result.Messages[2])
	}
	if result.Messages[2].Content == "" || result.Messages[2].Content == "approved" || result.Messages[2].Content == "tool says hello" {
		t.Fatalf("tool denial content = %q", result.Messages[2].Content)
	}
	if result.Messages[3].Role != "tool" || result.Messages[3].ToolCallID != "call-2" {
		t.Fatalf("skipped tool message = %#v", result.Messages[3])
	}
	if !strings.Contains(result.Messages[3].Content, "skipped because a previous tool call") {
		t.Fatalf("skipped tool content = %q", result.Messages[3].Content)
	}
	if len(store.messages) != len(result.Messages) {
		t.Fatalf("persisted messages = %#v, want %d messages", store.messages, len(result.Messages))
	}
	if store.messages[2].ToolCallID != "call-1" || store.messages[3].ToolCallID != "call-2" {
		t.Fatalf("persisted tool call ids = %#v", store.messages)
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
