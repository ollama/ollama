package agent

import (
	"context"
	"encoding/json"
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

type staticTool struct{}

type approvalTestTool struct {
	called *bool
}

type namedApprovalTestTool struct {
	name string
}

type cwdTestTool struct{}

type largeTool struct{}

type preTruncatedTool struct{}

type cancelingTool struct {
	cancel context.CancelFunc
}

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

func TestSessionEmitsToAllSinksAfterError(t *testing.T) {
	errSink := EventSinkFunc(func(Event) error {
		return errors.New("sink failed")
	})
	events := &recordingEventSink{}
	session := &Session{EventSinks: []EventSink{errSink, events}}

	err := session.emit(Event{Type: EventRunFinished})
	if err == nil {
		t.Fatal("emit should return the first sink error")
	}
	if !hasEventType(events.events, EventRunFinished) {
		t.Fatalf("later sink did not receive event after earlier error: %#v", events.events)
	}
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
		result.Messages = CompactionSummaryMessages("tool result summarized", false)
		result.Compacted = true
		result.Summary = "tool result summarized"
	}
	return result, nil
}

func (c *oversizedCompactor) MaybeCompact(_ context.Context, req CompactionRequest) (CompactionResult, error) {
	c.requests = append(c.requests, req)
	summary := strings.Repeat("oversized summary ", 300)
	return CompactionResult{
		Messages:  CompactionSummaryMessages(summary, req.ContinueTask),
		Compacted: true,
		Due:       true,
		Summary:   summary,
	}, nil
}

type recordingApprovalPrompter struct {
	requests []ApprovalRequest
	results  []Approval
}

func approvalStateForTest(allowAll bool, scopes map[string]bool) *ApprovalState {
	state := &ApprovalState{}
	state.Set(allowAll, scopes)
	return state
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

func (preTruncatedTool) Name() string {
	return "pre_truncated_tool"
}

func (preTruncatedTool) Description() string {
	return "returns a large result that is already marked as truncated"
}

func (preTruncatedTool) Schema() api.ToolFunction {
	return api.ToolFunction{
		Name:        "pre_truncated_tool",
		Description: "returns a large result that is already marked as truncated",
		Parameters: api.ToolFunctionParameters{
			Type: "object",
		},
	}
}

func (preTruncatedTool) Execute(context.Context, ToolContext, map[string]any) (ToolResult, error) {
	content := strings.Repeat("x", smallContextToolResultRunes) +
		"\n\n[tool output truncated: showing first ~1500 tokens; omitted ~999 tokens. Use a narrower command, line range, or search query if more detail is needed.]\n\n" +
		strings.Repeat("y", smallContextToolResultRunes)
	return ToolResult{Content: content}, nil
}

func (t cancelingTool) Name() string {
	return "cancel_tool"
}

func (t cancelingTool) Description() string {
	return "cancels while running"
}

func (t cancelingTool) Schema() api.ToolFunction {
	return api.ToolFunction{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters: api.ToolFunctionParameters{
			Type: "object",
		},
	}
}

func (t cancelingTool) Execute(ctx context.Context, _ ToolContext, _ map[string]any) (ToolResult, error) {
	t.cancel()
	<-ctx.Done()
	return ToolResult{}, ctx.Err()
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

func (t namedApprovalTestTool) Name() string {
	return t.name
}

func (t namedApprovalTestTool) Description() string {
	return "requires approval"
}

func (t namedApprovalTestTool) Schema() api.ToolFunction {
	return api.ToolFunction{
		Name:        t.name,
		Description: "requires approval",
		Parameters: api.ToolFunctionParameters{
			Type: "object",
		},
	}
}

func (t namedApprovalTestTool) RequiresApproval(map[string]any) bool {
	return true
}

func (t namedApprovalTestTool) Execute(context.Context, ToolContext, map[string]any) (ToolResult, error) {
	return ToolResult{Content: "approved"}, nil
}

func (p *recordingApprovalPrompter) PromptApproval(_ context.Context, req ApprovalRequest) (Approval, error) {
	p.requests = append(p.requests, req)
	if len(p.results) == 0 {
		return Approval{Allow: true}, nil
	}
	result := p.results[0]
	p.results = p.results[1:]
	return result, nil
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

func (cwdTestTool) RequiresApproval(map[string]any) bool {
	return true
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

	registry := &Registry{}
	registry.Register(staticTool{})

	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
	}

	result, err := session.Run(context.Background(), RunOptions{
		ChatID:      "chat-1",
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
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
}

func TestSessionAddsSystemPromptOnlyToRequest(t *testing.T) {
	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{{Message: api.Message{Role: "assistant", Content: "done"}}},
		},
	}
	session := &Session{Client: client}

	_, err := session.Run(context.Background(), RunOptions{
		ChatID:       "chat-1",
		Model:        "model",
		SystemPrompt: "available context: go-code",
		NewMessages:  []api.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
	reqMessages := client.requests[0].Messages
	if len(reqMessages) != 2 || reqMessages[0].Role != "system" || reqMessages[0].Content != "available context: go-code" {
		t.Fatalf("request messages = %#v", reqMessages)
	}
}

func TestSessionChatRequestMatchesRunRequest(t *testing.T) {
	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{{Message: api.Message{Role: "assistant", Content: "done"}}},
		},
	}
	registry := &Registry{}
	registry.Register(staticTool{})
	session := &Session{Client: client, Tools: registry}
	opts := RunOptions{
		ChatID:       "chat-1",
		Model:        "model",
		SystemPrompt: "available context: go-code",
		NewMessages:  []api.Message{{Role: "user", Content: "hello"}},
		Format:       "json",
		Options:      map[string]any{"temperature": 0.5},
	}

	want := buildChatRequest(opts, opts.NewMessages, registry.Tools())
	_, err := session.Run(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
	gotJSON, err := json.Marshal(client.requests[0])
	if err != nil {
		t.Fatal(err)
	}
	wantJSON, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	if string(gotJSON) != string(wantJSON) {
		t.Fatalf("ChatRequest mismatch\ngot:  %s\nwant: %s", gotJSON, wantJSON)
	}
}

func TestSessionAccumulatesStreamingAssistantMessage(t *testing.T) {
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

	session := &Session{
		Client: &fakeClient{responses: [][]api.ChatResponse{responses}},
	}

	result, err := session.Run(context.Background(), RunOptions{
		ChatID:      "chat-1",
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "stream"}},
	})
	if err != nil {
		t.Fatal(err)
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
	registry := &Registry{}
	registry.Register(staticTool{})
	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
	}

	result, err := session.Run(context.Background(), RunOptions{
		ChatID:      "chat-1",
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
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
}

func TestSessionKeepsPartialStreamOnCancellation(t *testing.T) {
	session := &Session{
		Client: &fakeClient{
			responses: [][]api.ChatResponse{{
				{Message: api.Message{Role: "assistant", Content: "partial "}},
				{Message: api.Message{Role: "assistant", Content: "answer"}},
			}},
			err: context.Canceled,
		},
	}

	result, err := session.Run(context.Background(), RunOptions{
		ChatID:      "chat-1",
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "cancel"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) != 2 || result.Messages[1].Content != "partial answer" {
		t.Fatalf("result messages = %#v", result.Messages)
	}
}

func TestSessionCancellationKeepsPartialResultWhenUISinkCancels(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	trace := &recordingEventSink{}
	session := &Session{
		Client: &fakeClient{
			responses: [][]api.ChatResponse{{
				{Message: api.Message{Role: "assistant", Content: "partial"}},
			}},
			err: context.Canceled,
		},
		EventSinks: []EventSink{
			EventSinkFunc(func(event Event) error {
				if event.Type == EventRunFinished {
					return context.Canceled
				}
				return nil
			}),
			trace,
		},
	}

	result, err := session.Run(ctx, RunOptions{
		ChatID:      "chat-1",
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "cancel"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result == nil || len(result.Messages) != 2 || result.Messages[1].Content != "partial" {
		t.Fatalf("result messages = %#v, want partial assistant result", result)
	}
	if !hasEventType(trace.events, EventRunFinished) {
		t.Fatalf("trace sink did not receive run finished event: %#v", trace.events)
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

func TestSessionDisabledToolsOmitToolsAndReturnsDisabledResults(t *testing.T) {
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
			{{Message: api.Message{Role: "assistant", Content: "tools are off"}}},
		},
	}
	registry := &Registry{}
	registry.Register(staticTool{})
	events := &recordingEventSink{}
	session := &Session{
		Client:       client,
		EventSinks:   []EventSink{events},
		Tools:        registry,
		DisableTools: true,
	}

	result, err := session.Run(context.Background(), RunOptions{
		ChatID:      "chat-1",
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(client.requests))
	}
	if got := len(client.requests[0].Tools); got != 0 {
		t.Fatalf("advertised tools = %d, want 0", got)
	}
	secondMessages := client.requests[1].Messages
	if len(secondMessages) != 3 {
		t.Fatalf("second request messages = %#v", secondMessages)
	}
	if secondMessages[2].Role != "tool" || secondMessages[2].ToolCallID != "call-1" || secondMessages[2].Content != toolExecutionDisabledMessage {
		t.Fatalf("disabled tool message = %#v", secondMessages[2])
	}
	if len(result.Messages) != 4 || result.Messages[2].Content != toolExecutionDisabledMessage {
		t.Fatalf("result messages = %#v", result.Messages)
	}
	var sawDetected, sawDisabled bool
	for _, event := range events.events {
		if event.Type == EventToolCallDetected {
			sawDetected = true
		}
		if event.Type == EventToolFinished && event.ToolStatus == ToolStatusDisabled && event.Content == toolExecutionDisabledMessage {
			sawDisabled = true
		}
	}
	if !sawDetected || !sawDisabled {
		t.Fatalf("events missing detected/disabled: %#v", events.events)
	}
}

func TestSessionCancellationAfterToolCallAppendsSkippedToolMessage(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	registry := &Registry{}
	registry.Register(staticTool{})
	session := &Session{
		Client:        cancelAfterToolCallClient{cancel: cancel},
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
	}

	result, err := session.Run(ctx, RunOptions{
		ChatID:      "chat-1",
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "cancel after tool"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) != 3 {
		t.Fatalf("messages = %#v", result.Messages)
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

func TestSessionCancellationDuringToolExecutionAppendsToolMessage(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	events := &recordingEventSink{}
	registry := &Registry{}
	registry.Register(cancelingTool{cancel: cancel})
	client := &fakeClient{responses: [][]api.ChatResponse{{
		{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
			ID: "call-1",
			Function: api.ToolCallFunction{
				Name: "cancel_tool",
			},
		}}}},
	}}}
	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
		EventSinks:    []EventSink{events},
	}

	result, err := session.Run(ctx, RunOptions{
		ChatID:      "chat-1",
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "cancel during tool"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) != 3 {
		t.Fatalf("messages = %#v", result.Messages)
	}
	if result.Messages[2].Role != "tool" || result.Messages[2].ToolCallID != "call-1" {
		t.Fatalf("tool message = %#v", result.Messages[2])
	}
	if !strings.Contains(result.Messages[2].Content, "context canceled") {
		t.Fatalf("tool content = %q", result.Messages[2].Content)
	}
	var finished *Event
	for i := range events.events {
		if events.events[i].Type == EventRunFinished {
			finished = &events.events[i]
		}
	}
	if finished == nil {
		t.Fatalf("run finished event missing: %#v", events.events)
	}
	if finished.Status != RunStatusCanceled {
		t.Fatalf("run status = %q, want canceled", finished.Status)
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
	registry := &Registry{}
	registry.Register(staticTool{})

	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
	}

	if _, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "keep going"}},
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
	registry := &Registry{}
	registry.Register(staticTool{})
	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
	}

	result, err := session.Run(context.Background(), RunOptions{
		ChatID:        "chat-1",
		Model:         "model",
		NewMessages:   []api.Message{{Role: "user", Content: "hit cap"}},
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
	registry := &Registry{}
	registry.Register(staticTool{})
	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
	}

	_, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "keep going"}},
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
	registry := &Registry{}
	registry.Register(staticTool{})
	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
	}

	if _, err := session.Run(context.Background(), RunOptions{
		Model:         "model",
		NewMessages:   []api.Message{{Role: "user", Content: "keep going"}},
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
	registry := &Registry{}
	registry.Register(largeTool{})
	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
	}

	result, err := session.Run(context.Background(), RunOptions{
		ChatID:      "chat-1",
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) < 3 {
		t.Fatalf("messages = %#v", result.Messages)
	}
	content := result.Messages[2].Content
	if !strings.Contains(content, "[tool output truncated: showing first ~") ||
		!strings.Contains(content, "omitted ~25 tokens") ||
		!strings.Contains(content, "Use a narrower command, line range, or search query") {
		t.Fatalf("tool content missing truncation marker: %q", content)
	}
	if strings.Count(content, "x") != maxToolResultRunes {
		t.Fatalf("truncated content x count = %d, want %d", strings.Count(content, "x"), maxToolResultRunes)
	}
	requestContent := client.requests[1].Messages[2].Content
	if !strings.Contains(requestContent, "[tool output truncated: showing first ~") {
		t.Fatalf("second model request did not use capped tool content: %q", requestContent)
	}
	if strings.Count(requestContent, "x") > maxToolResultRunes {
		t.Fatalf("request tool content x count = %d, want at most %d", strings.Count(requestContent, "x"), maxToolResultRunes)
	}
}

func TestSessionSmallContextUsesLowerToolResultPreviewCap(t *testing.T) {
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
	registry := &Registry{}
	registry.Register(largeTool{})
	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
		Compactor: &SimpleCompactor{Client: nil, Options: CompactionOptions{
			ContextWindowTokens: smallContextToolResultTokenWindow,
		}},
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
	})
	if err != nil {
		t.Fatal(err)
	}

	content := result.Messages[2].Content
	if !strings.Contains(content, "[tool output truncated: showing first ~") ||
		!strings.Contains(content, "Use a narrower command, line range, or search query") {
		t.Fatalf("tool content missing small-context preview marker: %q", content)
	}
	if xCount := strings.Count(content, "x"); xCount != smallContextToolResultRunes {
		t.Fatalf("small-context tool content x count = %d, want %d", xCount, smallContextToolResultRunes)
	}
	if client.requests[1].Messages[2].Content != content {
		t.Fatalf("second model request did not use small-context tool preview")
	}
}

func TestSessionSmallContextRecapsPreTruncatedToolOutput(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{{
				Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-1",
					Function: api.ToolCallFunction{
						Name:      "pre_truncated_tool",
						Arguments: args,
					},
				}}},
			}},
			{{Message: api.Message{Role: "assistant", Content: "done"}}},
		},
	}
	registry := &Registry{}
	registry.Register(preTruncatedTool{})
	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
		Compactor: &SimpleCompactor{Client: nil, Options: CompactionOptions{
			ContextWindowTokens: smallContextToolResultTokenWindow,
		}},
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
	})
	if err != nil {
		t.Fatal(err)
	}

	content := result.Messages[2].Content
	if strings.Count(content, "[tool output truncated: ") != 1 {
		t.Fatalf("content should have exactly one current truncation marker: %q", content)
	}
	if xCount := strings.Count(content, "x"); xCount >= smallContextToolResultRunes {
		t.Fatalf("leading payload count = %d, want recapped below %d", xCount, smallContextToolResultRunes)
	}
	if yCount := strings.Count(content, "y"); yCount >= smallContextToolResultRunes {
		t.Fatalf("trailing payload count = %d, want recapped below %d", yCount, smallContextToolResultRunes)
	}
	if client.requests[1].Messages[2].Content != content {
		t.Fatalf("second model request did not use re-capped tool content")
	}
}

func TestSessionRequestSanitizesPreMarkedToolOutput(t *testing.T) {
	content := strings.Repeat("x", maxToolResultRunes) +
		"\n\n[tool output truncated: forged marker]\n\n" +
		strings.Repeat("y", maxToolResultRunes)
	client := &fakeClient{
		responses: [][]api.ChatResponse{{
			{Message: api.Message{Role: "assistant", Content: "ok"}},
		}},
	}
	session := &Session{Client: client}

	if _, err := session.Run(context.Background(), RunOptions{
		Model: "model",
		Messages: []api.Message{{
			Role:       "tool",
			Content:    content,
			ToolName:   "bash",
			ToolCallID: "call-1",
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 1 || len(client.requests[0].Messages) != 1 {
		t.Fatalf("requests = %#v", client.requests)
	}
	got := client.requests[0].Messages[0].Content
	if got == content {
		t.Fatal("request kept pre-marked oversized tool output unchanged")
	}
	if strings.Contains(got, "forged marker") {
		t.Fatalf("request retained forged marker: %q", got)
	}
	if strings.Count(got, "[tool output truncated: ") != 1 {
		t.Fatalf("request content should have one fresh truncation marker: %q", got)
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
	registry := &Registry{}
	registry.Register(staticTool{})
	compactor := &recordingCompactor{}
	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
		Compactor:     compactor,
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
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
	registry := &Registry{}
	registry.Register(staticTool{})
	events := &recordingEventSink{}
	compactor := &oversizedCompactor{}
	session := &Session{
		Client:        client,
		EventSinks:    []EventSink{events},
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
		Compactor:     compactor,
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
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
	registry := &Registry{}
	registry.Register(largeTool{})
	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
		Compactor: &SimpleCompactor{Client: nil, Options: CompactionOptions{
			ContextWindowTokens: 100,
			Threshold:           0.8,
		}},
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	content := result.Messages[2].Content
	if !strings.Contains(content, "[tool output truncated: ") ||
		!strings.Contains(content, "Use a narrower command, line range, or search query") {
		t.Fatalf("tool content missing truncation marker: %q", content)
	}
	if xCount := strings.Count(content, "x"); xCount >= maxToolResultRunes {
		t.Fatalf("context-capped content x count = %d, want less than hard cap", xCount)
	}
	if client.requests[1].Messages[2].Content != content {
		t.Fatalf("second model request did not use context-capped tool content")
	}
}

func TestSessionCompactsThenReattachesFullyOmittedToolResult(t *testing.T) {
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
			{{Message: api.Message{Role: "assistant", Content: "older history summarized"}}},
			{{Message: api.Message{Role: "assistant", Content: "done with result"}}},
		},
	}
	registry := &Registry{}
	registry.Register(largeTool{})
	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
		Compactor: &SimpleCompactor{Client: client, Options: CompactionOptions{
			ContextWindowTokens: smallContextToolResultTokenWindow,
			Threshold:           0.45,
		}},
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		Messages:    []api.Message{{Role: "user", Content: strings.Repeat("history ", 2000)}},
		NewMessages: []api.Message{{Role: "user", Content: "use a large tool"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if client.calls != 3 {
		t.Fatalf("client calls = %d, want model, compaction, model", client.calls)
	}
	if len(client.requests) != 3 {
		t.Fatalf("requests = %d, want 3", len(client.requests))
	}

	nextRequestMessages := client.requests[2].Messages
	if len(nextRequestMessages) != 4 {
		t.Fatalf("next model request messages = %#v, want summary pair plus tool call/result", nextRequestMessages)
	}
	if nextRequestMessages[0].Role != "assistant" || len(nextRequestMessages[0].ToolCalls) != 1 || nextRequestMessages[0].ToolCalls[0].Function.Name != CompactionToolName {
		t.Fatalf("first message should be compaction summary tool call: %#v", nextRequestMessages[0])
	}
	if nextRequestMessages[1].Role != "tool" || nextRequestMessages[1].ToolName != CompactionToolName || !strings.Contains(nextRequestMessages[1].Content, "older history summarized") {
		t.Fatalf("second message should be compaction summary result: %#v", nextRequestMessages[1])
	}
	if nextRequestMessages[2].Role != "assistant" || len(nextRequestMessages[2].ToolCalls) != 1 || nextRequestMessages[2].ToolCalls[0].ID != "call-1" {
		t.Fatalf("third message should be original assistant tool call: %#v", nextRequestMessages[2])
	}
	toolResult := nextRequestMessages[3]
	if toolResult.Role != "tool" || toolResult.ToolName != "large_tool" || toolResult.ToolCallID != "call-1" {
		t.Fatalf("fourth message should be reattached large tool result: %#v", toolResult)
	}
	if toolOutputFullyOmitted(toolResult.Content) {
		t.Fatalf("tool result should be re-fitted after compaction, got full omission marker: %q", toolResult.Content)
	}
	if !strings.Contains(toolResult.Content, "[tool output truncated: showing first ~") {
		t.Fatalf("tool result should still be bounded after compaction: %q", toolResult.Content)
	}
	if strings.Count(toolResult.Content, "x") != smallContextToolResultRunes {
		t.Fatalf("tool result x count = %d, want %d", strings.Count(toolResult.Content, "x"), smallContextToolResultRunes)
	}
	if got := result.Messages[len(result.Messages)-1].Content; got != "done with result" {
		t.Fatalf("final response = %q", got)
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
	registry := &Registry{}
	registry.Register(largeTool{})
	events := &recordingEventSink{}
	session := &Session{
		Client:        client,
		EventSinks:    []EventSink{events},
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
		Compactor: &SimpleCompactor{Client: client, Options: CompactionOptions{
			ContextWindowTokens: 300,
			Threshold:           0.3,
		}},
	}

	if _, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
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
	session := &Session{
		Client: client,
	}

	result, err := session.Run(context.Background(), RunOptions{
		ChatID: "chat-1",
		Model:  "model",
		NewMessages: []api.Message{
			{Role: "user", Content: "use seeded tool"},
			{Role: "tool", ToolName: "example_tool", ToolCallID: "call-1", Content: largeContent},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) < 2 {
		t.Fatalf("messages = %#v", result.Messages)
	}
	content := result.Messages[1].Content
	if !strings.Contains(content, "[tool output truncated: showing first ~") ||
		!strings.Contains(content, "omitted ~25 tokens") {
		t.Fatalf("seeded tool content missing truncation marker: %q", content)
	}
	requestContent := client.requests[0].Messages[1].Content
	if !strings.Contains(requestContent, "[tool output truncated: showing first ~") {
		t.Fatalf("model request did not use capped seeded tool content: %q", requestContent)
	}
	if strings.Count(requestContent, "x") > maxToolResultRunes {
		t.Fatalf("request seeded tool content x count = %d, want at most %d", strings.Count(requestContent, "x"), maxToolResultRunes)
	}
}

func TestSessionPreflightRejectsOversizedFirstRequest(t *testing.T) {
	client := &fakeClient{
		responses: [][]api.ChatResponse{{
			{Message: api.Message{Role: "assistant", Content: "should not run"}},
		}},
	}
	events := &recordingEventSink{}
	session := &Session{
		Client:     client,
		EventSinks: []EventSink{events},
		Compactor: &SimpleCompactor{Client: nil, Options: CompactionOptions{
			ContextWindowTokens: 128,
		}},
	}

	_, err := session.Run(context.Background(), RunOptions{
		ChatID:       "chat-1",
		Model:        "model",
		SystemPrompt: strings.Repeat("system instructions ", 200),
		NewMessages:  []api.Message{{Role: "user", Content: "hello"}},
	})
	if err == nil {
		t.Fatal("expected preflight context error")
	}
	if !strings.Contains(err.Error(), "Reduce the system prompt or message history") || !strings.Contains(err.Error(), "compact the conversation") {
		t.Fatalf("error = %q, want actionable prompt guidance", err.Error())
	}
	if len(client.requests) != 0 {
		t.Fatalf("chat requests = %d, want none before preflight passes", len(client.requests))
	}
	if !hasEventType(events.events, EventError) {
		t.Fatalf("events missing error: %#v", events.events)
	}
}

func TestSessionPreflightIgnoresRawImageBytes(t *testing.T) {
	client := &fakeClient{
		responses: [][]api.ChatResponse{{
			{Message: api.Message{Role: "assistant", Content: "image received"}},
		}},
	}
	session := &Session{
		Client: client,
		Compactor: &SimpleCompactor{Client: nil, Options: CompactionOptions{
			ContextWindowTokens: 128,
		}},
	}

	image := make(api.ImageData, 64*1024)
	result, err := session.Run(context.Background(), RunOptions{
		ChatID: "chat-1",
		Model:  "model",
		NewMessages: []api.Message{{
			Role:    "user",
			Content: "describe this image",
			Images:  []api.ImageData{image},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 1 {
		t.Fatalf("chat requests = %d, want 1", len(client.requests))
	}
	if got := client.requests[0].Messages[0].Images; len(got) != 1 || len(got[0]) != len(image) {
		t.Fatalf("request images = %#v, want original image payload", got)
	}
	if len(result.Messages) == 0 || result.Messages[len(result.Messages)-1].Content != "image received" {
		t.Fatalf("result messages = %#v", result.Messages)
	}
}

func TestSessionFreezesBatchToolWorkingDirAfterApproval(t *testing.T) {
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
	registry := &Registry{}
	registry.Register(cwdTestTool{})
	prompter := &recordingApprovalPrompter{results: []Approval{{Allow: true}}}
	session := &Session{
		Client:           client,
		Tools:            registry,
		ApprovalPrompter: prompter,
		WorkingDir:       root,
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use cwd"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want, err := filepath.EvalSymlinks(filepath.Join(root, "sub"))
	if err != nil {
		t.Fatal(err)
	}
	approvedRoot := root
	if len(prompter.requests) != 1 {
		t.Fatalf("approval requests = %d, want 1", len(prompter.requests))
	}
	if prompter.requests[0].WorkingDir != approvedRoot {
		t.Fatalf("approval cwd = %q, want %q", prompter.requests[0].WorkingDir, approvedRoot)
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
	if result.Messages[3].Content != approvedRoot {
		t.Fatalf("second tool saw cwd %q, want approved cwd %q", result.Messages[3].Content, approvedRoot)
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
	registry := &Registry{}
	registry.Register(cwdTestTool{})
	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
		WorkingDir:    root,
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use cwd"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want, err := filepath.EvalSymlinks(filepath.Dir(root))
	if err != nil {
		t.Fatal(err)
	}
	approvedRoot := root
	if session.WorkingDir != want {
		t.Fatalf("session cwd = %q, want %q", session.WorkingDir, want)
	}
	if result.Messages[2].Content != "escaped" {
		t.Fatalf("escape tool content = %q, want unchanged output", result.Messages[2].Content)
	}
	if result.Messages[3].Content != approvedRoot {
		t.Fatalf("second tool saw cwd %q, want original cwd %q", result.Messages[3].Content, approvedRoot)
	}
}

func TestSessionDeniesWithoutApprovalPrompter(t *testing.T) {
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
	registry := &Registry{}
	registry.Register(approvalTestTool{called: &called})
	registry.Register(staticTool{})
	session := &Session{
		Client: client,
		Tools:  registry,
	}

	result, err := session.Run(context.Background(), RunOptions{
		ChatID:      "chat-1",
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
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
		t.Fatalf("second denial tool message = %#v", result.Messages[3])
	}
	if result.Messages[3].Content == "" || result.Messages[3].Content == "tool says hello" {
		t.Fatalf("second denial content = %q", result.Messages[3].Content)
	}
}

func TestSessionPromptsOnceForApprovalBatch(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
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
							Name:      "approval_tool",
							Arguments: args,
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
	registry := &Registry{}
	registry.Register(approvalTestTool{called: &called})
	prompter := &recordingApprovalPrompter{
		results: []Approval{{Reason: "denied"}},
	}
	session := &Session{
		Client:           client,
		Tools:            registry,
		ApprovalPrompter: prompter,
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use tools"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(prompter.requests) != 1 {
		t.Fatalf("approval prompts = %d, want 1", len(prompter.requests))
	}
	if len(prompter.requests[0].Calls) != 2 {
		t.Fatalf("approval calls = %#v, want both tool calls", prompter.requests[0].Calls)
	}
	if called {
		t.Fatal("tool ran despite denied approval")
	}
	if client.calls != 1 {
		t.Fatalf("client calls = %d, want 1 after denial", client.calls)
	}
	if len(result.Messages) != 4 || result.Messages[2].Role != "tool" || result.Messages[3].Role != "tool" {
		t.Fatalf("messages = %#v", result.Messages)
	}
}

func TestSessionRunsFullApprovedToolBatchBeforeNextModelStep(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
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
							Name:      "approval_tool",
							Arguments: args,
						},
					},
				}}},
			},
			{
				{Message: api.Message{Role: "assistant", Content: "done"}},
			},
		},
	}
	registry := &Registry{}
	registry.Register(approvalTestTool{})
	prompter := &recordingApprovalPrompter{
		results: []Approval{{Allow: true}},
	}
	events := &recordingEventSink{}
	session := &Session{
		Client:           client,
		Tools:            registry,
		ApprovalPrompter: prompter,
		EventSinks:       []EventSink{events},
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use tools"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if client.calls != 2 {
		t.Fatalf("client calls = %d, want second model step only after tool batch", client.calls)
	}
	if len(result.Messages) != 5 {
		t.Fatalf("messages = %#v, want user, assistant tool calls, two tool results, final assistant", result.Messages)
	}
	if result.Messages[2].Role != "tool" || result.Messages[2].ToolCallID != "call-1" {
		t.Fatalf("first tool result = %#v", result.Messages[2])
	}
	if result.Messages[3].Role != "tool" || result.Messages[3].ToolCallID != "call-2" {
		t.Fatalf("second tool result = %#v", result.Messages[3])
	}
	if result.Messages[4].Role != "assistant" || result.Messages[4].Content != "done" {
		t.Fatalf("final assistant = %#v", result.Messages[4])
	}

	var finishedBeforeDelta []string
	for _, event := range events.events {
		if event.Type == EventMessageDelta && event.Content == "done" {
			break
		}
		if event.Type == EventToolFinished {
			finishedBeforeDelta = append(finishedBeforeDelta, event.ToolCallID)
		}
	}
	if strings.Join(finishedBeforeDelta, ",") != "call-1,call-2" {
		t.Fatalf("tool finishes before final model delta = %#v, want full batch before model", finishedBeforeDelta)
	}
}

func TestSessionAllowAllApprovalSkipsFuturePrompts(t *testing.T) {
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
			{
				{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-2",
					Function: api.ToolCallFunction{
						Name:      "approval_tool",
						Arguments: args,
					},
				}}}},
			},
			{
				{Message: api.Message{Role: "assistant", Content: "done again"}},
			},
		},
	}
	registry := &Registry{}
	registry.Register(approvalTestTool{})
	prompter := &recordingApprovalPrompter{
		results: []Approval{{AllowAll: true}},
	}
	session := &Session{
		Client:           client,
		Tools:            registry,
		ApprovalPrompter: prompter,
	}

	for range 2 {
		if _, err := session.Run(context.Background(), RunOptions{
			Model:       "model",
			NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	if !session.ApprovalState.AllowAll() {
		t.Fatal("session did not remember allow all")
	}
	if len(prompter.requests) != 1 {
		t.Fatalf("approval prompts = %d, want 1", len(prompter.requests))
	}
}

func TestSessionAllowToolApprovalSkipsFuturePromptForSameTool(t *testing.T) {
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
			{
				{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-2",
					Function: api.ToolCallFunction{
						Name:      "approval_tool",
						Arguments: args,
					},
				}}}},
			},
			{
				{Message: api.Message{Role: "assistant", Content: "done again"}},
			},
		},
	}
	registry := &Registry{}
	registry.Register(approvalTestTool{})
	prompter := &recordingApprovalPrompter{
		results: []Approval{{AllowScopes: []string{"approval_tool"}}},
	}
	session := &Session{
		Client:           client,
		Tools:            registry,
		ApprovalPrompter: prompter,
	}

	for range 2 {
		if _, err := session.Run(context.Background(), RunOptions{
			Model:       "model",
			NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	if session.ApprovalState.AllowAll() {
		t.Fatal("allowing one tool enabled full access")
	}
	if !session.ApprovalState.Allows("approval_tool") {
		t.Fatal("approval_tool scope was not saved")
	}
	if len(prompter.requests) != 1 {
		t.Fatalf("approval prompts = %d, want 1", len(prompter.requests))
	}
}

func TestSessionAllowShellApprovalScopesToExactCommand(t *testing.T) {
	pwdArgs := api.NewToolCallFunctionArguments()
	pwdArgs.Set("command", "pwd")
	lsArgs := api.NewToolCallFunctionArguments()
	lsArgs.Set("command", "ls")
	client := &fakeClient{
		responses: [][]api.ChatResponse{
			{
				{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-1",
					Function: api.ToolCallFunction{
						Name:      "bash",
						Arguments: pwdArgs,
					},
				}}}},
			},
			{
				{Message: api.Message{Role: "assistant", Content: "done"}},
			},
			{
				{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-2",
					Function: api.ToolCallFunction{
						Name:      "bash",
						Arguments: pwdArgs,
					},
				}}}},
			},
			{
				{Message: api.Message{Role: "assistant", Content: "done again"}},
			},
			{
				{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID: "call-3",
					Function: api.ToolCallFunction{
						Name:      "bash",
						Arguments: lsArgs,
					},
				}}}},
			},
			{
				{Message: api.Message{Role: "assistant", Content: "done finally"}},
			},
		},
	}
	registry := &Registry{}
	registry.Register(namedApprovalTestTool{name: "bash"})
	prompter := &recordingApprovalPrompter{
		results: []Approval{
			{AllowScopes: []string{toolApprovalScope("bash", map[string]any{"command": "pwd"})}},
			{Allow: true},
		},
	}
	session := &Session{
		Client:           client,
		Tools:            registry,
		ApprovalPrompter: prompter,
	}

	for range 3 {
		if _, err := session.Run(context.Background(), RunOptions{
			Model:       "model",
			NewMessages: []api.Message{{Role: "user", Content: "use a command"}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	if !session.ApprovalState.Allows("bash\x00pwd") {
		t.Fatal("pwd command scope was not saved")
	}
	if session.ApprovalState.Allows("bash") || session.ApprovalState.Allows("bash\x00ls") {
		t.Fatal("shell approval was too broad")
	}
	if len(prompter.requests) != 2 {
		t.Fatalf("approval prompts = %d, want first pwd and later ls", len(prompter.requests))
	}
	if got := prompter.requests[0].Calls[0].ApprovalScope; got != "bash\x00pwd" {
		t.Fatalf("first approval scope = %q, want pwd command scope", got)
	}
	if got := prompter.requests[1].Calls[0].ApprovalScope; got != "bash\x00ls" {
		t.Fatalf("second approval scope = %q, want ls command scope", got)
	}
}

func TestSessionAllowAllToolsExecutesApprovalTool(t *testing.T) {
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
	registry := &Registry{}
	registry.Register(approvalTestTool{called: &called})
	session := &Session{
		Client:        client,
		Tools:         registry,
		ApprovalState: approvalStateForTest(true, nil),
	}

	result, err := session.Run(context.Background(), RunOptions{
		Model:       "model",
		NewMessages: []api.Message{{Role: "user", Content: "use a tool"}},
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
