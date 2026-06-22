package agent

import (
	"context"
	"time"

	"github.com/ollama/ollama/api"
)

type EventType string

const (
	EventRunStarted         EventType = "run_started"
	EventMessageStarted     EventType = "message_started"
	EventMessageDelta       EventType = "message_delta"
	EventThinkingDelta      EventType = "thinking_delta"
	EventToolCallDetected   EventType = "tool_call_detected"
	EventToolStarted        EventType = "tool_started"
	EventToolFinished       EventType = "tool_finished"
	EventToolsUnavailable   EventType = "tools_unavailable"
	EventCompactionStarted  EventType = "compaction_started"
	EventCompactionProgress EventType = "compaction_progress"
	EventCompacted          EventType = "compacted"
	EventCompactionSkipped  EventType = "compaction_skipped"
	EventLoopStep           EventType = "loop_step"
	EventRequestBuilt       EventType = "request_built"
	EventModelStreamDone    EventType = "model_stream_done"
	EventRunFinished        EventType = "run_finished"
	EventError              EventType = "error"
)

type Event struct {
	Type                      EventType         `json:"type"`
	RunID                     string            `json:"runId,omitempty"`
	ChatID                    string            `json:"chatId,omitempty"`
	MessageID                 string            `json:"messageId,omitempty"`
	Model                     string            `json:"model,omitempty"`
	Status                    string            `json:"status,omitempty"`
	ToolCallID                string            `json:"toolCallId,omitempty"`
	ToolName                  string            `json:"toolName,omitempty"`
	WorkingDir                string            `json:"workingDir,omitempty"`
	Content                   string            `json:"content,omitempty"`
	Thinking                  string            `json:"thinking,omitempty"`
	ToolCalls                 []api.ToolCall    `json:"toolCalls,omitempty"`
	Messages                  []api.Message     `json:"messages,omitempty"`
	Args                      map[string]any    `json:"args,omitempty"`
	Tokens                    int               `json:"tokens,omitempty"`
	PromptTokens              int               `json:"promptTokens,omitempty"`
	ContextWindowTokens       int               `json:"contextWindowTokens,omitempty"`
	CompactionThresholdTokens int               `json:"compactionThresholdTokens,omitempty"`
	MessageCount              int               `json:"messageCount,omitempty"`
	UserMessageCount          int               `json:"userMessageCount,omitempty"`
	AssistantMessageCount     int               `json:"assistantMessageCount,omitempty"`
	ToolMessageCount          int               `json:"toolMessageCount,omitempty"`
	SystemMessageCount        int               `json:"systemMessageCount,omitempty"`
	ToolCallCount             int               `json:"toolCallCount,omitempty"`
	ToolCount                 int               `json:"toolCount,omitempty"`
	ToolNames                 []string          `json:"toolNames,omitempty"`
	ToolRound                 int               `json:"toolRound,omitempty"`
	ToolRoundLimit            int               `json:"toolRoundLimit,omitempty"`
	Error                     string            `json:"error,omitempty"`
	StartedAt                 time.Time         `json:"startedAt,omitempty"`
	FinishedAt                time.Time         `json:"finishedAt,omitempty"`
	Response                  *api.ChatResponse `json:"-"`
}

type EventSink interface {
	Emit(Event) error
}

type MultiEventSink []EventSink

func (s MultiEventSink) Emit(event Event) error {
	var firstErr error
	for _, sink := range s {
		if sink == nil {
			continue
		}
		if err := sink.Emit(event); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

type EventSinkFunc func(Event) error

func (fn EventSinkFunc) Emit(event Event) error {
	if fn == nil {
		return nil
	}
	return fn(event)
}

func emit(sink EventSink, event Event) error {
	if sink == nil {
		return nil
	}
	return sink.Emit(event)
}

func emitIgnoringCanceled(ctx context.Context, sink EventSink, event Event) error {
	err := emit(sink, event)
	if err != nil && ctx != nil && ctx.Err() != nil {
		//nolint:nilerr // Event sinks may close during cancellation; cancellation is not a user-facing emit failure.
		return nil
	}
	return err
}
