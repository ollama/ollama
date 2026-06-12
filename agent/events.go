package agent

import (
	"time"

	"github.com/ollama/ollama/api"
)

type EventType string

const (
	EventRunStarted        EventType = "run_started"
	EventMessageStarted    EventType = "message_started"
	EventMessageDelta      EventType = "message_delta"
	EventThinkingDelta     EventType = "thinking_delta"
	EventToolCallDetected  EventType = "tool_call_detected"
	EventToolStarted       EventType = "tool_started"
	EventToolFinished      EventType = "tool_finished"
	EventToolsUnavailable  EventType = "tools_unavailable"
	EventCompacted         EventType = "compacted"
	EventCompactionSkipped EventType = "compaction_skipped"
	EventRunFinished       EventType = "run_finished"
	EventError             EventType = "error"
)

type Event struct {
	Type       EventType         `json:"type"`
	RunID      string            `json:"runId,omitempty"`
	ChatID     string            `json:"chatId,omitempty"`
	MessageID  string            `json:"messageId,omitempty"`
	ToolCallID string            `json:"toolCallId,omitempty"`
	ToolName   string            `json:"toolName,omitempty"`
	WorkingDir string            `json:"workingDir,omitempty"`
	Content    string            `json:"content,omitempty"`
	Thinking   string            `json:"thinking,omitempty"`
	ToolCalls  []api.ToolCall    `json:"toolCalls,omitempty"`
	Messages   []api.Message     `json:"messages,omitempty"`
	Args       map[string]any    `json:"args,omitempty"`
	Error      string            `json:"error,omitempty"`
	StartedAt  time.Time         `json:"startedAt,omitempty"`
	FinishedAt time.Time         `json:"finishedAt,omitempty"`
	Response   *api.ChatResponse `json:"-"`
}

type EventSink interface {
	Emit(Event) error
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
