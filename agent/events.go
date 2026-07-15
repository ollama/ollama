package agent

import (
	"context"
	"errors"

	"github.com/ollama/ollama/api"
)

type EventType string

const (
	EventMessageDelta       EventType = "message_delta"
	EventThinkingDelta      EventType = "thinking_delta"
	EventToolCallDetected   EventType = "tool_call_detected"
	EventToolStarted        EventType = "tool_started"
	EventToolFinished       EventType = "tool_finished"
	EventCompactionStarted  EventType = "compaction_started"
	EventCompactionProgress EventType = "compaction_progress"
	EventCompacted          EventType = "compacted"
	EventCompactionSkipped  EventType = "compaction_skipped"
	EventRunFinished        EventType = "run_finished"
	EventError              EventType = "error"
)

type Event struct {
	Type       EventType      `json:"type"`
	RunID      string         `json:"runId,omitempty"`
	ChatID     string         `json:"chatId,omitempty"`
	Model      string         `json:"model,omitempty"`
	Status     string         `json:"status,omitempty"`
	ToolCallID string         `json:"toolCallId,omitempty"`
	ToolName   string         `json:"toolName,omitempty"`
	WorkingDir string         `json:"workingDir,omitempty"`
	Content    string         `json:"content,omitempty"`
	Thinking   string         `json:"thinking,omitempty"`
	ToolCalls  []api.ToolCall `json:"toolCalls,omitempty"`
	Messages   []api.Message  `json:"messages,omitempty"`
	Args       map[string]any `json:"args,omitempty"`
	Tokens     int            `json:"tokens,omitempty"`
	Error      string         `json:"error,omitempty"`
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

func (s *Session) emit(event Event) error {
	if s == nil {
		return nil
	}
	var errs []error
	for _, sink := range s.EventSinks {
		if sink == nil {
			continue
		}
		if err := sink.Emit(event); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (s *Session) emitIgnoringCanceled(ctx context.Context, event Event) error {
	err := s.emit(event)
	if err != nil && ctx != nil && ctx.Err() != nil {
		//nolint:nilerr // Event sinks may close during cancellation; cancellation is not a user-facing emit failure.
		return nil
	}
	return err
}
