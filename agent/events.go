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

// eventMetadata carries the run identification fields shared by all events.
type eventMetadata struct {
	runID  string
	chatID string
	model  string
}

func newEventMetadata(runID string, opts RunOptions) eventMetadata {
	return eventMetadata{runID: runID, chatID: opts.ChatID, model: opts.Model}
}

func newMessageDelta(m eventMetadata, content string) Event {
	return Event{Type: EventMessageDelta, RunID: m.runID, ChatID: m.chatID, Model: m.model, Content: content}
}

func newThinkingDelta(m eventMetadata, thinking string) Event {
	return Event{Type: EventThinkingDelta, RunID: m.runID, ChatID: m.chatID, Model: m.model, Thinking: thinking}
}

func newToolCallDetected(m eventMetadata, calls []api.ToolCall) Event {
	return Event{Type: EventToolCallDetected, RunID: m.runID, ChatID: m.chatID, Model: m.model, ToolCalls: calls}
}

func newToolStarted(m eventMetadata, callID, toolName, workingDir string, args map[string]any) Event {
	return Event{Type: EventToolStarted, RunID: m.runID, ChatID: m.chatID, Model: m.model, Status: "running", ToolCallID: callID, ToolName: toolName, WorkingDir: workingDir, Args: args}
}

func newToolFinished(m eventMetadata, status, callID, toolName, workingDir string, args map[string]any, content, errMsg string) Event {
	ev := Event{Type: EventToolFinished, RunID: m.runID, ChatID: m.chatID, Model: m.model, Status: status, ToolCallID: callID, ToolName: toolName, WorkingDir: workingDir, Args: args, Content: content}
	if errMsg != "" {
		ev.Error = errMsg
	}
	return ev
}

func newRunFinished(m eventMetadata, status string) Event {
	return Event{Type: EventRunFinished, RunID: m.runID, ChatID: m.chatID, Model: m.model, Status: status}
}

func newErrorEvent(m eventMetadata, errMsg string) Event {
	return Event{Type: EventError, RunID: m.runID, ChatID: m.chatID, Model: m.model, Error: errMsg}
}

func newCompactionProgress(m eventMetadata, tokens int) Event {
	return Event{Type: EventCompactionProgress, RunID: m.runID, ChatID: m.chatID, Model: m.model, Tokens: tokens}
}

func newCompactionStarted(m eventMetadata, status string) Event {
	return Event{Type: EventCompactionStarted, RunID: m.runID, ChatID: m.chatID, Model: m.model, Status: status}
}

func newCompactionSkipped(m eventMetadata, status, content string) Event {
	return Event{Type: EventCompactionSkipped, RunID: m.runID, ChatID: m.chatID, Model: m.model, Status: status, Content: content}
}

func newCompacted(m eventMetadata, messages []api.Message, status, content string) Event {
	return Event{Type: EventCompacted, RunID: m.runID, ChatID: m.chatID, Model: m.model, Status: status, Content: content, Messages: messages}
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
