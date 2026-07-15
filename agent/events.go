package agent

import (
	"context"

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

// ToolStatus is the typed lifecycle state for a tool call, carried on
// Event.ToolStatus for tool events.
type ToolStatus string

const (
	ToolStatusRunning  ToolStatus = "running"
	ToolStatusDone     ToolStatus = "done"
	ToolStatusFailed   ToolStatus = "failed"
	ToolStatusDenied   ToolStatus = "denied"
	ToolStatusDisabled ToolStatus = "disabled"
	ToolStatusSkipped  ToolStatus = "skipped"
)

// RunStatus is the typed terminal outcome of a run, carried on Event.Status for
// run_finished events.
type RunStatus string

const (
	RunStatusDone     RunStatus = "done"
	RunStatusDenied   RunStatus = "denied"
	RunStatusCanceled RunStatus = "canceled"
)

// CompactionTrigger is the typed reason a compaction ran or was attempted,
// carried on Event.CompactionTrigger for compaction events.
type CompactionTrigger string

const (
	CompactionTriggerForce      CompactionTrigger = "force"
	CompactionTriggerPromptEval CompactionTrigger = "prompt_eval"
	CompactionTriggerEstimate   CompactionTrigger = "estimate"
	CompactionTriggerToolOutput CompactionTrigger = "tool_output"
	CompactionTriggerError      CompactionTrigger = "error"
	CompactionTriggerDue        CompactionTrigger = "due"
)

type Event struct {
	Type              EventType         `json:"type"`
	RunID             string            `json:"runId,omitempty"`
	ChatID            string            `json:"chatId,omitempty"`
	Model             string            `json:"model,omitempty"`
	Status            RunStatus         `json:"status,omitempty"`
	ToolStatus        ToolStatus        `json:"toolStatus,omitempty"`
	CompactionTrigger CompactionTrigger `json:"compactionTrigger,omitempty"`
	ToolCallID        string            `json:"toolCallId,omitempty"`
	ToolName          string            `json:"toolName,omitempty"`
	WorkingDir        string            `json:"workingDir,omitempty"`
	Content           string            `json:"content,omitempty"`
	Thinking          string            `json:"thinking,omitempty"`
	ToolCalls         []api.ToolCall    `json:"toolCalls,omitempty"`
	Messages          []api.Message     `json:"messages,omitempty"`
	Args              map[string]any    `json:"args,omitempty"`
	Tokens            int               `json:"tokens,omitempty"`
	Error             string            `json:"error,omitempty"`
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
	var firstErr error
	for _, sink := range s.EventSinks {
		if sink == nil {
			continue
		}
		if err := sink.Emit(event); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func (s *Session) emitIgnoringCanceled(ctx context.Context, event Event) error {
	err := s.emit(event)
	if err != nil && ctx != nil && ctx.Err() != nil {
		//nolint:nilerr // Event sinks may close during cancellation; cancellation is not a user-facing emit failure.
		return nil
	}
	return err
}
