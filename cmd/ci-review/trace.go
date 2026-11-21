package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/agent"
)

type traceSink struct {
	mu        sync.Mutex
	file      *os.File
	encoder   *json.Encoder
	assistant map[string]string
}

type traceRecord struct {
	Timestamp         time.Time      `json:"timestamp"`
	Type              string         `json:"type"`
	RunID             string         `json:"run_id,omitempty"`
	Model             string         `json:"model,omitempty"`
	Status            string         `json:"status,omitempty"`
	ToolStatus        string         `json:"tool_status,omitempty"`
	ToolCallID        string         `json:"tool_call_id,omitempty"`
	ToolName          string         `json:"tool_name,omitempty"`
	Args              map[string]any `json:"args,omitempty"`
	OutputPreview     string         `json:"output_preview,omitempty"`
	Tokens            int            `json:"tokens,omitempty"`
	CompactionTrigger string         `json:"compaction_trigger,omitempty"`
	Error             string         `json:"error,omitempty"`
}

func newTraceSink(path string) (*traceSink, error) {
	if path == "" {
		return nil, nil
	}
	file, err := os.OpenFile(filepath.Clean(path), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	return &traceSink{file: file, encoder: json.NewEncoder(file), assistant: make(map[string]string)}, nil
}

func (t *traceSink) Close() error {
	if t == nil || t.file == nil {
		return nil
	}
	return t.file.Close()
}

func (t *traceSink) Emit(event agent.Event) error {
	if t == nil {
		return nil
	}
	if event.Type == agent.EventThinkingDelta {
		return nil
	}
	if event.Type == agent.EventMessageDelta {
		t.appendAssistant(event.RunID, event.Content)
		return nil
	}
	if event.Type == agent.EventToolCallDetected {
		return t.flushAssistant(event, "assistant_message", false)
	}
	if event.Type == agent.EventRunFinished {
		if err := t.flushAssistant(event, "assistant_completion", true); err != nil {
			return err
		}
	}
	typeName := string(event.Type)
	if event.Type == agent.EventToolStarted {
		typeName = "tool_call"
	}
	if event.Type == agent.EventToolFinished {
		typeName = "tool_result"
		if event.ToolName == "final_review" {
			if event.ToolStatus == agent.ToolStatusDone {
				typeName = "final_review_accepted"
			} else {
				typeName = "final_review_rejected"
			}
		}
	}
	record := traceRecord{
		Timestamp:         time.Now().UTC(),
		Type:              typeName,
		RunID:             event.RunID,
		Model:             event.Model,
		Status:            string(event.Status),
		ToolStatus:        string(event.ToolStatus),
		ToolCallID:        event.ToolCallID,
		ToolName:          event.ToolName,
		Args:              sanitizeTraceArgs(event.Args),
		OutputPreview:     tracePreview(event.Content),
		Tokens:            event.Tokens,
		CompactionTrigger: string(event.CompactionTrigger),
		Error:             redactTraceText(event.Error),
	}
	return t.write(record)
}

func (t *traceSink) Record(kind string, values map[string]any) error {
	if t == nil {
		return nil
	}
	return t.write(traceRecord{Timestamp: time.Now().UTC(), Type: kind, Args: sanitizeTraceArgs(values)})
}

func (t *traceSink) write(record traceRecord) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.writeLocked(record)
}

func (t *traceSink) writeLocked(record traceRecord) error {
	if err := t.encoder.Encode(record); err != nil {
		return fmt.Errorf("write trace: %w", err)
	}
	return nil
}

func (t *traceSink) appendAssistant(runID, content string) {
	if content == "" {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.assistant[runID] += content
}

func (t *traceSink) flushAssistant(event agent.Event, kind string, includeEmpty bool) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	content := t.assistant[event.RunID]
	delete(t.assistant, event.RunID)
	if strings.TrimSpace(content) == "" && !includeEmpty {
		return nil
	}
	if strings.TrimSpace(content) == "" {
		content = "[empty]"
	}
	return t.writeLocked(traceRecord{
		Timestamp:     time.Now().UTC(),
		Type:          kind,
		RunID:         event.RunID,
		Model:         event.Model,
		OutputPreview: tracePreview(content),
	})
}

func sanitizeTraceArgs(values map[string]any) map[string]any {
	if len(values) == 0 {
		return nil
	}
	result := make(map[string]any, len(values))
	for key, value := range values {
		if secretTraceKey(key) {
			result[key] = "[redacted]"
			continue
		}
		result[key] = sanitizeTraceValue(value)
	}
	return result
}

func sanitizeTraceValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return sanitizeTraceArgs(value)
	case []any:
		result := make([]any, len(value))
		for i, item := range value {
			result[i] = sanitizeTraceValue(item)
		}
		return result
	case []map[string]any:
		result := make([]map[string]any, len(value))
		for i, item := range value {
			result[i] = sanitizeTraceArgs(item)
		}
		return result
	case string:
		return redactTraceText(value)
	default:
		return value
	}
}

func secretTraceKey(key string) bool {
	key = strings.ToLower(key)
	return strings.Contains(key, "token") || strings.Contains(key, "secret") || strings.Contains(key, "password") || strings.Contains(key, "api_key")
}

func redactTraceText(value string) string {
	if strings.Contains(strings.ToLower(value), "authorization: bearer") || strings.Contains(strings.ToLower(value), "ollama_api_key") || strings.Contains(strings.ToLower(value), "private key") {
		return "[redacted]"
	}
	return value
}

func tracePreview(value string) string {
	value = redactTraceText(value)
	const limit = 1200
	if len(value) <= limit {
		return value
	}
	return value[:limit] + "\n[truncated]"
}
