package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/agent"
)

func TestTraceSinkCapturesToolResultsWithoutThinkingOrSecrets(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trace.jsonl")
	trace, err := newTraceSink(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := trace.Emit(agent.Event{
		Type:     agent.EventToolFinished,
		ToolName: "search_text",
		Content:  "untrusted tool output",
		Thinking: "private reasoning",
		Args:     map[string]any{"query": "listen", "api_token": "secret"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := trace.Close(); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "untrusted tool output") || strings.Contains(string(data), "private reasoning") || strings.Contains(string(data), "secret") {
		t.Fatalf("unexpected trace content: %s", data)
	}
	var record traceRecord
	if err := json.Unmarshal(data, &record); err != nil {
		t.Fatal(err)
	}
	if record.Type != "tool_result" || record.OutputPreview == "" || record.Args["api_token"] != "[redacted]" {
		t.Fatalf("trace record = %#v", record)
	}
}

func TestTraceSinkCapturesTerminalAssistantCompletion(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trace.jsonl")
	trace, err := newTraceSink(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := trace.Emit(agent.Event{Type: agent.EventMessageDelta, RunID: "run", Model: "glm-5.2", Content: "I found no actionable issues."}); err != nil {
		t.Fatal(err)
	}
	if err := trace.Emit(agent.Event{Type: agent.EventRunFinished, RunID: "run", Model: "glm-5.2", Status: agent.RunStatusDone}); err != nil {
		t.Fatal(err)
	}
	if err := trace.Close(); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"type":"assistant_completion"`) || !strings.Contains(string(data), "I found no actionable issues.") {
		t.Fatalf("terminal completion missing from trace: %s", data)
	}
}

func TestTraceSinkMarksEmptyTerminalAssistantCompletion(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trace.jsonl")
	trace, err := newTraceSink(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := trace.Emit(agent.Event{Type: agent.EventRunFinished, RunID: "run", Model: "glm-5.2", Status: agent.RunStatusDone}); err != nil {
		t.Fatal(err)
	}
	if err := trace.Close(); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"type":"assistant_completion"`) || !strings.Contains(string(data), "[empty]") {
		t.Fatalf("empty terminal completion missing from trace: %s", data)
	}
}
