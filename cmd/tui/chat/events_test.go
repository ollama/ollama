package chat

import (
	"strings"
	"testing"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

func TestApplyAgentEventStreamsAssistantContent(t *testing.T) {
	m := chatModel{running: true}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "hello"})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: " world"})

	if len(m.entries) != 1 || m.entries[0].role != "assistant" || m.entries[0].content != "hello world" {
		t.Fatalf("entries = %#v", m.entries)
	}
	if len(m.liveMessages) != 1 || m.liveMessages[0].Content != "hello world" {
		t.Fatalf("live messages = %#v", m.liveMessages)
	}
}

func TestApplyAgentEventTracksToolLifecycle(t *testing.T) {
	m := chatModel{running: true}
	args := map[string]any{"command": "pwd"}

	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolStarted,
		ToolCallID: "call-1",
		ToolName:   "bash",
		Args:       args,
	})
	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolFinished,
		ToolCallID: "call-1",
		ToolName:   "bash",
		Args:       args,
		Content:    "ok",
	})

	if len(m.entries) != 1 {
		t.Fatalf("entries = %#v", m.entries)
	}
	entry := m.entries[0]
	if entry.status != "done" || entry.content != "ok" || !strings.Contains(entry.label, "Bash") {
		t.Fatalf("tool entry = %#v", entry)
	}
	if len(m.liveMessages) != 1 || m.liveMessages[0].Role != "tool" || m.liveMessages[0].Content != "ok" {
		t.Fatalf("live messages = %#v", m.liveMessages)
	}
}

func TestMessagesEndWithCompactionResult(t *testing.T) {
	messages := []api.Message{{
		Role:       "tool",
		ToolName:   coreagent.CompactionToolName,
		ToolCallID: coreagent.CompactionToolCallID,
		Content:    coreagent.CompactionSummaryMessagePrefix + "summary",
	}}
	if !messagesEndWithCompactionResult(messages) {
		t.Fatal("expected compaction result")
	}
}
