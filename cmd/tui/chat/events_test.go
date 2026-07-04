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
	if line := stripANSI(toolStatusLine(entry)); line != `Bash("pwd")` {
		t.Fatalf("tool status line = %q, want command label", line)
	}
	if len(m.liveMessages) != 1 || m.liveMessages[0].Role != "tool" || m.liveMessages[0].Content != "ok" {
		t.Fatalf("live messages = %#v", m.liveMessages)
	}
}

func TestApplyAgentEventRendersDeniedCommandAsDenied(t *testing.T) {
	m := chatModel{running: true}
	args := map[string]any{"command": "pwd"}

	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolFinished,
		Status:     "denied",
		ToolCallID: "call-1",
		ToolName:   "bash",
		Args:       args,
		Content:    "Tool execution denied.",
		Error:      "Tool execution denied.",
	})

	if len(m.entries) != 1 {
		t.Fatalf("entries = %#v", m.entries)
	}
	entry := m.entries[0]
	if entry.status != "denied" {
		t.Fatalf("tool status = %q, want denied: %#v", entry.status, entry)
	}
	if line := stripANSI(toolStatusLine(entry)); line != `Bash("pwd") denied` {
		t.Fatalf("tool status line = %q, want denied command label", line)
	}
}

func TestApplyAgentEventShowsWorkingWhileAwaitingCloudToolStart(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("command", "pwd")
	m := chatModel{
		running: true,
	}

	m.applyAgentEvent(coreagent.Event{
		Type: coreagent.EventToolCallDetected,
		ToolCalls: []api.ToolCall{{
			ID: "call-1",
			Function: api.ToolCallFunction{
				Name:      "bash",
				Arguments: args,
			},
		}},
	})

	if line := stripANSI(m.activityLine()); !strings.Contains(line, "Working") {
		t.Fatalf("activityLine = %q, want Working while tool call is pending", line)
	}
}

func TestActivityLineShowsWorkingWhileAwaitingModelBeforeFirstEvent(t *testing.T) {
	m := chatModel{
		running:       true,
		awaitingModel: true,
		spinner:       0,
	}

	if line := stripANSI(m.activityLine()); !strings.Contains(line, "Working") {
		t.Fatalf("activityLine = %q, want Working while stream is open before first event", line)
	}
}

func TestApplyAgentEventCountsHiddenDetectedCommandsInGroupSummary(t *testing.T) {
	firstArgs := api.NewToolCallFunctionArguments()
	firstArgs.Set("command", "pwd")
	secondArgs := api.NewToolCallFunctionArguments()
	secondArgs.Set("command", "ls")
	m := chatModel{
		running: true,
		spinner: waitingSpinnerTicks,
	}

	m.applyAgentEvent(coreagent.Event{
		Type: coreagent.EventToolCallDetected,
		ToolCalls: []api.ToolCall{
			{ID: "call-1", Function: api.ToolCallFunction{Name: "bash", Arguments: firstArgs}},
			{ID: "call-2", Function: api.ToolCallFunction{Name: "bash", Arguments: secondArgs}},
		},
	})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-1", ToolName: "bash", Args: firstArgs.ToMap()})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-1", ToolName: "bash", Args: firstArgs.ToMap(), Content: "one"})

	if len(m.entries) != 1 {
		t.Fatalf("entries = %d, want one grouped entry: %#v", len(m.entries), m.entries)
	}
	if m.entries[0].role != "tool_group" || len(m.entries[0].tools) != 1 {
		t.Fatalf("visible history should keep only the finished child: %#v", m.entries[0])
	}
	if line := stripANSI(toolGroupStatusLine(m.entries[0])); line != "Ran 2 commands" {
		t.Fatalf("grouped command line = %q", line)
	}
	if line := stripANSI(m.activityLine()); !strings.Contains(line, "Working") {
		t.Fatalf("activityLine = %q, want Working while second command is pending", line)
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-2", ToolName: "bash", Args: secondArgs.ToMap()})
	if len(m.entries) != 2 {
		t.Fatalf("entries after second start = %d, want finished command plus running command: %#v", len(m.entries), m.entries)
	}
	if line := stripANSI(toolStatusLine(m.entries[0])); line != `Bash("pwd")` {
		t.Fatalf("finished command line after second start = %q", line)
	}
	if line := stripANSI(toolStatusLine(m.entries[1])); line != `Bash("ls")` {
		t.Fatalf("running command line = %q", line)
	}
}

func TestApplyAgentEventGroupsCompletedCommandsImmediately(t *testing.T) {
	m := chatModel{running: true}
	firstArgs := map[string]any{"command": "pwd"}
	secondArgs := map[string]any{"command": "ls"}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-1", ToolName: "bash", Args: firstArgs})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-1", ToolName: "bash", Args: firstArgs, Content: "one"})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-2", ToolName: "bash", Args: secondArgs})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-2", ToolName: "bash", Args: secondArgs, Content: "two"})

	if len(m.entries) != 1 {
		t.Fatalf("entries = %d, want one grouped command entry: %#v", len(m.entries), m.entries)
	}
	if m.entries[0].role != "tool_group" || len(m.entries[0].tools) != 2 {
		t.Fatalf("completed commands should be grouped immediately: %#v", m.entries[0])
	}
	if line := stripANSI(toolGroupStatusLine(m.entries[0])); line != "Ran 2 commands" {
		t.Fatalf("grouped command line = %q", line)
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "done"})
	transcript := stripANSI(m.renderTranscript(100))
	if !strings.Contains(transcript, "• Ran 2 commands\n\n  done") {
		t.Fatalf("grouped command should be visually separated from assistant content:\n%s", transcript)
	}
}

func TestApplyAgentEventGroupsDeniedCommandsAsDenied(t *testing.T) {
	m := chatModel{running: true}
	firstArgs := map[string]any{"command": "pwd"}
	secondArgs := map[string]any{"command": "ls"}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, Status: "denied", ToolCallID: "call-1", ToolName: "bash", Args: firstArgs, Content: "Tool execution denied.", Error: "Tool execution denied."})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, Status: "denied", ToolCallID: "call-2", ToolName: "bash", Args: secondArgs, Content: "Tool execution denied.", Error: "Tool execution denied."})

	if len(m.entries) != 1 {
		t.Fatalf("entries = %d, want one grouped command entry: %#v", len(m.entries), m.entries)
	}
	if m.entries[0].role != "tool_group" || len(m.entries[0].tools) != 2 {
		t.Fatalf("denied commands should be grouped immediately: %#v", m.entries[0])
	}
	if line := stripANSI(toolGroupStatusLine(m.entries[0])); line != "Denied 2 commands" {
		t.Fatalf("grouped command line = %q", line)
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
