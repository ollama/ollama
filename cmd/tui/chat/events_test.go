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
		ToolStatus: coreagent.ToolStatusDenied,
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

func TestActivityLineShowsWorkingAfterAssistantContentGoesIdle(t *testing.T) {
	m := chatModel{
		running: true,
		spinner: idleWorkingDelayTicks,
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "I will inspect that next."})
	if line := strings.TrimSpace(stripANSI(m.activityLine())); line != "" {
		t.Fatalf("activityLine immediately after content = %q, want quiet until the idle delay", line)
	}

	m.spinner = idleWorkingDelayTicks
	if line := stripANSI(m.activityLine()); !strings.Contains(line, "Working") {
		t.Fatalf("activityLine after idle content stream = %q, want Working while stream remains open", line)
	}
}

func TestApplyAgentEventKeepsDetectedBatchStableUntilComplete(t *testing.T) {
	firstArgs := api.NewToolCallFunctionArguments()
	firstArgs.Set("command", "pwd")
	secondArgs := api.NewToolCallFunctionArguments()
	secondArgs.Set("command", "ls")
	m := chatModel{
		running: true,
		spinner: idleWorkingDelayTicks,
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
		t.Fatalf("entries = %d, want first completed command row: %#v", len(m.entries), m.entries)
	}
	if m.entries[0].role != "tool" || m.entries[0].status != "done" {
		t.Fatalf("first command should remain stable while second is pending: %#v", m.entries[0])
	}
	if line := stripANSI(toolStatusLine(m.entries[0])); line != `Bash("pwd")` {
		t.Fatalf("completed command line = %q", line)
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

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-2", ToolName: "bash", Args: secondArgs.ToMap(), Content: "two"})
	if line := stripANSI(m.activityLine()); !strings.Contains(line, "Working") {
		t.Fatalf("activityLine after completed batch = %q, want Working while waiting for next model response", line)
	}
	if len(m.entries) != 2 {
		t.Fatalf("entries after batch completion = %d, want stable command rows until the next tool boundary: %#v", len(m.entries), m.entries)
	}
	for i, want := range []string{`Bash("pwd")`, `Bash("ls")`} {
		if line := stripANSI(toolStatusLine(m.entries[i])); line != want {
			t.Fatalf("completed command row %d = %q, want %q", i, line, want)
		}
	}

	thirdArgs := api.NewToolCallFunctionArguments()
	thirdArgs.Set("command", "date")
	m.applyAgentEvent(coreagent.Event{
		Type: coreagent.EventToolCallDetected,
		ToolCalls: []api.ToolCall{
			{ID: "call-3", Function: api.ToolCallFunction{Name: "bash", Arguments: thirdArgs}},
		},
	})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-3", ToolName: "bash", Args: thirdArgs.ToMap()})

	if len(m.entries) != 2 {
		t.Fatalf("entries after next tool boundary = %d, want grouped history plus running command: %#v", len(m.entries), m.entries)
	}
	if m.entries[0].role != "tool_group" || len(m.entries[0].tools) != 2 {
		t.Fatalf("completed detected batch should collapse at the next tool boundary: %#v", m.entries[0])
	}
	if line := stripANSI(toolGroupStatusLine(m.entries[0])); line != "Ran 2 commands" {
		t.Fatalf("grouped command line = %q", line)
	}
	if line := stripANSI(toolStatusLine(m.entries[1])); line != `Bash("date")` {
		t.Fatalf("running command line = %q", line)
	}
}

func TestApplyAgentEventDoesNotCollapsePartialDetectedBatch(t *testing.T) {
	firstArgs := api.NewToolCallFunctionArguments()
	firstArgs.Set("command", "pwd")
	secondArgs := api.NewToolCallFunctionArguments()
	secondArgs.Set("command", "ls")
	thirdArgs := api.NewToolCallFunctionArguments()
	thirdArgs.Set("command", "date")
	m := chatModel{running: true}

	m.applyAgentEvent(coreagent.Event{
		Type: coreagent.EventToolCallDetected,
		ToolCalls: []api.ToolCall{
			{ID: "call-1", Function: api.ToolCallFunction{Name: "bash", Arguments: firstArgs}},
			{ID: "call-2", Function: api.ToolCallFunction{Name: "bash", Arguments: secondArgs}},
			{ID: "call-3", Function: api.ToolCallFunction{Name: "bash", Arguments: thirdArgs}},
		},
	})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-1", ToolName: "bash", Args: firstArgs.ToMap()})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-1", ToolName: "bash", Args: firstArgs.ToMap(), Content: "one"})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-2", ToolName: "bash", Args: secondArgs.ToMap()})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-2", ToolName: "bash", Args: secondArgs.ToMap(), Content: "two"})

	if line := stripANSI(m.activityLine()); !strings.Contains(line, "Working") {
		t.Fatalf("activityLine before final detected call = %q, want Working while final tool is pending", line)
	}
	if len(m.entries) != 2 {
		t.Fatalf("entries before final detected call = %d, want two stable rows: %#v", len(m.entries), m.entries)
	}
	for i, want := range []string{`Bash("pwd")`, `Bash("ls")`} {
		if line := stripANSI(toolStatusLine(m.entries[i])); line != want {
			t.Fatalf("tool row %d = %q, want %q", i, line, want)
		}
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-3", ToolName: "bash", Args: thirdArgs.ToMap()})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-3", ToolName: "bash", Args: thirdArgs.ToMap(), Content: "three"})

	if len(m.entries) != 3 {
		t.Fatalf("entries after full detected batch = %#v, want stable tool rows until the next tool boundary", m.entries)
	}
	for i, want := range []string{`Bash("pwd")`, `Bash("ls")`, `Bash("date")`} {
		if line := stripANSI(toolStatusLine(m.entries[i])); line != want {
			t.Fatalf("tool row %d = %q, want %q", i, line, want)
		}
	}

	fourthArgs := api.NewToolCallFunctionArguments()
	fourthArgs.Set("command", "whoami")
	m.applyAgentEvent(coreagent.Event{
		Type: coreagent.EventToolCallDetected,
		ToolCalls: []api.ToolCall{
			{ID: "call-4", Function: api.ToolCallFunction{Name: "bash", Arguments: fourthArgs}},
		},
	})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-4", ToolName: "bash", Args: fourthArgs.ToMap()})

	if len(m.entries) != 2 || m.entries[0].role != "tool_group" || len(m.entries[0].tools) != 3 {
		t.Fatalf("entries after next detected batch starts = %#v, want one grouped history entry plus active tool", m.entries)
	}
	if line := stripANSI(toolGroupStatusLine(m.entries[0])); line != "Ran 3 commands" {
		t.Fatalf("grouped command line = %q", line)
	}
}

func TestApplyAgentEventGroupsCompletedCommandsAtNextToolBoundary(t *testing.T) {
	m := chatModel{running: true}
	firstArgs := map[string]any{"command": "pwd"}
	secondArgs := map[string]any{"command": "ls"}
	thirdArgs := map[string]any{"command": "date"}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-1", ToolName: "bash", Args: firstArgs})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-1", ToolName: "bash", Args: firstArgs, Content: "one"})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-2", ToolName: "bash", Args: secondArgs})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-2", ToolName: "bash", Args: secondArgs, Content: "two"})

	if len(m.entries) != 2 {
		t.Fatalf("entries after second finish = %d, want two stable command rows: %#v", len(m.entries), m.entries)
	}
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-3", ToolName: "bash", Args: thirdArgs})

	if len(m.entries) != 2 {
		t.Fatalf("entries = %d, want grouped command history plus active command: %#v", len(m.entries), m.entries)
	}
	if m.entries[0].role != "tool_group" || len(m.entries[0].tools) != 2 {
		t.Fatalf("completed commands should be grouped when the next command starts: %#v", m.entries[0])
	}
	if line := stripANSI(toolGroupStatusLine(m.entries[0])); line != "Ran 2 commands" {
		t.Fatalf("grouped command line = %q", line)
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-3", ToolName: "bash", Args: thirdArgs, Content: "three"})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "done"})

	if len(m.entries) != 3 {
		t.Fatalf("entries after assistant content = %d, want grouped history, last command, assistant: %#v", len(m.entries), m.entries)
	}
	transcript := stripANSI(m.renderTranscript(100))
	if !strings.Contains(transcript, "• Ran 2 commands\n\n• Bash(\"date\")\n\n  done") {
		t.Fatalf("tool history should stay visually separated from assistant content:\n%s", transcript)
	}
}

func TestApplyAgentEventDoesNotGroupCompletedCommandsOnMessageDelta(t *testing.T) {
	m := chatModel{running: true}
	firstArgs := map[string]any{"command": "pwd"}
	secondArgs := map[string]any{"command": "ls"}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-1", ToolName: "bash", Args: firstArgs})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-1", ToolName: "bash", Args: firstArgs, Content: "one"})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-2", ToolName: "bash", Args: secondArgs})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-2", ToolName: "bash", Args: secondArgs, Content: "two"})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "done"})

	if len(m.entries) != 3 {
		t.Fatalf("entries = %d, want two command rows plus assistant content: %#v", len(m.entries), m.entries)
	}
	if m.entries[0].role != "tool" || m.entries[1].role != "tool" || m.entries[2].role != "assistant" {
		t.Fatalf("completed commands should not collapse on assistant content: %#v", m.entries)
	}
}

func TestApplyAgentEventGroupsPreviouslyDeniedCommandsAtNextToolBoundary(t *testing.T) {
	m := chatModel{running: true}
	firstArgs := map[string]any{"command": "pwd"}
	secondArgs := map[string]any{"command": "ls"}
	thirdArgs := map[string]any{"command": "date"}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolStatus: coreagent.ToolStatusDenied, ToolCallID: "call-1", ToolName: "bash", Args: firstArgs, Content: "Tool execution denied.", Error: "Tool execution denied."})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolStatus: coreagent.ToolStatusDenied, ToolCallID: "call-2", ToolName: "bash", Args: secondArgs, Content: "Tool execution denied.", Error: "Tool execution denied."})

	if len(m.entries) != 2 {
		t.Fatalf("entries = %d, want two stable denied command rows: %#v", len(m.entries), m.entries)
	}
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-3", ToolName: "bash", Args: thirdArgs})

	if len(m.entries) != 2 {
		t.Fatalf("entries = %d, want grouped denied command entry plus active command: %#v", len(m.entries), m.entries)
	}
	if m.entries[0].role != "tool_group" || len(m.entries[0].tools) != 2 {
		t.Fatalf("denied commands should be grouped at the next tool boundary: %#v", m.entries[0])
	}
	if line := stripANSI(toolGroupStatusLine(m.entries[0])); line != "Denied 2 commands" {
		t.Fatalf("grouped command line = %q", line)
	}
	if line := stripANSI(toolStatusLine(m.entries[1])); line != `Bash("date")` {
		t.Fatalf("running command line = %q", line)
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
