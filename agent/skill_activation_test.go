package agent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

type skillTestClient struct{ requests []*api.ChatRequest }

func (c *skillTestClient) Chat(_ context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
	c.requests = append(c.requests, req)
	return fn(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "Done."}})
}

func testSkillCatalog(t *testing.T) *SkillCatalog {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "release-notes")
	if err := os.Mkdir(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "SKILL.md"), []byte("---\nname: release-notes\ndescription: Draft release notes.\n---\nUse concise bullets."), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := DiscoverSkills(dir)
	if err != nil {
		t.Fatal(err)
	}
	return catalog
}

func TestSessionSkillActivationPreservesCallAndResultOrder(t *testing.T) {
	catalog := testSkillCatalog(t)
	client := &skillTestClient{}
	events := &recordingEventSink{}
	result, err := (&Session{Client: client, Skills: catalog, EventSinks: []EventSink{events}}).Run(context.Background(), RunOptions{
		Model:       "test",
		NewMessages: []api.Message{{Role: "user", Content: "draft release notes"}},
		SkillName:   "release-notes",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) != 4 {
		t.Fatalf("transcript = %#v", result.Messages)
	}
	call, toolTranscript := result.Messages[1], result.Messages[2]
	if call.Role != "assistant" || len(call.ToolCalls) != 1 || call.ToolCalls[0].Function.Name != "skill" || !strings.HasPrefix(call.ToolCalls[0].ID, "call_skill_") {
		t.Fatalf("call message = %#v", call)
	}
	if toolTranscript.Role != "tool" || toolTranscript.ToolName != "skill" || toolTranscript.ToolCallID != call.ToolCalls[0].ID || !strings.Contains(toolTranscript.Content, "Use concise bullets.") {
		t.Fatalf("tool result = %#v", toolTranscript)
	}
	if len(client.requests) != 1 || len(client.requests[0].Messages) != 3 || client.requests[0].Messages[2].ToolCallID != call.ToolCalls[0].ID {
		t.Fatalf("model request did not preserve transcript: %#v", client.requests)
	}
	var skillEvents []EventType
	for _, event := range events.events {
		if event.ToolName == "skill" || event.Type == EventToolCallDetected {
			skillEvents = append(skillEvents, event.Type)
		}
	}
	if len(skillEvents) < 3 {
		t.Fatalf("skill event order = %#v, want tool_call_detected,tool_started,tool_finished", skillEvents)
	}
	if got, want := strings.Join([]string{string(skillEvents[0]), string(skillEvents[1]), string(skillEvents[2])}, ","), "tool_call_detected,tool_started,tool_finished"; got != want {
		t.Fatalf("skill event order = %#v, want %s", skillEvents, want)
	}
}
