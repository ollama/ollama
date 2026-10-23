package openai

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestCompactionRejectsHistoricalToolsWithoutChangingRetainedState(t *testing.T) {
	for _, name := range []string{"read_fixture", "run_tests"} {
		t.Run(name, func(t *testing.T) {
			plan, err := PrepareStandaloneCompaction([]byte(`{"model":"test","input":[
				{"type":"compaction","encrypted_content":"synthetic-foreign-checkpoint"},
				{"role":"user","content":"The project label is kiwi-42."},
				{"type":"custom_tool_call","call_id":"call_fixture","name":"read_fixture","input":"fixture.txt"},
				{"type":"custom_tool_call_output","call_id":"call_fixture","output":"plum-73"}
			]}`))
			if err != nil {
				t.Fatal(err)
			}
			response, err := json.Marshal(map[string]any{"output": []any{map[string]any{
				"type": "function_call", "name": name, "call_id": "call_wrong", "arguments": `{"input":"fixture.txt"}`,
			}}})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := plan.Complete(response); err == nil || !strings.Contains(err.Error(), "expected one create_summary call, got 0") {
				t.Fatalf("historical tool response was not rejected: %v", err)
			}
			if _, err := plan.SummaryRequest("expected one create_summary call, got 0"); err != nil {
				t.Fatal(err)
			}
			result, err := plan.Complete(compactionResponseBody(t, map[string]any{
				"summary":         "Project kiwi-42; fixture result plum-73.",
				"retain_item_ids": []string{"item_000001", "item_000002", "item_000003"},
			}))
			if err != nil {
				t.Fatal(err)
			}
			payload := decodeResultPayload(t, result)
			if len(payload.Retained) != 3 || payload.Retained[0].Content != "The project label is kiwi-42." ||
				len(payload.Retained[1].ToolCalls) != 1 || payload.Retained[1].ToolCalls[0].ID != "call_fixture" ||
				payload.Retained[2].ToolCallID != "call_fixture" || payload.Retained[2].Content != "plum-73" {
				t.Fatalf("retained conversation changed after the rejected summary: %#v", payload.Retained)
			}
		})
	}
}
