package openai

import (
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/renderers"
)

func TestResponsesPreserveDeveloperRole(t *testing.T) {
	body := `{"model":"test-model","instructions":"Base instructions","stream":true,"input":[
        {"type":"message","role":"system","content":"System policy"},
        {"role":"developer","content":"Use the scratch project."},
        {"role":"user","content":"Read the fixture"},
        {"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"fixture.txt\"}"},
        {"type":"function_call_output","call_id":"call_1","output":"fixture contents"},
        {"type":"message","role":"developer","content":[{"type":"input_text","text":"Keep the file"},{"type":"input_text","text":" unchanged."}]},
        {"role":"assistant","content":"I read the fixture."},
        {"role":"user","content":"Continue"}
    ]}`
	var request ResponsesRequest
	if err := json.Unmarshal([]byte(body), &request); err != nil {
		t.Fatal(err)
	}
	got, err := FromResponsesRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	var roles []string
	for _, message := range got.Messages {
		roles = append(roles, message.Role)
	}
	want := []string{"system", "system", "developer", "user", "assistant", "tool", "developer", "assistant", "user"}
	if !slices.Equal(roles, want) {
		t.Fatalf("roles = %v, want %v", roles, want)
	}
	if got.Messages[2].Content != "Use the scratch project." || got.Messages[6].Content != "Keep the file unchanged." {
		t.Fatal("developer content changed")
	}
	if got.Messages[4].ToolCalls[0].ID != "call_1" || got.Messages[5].ToolCallID != "call_1" {
		t.Fatal("tool history lost its call/result pairing")
	}
}

func TestCompactionPreservesInstructionRoles(t *testing.T) {
	body := []byte(`{"model":"test-model","input":[
        {"role":"system","content":"System policy."},
        {"role":"developer","content":"Retain the project policy."},
        {"role":"user","content":"Read the project status"},
        {"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"status.txt\"}"},
        {"type":"function_call_output","call_id":"call_1","output":"The project status check passed."},
        {"role":"assistant","content":"I read the status."},
        {"role":"developer","content":"Preserve the scratch fixture."},
        {"role":"assistant","content":"I will continue."}
    ]}`)
	want := []api.Message{
		{Role: "system", Content: "System policy."},
		{Role: "developer", Content: "Retain the project policy."},
		{Role: "developer", Content: "Preserve the scratch fixture."},
	}
	for cycle := range 2 {
		plan, err := PrepareStandaloneCompaction(body)
		if err != nil {
			t.Fatal(err)
		}
		request, err := plan.SummaryRequest("")
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(request), `\"role\":\"developer\"`) {
			t.Fatal("compaction transcript lost the developer role")
		}
		// Retain policies even when the summarizer selects no exact items.
		result, err := plan.Complete(compactionResponseBody(t, map[string]any{
			"summary": "Continue the project.", "retain_item_ids": []string{},
		}))
		if err != nil {
			t.Fatal(err)
		}
		payload := decodeResultPayload(t, result)
		if !reflect.DeepEqual(payload.Retained, want) {
			t.Fatalf("cycle %d: retained %#v, want %#v", cycle, payload.Retained, want)
		}
		replay, err := json.Marshal(map[string]any{"model": "test-model", "input": []any{
			result.Item, map[string]any{"role": "user", "content": "Continue"},
		}})
		if err != nil {
			t.Fatal(err)
		}
		expanded, changed, err := ExpandResponsesCompactionInput(replay)
		if err != nil || !changed {
			t.Fatalf("expand = %v, %v", changed, err)
		}
		var decoded ResponsesRequest
		if err := json.Unmarshal(expanded, &decoded); err != nil {
			t.Fatal(err)
		}
		chat, err := FromResponsesRequest(decoded)
		if err != nil {
			t.Fatal(err)
		}
		if len(chat.Messages) != len(want)+3 || !reflect.DeepEqual(chat.Messages[:len(want)], want) {
			t.Fatalf("cycle %d: replay lost or reordered the instruction prefix: %#v", cycle, chat.Messages)
		}
		call, resultMessage := chat.Messages[len(want)], chat.Messages[len(want)+1]
		if len(call.ToolCalls) != 1 || call.ToolCalls[0].ID != resultMessage.ToolCallID {
			t.Fatal("summary call/result pairing changed")
		}
		original := slices.Clone(chat.Messages)
		for _, renderer := range []string{"qwen3-coder", "cogito", "functiongemma"} {
			prompt, err := renderers.RenderWithRenderer(renderer, chat.Messages, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			position := 0
			for _, instruction := range want {
				index := strings.Index(prompt[position:], instruction.Content)
				if index < 0 {
					t.Fatalf("cycle %d, %s: lost or reordered %q: %q", cycle, renderer, instruction.Content, prompt)
				}
				position += index + len(instruction.Content)
			}
			if !reflect.DeepEqual(chat.Messages, original) {
				t.Fatal("rendering mutated the compacted history")
			}
		}
		body = replay
	}
}

func TestCompactionSummaryPreservesRetainedMessageOrder(t *testing.T) {
	retained := []api.Message{
		{Role: "system", Content: "Base policy"},
		{Role: "developer", Content: "Project policy"},
		{Role: "user", Content: "Earlier request"},
		{Role: "developer", Content: "Later policy"},
		{Role: "user", Content: "Continue"},
	}
	items, err := payloadToResponsesItems(OllamaCompactionPayload{
		Type: OllamaCompactionPayloadType, Version: OllamaCompactionPayloadVersion,
		Summary: "Earlier work is complete.", Retained: retained,
	})
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]any{"model": "test", "input": items})
	if err != nil {
		t.Fatal(err)
	}
	var request ResponsesRequest
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatal(err)
	}
	chat, err := FromResponsesRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 7 || !reflect.DeepEqual(chat.Messages[:2], retained[:2]) || !reflect.DeepEqual(chat.Messages[4:], retained[2:]) {
		t.Fatalf("summary displaced the leading prefix or reordered later instructions: %#v", chat.Messages)
	}
	if len(chat.Messages[2].ToolCalls) != 1 || chat.Messages[2].ToolCalls[0].ID != chat.Messages[3].ToolCallID {
		t.Fatal("summary call/result pairing changed")
	}
}
