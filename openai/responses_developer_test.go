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
		var instructions []api.Message
		for _, message := range chat.Messages {
			if message.Role == "system" || message.Role == "developer" {
				instructions = append(instructions, message)
			}
		}
		if !reflect.DeepEqual(instructions, want) {
			t.Fatalf("replay instructions = %#v", instructions)
		}
		body = replay
	}
}

func TestCompactedInstructionsReachRenderer(t *testing.T) {
	for _, role := range []string{"system", "developer"} {
		t.Run(role, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{"model": "test", "input": []any{
				map[string]any{"role": role, "content": "LEADING_POLICY"},
				map[string]any{"role": "user", "content": "Earlier request"},
			}})
			if err != nil {
				t.Fatal(err)
			}
			for cycle := range 2 {
				plan, err := PrepareStandaloneCompaction(body)
				if err != nil {
					t.Fatal(err)
				}
				result, err := plan.Complete(compactionResponseBody(t, map[string]any{
					"summary": "Earlier work is complete.", "retain_item_ids": []string{},
				}))
				if err != nil {
					t.Fatal(err)
				}
				body, err = json.Marshal(map[string]any{"model": "test", "input": []any{
					result.Item, map[string]any{"role": "user", "content": "Continue"},
				}})
				if err != nil {
					t.Fatal(err)
				}
				expanded, _, err := ExpandResponsesCompactionInput(body)
				if err != nil {
					t.Fatal(err)
				}
				var request ResponsesRequest
				if err := json.Unmarshal(expanded, &request); err != nil {
					t.Fatal(err)
				}
				chat, err := FromResponsesRequest(request)
				if err != nil {
					t.Fatal(err)
				}
				for _, renderer := range []string{"cogito", "functiongemma"} {
					prompt, err := renderers.RenderWithRenderer(renderer, chat.Messages, nil, nil)
					if err != nil {
						t.Fatal(err)
					}
					if !strings.Contains(prompt, "LEADING_POLICY") {
						t.Errorf("cycle %d, %s: compacted instructions missing from prompt: %q", cycle, renderer, prompt)
					}
				}
				if len(chat.Messages) != 4 || chat.Messages[0].Role != role || chat.Messages[0].Content != "LEADING_POLICY" {
					t.Fatalf("cycle %d: leading instruction position or role changed: %#v", cycle, chat.Messages)
				}
				if len(chat.Messages[1].ToolCalls) != 1 || chat.Messages[1].ToolCalls[0].ID != chat.Messages[2].ToolCallID {
					t.Fatal("summary call/result pairing changed")
				}
			}
		})
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
