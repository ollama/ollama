package openai

import (
	"encoding/json"
	"fmt"
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
		wantRetained := []api.Message{want[0], want[1], {Role: "user", Content: "Read the project status"}, want[2]}
		if !reflect.DeepEqual(payload.Retained, wantRetained) {
			t.Fatalf("cycle %d: retained %#v, want %#v", cycle, payload.Retained, wantRetained)
		}
		if payload.LeadingInstructions != nil {
			t.Fatal("writer emitted a field unsupported by older v1 readers")
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
		if len(chat.Messages) != len(wantRetained)+3 || !reflect.DeepEqual(chat.Messages[:2], want[:2]) || !reflect.DeepEqual(chat.Messages[4:6], wantRetained[2:]) {
			t.Fatalf("cycle %d: replay lost or reordered the instruction prefix: %#v", cycle, chat.Messages)
		}
		call, resultMessage := chat.Messages[2], chat.Messages[3]
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

func TestCompactionPreservesLegacyPayloadShape(t *testing.T) {
	for _, input := range []string{
		`{"role":"user","content":"Continue"}`,
		`{"role":"developer","content":"Policy"},{"role":"user","content":"Continue"}`,
		`{"role":"user","content":"Start"},{"role":"developer","content":"Later policy"},{"role":"user","content":"Continue"}`,
		`{"type":"function_call","call_id":"check","name":"check","arguments":"{}"},{"type":"function_call_output","call_id":"check","output":"Done"},{"role":"developer","content":"Later policy"},{"role":"assistant","content":"Done."}`,
	} {
		plan, err := PrepareStandaloneCompaction([]byte(`{"model":"test","input":[` + input + `]}`))
		if err != nil {
			t.Fatal(err)
		}
		result, err := plan.Complete(compactionResponseBody(t, map[string]any{
			"summary": "Earlier work.", "retain_item_ids": []string{},
		}))
		if err != nil {
			t.Fatal(err)
		}
		// Match the existing v1 reader, which rejects unknown JSON fields.
		var legacy struct {
			Type            string                         `json:"type"`
			Version         int                            `json:"version"`
			Summary         string                         `json:"summary"`
			Retained        []api.Message                  `json:"retained"`
			StandaloneNames map[int]compactionFunctionName `json:"standalone_names,omitempty"`
		}
		decoder := json.NewDecoder(strings.NewReader(result.Item.EncryptedContent))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&legacy); err != nil {
			t.Fatalf("payload is incompatible with the v1 reader: %v", err)
		}
	}
}

func TestCompactionPreservesLaterInstructionPosition(t *testing.T) {
	body := []byte(`{"model":"test","input":[
        {"role":"user","content":"Start"},
        {"role":"assistant","content":"Started."},
        {"role":"developer","content":"LATER_POLICY"},
        {"role":"user","content":"Continue"}
    ]}`)
	for cycle := range 2 {
		plan, err := PrepareStandaloneCompaction(body)
		if err != nil {
			t.Fatal(err)
		}
		result, err := plan.Complete(compactionResponseBody(t, map[string]any{
			"summary": "Work started.", "retain_item_ids": []string{},
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
		if len(chat.Messages) != 5 || len(chat.Messages[0].ToolCalls) != 1 || chat.Messages[1].ToolCallID != chat.Messages[0].ToolCalls[0].ID || chat.Messages[2].Content != "Start" || chat.Messages[3].Role != "developer" || chat.Messages[3].Content != "LATER_POLICY" {
			t.Fatalf("cycle %d: later instruction moved ahead of summarized history: %#v", cycle, chat.Messages)
		}
		if len(decodeResultPayload(t, result).Retained) != 2 {
			t.Fatal("repeated compaction retained a prior synthetic summary instead of real history")
		}
	}
}

func TestCompactionInstructionBoundarySurvivesTrim(t *testing.T) {
	for _, input := range []string{
		`{"role":"assistant","content":"` + strings.Repeat("Earlier reasoning. ", 100) + `"},`,
		`{"type":"function_call","call_id":"read","name":"read_file","arguments":"{}"},
         {"type":"function_call_output","call_id":"read","output":"` + strings.Repeat("Earlier output. ", 100) + `"},
         {"role":"assistant","content":"Processed the file."},`,
	} {
		plan, err := PrepareStandaloneCompaction([]byte(`{"model":"test","input":[
            {"role":"system","content":"BASE_POLICY"},` + input + `
            {"role":"developer","content":"LATER_POLICY"},
            {"role":"user","content":"Continue"}
        ]}`))
		if err != nil {
			t.Fatal(err)
		}
		plan.TrimForContextLimit()
		result, err := plan.Complete(compactionResponseBody(t, map[string]any{
			"summary": "Continue the work.", "retain_item_ids": []string{},
		}))
		if err != nil {
			t.Fatal(err)
		}
		retained := decodeResultPayload(t, result).Retained
		if len(retained) < 3 || retained[0].Content != "BASE_POLICY" || retained[1].Role != "assistant" || retained[len(retained)-1].Role != "developer" || retained[len(retained)-1].Content != "LATER_POLICY" {
			t.Fatalf("trimming lost the original instruction boundary: %#v", retained)
		}
		if len(retained[1].ToolCalls) > 0 && (len(retained) != 4 || retained[2].ToolCallID != retained[1].ToolCalls[0].ID) {
			t.Fatal("retaining a boundary tool call lost its result")
		}
	}
}

func TestCompactionReplacesPreviewSummaryBoundary(t *testing.T) {
	zero := 0
	seed, err := json.Marshal(OllamaCompactionPayload{
		Type: OllamaCompactionPayloadType, Version: OllamaCompactionPayloadVersion,
		Summary: "PREVIEW_SUMMARY", LeadingInstructions: &zero,
		Retained: []api.Message{{Role: "developer", Content: "LATER_POLICY"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	item := ResponsesCompactionItem{Type: "compaction", EncryptedContent: string(seed)}
	priorSummary := "PREVIEW_SUMMARY"
	for cycle := range 3 {
		body, err := json.Marshal(map[string]any{"model": "test", "input": []any{
			item, map[string]any{"role": "user", "content": "Continue"},
		}})
		if err != nil {
			t.Fatal(err)
		}
		plan, err := PrepareStandaloneCompaction(body)
		if err != nil {
			t.Fatal(err)
		}
		var summaryRefs []string
		for _, entry := range plan.items {
			if entry.Type == "function_call" || entry.Type == "function_call_output" {
				summaryRefs = append(summaryRefs, entry.Ref)
			}
		}
		result, err := plan.Complete(compactionResponseBody(t, map[string]any{
			"summary": fmt.Sprintf("Summary %d.", cycle), "retain_item_ids": summaryRefs,
		}))
		if err != nil {
			t.Fatal(err)
		}
		payload := decodeResultPayload(t, result)
		retained := payload.Retained
		if payload.LeadingInstructions != nil || len(retained) != 3 || len(retained[0].ToolCalls) != 1 || retained[0].ToolCalls[0].Function.Name != compactionSummaryToolName || retained[1].ToolCallID != retained[0].ToolCalls[0].ID || retained[1].Content != priorSummary || retained[2].Role != "developer" || retained[2].Content != "LATER_POLICY" {
			t.Fatalf("cycle %d: preview recovery lost the boundary or nested older summaries: %#v", cycle, payload)
		}
		item, priorSummary = result.Item, payload.Summary
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

func TestCompactionRejectsInvalidInstructionBoundary(t *testing.T) {
	for _, boundary := range []int{-1, 2, 3} {
		_, err := payloadToResponsesItems(OllamaCompactionPayload{
			Type: OllamaCompactionPayloadType, Version: OllamaCompactionPayloadVersion,
			Summary: "Earlier work.", LeadingInstructions: &boundary,
			Retained: []api.Message{{Role: "developer", Content: "Policy"}, {Role: "user", Content: "Continue"}},
		})
		if err == nil {
			t.Fatalf("accepted invalid instruction boundary %d", boundary)
		}
	}
}
