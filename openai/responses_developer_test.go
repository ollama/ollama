package openai

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/renderers"
)

func replayCompaction(t *testing.T, item ResponsesCompactionItem) ([]byte, []api.Message) {
	t.Helper()
	body, err := json.Marshal(map[string]any{"model": "test", "input": []any{
		item, map[string]any{"role": "user", "content": "Continue"},
	}})
	require.NoError(t, err)
	expanded, changed, err := ExpandResponsesCompactionInput(body)
	require.NoError(t, err)
	require.True(t, changed, "compaction input must expand")
	var request ResponsesRequest
	require.NoError(t, json.Unmarshal(expanded, &request))
	chat, err := FromResponsesRequest(request)
	require.NoError(t, err)
	return body, chat.Messages
}

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
	require.NoError(t, json.Unmarshal([]byte(body), &request))
	got, err := FromResponsesRequest(request)
	require.NoError(t, err)
	var roles []string
	for _, message := range got.Messages {
		roles = append(roles, message.Role)
	}
	want := []string{"system", "system", "developer", "user", "assistant", "tool", "developer", "assistant", "user"}
	require.Equal(t, want, roles)
	require.Equal(t, "Use the scratch project.", got.Messages[2].Content)
	require.Equal(t, "Keep the file unchanged.", got.Messages[6].Content)
	require.Equal(t, "call_1", got.Messages[4].ToolCalls[0].ID)
	require.Equal(t, "call_1", got.Messages[5].ToolCallID)
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
		require.NoError(t, err)
		request, err := plan.SummaryRequest("")
		require.NoError(t, err)
		require.Contains(t, string(request), `\"role\":\"developer\"`)
		// Retain policies even when the summarizer selects no exact items.
		result, err := plan.Complete(compactionResponseBody(t, map[string]any{
			"summary": "Continue the project.", "retain_item_ids": []string{},
		}))
		require.NoError(t, err)
		payload := decodeResultPayload(t, result)
		wantRetained := []api.Message{want[0], want[1], {Role: "user", Content: "Read the project status"}, want[2]}
		require.Equal(t, wantRetained, payload.Retained, "cycle %d", cycle)
		require.Nil(t, payload.LeadingInstructions, "writer must support older v1 readers")
		replay, messages := replayCompaction(t, result.Item)
		require.Len(t, messages, len(wantRetained)+3)
		require.Equal(t, want[:2], messages[:2])
		require.Equal(t, wantRetained[2:], messages[4:6])
		call, resultMessage := messages[2], messages[3]
		require.Len(t, call.ToolCalls, 1)
		require.Equal(t, call.ToolCalls[0].ID, resultMessage.ToolCallID)
		original := slices.Clone(messages)
		for _, renderer := range []string{"qwen3-coder", "cogito", "functiongemma"} {
			prompt, err := renderers.RenderWithRenderer(renderer, messages, nil, nil)
			require.NoError(t, err)
			position := 0
			for _, instruction := range want {
				index := strings.Index(prompt[position:], instruction.Content)
				if index < 0 {
					t.Fatalf("cycle %d, %s: lost or reordered %q: %q", cycle, renderer, instruction.Content, prompt)
				}
				position += index + len(instruction.Content)
			}
			require.Equal(t, original, messages, "rendering mutated the compacted history")
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
		require.NoError(t, err)
		result, err := plan.Complete(compactionResponseBody(t, map[string]any{
			"summary": "Earlier work.", "retain_item_ids": []string{},
		}))
		require.NoError(t, err)
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
		require.NoError(t, decoder.Decode(&legacy), "payload must support the v1 reader")
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
		require.NoError(t, err)
		result, err := plan.Complete(compactionResponseBody(t, map[string]any{
			"summary": "Work started.", "retain_item_ids": []string{},
		}))
		require.NoError(t, err)
		replay, messages := replayCompaction(t, result.Item)
		body = replay
		require.Len(t, messages, 5, "cycle %d", cycle)
		require.Len(t, messages[0].ToolCalls, 1)
		require.Equal(t, messages[0].ToolCalls[0].ID, messages[1].ToolCallID)
		require.Equal(t, "Start", messages[2].Content)
		require.Equal(t, api.Message{Role: "developer", Content: "LATER_POLICY"}, messages[3])
		require.Len(t, decodeResultPayload(t, result).Retained, 2, "must not retain a prior synthetic summary")
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
		require.NoError(t, err)
		plan.TrimForContextLimit()
		result, err := plan.Complete(compactionResponseBody(t, map[string]any{
			"summary": "Continue the work.", "retain_item_ids": []string{},
		}))
		require.NoError(t, err)
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
	require.NoError(t, err)
	item := ResponsesCompactionItem{Type: "compaction", EncryptedContent: string(seed)}
	priorSummary := "PREVIEW_SUMMARY"
	for cycle := range 3 {
		body, _ := replayCompaction(t, item)
		plan, err := PrepareStandaloneCompaction(body)
		require.NoError(t, err)
		var summaryRefs []string
		for _, entry := range plan.items {
			if entry.Type == "function_call" || entry.Type == "function_call_output" {
				summaryRefs = append(summaryRefs, entry.Ref)
			}
		}
		result, err := plan.Complete(compactionResponseBody(t, map[string]any{
			"summary": fmt.Sprintf("Summary %d.", cycle), "retain_item_ids": summaryRefs,
		}))
		require.NoError(t, err)
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
	require.NoError(t, err)
	body, err := json.Marshal(map[string]any{"model": "test", "input": items})
	require.NoError(t, err)
	var request ResponsesRequest
	require.NoError(t, json.Unmarshal(body, &request))
	chat, err := FromResponsesRequest(request)
	require.NoError(t, err)
	require.Len(t, chat.Messages, 7)
	require.Equal(t, retained[:2], chat.Messages[:2])
	require.Equal(t, retained[2:], chat.Messages[4:])
	require.Len(t, chat.Messages[2].ToolCalls, 1)
	require.Equal(t, chat.Messages[2].ToolCalls[0].ID, chat.Messages[3].ToolCallID)
}

func TestCompactionRejectsInvalidInstructionBoundary(t *testing.T) {
	for _, boundary := range []int{-1, 2, 3} {
		_, err := payloadToResponsesItems(OllamaCompactionPayload{
			Type: OllamaCompactionPayloadType, Version: OllamaCompactionPayloadVersion,
			Summary: "Earlier work.", LeadingInstructions: &boundary,
			Retained: []api.Message{{Role: "developer", Content: "Policy"}, {Role: "user", Content: "Continue"}},
		})
		require.Error(t, err, "invalid instruction boundary %d", boundary)
	}
}
