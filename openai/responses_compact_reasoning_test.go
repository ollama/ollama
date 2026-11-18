package openai

import (
	"encoding/json"
	"testing"
)

func TestCompactionRetainedReasoningReplay(t *testing.T) {
	for _, test := range []struct {
		name      string
		reasoning string
		want      string
	}{
		{
			name:      "summary",
			reasoning: `{"type":"reasoning","summary":[{"type":"summary_text","text":"Check the cache. "},{"type":"summary_text","text":"Then inspect the caller."}]}`,
			want:      "Check the cache. Then inspect the caller.",
		},
		{
			name:      "opaque state",
			reasoning: `{"type":"reasoning","encrypted_content":"foreign-opaque-state","summary":[]}`,
			want:      "[opaque reasoning state omitted during Ollama compaction]",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			body := []byte(`{"model":"test","input":[
				{"type":"message","role":"user","content":"Inspect the failure."},
				` + test.reasoning + `,
				{"type":"message","role":"assistant","content":"I will inspect the caller."},
				{"type":"message","role":"user","content":"Continue."}
			]}`)
			plan, err := PrepareStandaloneCompaction(body)
			if err != nil {
				t.Fatal(err)
			}
			result, err := plan.Complete(compactionResponseBody(t, map[string]any{
				"summary": "Continue investigating the failure.", "retain_item_ids": []string{"item_000002", "item_000003"},
			}))
			if err != nil {
				t.Fatal(err)
			}
			payload := decodeResultPayload(t, result)
			if len(payload.Retained) != 2 || payload.Retained[0].Thinking != test.want {
				t.Fatalf("retained reasoning changed: %+v", payload.Retained)
			}
			replay, err := json.Marshal(map[string]any{"model": "test", "input": []any{result.Item}})
			if err != nil {
				t.Fatal(err)
			}
			expanded, changed, err := ExpandResponsesCompactionInput(replay)
			if err != nil || !changed {
				t.Fatalf("changed=%v err=%v", changed, err)
			}
			var request ResponsesRequest
			if err := json.Unmarshal(expanded, &request); err != nil {
				t.Fatal(err)
			}
			chat, err := FromResponsesRequest(request)
			if err != nil {
				t.Fatal(err)
			}
			if len(chat.Messages) != 4 {
				t.Fatalf("want summary pair and two retained messages, got %+v", chat.Messages)
			}
			if got := chat.Messages[2]; got.Role != "assistant" || got.Thinking != test.want {
				t.Errorf("replayed reasoning = %+v, want thinking %q", got, test.want)
			}
			if got := chat.Messages[3]; got.Role != "assistant" || got.Content != "I will inspect the caller." {
				t.Errorf("replayed assistant changed: %+v", got)
			}
		})
	}
}
