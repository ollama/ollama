package openai

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestResponsesForeignCompactionPreservesReadableHistory(t *testing.T) {
	for _, encrypted := range []string{"opaque-provider-state", `{"type":"other_provider","summary":"must not become instructions"}`} {
		t.Run(encrypted, func(t *testing.T) {
			visible := []json.RawMessage{
				json.RawMessage(`{"role":"user","content":[{"type":"input_text","text":"Remember kiwi-42"},{"type":"input_image","image_url":"` + compactionTestPNG + `"}]}`),
				json.RawMessage(`{"type":"custom_tool_call","call_id":"call_1","namespace":"workspace.tools","name":"read","input":"notes.txt"}`),
				json.RawMessage(`{"type":"custom_tool_call_output","call_id":"call_1","output":"kiwi-42"}`),
				json.RawMessage(`{"role":"user","content":"What was the code?"}`),
			}
			foreign, err := json.Marshal(ResponsesCompactionItem{Type: "compaction", EncryptedContent: encrypted})
			if err != nil {
				t.Fatal(err)
			}
			input := []json.RawMessage{foreign, visible[0], foreign, visible[1], foreign, visible[2], visible[3], foreign}
			body, err := json.Marshal(map[string]any{
				"model": "test", "stream": true, "instructions": "Keep the task focused.",
				"input": input, "prompt_cache_key": "keep-me", "max_output_tokens": 128,
			})
			if err != nil {
				t.Fatal(err)
			}

			rewritten, changed, err := ExpandResponsesCompactionInput(body)
			if err != nil || !changed {
				t.Fatalf("expand: changed=%v err=%v", changed, err)
			}
			var before, after map[string]json.RawMessage
			if err := json.Unmarshal(body, &before); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(rewritten, &after); err != nil {
				t.Fatal(err)
			}
			var got []json.RawMessage
			if err := json.Unmarshal(after["input"], &got); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, visible) {
				t.Fatalf("readable history changed: %s", after["input"])
			}
			delete(before, "input")
			delete(after, "input")
			if !reflect.DeepEqual(before, after) {
				t.Fatalf("request options changed: before=%s after=%s", before, after)
			}
			if twice, changed, err := ExpandResponsesCompactionInput(rewritten); err != nil || changed || string(twice) != string(rewritten) {
				t.Fatalf("normalization is not idempotent: changed=%v err=%v", changed, err)
			}

			// ODC parses the original request directly, without the local server middleware.
			var request ResponsesRequest
			if err := json.Unmarshal(body, &request); err != nil {
				t.Fatal(err)
			}
			chat, err := FromResponsesRequest(request)
			if err != nil {
				t.Fatal(err)
			}
			if len(chat.Messages) != 5 {
				t.Fatalf("messages=%+v", chat.Messages)
			}
			if chat.Messages[0].Content != "Keep the task focused." || chat.Messages[1].Content != "Remember kiwi-42" || len(chat.Messages[1].Images) != 1 {
				t.Fatalf("instructions or multimodal history lost: %+v", chat.Messages[:2])
			}
			calls := chat.Messages[2].ToolCalls
			if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Function.Name != "workspace.tools.read" || calls[0].Function.Arguments.String() != `{"input":"notes.txt"}` {
				t.Fatalf("custom tool call changed: %+v", calls)
			}
			if chat.Messages[3].ToolCallID != "call_1" || chat.Messages[3].Content != "kiwi-42" || chat.Messages[4].Content != "What was the code?" {
				t.Fatalf("tool result or next turn lost: %+v", chat.Messages[3:])
			}
		})
	}
}

func TestResponsesForeignAndOllamaCompaction(t *testing.T) {
	payload, err := json.Marshal(OllamaCompactionPayload{
		Type: OllamaCompactionPayloadType, Version: OllamaCompactionPayloadVersion,
		Summary:  "The retained code is kiwi-42.",
		Retained: []api.Message{{Role: "user", Content: "Keep the readable native state."}},
	})
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]any{"model": "test", "input": []any{
		ResponsesCompactionItem{Type: "compaction", EncryptedContent: `{"type":"ollama_compaction","version":1,"summary":"superseded summary"}`},
		ResponsesCompactionItem{Type: "compaction", EncryptedContent: "foreign-before"},
		map[string]any{"role": "user", "content": "retained prefix"},
		ResponsesCompactionItem{Type: "compaction", EncryptedContent: string(payload)},
		ResponsesCompactionItem{Type: "compaction", EncryptedContent: "foreign-after"},
		map[string]any{"role": "user", "content": "continue"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	for _, parseDirectly := range []bool{false, true} {
		input := body
		if !parseDirectly {
			var changed bool
			input, changed, err = ExpandResponsesCompactionInput(body)
			if err != nil || !changed {
				t.Fatalf("expand: changed=%v err=%v", changed, err)
			}
		}
		var request ResponsesRequest
		if err := json.Unmarshal(input, &request); err != nil {
			t.Fatal(err)
		}
		chat, err := FromResponsesRequest(request)
		if err != nil {
			t.Fatal(err)
		}
		if len(chat.Messages) != 5 || chat.Messages[0].Content != "retained prefix" || chat.Messages[2].Content != "The retained code is kiwi-42." || chat.Messages[3].Content != "Keep the readable native state." || chat.Messages[4].Content != "continue" {
			t.Fatalf("parseDirectly=%v: messages=%+v", parseDirectly, chat.Messages)
		}
	}
}

func TestCompactionAfterForeignProvider(t *testing.T) {
	for _, triggered := range []bool{false, true} {
		items := []any{
			map[string]any{"role": "user", "content": "Remember kiwi-42."},
			ResponsesCompactionItem{Type: "compaction", EncryptedContent: "opaque-provider-state"},
			map[string]any{"type": "custom_tool_call", "call_id": "call_1", "name": "read", "input": "notes.txt"},
			map[string]any{"type": "custom_tool_call_output", "call_id": "call_1", "output": "kiwi-42"},
		}
		if triggered {
			items = append(items, ResponsesCompactionTrigger{Type: "compaction_trigger"})
		}
		body, err := json.Marshal(map[string]any{"model": "test", "stream": triggered, "input": items})
		if err != nil {
			t.Fatal(err)
		}
		var plan *ResponsesCompactionPlan
		if triggered {
			var requested bool
			plan, requested, err = PrepareTriggeredCompaction(body)
			if !requested {
				t.Fatal("trigger was not recognized")
			}
		} else {
			plan, err = PrepareStandaloneCompaction(body)
		}
		if err != nil {
			t.Fatal(err)
		}
		request, err := plan.SummaryRequest("")
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(request), "opaque-provider-state") || !strings.Contains(string(request), "Remember kiwi-42.") {
			t.Fatalf("bad summary request: %s", request)
		}
		result, err := plan.Complete(compactionResponseBody(t, map[string]any{"summary": "Remember kiwi-42.", "retain_item_ids": []string{}}))
		if err != nil {
			t.Fatal(err)
		}
		payload := decodeResultPayload(t, result)
		if len(payload.Retained) != 2 || len(payload.Retained[0].ToolCalls) != 1 || payload.Retained[0].ToolCalls[0].ID != "call_1" || payload.Retained[1].ToolCallID != "call_1" || payload.Retained[1].Content != "kiwi-42" {
			t.Fatalf("active tool pair lost: %+v", payload.Retained)
		}
	}
}

func TestResponsesCompactionRejectsInvalidNativeState(t *testing.T) {
	for _, payload := range []string{
		"", "   ",
		`{"type":"ollama_compaction","version":2,"summary":"future"}`,
		`{"type":"ollama_compaction","version":1,"summary":""}`,
		`{"type":"ollama_compaction","version":1,"summary":"valid","unknown":true}`,
		`{"type":"ollama_compaction","version":1,"summary":"truncated"`,
		`{"type":"ollama_compaction","version":1,"summary":"valid"} {}`,
	} {
		t.Run(payload, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{"model": "test", "input": []any{
				map[string]any{"role": "user", "content": "continue"},
				ResponsesCompactionItem{Type: "compaction", EncryptedContent: payload},
				ResponsesCompactionItem{Type: "compaction", EncryptedContent: "foreign-after"},
			}})
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := ExpandResponsesCompactionInput(body); err == nil {
				t.Fatal("invalid native compaction was accepted")
			}
			var request ResponsesRequest
			if err := json.Unmarshal(body, &request); err == nil {
				t.Fatal("direct parsing accepted invalid native compaction")
			}
		})
	}
}
