package openai

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestCompactionPreservesAmbiguousStandaloneNames(t *testing.T) {
	body := []byte(`{"model":"test","input":[
		{"type":"message","role":"user","content":"Continue."},
		{"type":"function_call_output","namespace":"a.b","name":"c","output":"first"},
		{"type":"message","role":"assistant","content":"Acknowledged."},
		{"type":"function_call_output","namespace":"a","name":"b.c","output":"second"},
		{"type":"function_call_output","name":"a.b.c","output":"third"},
		{"type":"message","role":"user","content":"Keep working."}
	]}`)
	wantNames := []compactionFunctionName{
		{Name: "c", Namespace: "a.b"},
		{Name: "b.c", Namespace: "a"},
		{Name: "a.b.c"},
	}
	wantContent := []string{"first", "second", "third"}
	standaloneIndices := []int{1, 3, 4}
	wantOrdinary := map[int]api.Message{
		0: {Role: "user", Content: "Continue."},
		2: {Role: "assistant", Content: "Acknowledged."},
	}
	wantRetainedCount := len(wantNames) + len(wantOrdinary)
	for cycle := range 2 {
		plan, err := PrepareStandaloneCompaction(body)
		if err != nil {
			t.Fatal(err)
		}
		var retainRefs []string
		for _, item := range plan.items {
			for _, want := range wantOrdinary {
				if item.Message.Role == want.Role && item.Message.Content == want.Content {
					retainRefs = append(retainRefs, item.Ref)
				}
			}
		}
		if len(retainRefs) != len(wantOrdinary) {
			t.Fatalf("cycle %d: ordinary messages missing from the transcript", cycle)
		}
		result, err := plan.Complete(compactionResponseBody(t, map[string]any{
			"summary": "Continue the task.", "retain_item_ids": retainRefs,
		}))
		if err != nil {
			t.Fatal(err)
		}
		payload := decodeResultPayload(t, result)
		if len(payload.Retained) != wantRetainedCount || len(payload.StandaloneNames) != len(wantNames) {
			t.Fatalf("cycle %d: retained messages or names changed: %+v", cycle, payload)
		}
		for i, want := range wantNames {
			index := standaloneIndices[i]
			if got := payload.StandaloneNames[index]; got != want {
				t.Errorf("cycle %d: retained name %d = %+v, want %+v", cycle, index, got, want)
			}
			if got := payload.Retained[index]; got.ToolName != "a.b.c" || got.ToolCallID != "" || got.Content != wantContent[i] {
				t.Errorf("cycle %d: retained message %d changed: %+v", cycle, index, got)
			}
		}
		for index, want := range wantOrdinary {
			if got := payload.Retained[index]; got.Role != want.Role || got.Content != want.Content {
				t.Errorf("cycle %d: ordinary retained message %d changed: %+v", cycle, index, got)
			}
		}
		replay, err := json.Marshal(map[string]any{"model": "test", "input": []any{result.Item}})
		if err != nil {
			t.Fatal(err)
		}
		expanded, changed, err := ExpandResponsesCompactionInput(replay)
		if err != nil || !changed {
			t.Fatalf("cycle %d: changed=%v err=%v", cycle, changed, err)
		}
		var request ResponsesRequest
		if err := json.Unmarshal(expanded, &request); err != nil {
			t.Fatal(err)
		}
		if len(request.Input.Items) != 2+wantRetainedCount {
			t.Fatalf("cycle %d: got %d items, want summary pair and five retained messages", cycle, len(request.Input.Items))
		}
		for i, want := range wantNames {
			index := standaloneIndices[i] + 2
			output, ok := request.Input.Items[index].(ResponsesFunctionCallOutput)
			if !ok || output.CallID != "" || output.Name != want.Name || output.Namespace != want.Namespace || output.Output != wantContent[i] {
				t.Errorf("cycle %d: output %d lost its original identity or order: %+v", cycle, index, request.Input.Items[index])
			}
		}
		chat, err := FromResponsesRequest(request)
		if err != nil {
			t.Fatal(err)
		}
		if len(chat.Messages) != 2+wantRetainedCount {
			t.Fatalf("cycle %d: got %d native messages", cycle, len(chat.Messages))
		}
		for i := range wantNames {
			index := standaloneIndices[i] + 2
			if got := chat.Messages[index]; got.ToolName != "a.b.c" || got.ToolCallID != "" || got.Content != wantContent[i] {
				t.Errorf("cycle %d: native output %d changed: %+v", cycle, index, got)
			}
		}
		for index, want := range wantOrdinary {
			if got := chat.Messages[index+2]; got.Role != want.Role || got.Content != want.Content {
				t.Errorf("cycle %d: ordinary native message %d changed: %+v", cycle, index+2, got)
			}
		}
		body = expanded
	}
}

func TestCompactionRejectsInvalidStandaloneNames(t *testing.T) {
	standalone := []api.Message{{Role: "tool", ToolName: "workspace.handoff", Content: "Continue the task."}}
	name := compactionFunctionName{Name: "handoff", Namespace: "workspace"}
	for _, test := range []struct {
		name     string
		retained []api.Message
		names    map[int]compactionFunctionName
	}{
		{name: "negative index", retained: standalone, names: map[int]compactionFunctionName{-1: name}},
		{name: "index past retained", retained: standalone, names: map[int]compactionFunctionName{1: name}},
		{name: "missing identity", retained: standalone},
		{name: "different native name", retained: standalone, names: map[int]compactionFunctionName{0: {Name: "other", Namespace: "workspace"}}},
		{name: "blank name", retained: standalone, names: map[int]compactionFunctionName{0: {Name: " ", Namespace: "workspace"}}},
		{name: "user message", retained: []api.Message{{Role: "user", ToolName: "workspace.handoff", Content: "Continue."}}, names: map[int]compactionFunctionName{0: name}},
		{
			name: "paired output",
			retained: []api.Message{
				{Role: "assistant", ToolCalls: []api.ToolCall{{ID: "paired", Function: api.ToolCallFunction{Name: "workspace.handoff", Arguments: api.NewToolCallFunctionArguments()}}}},
				{Role: "tool", ToolCallID: "paired", ToolName: "workspace.handoff", Content: "Finished."},
			},
			names: map[int]compactionFunctionName{1: name},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			payload, err := json.Marshal(OllamaCompactionPayload{
				Type: OllamaCompactionPayloadType, Version: OllamaCompactionPayloadVersion,
				Summary: "Continue the task.", Retained: test.retained, StandaloneNames: test.names,
			})
			if err != nil {
				t.Fatal(err)
			}
			body, err := json.Marshal(map[string]any{
				"model": "test", "input": []any{ResponsesCompactionItem{Type: "compaction", EncryptedContent: string(payload)}},
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := ExpandResponsesCompactionInput(body); err == nil {
				t.Fatal("accepted invalid standalone name metadata")
			}
		})
	}
}

func TestCompactionReplaysLegacyPairedPayload(t *testing.T) {
	// Version 1 payloads written before standalone outputs have no name metadata.
	payload := `{"type":"ollama_compaction","version":1,"summary":"The file was read.","retained":[
		{"role":"assistant","tool_calls":[{"id":"paired","function":{"name":"workspace.read","arguments":{}}}]},
		{"role":"tool","tool_call_id":"paired","tool_name":"workspace.read","content":"File contents."}
	]}`
	body, err := json.Marshal(map[string]any{
		"model": "test", "input": []any{ResponsesCompactionItem{Type: "compaction", EncryptedContent: payload}},
	})
	if err != nil {
		t.Fatal(err)
	}
	expanded, changed, err := ExpandResponsesCompactionInput(body)
	if err != nil || !changed {
		t.Fatalf("changed=%v err=%v", changed, err)
	}
	var request ResponsesRequest
	if err := json.Unmarshal(expanded, &request); err != nil {
		t.Fatal(err)
	}
	if len(request.Input.Items) != 4 {
		t.Fatalf("got %d items, want summary and retained pairs", len(request.Input.Items))
	}
	call, ok := request.Input.Items[2].(ResponsesFunctionCall)
	if !ok || call.CallID != "paired" || call.Name != "workspace.read" {
		t.Fatalf("legacy call changed: %+v", request.Input.Items[2])
	}
	output, ok := request.Input.Items[3].(ResponsesFunctionCallOutput)
	if !ok || output.CallID != "paired" || output.Output != "File contents." {
		t.Fatalf("legacy output changed: %+v", request.Input.Items[3])
	}
	if _, err := PrepareStandaloneCompaction(expanded); err != nil {
		t.Fatalf("cannot compact legacy replay: %v", err)
	}
}
