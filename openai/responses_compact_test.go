package openai

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func compactionResponseBody(t *testing.T, selection map[string]any) []byte {
	t.Helper()
	arguments, err := json.Marshal(selection)
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]any{
		"id":     "resp_summary",
		"object": "response",
		"status": "completed",
		"output": []any{map[string]any{
			"id": "fc_summary", "type": "function_call", "status": "completed",
			"call_id": "call_summary", "name": CreateSummaryToolName, "arguments": string(arguments),
		}},
		"usage": map[string]any{"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
	})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func decodeResultPayload(t *testing.T, result ResponsesCompactionResult) OllamaCompactionPayload {
	t.Helper()
	item, err := json.Marshal(result.Item)
	if err != nil {
		t.Fatal(err)
	}
	payload, err := decodeOllamaCompactionItem(item)
	if err != nil {
		t.Fatal(err)
	}
	return payload
}

func summaryTranscriptText(t *testing.T, requestBody []byte) string {
	t.Helper()
	var request struct {
		Input []struct {
			Content json.RawMessage `json:"content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(requestBody, &request); err != nil {
		t.Fatal(err)
	}
	var transcript strings.Builder
	for _, input := range request.Input[1:] {
		var text string
		if json.Unmarshal(input.Content, &text) == nil {
			transcript.WriteString(text)
			continue
		}
		var blocks []struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(input.Content, &blocks); err != nil {
			t.Fatal(err)
		}
		for _, block := range blocks {
			transcript.WriteString(block.Text)
		}
	}
	return transcript.String()
}

func TestPrepareTriggeredCompactionBuildsSummaryRequest(t *testing.T) {
	description := "Read a file"
	body := []byte(`{
		"model":"qwen3:4b",
		"stream":true,
		"instructions":"do not send this to the compactor",
		"tools":[{"type":"function","name":"read_file","description":"Read a file","strict":false,"parameters":{"type":"object"}}],
		"input":[
			{"type":"message","role":"user","content":"inspect main.go"},
			{"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"main.go\"}"},
			{"type":"function_call_output","call_id":"call_1","output":"package main"},
			{"type":"compaction_trigger"}
		]
	}`)

	plan, requested, err := PrepareTriggeredCompaction(body)
	if err != nil {
		t.Fatal(err)
	}
	if !requested {
		t.Fatal("expected compaction request")
	}
	if plan.Model != "qwen3:4b" || !plan.Stream {
		t.Fatalf("unexpected plan: model=%q stream=%v", plan.Model, plan.Stream)
	}

	requestBody, err := plan.SummaryRequest("")
	if err != nil {
		t.Fatal(err)
	}
	var request struct {
		Input []struct {
			Role string `json:"role"`
		} `json:"input"`
		Tools        []ResponsesTool `json:"tools"`
		Instructions string          `json:"instructions"`
		Stream       bool            `json:"stream"`
	}
	if err := json.Unmarshal(requestBody, &request); err != nil {
		t.Fatal(err)
	}
	if request.Stream {
		t.Fatal("summary inference must be non-streaming")
	}
	if request.Instructions != "" {
		t.Fatalf("original instructions leaked into summary request: %q", request.Instructions)
	}
	if len(request.Tools) != 1 || request.Tools[0].Name != CreateSummaryToolName {
		t.Fatalf("unexpected callable tools: %+v", request.Tools)
	}
	transcript := summaryTranscriptText(t, requestBody)
	if len(request.Input) != 5 || !strings.Contains(transcript, description) || !strings.Contains(transcript, "read_file") {
		t.Fatalf("summary transcript is missing tool metadata: %+v", request.Input)
	}
	if strings.Contains(transcript, `"instructions"`) {
		t.Fatalf("original instructions leaked into transcript: %s", transcript)
	}

	result, err := plan.Complete(compactionResponseBody(t, map[string]any{
		"summary": "Continue inspecting main.go.", "retain_item_ids": []string{},
	}))
	if err != nil {
		t.Fatal(err)
	}
	payload := decodeResultPayload(t, result)
	if payload.Summary != "Continue inspecting main.go." {
		t.Fatalf("unexpected summary: %q", payload.Summary)
	}
	if len(payload.Retained) != 2 {
		t.Fatalf("expected active call and result to be forced, got %+v", payload.Retained)
	}
	if payload.Retained[0].ToolCalls[0].ID != "call_1" || payload.Retained[1].ToolCallID != "call_1" {
		t.Fatalf("tool state was not retained in order: %+v", payload.Retained)
	}
}

func TestCompactionNamesBuiltInSearchToolMetadata(t *testing.T) {
	body := []byte(`{
		"model":"test",
		"stream":true,
		"tools":[{"type":"tool_search"},{"type":"web_search"}],
		"input":[
			{"type":"message","role":"user","content":"find current information"},
			{"type":"compaction_trigger"}
		]
	}`)
	plan, requested, err := PrepareTriggeredCompaction(body)
	if err != nil || !requested {
		t.Fatalf("prepare: requested=%v err=%v", requested, err)
	}

	requestBody, err := plan.SummaryRequest("")
	if err != nil {
		t.Fatal(err)
	}
	transcript := summaryTranscriptText(t, requestBody)
	if !strings.Contains(transcript, `"name":"tool_search"`) ||
		!strings.Contains(transcript, `"name":"web_search"`) {
		t.Fatalf("summary transcript is missing built-in tool names: %s", requestBody)
	}
	if strings.Contains(transcript, `"name":""`) {
		t.Fatalf("summary transcript contains unnamed tool metadata: %s", requestBody)
	}
}

const compactionTestPNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

func TestCompactionSendsImagesAsMultimodalInput(t *testing.T) {
	body := []byte(`{
		"model":"test","stream":true,
		"input":[
			{"type":"message","role":"user","content":[
				{"type":"input_text","text":"describe this"},
				{"type":"input_image","detail":"auto","image_url":"` + compactionTestPNG + `"}
			]},
			{"type":"compaction_trigger"}
		]
	}`)
	plan, requested, err := PrepareTriggeredCompaction(body)
	if err != nil || !requested {
		t.Fatalf("prepare: requested=%v err=%v", requested, err)
	}

	requestBody, err := plan.SummaryRequest("")
	if err != nil {
		t.Fatal(err)
	}
	var request struct {
		Input []struct {
			Content json.RawMessage `json:"content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(requestBody, &request); err != nil {
		t.Fatal(err)
	}
	if len(request.Input) != 2 {
		t.Fatalf("input count=%d, want 2", len(request.Input))
	}
	var blocks []struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL string `json:"image_url"`
	}
	if err := json.Unmarshal(request.Input[1].Content, &blocks); err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 2 || blocks[0].Type != "input_text" || blocks[1].Type != "input_image" {
		t.Fatalf("unexpected transcript blocks: %+v", blocks)
	}
	if !strings.Contains(blocks[0].Text, `"ref":"item_000001"`) || !strings.Contains(blocks[0].Text, `"image_count":1`) {
		t.Fatalf("image is not associated with its source item: %s", blocks[0].Text)
	}
	if strings.Contains(blocks[0].Text, "iVBOR") || blocks[1].ImageURL != compactionTestPNG {
		t.Fatalf("image must be a real image block, not transcript text: %+v", blocks)
	}
	var responsesRequest ResponsesRequest
	if err := json.Unmarshal(requestBody, &responsesRequest); err != nil {
		t.Fatal(err)
	}
	chatRequest, err := FromResponsesRequest(responsesRequest)
	if err != nil {
		t.Fatal(err)
	}
	if len(chatRequest.Messages) != 2 || len(chatRequest.Messages[1].Images) != 1 || !strings.Contains(chatRequest.Messages[1].Content, "item_000001") {
		t.Fatalf("local Responses conversion lost the image-to-item association: %+v", chatRequest.Messages)
	}
}

func TestCompactionImageRetentionIsModelSelectedAndReplayed(t *testing.T) {
	body := []byte(`{
		"model":"test","stream":true,
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_image","detail":"auto","image_url":"` + compactionTestPNG + `"}]},
			{"type":"message","role":"assistant","content":"I inspected it."},
			{"type":"compaction_trigger"}
		]
	}`)
	plan, requested, err := PrepareTriggeredCompaction(body)
	if err != nil || !requested {
		t.Fatalf("prepare: requested=%v err=%v", requested, err)
	}

	t.Run("selected", func(t *testing.T) {
		result, err := plan.Complete(compactionResponseBody(t, map[string]any{
			"summary": "The image was inspected.", "retain_item_ids": []string{"item_000001"},
		}))
		if err != nil {
			t.Fatal(err)
		}
		payload := decodeResultPayload(t, result)
		if len(payload.Retained) != 1 || len(payload.Retained[0].Images) != 1 {
			t.Fatalf("selected image was not retained: %+v", payload.Retained)
		}
		request, err := json.Marshal(map[string]any{"model": "test", "input": []any{result.Item}})
		if err != nil {
			t.Fatal(err)
		}
		expanded, changed, err := ExpandResponsesCompactionInput(request)
		if err != nil || !changed {
			t.Fatalf("expand: changed=%v err=%v", changed, err)
		}
		if !bytes.Contains(expanded, []byte(`"type":"input_image"`)) || !bytes.Contains(expanded, []byte(compactionTestPNG)) {
			t.Fatalf("retained image was not replayed as Responses content: %s", expanded)
		}
	})

	t.Run("not selected", func(t *testing.T) {
		result, err := plan.Complete(compactionResponseBody(t, map[string]any{
			"summary": "The image was inspected.", "retain_item_ids": []string{},
		}))
		if err != nil {
			t.Fatal(err)
		}
		if payload := decodeResultPayload(t, result); len(payload.Retained) != 0 {
			t.Fatalf("unselected image was retained: %+v", payload.Retained)
		}
	})
}

func TestCompactionReplaysRetainedToolOutputImagesWithTheirCall(t *testing.T) {
	body := []byte(`{
		"model":"test","stream":true,
		"input":[
			{"type":"function_call","call_id":"call_1","name":"screenshot","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_1","output":[
				{"type":"input_text","text":"captured"},
				{"type":"input_image","detail":"auto","image_url":"` + compactionTestPNG + `"}
			]},
			{"type":"message","role":"assistant","content":"done"},
			{"type":"compaction_trigger"}
		]
	}`)
	plan, requested, err := PrepareTriggeredCompaction(body)
	if err != nil || !requested {
		t.Fatalf("prepare: requested=%v err=%v", requested, err)
	}
	summaryRequest, err := plan.SummaryRequest("")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(summaryRequest, []byte(compactionTestPNG)) || !bytes.Contains(summaryRequest, []byte(`image_count\":1`)) {
		t.Fatalf("compactor request did not associate the tool image with its transcript item: %s", summaryRequest)
	}
	result, err := plan.Complete(compactionResponseBody(t, map[string]any{
		"summary": "A screenshot was captured.", "retain_item_ids": []string{"item_000002"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	payload := decodeResultPayload(t, result)
	if len(payload.Retained) != 2 || len(payload.Retained[0].ToolCalls) != 1 || len(payload.Retained[1].Images) != 1 {
		t.Fatalf("tool call and image result were not retained together: %+v", payload.Retained)
	}
	items, err := payloadToResponsesItems(payload)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 4 || rawInputItemType(items[2]) != "function_call" || rawInputItemType(items[3]) != "function_call_output" {
		t.Fatalf("tool pairing or order changed: %s", items)
	}
	if !bytes.Contains(items[3], []byte(`"type":"input_image"`)) || !bytes.Contains(items[3], []byte(compactionTestPNG)) {
		t.Fatalf("tool output image was not replayed: %s", items[3])
	}
}

func TestCompactionForcesActiveToolOutputImage(t *testing.T) {
	body := []byte(`{
		"model":"test","stream":true,
		"input":[
			{"type":"function_call","call_id":"call_1","name":"screenshot","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_1","output":[{"type":"input_image","detail":"auto","image_url":"` + compactionTestPNG + `"}]},
			{"type":"compaction_trigger"}
		]
	}`)
	plan, requested, err := PrepareTriggeredCompaction(body)
	if err != nil || !requested {
		t.Fatalf("prepare: requested=%v err=%v", requested, err)
	}
	result, err := plan.Complete(compactionResponseBody(t, map[string]any{
		"summary": "The screenshot tool just completed.", "retain_item_ids": []string{},
	}))
	if err != nil {
		t.Fatal(err)
	}
	payload := decodeResultPayload(t, result)
	if len(payload.Retained) != 2 || len(payload.Retained[1].Images) != 1 {
		t.Fatalf("active tool state and its image must be forced: %+v", payload.Retained)
	}
}

func TestCompactionRejectsUnsupportedImageSources(t *testing.T) {
	for name, image := range map[string]string{
		"file id":    `{"type":"input_image","detail":"auto","file_id":"file_123"}`,
		"remote URL": `{"type":"input_image","detail":"auto","image_url":"https://example.com/image.png"}`,
	} {
		t.Run(name, func(t *testing.T) {
			body := []byte(`{"model":"test","stream":true,"input":[{"type":"message","role":"user","content":[` + image + `]},{"type":"compaction_trigger"}]}`)
			_, requested, err := PrepareTriggeredCompaction(body)
			if !requested || err == nil {
				t.Fatalf("requested=%v err=%v", requested, err)
			}
		})
	}
}

func TestPrepareTriggeredCompactionRequiresStreaming(t *testing.T) {
	body := []byte(`{"model":"test","input":[{"type":"message","role":"user","content":"hi"},{"type":"compaction_trigger"}]}`)
	_, requested, err := PrepareTriggeredCompaction(body)
	if !requested || err == nil || !strings.Contains(err.Error(), "stream=true") {
		t.Fatalf("requested=%v err=%v", requested, err)
	}
}

func TestCompactionSelectionKeepsCompleteToolGroupInOriginalOrder(t *testing.T) {
	body := []byte(`{
		"model":"test",
		"stream":true,
		"input":[
			{"type":"message","role":"user","content":"run it"},
			{"type":"function_call","call_id":"call_1","name":"shell","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_1","output":"ok"},
			{"type":"message","role":"assistant","content":"done"},
			{"type":"compaction_trigger"}
		]
	}`)
	plan, requested, err := PrepareTriggeredCompaction(body)
	if err != nil || !requested {
		t.Fatalf("prepare: requested=%v err=%v", requested, err)
	}

	result, err := plan.Complete(compactionResponseBody(t, map[string]any{
		"summary": "The command completed.", "retain_item_ids": []string{"item_000003"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	payload := decodeResultPayload(t, result)
	if len(payload.Retained) != 2 {
		t.Fatalf("expected complete tool group, got %+v", payload.Retained)
	}
	if len(payload.Retained[0].ToolCalls) != 1 || payload.Retained[1].Role != "tool" {
		t.Fatalf("tool group order changed: %+v", payload.Retained)
	}
}

func TestCompactionPreservesNamespacedFunctionCallIdentity(t *testing.T) {
	body := []byte(`{
		"model":"test",
		"stream":true,
		"tools":[{"type":"namespace","name":"mcp__codex_apps__github","tools":[{"type":"function","name":"_get_repo","parameters":{"type":"object"}}]}],
		"input":[
			{"type":"function_call","call_id":"call_repo","namespace":"mcp__codex_apps__github","name":"_get_repo","arguments":"{\"repo\":\"ollama/ollama\"}"},
			{"type":"function_call_output","call_id":"call_repo","output":"found"},
			{"type":"message","role":"assistant","content":"done"},
			{"type":"compaction_trigger"}
		]
	}`)
	plan, requested, err := PrepareTriggeredCompaction(body)
	if err != nil || !requested {
		t.Fatalf("prepare: requested=%v err=%v", requested, err)
	}

	requestBody, err := plan.SummaryRequest("")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(requestBody, []byte(`mcp__codex_apps__github_get_repo`)) {
		t.Fatalf("summary transcript lost the namespaced tool identity: %s", requestBody)
	}

	result, err := plan.Complete(compactionResponseBody(t, map[string]any{
		"summary": "The repository was found.", "retain_item_ids": []string{"item_000002"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	payload := decodeResultPayload(t, result)
	if len(payload.Retained) != 2 {
		t.Fatalf("expected complete namespaced tool group, got %+v", payload.Retained)
	}
	if got := payload.Retained[0].ToolCalls[0].Function.Name; got != "mcp__codex_apps__github_get_repo" {
		t.Fatalf("retained tool name = %q", got)
	}
}

func TestCompactionPreservesToolSearchStateAndAcceptsWebSearchHistory(t *testing.T) {
	body := []byte(`{
		"model":"test",
		"stream":true,
		"input":[
			{"type":"message","role":"user","content":"find the issue"},
			{"type":"tool_search_call","call_id":"call_search","execution":"client","status":"completed","arguments":{"query":"github"}},
			{"type":"tool_search_output","call_id":"call_search","execution":"client","status":"completed","tools":[{"type":"namespace","name":"mcp__codex_apps__github","tools":[{"type":"function","name":"_search"}]}]},
			{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"current issue"}},
			{"type":"message","role":"assistant","content":"I found it."},
			{"type":"compaction_trigger"}
		]
	}`)
	plan, requested, err := PrepareTriggeredCompaction(body)
	if err != nil || !requested {
		t.Fatalf("prepare: requested=%v err=%v", requested, err)
	}

	result, err := plan.Complete(compactionResponseBody(t, map[string]any{
		"summary": "The issue was found.", "retain_item_ids": []string{"item_000003"},
	}))
	if err != nil {
		t.Fatal(err)
	}
	payload := decodeResultPayload(t, result)
	if len(payload.Retained) != 2 || payload.Retained[0].ToolCalls[0].Function.Name != "tool_search" || payload.Retained[1].ToolName != "tool_search" {
		t.Fatalf("tool search state was not retained as a pair: %+v", payload.Retained)
	}

	expandedBody, err := json.Marshal(map[string]any{
		"model": "test",
		"input": []any{result.Item, map[string]any{"type": "message", "role": "user", "content": "continue"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	rewritten, changed, err := ExpandResponsesCompactionInput(expandedBody)
	if err != nil || !changed {
		t.Fatalf("expand: changed=%v err=%v", changed, err)
	}
	var request struct {
		Input []json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(rewritten, &request); err != nil {
		t.Fatal(err)
	}
	wantTypes := []string{"function_call", "function_call_output", "tool_search_call", "tool_search_output", "message"}
	if len(request.Input) != len(wantTypes) {
		t.Fatalf("expanded input = %s", rewritten)
	}
	for i, want := range wantTypes {
		if got := rawInputItemType(request.Input[i]); got != want {
			t.Fatalf("expanded input[%d] type = %q, want %q: %s", i, got, want, rewritten)
		}
	}
}

func TestCompactionRejectsInvalidToolStateAndSelections(t *testing.T) {
	t.Run("orphan result", func(t *testing.T) {
		body := []byte(`{"model":"test","stream":true,"input":[{"type":"function_call_output","call_id":"missing","output":"x"},{"type":"compaction_trigger"}]}`)
		_, requested, err := PrepareTriggeredCompaction(body)
		if !requested || err == nil || !strings.Contains(err.Error(), "no matching call") {
			t.Fatalf("requested=%v err=%v", requested, err)
		}
	})

	t.Run("unknown ref", func(t *testing.T) {
		body := []byte(`{"model":"test","stream":true,"input":[{"type":"message","role":"user","content":"hi"},{"type":"compaction_trigger"}]}`)
		plan, _, err := PrepareTriggeredCompaction(body)
		if err != nil {
			t.Fatal(err)
		}
		_, err = plan.Complete(compactionResponseBody(t, map[string]any{
			"summary": "hi", "retain_item_ids": []string{"missing"},
		}))
		if err == nil || !strings.Contains(err.Error(), "unknown item") {
			t.Fatalf("expected unknown item error, got %v", err)
		}
	})
}

func TestExpandResponsesCompactionInputPreservesCodexRetainedPrefix(t *testing.T) {
	payload := OllamaCompactionPayload{
		Type: OllamaCompactionPayloadType, Version: OllamaCompactionPayloadVersion,
		Summary: "The user chose option A.",
		Retained: []api.Message{{Role: "assistant", ToolCalls: []api.ToolCall{{
			ID: "call_live", Function: api.ToolCallFunction{Name: "shell", Arguments: api.NewToolCallFunctionArguments()},
		}}}},
	}
	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]any{
		"model": "test",
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "first user turn retained by Codex"},
			map[string]any{"type": "message", "role": "user", "content": "second user turn retained by Codex"},
			ResponsesCompactionItem{Type: "compaction", EncryptedContent: string(payloadJSON)},
			map[string]any{"type": "message", "role": "user", "content": "new turn"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	rewritten, changed, err := ExpandResponsesCompactionInput(body)
	if err != nil {
		t.Fatal(err)
	}
	if !changed {
		t.Fatal("expected rewritten input")
	}
	var request struct {
		Input []json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(rewritten, &request); err != nil {
		t.Fatal(err)
	}
	if len(request.Input) != 6 {
		t.Fatalf("expected retained prefix, summary pair, retained call, and new turn; got %s", rewritten)
	}
	var types []string
	for _, item := range request.Input {
		types = append(types, rawInputItemType(item))
	}
	want := []string{"message", "message", "function_call", "function_call_output", "function_call", "message"}
	for i := range want {
		if types[i] != want[i] {
			t.Fatalf("types=%v want=%v", types, want)
		}
	}
	for _, marker := range []string{"first user turn retained by Codex", "second user turn retained by Codex"} {
		if strings.Count(string(rewritten), marker) != 1 {
			t.Fatalf("retained user message %q was lost or duplicated: %s", marker, rewritten)
		}
	}
	if !strings.Contains(string(rewritten), "new turn") {
		t.Fatalf("post-boundary input was lost: %s", rewritten)
	}
}

func TestExpandResponsesCompactionInputRejectsForeignPayload(t *testing.T) {
	body := []byte(`{"model":"test","input":[{"type":"compaction","encrypted_content":"opaque-provider-state"}]}`)
	_, changed, err := ExpandResponsesCompactionInput(body)
	if changed || err == nil || !strings.Contains(err.Error(), "not an Ollama payload") {
		t.Fatalf("changed=%v err=%v", changed, err)
	}
}

func TestRepeatedCompactionReplacesRatherThanNestsPayload(t *testing.T) {
	oldPayload, err := json.Marshal(OllamaCompactionPayload{
		Type: OllamaCompactionPayloadType, Version: OllamaCompactionPayloadVersion, Summary: "old summary",
	})
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]any{
		"model": "test", "stream": true,
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "retained prefix"},
			ResponsesCompactionItem{Type: "compaction", EncryptedContent: string(oldPayload)},
			map[string]any{"type": "message", "role": "user", "content": "new work"},
			ResponsesCompactionTrigger{Type: "compaction_trigger"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	plan, requested, err := PrepareTriggeredCompaction(body)
	if err != nil || !requested {
		t.Fatalf("prepare: requested=%v err=%v", requested, err)
	}
	result, err := plan.Complete(compactionResponseBody(t, map[string]any{
		"summary": "replacement summary", "retain_item_ids": []string{},
	}))
	if err != nil {
		t.Fatal(err)
	}
	payload := decodeResultPayload(t, result)
	if len(payload.Retained) != 0 {
		t.Fatalf("old synthetic summary pair was retained: %+v", payload.Retained)
	}
	if strings.Contains(result.Item.EncryptedContent, "old summary") || strings.Contains(result.Item.EncryptedContent, "retained prefix") {
		t.Fatalf("old compaction was nested: %s", result.Item.EncryptedContent)
	}
}

func TestCompactionStreamContainsExactlyOneCompletedItem(t *testing.T) {
	result := ResponsesCompactionResult{Item: ResponsesCompactionItem{Type: "compaction", EncryptedContent: `{"type":"ollama_compaction","version":1}`}}
	events := NewResponsesCompactionStreamEvents("resp_1", "test", result)
	var done, completed int
	for _, event := range events {
		switch event.Event {
		case "response.output_item.done":
			done++
			data := event.Data.(map[string]any)
			item := data["item"].(ResponsesCompactionItem)
			if item.Type != "compaction" {
				t.Fatalf("unexpected item: %+v", item)
			}
		case "response.completed":
			completed++
		}
	}
	if done != 1 || completed != 1 {
		t.Fatalf("done=%d completed=%d events=%+v", done, completed, events)
	}
}
