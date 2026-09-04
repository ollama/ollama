package openai

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	OllamaCompactionPayloadType    = "ollama_compaction"
	OllamaCompactionPayloadVersion = 1
	CreateSummaryToolName          = "create_summary"
	compactionSummaryToolName      = "ollama_compaction_summary"
)

// ResponsesCompactionTrigger is the terminal input item sent by current Codex
// clients when they request remote compaction through POST /v1/responses.
type ResponsesCompactionTrigger struct {
	Type string `json:"type"`
}

func (ResponsesCompactionTrigger) responsesInputItem() {}

// ResponsesCompactionItem is the opaque continuation item understood by Codex.
// Ollama stores a versioned JSON payload in EncryptedContent; the payload is not
// encrypted and must be consumed by Ollama before another model request.
type ResponsesCompactionItem struct {
	Type             string `json:"type"`
	EncryptedContent string `json:"encrypted_content"`
}

func (ResponsesCompactionItem) responsesInputItem() {}

// OllamaCompactionPayload is Ollama's stateless continuation format.
type OllamaCompactionPayload struct {
	Type     string        `json:"type"`
	Version  int           `json:"version"`
	Summary  string        `json:"summary"`
	Retained []api.Message `json:"retained"`
}

// CompactionTranscriptItem is one ordered input item shown to the compaction
// model. Ref is request-local and is the only value the model may select.
type CompactionTranscriptItem struct {
	Ref     string      `json:"ref"`
	Type    string      `json:"type"`
	Message api.Message `json:"message"`
}

type compactionToolMetadata struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

type compactionTranscript struct {
	Items []CompactionTranscriptItem `json:"items"`
	Tools []compactionToolMetadata   `json:"tools,omitempty"`
}

type compactionToolGroup struct {
	CallRef   string
	ResultRef string
}

// ResponsesCompactionPlan contains the validated, ordered state required to
// make and verify a compaction-model request.
type ResponsesCompactionPlan struct {
	Model  string
	Stream bool

	items      []CompactionTranscriptItem
	tools      []compactionToolMetadata
	groups     []compactionToolGroup
	forcedRefs map[string]struct{}
}

// ResponsesCompactionResult is the validated result of a compaction-model call.
type ResponsesCompactionResult struct {
	Item  ResponsesCompactionItem
	Usage *ResponsesUsage
}

type summarySelection struct {
	Summary       string   `json:"summary"`
	RetainItemIDs []string `json:"retain_item_ids"`
}

type rawResponsesRequest struct {
	Model  string                     `json:"model"`
	Input  json.RawMessage            `json:"input"`
	Stream *bool                      `json:"stream,omitempty"`
	Tools  []ResponsesTool            `json:"tools,omitempty"`
	Fields map[string]json.RawMessage `json:"-"`
}

// PrepareTriggeredCompaction recognizes the Codex v2 terminal trigger. It
// returns requested=false without changing ordinary Responses requests.
func PrepareTriggeredCompaction(body []byte) (*ResponsesCompactionPlan, bool, error) {
	req, items, err := decodeRawResponsesRequest(body)
	if err != nil {
		return nil, false, err
	}
	if len(items) == 0 || rawInputItemType(items[len(items)-1]) != "compaction_trigger" {
		return nil, false, nil
	}
	if req.Stream == nil || !*req.Stream {
		return nil, true, errors.New("compaction_trigger requires stream=true")
	}
	for _, item := range items[:len(items)-1] {
		if rawInputItemType(item) == "compaction_trigger" {
			return nil, true, errors.New("compaction_trigger must be the final and only compaction trigger")
		}
	}

	items, _, err = expandOllamaCompactionItems(items[:len(items)-1])
	if err != nil {
		return nil, true, err
	}
	plan, err := newResponsesCompactionPlan(req, items)
	return plan, true, err
}

// PrepareStandaloneCompaction validates a POST /v1/responses/compact body.
func PrepareStandaloneCompaction(body []byte) (*ResponsesCompactionPlan, error) {
	req, items, err := decodeRawResponsesRequest(body)
	if err != nil {
		return nil, err
	}
	for _, item := range items {
		if rawInputItemType(item) == "compaction_trigger" {
			return nil, errors.New("compaction_trigger is only valid on POST /v1/responses")
		}
	}
	items, _, err = expandOllamaCompactionItems(items)
	if err != nil {
		return nil, err
	}
	return newResponsesCompactionPlan(req, items)
}

// ExpandResponsesCompactionInput replaces the newest Ollama compaction item
// with its summary and retained messages. Items on either side of the
// compaction item remain in their original order.
func ExpandResponsesCompactionInput(body []byte) ([]byte, bool, error) {
	req, items, err := decodeRawResponsesRequest(body)
	if err != nil {
		return nil, false, err
	}
	items, changed, err := expandOllamaCompactionItems(items)
	if err != nil || !changed {
		return body, changed, err
	}

	input, err := json.Marshal(items)
	if err != nil {
		return nil, false, err
	}
	req.Fields["input"] = input
	rewritten, err := json.Marshal(req.Fields)
	return rewritten, true, err
}

func decodeRawResponsesRequest(body []byte) (rawResponsesRequest, []json.RawMessage, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return rawResponsesRequest{}, nil, fmt.Errorf("invalid Responses request: %w", err)
	}

	var req rawResponsesRequest
	if raw, ok := fields["model"]; !ok || json.Unmarshal(raw, &req.Model) != nil || strings.TrimSpace(req.Model) == "" {
		return rawResponsesRequest{}, nil, errors.New("model is required")
	}
	req.Model = strings.TrimSpace(req.Model)
	req.Fields = fields
	req.Input = fields["input"]
	if raw := fields["stream"]; len(raw) > 0 {
		if err := json.Unmarshal(raw, &req.Stream); err != nil {
			return rawResponsesRequest{}, nil, fmt.Errorf("invalid stream value: %w", err)
		}
	}
	if raw := fields["tools"]; len(raw) > 0 {
		if err := json.Unmarshal(raw, &req.Tools); err != nil {
			return rawResponsesRequest{}, nil, fmt.Errorf("invalid tools: %w", err)
		}
	}

	items, err := decodeRawResponsesInput(req.Input)
	if err != nil {
		return rawResponsesRequest{}, nil, err
	}
	return req, items, nil
}

func decodeRawResponsesInput(input json.RawMessage) ([]json.RawMessage, error) {
	if len(input) == 0 || bytes.Equal(bytes.TrimSpace(input), []byte("null")) {
		return nil, errors.New("input is required")
	}

	var text string
	if err := json.Unmarshal(input, &text); err == nil {
		item, err := json.Marshal(map[string]any{
			"type": "message", "role": "user", "content": text,
		})
		if err != nil {
			return nil, err
		}
		return []json.RawMessage{item}, nil
	}

	var items []json.RawMessage
	if err := json.Unmarshal(input, &items); err != nil {
		return nil, fmt.Errorf("input must be a string or array: %w", err)
	}
	if items == nil {
		return nil, errors.New("input must be a string or array")
	}
	return items, nil
}

func rawInputItemType(item json.RawMessage) string {
	var header struct {
		Type string `json:"type"`
		Role string `json:"role"`
	}
	if json.Unmarshal(item, &header) != nil {
		return ""
	}
	if header.Type == "" && header.Role != "" {
		return "message"
	}
	return header.Type
}

func expandOllamaCompactionItems(items []json.RawMessage) ([]json.RawMessage, bool, error) {
	boundary := -1
	for i := range items {
		if rawInputItemType(items[i]) == "compaction" {
			boundary = i
		}
	}
	if boundary < 0 {
		return items, false, nil
	}

	payload, err := decodeOllamaCompactionItem(items[boundary])
	if err != nil {
		return nil, false, err
	}
	expanded, err := payloadToResponsesItems(payload)
	if err != nil {
		return nil, false, err
	}
	rewritten := make([]json.RawMessage, 0, len(items)-1+len(expanded))
	rewritten = append(rewritten, items[:boundary]...)
	rewritten = append(rewritten, expanded...)
	rewritten = append(rewritten, items[boundary+1:]...)
	return rewritten, true, nil
}

func decodeOllamaCompactionItem(item json.RawMessage) (OllamaCompactionPayload, error) {
	var wire ResponsesCompactionItem
	if err := json.Unmarshal(item, &wire); err != nil {
		return OllamaCompactionPayload{}, fmt.Errorf("invalid compaction item: %w", err)
	}
	if wire.Type != "compaction" || wire.EncryptedContent == "" {
		return OllamaCompactionPayload{}, errors.New("invalid compaction item")
	}

	var payload OllamaCompactionPayload
	decoder := json.NewDecoder(strings.NewReader(wire.EncryptedContent))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&payload); err != nil {
		return OllamaCompactionPayload{}, errors.New("unsupported compaction item: encrypted_content is not an Ollama payload")
	}
	if payload.Type != OllamaCompactionPayloadType || payload.Version != OllamaCompactionPayloadVersion {
		return OllamaCompactionPayload{}, fmt.Errorf("unsupported Ollama compaction payload type or version")
	}
	if strings.TrimSpace(payload.Summary) == "" {
		return OllamaCompactionPayload{}, errors.New("Ollama compaction payload has an empty summary")
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return OllamaCompactionPayload{}, errors.New("unsupported compaction item: encrypted_content contains more than one JSON value")
	}
	return payload, nil
}

func payloadToResponsesItems(payload OllamaCompactionPayload) ([]json.RawMessage, error) {
	b, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	hash := sha256.Sum256(b)
	callID := "call_ollama_compaction_" + hex.EncodeToString(hash[:6])
	args, _ := json.Marshal(map[string]any{"version": payload.Version})

	items := make([]json.RawMessage, 0, 2+len(payload.Retained))
	call, err := json.Marshal(map[string]any{
		"type": "function_call", "call_id": callID, "name": compactionSummaryToolName, "arguments": string(args),
	})
	if err != nil {
		return nil, err
	}
	result, err := json.Marshal(map[string]any{
		"type": "function_call_output", "call_id": callID, "output": payload.Summary,
	})
	if err != nil {
		return nil, err
	}
	items = append(items, call, result)
	for _, message := range payload.Retained {
		converted, err := messageToResponsesItems(message)
		if err != nil {
			return nil, fmt.Errorf("invalid retained message: %w", err)
		}
		items = append(items, converted...)
	}
	return items, nil
}

func messageToResponsesItems(message api.Message) ([]json.RawMessage, error) {
	if len(message.Images) > 0 {
		return nil, errors.New("retained images are not supported")
	}

	var values []any
	if message.Thinking != "" {
		values = append(values, map[string]any{
			"type":    "reasoning",
			"summary": []map[string]string{{"type": "summary_text", "text": message.Thinking}},
		})
	}
	if message.Role == "tool" {
		if message.ToolCallID == "" {
			return nil, errors.New("retained tool message is missing tool_call_id")
		}
		values = append(values, map[string]any{
			"type": "function_call_output", "call_id": message.ToolCallID, "output": message.Content,
		})
	} else if message.Content != "" || len(message.ToolCalls) == 0 {
		values = append(values, map[string]any{
			"type": "message", "role": message.Role, "content": message.Content,
		})
	}
	for _, call := range message.ToolCalls {
		if call.ID == "" || call.Function.Name == "" {
			return nil, errors.New("retained function call is missing call_id or name")
		}
		arguments, err := json.Marshal(call.Function.Arguments)
		if err != nil {
			return nil, err
		}
		values = append(values, map[string]any{
			"type": "function_call", "call_id": call.ID, "name": call.Function.Name, "arguments": string(arguments),
		})
	}

	items := make([]json.RawMessage, 0, len(values))
	for _, value := range values {
		item, err := json.Marshal(value)
		if err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, nil
}

func newResponsesCompactionPlan(req rawResponsesRequest, rawItems []json.RawMessage) (*ResponsesCompactionPlan, error) {
	if len(rawItems) == 0 {
		return nil, errors.New("compaction input is empty")
	}

	items := make([]CompactionTranscriptItem, 0, len(rawItems))
	for i, raw := range rawItems {
		item, err := unmarshalResponsesInputItem(raw)
		if err != nil {
			return nil, fmt.Errorf("input[%d]: %w", i, err)
		}
		message, kind, err := compactionMessage(item)
		if err != nil {
			return nil, fmt.Errorf("input[%d]: %w", i, err)
		}
		if len(message.Images) > 0 {
			return nil, fmt.Errorf("input[%d]: image inputs are not supported by compaction", i)
		}
		items = append(items, CompactionTranscriptItem{
			Ref: fmt.Sprintf("item_%06d", i+1), Type: kind, Message: message,
		})
	}

	groups, forced, err := analyzeCompactionToolState(items)
	if err != nil {
		return nil, err
	}
	stream := req.Stream != nil && *req.Stream
	return &ResponsesCompactionPlan{
		Model: req.Model, Stream: stream, items: items, tools: collectCompactionToolMetadata(req.Tools), groups: groups, forcedRefs: forced,
	}, nil
}

func compactionMessage(item ResponsesInputItem) (api.Message, string, error) {
	switch value := item.(type) {
	case ResponsesInputMessage:
		message, err := convertInputMessage(value)
		return message, "message", err
	case ResponsesFunctionCall:
		var arguments api.ToolCallFunctionArguments
		if value.Arguments != "" {
			if err := json.Unmarshal([]byte(value.Arguments), &arguments); err != nil {
				return api.Message{}, "", fmt.Errorf("invalid function arguments: %w", err)
			}
		}
		return api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
			ID: value.CallID, Function: api.ToolCallFunction{Name: value.Name, Arguments: arguments},
		}}}, "function_call", nil
	case ResponsesFunctionCallOutput:
		content := value.Output
		var images []api.ImageData
		if len(value.OutputItems) > 0 {
			var err error
			content, images, err = convertResponsesContent(value.OutputItems)
			if err != nil {
				return api.Message{}, "", err
			}
		}
		return api.Message{Role: "tool", Content: content, Images: images, ToolCallID: value.CallID}, "function_call_output", nil
	case ResponsesReasoningInput:
		var summary strings.Builder
		for _, part := range value.Summary {
			summary.WriteString(part.Text)
		}
		thinking := summary.String()
		if thinking == "" && value.EncryptedContent != "" {
			thinking = "[opaque reasoning state omitted during Ollama compaction]"
		}
		return api.Message{Role: "assistant", Thinking: thinking}, "reasoning", nil
	case ResponsesCompactionItem, ResponsesCompactionTrigger:
		return api.Message{}, "", errors.New("unexpected compaction control item")
	default:
		return api.Message{}, "", fmt.Errorf("unsupported compaction input type %T", item)
	}
}

func analyzeCompactionToolState(items []CompactionTranscriptItem) ([]compactionToolGroup, map[string]struct{}, error) {
	type pendingGroup struct {
		callIndex   int
		resultIndex int
		group       compactionToolGroup
	}
	byCallID := make(map[string]*pendingGroup)
	ignoredCallIDs := make(map[string]struct{})
	var ordered []*pendingGroup

	for i, item := range items {
		switch item.Type {
		case "function_call":
			call := item.Message.ToolCalls[0]
			if call.ID == "" {
				return nil, nil, fmt.Errorf("%s: function call is missing call_id", item.Ref)
			}
			if call.Function.Name == compactionSummaryToolName && strings.HasPrefix(call.ID, "call_ollama_compaction_") {
				ignoredCallIDs[call.ID] = struct{}{}
				continue
			}
			if _, exists := byCallID[call.ID]; exists {
				return nil, nil, fmt.Errorf("%s: duplicate function call_id %q", item.Ref, call.ID)
			}
			group := &pendingGroup{callIndex: i, resultIndex: -1, group: compactionToolGroup{CallRef: item.Ref}}
			byCallID[call.ID] = group
			ordered = append(ordered, group)
		case "function_call_output":
			callID := item.Message.ToolCallID
			if _, ignored := ignoredCallIDs[callID]; ignored {
				continue
			}
			group := byCallID[callID]
			if group == nil {
				return nil, nil, fmt.Errorf("%s: function output has no matching call %q", item.Ref, callID)
			}
			if group.resultIndex >= 0 {
				return nil, nil, fmt.Errorf("%s: duplicate function output for call %q", item.Ref, callID)
			}
			group.resultIndex = i
			group.group.ResultRef = item.Ref
		}
	}

	forced := make(map[string]struct{})
	groups := make([]compactionToolGroup, 0, len(ordered))
	for _, candidate := range ordered {
		groups = append(groups, candidate.group)
		active := candidate.resultIndex < 0
		if !active {
			active = true
			for _, item := range items[candidate.resultIndex+1:] {
				if item.Type == "reasoning" || item.Type == "function_call" || (item.Type == "message" && item.Message.Role == "assistant") {
					active = false
					break
				}
			}
		}
		if active {
			forced[candidate.group.CallRef] = struct{}{}
			if candidate.group.ResultRef != "" {
				forced[candidate.group.ResultRef] = struct{}{}
			}
		}
	}
	return groups, forced, nil
}

func collectCompactionToolMetadata(tools []ResponsesTool) []compactionToolMetadata {
	var metadata []compactionToolMetadata
	var add func(prefix string, tools []ResponsesTool)
	add = func(prefix string, tools []ResponsesTool) {
		for _, tool := range tools {
			name := tool.Name
			if prefix != "" && !strings.HasPrefix(name, prefix+".") {
				name = prefix + "." + name
			}
			if tool.Type == "namespace" {
				add(name, tool.Tools)
				continue
			}
			description := ""
			if tool.Description != nil {
				description = *tool.Description
			}
			metadata = append(metadata, compactionToolMetadata{Name: name, Description: description})
		}
	}
	add("", tools)
	return metadata
}

// SummaryRequest returns an ordinary non-streaming Responses request. A repair
// request includes the validation error from the first model response.
func (p *ResponsesCompactionPlan) SummaryRequest(repairError string) ([]byte, error) {
	// TODO(compaction): enforce the 40% retained-context target after the
	// selected model's prompt renderer and token budget are available here.
	transcript, err := json.Marshal(compactionTranscript{Items: p.items, Tools: p.tools})
	if err != nil {
		return nil, err
	}
	prompt := `Summarize the conversation for another coding agent. Preserve the goal, decisions, constraints, repository state, changed files, test results, failures, active work, and next actions. Use retain_item_ids only for exact source items that cannot safely be paraphrased. Tool calls and results are execution state; do not invent or edit them. Call create_summary exactly once.`
	if repairError != "" {
		prompt += " Your previous create_summary call was invalid: " + repairError + ". Return one corrected create_summary call."
	}

	description := "Return the compact summary and the exact input item references that must remain verbatim."
	strict := true
	request := map[string]any{
		"model": p.Model,
		"input": []any{
			map[string]any{"type": "message", "role": "system", "content": prompt},
			map[string]any{"type": "message", "role": "user", "content": string(transcript)},
		},
		"tools": []ResponsesTool{{
			Type: "function", Name: CreateSummaryToolName, Description: &description, Strict: &strict,
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"summary":         map[string]any{"type": "string"},
					"retain_item_ids": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
				},
				"required":             []string{"summary", "retain_item_ids"},
				"additionalProperties": false,
			},
		}},
		"tool_choice":         map[string]any{"type": "function", "name": CreateSummaryToolName},
		"parallel_tool_calls": false,
		"store":               false,
		"stream":              false,
	}
	return json.Marshal(request)
}

// Complete validates create_summary and builds the stateless continuation.
func (p *ResponsesCompactionPlan) Complete(body []byte) (ResponsesCompactionResult, error) {
	var response ResponsesResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return ResponsesCompactionResult{}, fmt.Errorf("invalid summary response: %w", err)
	}

	var calls []ResponsesOutputItem
	for _, item := range response.Output {
		if item.Type == "function_call" && item.Name == CreateSummaryToolName {
			calls = append(calls, item)
		}
	}
	if len(calls) != 1 {
		return ResponsesCompactionResult{}, fmt.Errorf("expected one %s call, got %d", CreateSummaryToolName, len(calls))
	}

	var selection summarySelection
	if err := json.Unmarshal([]byte(calls[0].Arguments), &selection); err != nil {
		return ResponsesCompactionResult{}, fmt.Errorf("invalid %s arguments: %w", CreateSummaryToolName, err)
	}
	selection.Summary = strings.TrimSpace(selection.Summary)
	if selection.Summary == "" {
		return ResponsesCompactionResult{}, errors.New("create_summary returned an empty summary")
	}

	known := make(map[string]struct{}, len(p.items))
	for _, item := range p.items {
		known[item.Ref] = struct{}{}
	}
	selected := make(map[string]struct{}, len(selection.RetainItemIDs)+len(p.forcedRefs))
	for _, ref := range selection.RetainItemIDs {
		if _, ok := known[ref]; !ok {
			return ResponsesCompactionResult{}, fmt.Errorf("create_summary selected unknown item %q", ref)
		}
		if _, duplicate := selected[ref]; duplicate {
			return ResponsesCompactionResult{}, fmt.Errorf("create_summary selected duplicate item %q", ref)
		}
		selected[ref] = struct{}{}
	}
	for ref := range p.forcedRefs {
		selected[ref] = struct{}{}
	}
	for _, group := range p.groups {
		_, callSelected := selected[group.CallRef]
		_, resultSelected := selected[group.ResultRef]
		if callSelected || resultSelected {
			selected[group.CallRef] = struct{}{}
			if group.ResultRef != "" {
				selected[group.ResultRef] = struct{}{}
			}
		}
	}

	retained := make([]api.Message, 0, len(selected))
	for _, item := range p.items {
		if _, ok := selected[item.Ref]; ok {
			retained = append(retained, item.Message)
		}
	}
	payload := OllamaCompactionPayload{
		Type: OllamaCompactionPayloadType, Version: OllamaCompactionPayloadVersion, Summary: selection.Summary, Retained: retained,
	}
	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		return ResponsesCompactionResult{}, err
	}
	return ResponsesCompactionResult{
		Item: ResponsesCompactionItem{Type: "compaction", EncryptedContent: string(payloadJSON)}, Usage: response.Usage,
	}, nil
}

// ResponsesCompactedResponse is returned by POST /v1/responses/compact.
type ResponsesCompactedResponse struct {
	ID        string                    `json:"id"`
	Object    string                    `json:"object"`
	CreatedAt int64                     `json:"created_at"`
	Output    []ResponsesCompactionItem `json:"output"`
	Usage     *ResponsesUsage           `json:"usage"`
}

// NewResponsesCompactedResponse builds the standalone compact response.
func NewResponsesCompactedResponse(id string, result ResponsesCompactionResult) ResponsesCompactedResponse {
	return ResponsesCompactedResponse{
		ID: id, Object: "response.compaction", CreatedAt: time.Now().Unix(), Output: []ResponsesCompactionItem{result.Item}, Usage: result.Usage,
	}
}

// NewResponsesCompactionStreamEvents builds the Codex v2 stream. Codex uses
// output_item.done and requires exactly one compaction item before completed.
func NewResponsesCompactionStreamEvents(id, model string, result ResponsesCompactionResult) []ResponsesStreamEvent {
	converter := NewResponsesStreamConverter(id, "", model, ResponsesRequest{Model: model})
	created := converter.buildResponseObject("in_progress", []any{}, nil)
	completedUsage := map[string]any(nil)
	if result.Usage != nil {
		completedUsage = map[string]any{
			"input_tokens":          result.Usage.InputTokens,
			"output_tokens":         result.Usage.OutputTokens,
			"total_tokens":          result.Usage.TotalTokens,
			"input_tokens_details":  result.Usage.InputTokensDetails,
			"output_tokens_details": result.Usage.OutputTokensDetails,
		}
	}
	completed := converter.buildResponseObject("completed", []any{result.Item}, completedUsage)
	completed["completed_at"] = time.Now().Unix()

	return []ResponsesStreamEvent{
		converter.newEvent("response.created", map[string]any{"response": created}),
		converter.newEvent("response.in_progress", map[string]any{"response": created}),
		converter.newEvent("response.output_item.added", map[string]any{"output_index": 0, "item": result.Item}),
		converter.newEvent("response.output_item.done", map[string]any{"output_index": 0, "item": result.Item}),
		converter.newEvent("response.completed", map[string]any{"response": completed}),
	}
}
