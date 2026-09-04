package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const (
	// autoReviewModel is the native Codex reviewer alias. The routing catalog
	// may map requests for this alias to a selected Ollama model.
	autoReviewModel          = "codex-auto-review"
	guardianDecisionToolName = "submit_guardian_decision"

	guardianDecisionInstruction = `

When you have finished reviewing the action, call submit_guardian_decision exactly once with your final decision. Do not return the final decision as assistant text.`
)

var guardianDecisionTool = map[string]any{
	"type":        "function",
	"name":        guardianDecisionToolName,
	"description": "Submit the final Codex Guardian approval decision after completing any necessary investigation.",
	"strict":      true,
	"parameters": map[string]any{
		"type": "object",
		"properties": map[string]any{
			"risk_level": map[string]any{
				"type": "string",
				"enum": []string{"low", "medium", "high", "critical"},
			},
			"user_authorization": map[string]any{
				"type": "string",
				"enum": []string{"unknown", "low", "medium", "high"},
			},
			"outcome": map[string]any{
				"type": "string",
				"enum": []string{"allow", "deny"},
			},
			"rationale": map[string]any{
				"type":      "string",
				"minLength": 1,
			},
		},
		"required":             []string{"risk_level", "user_authorization", "outcome", "rationale"},
		"additionalProperties": false,
	},
}

// autoReviewState is the per-request seam between the core proxy and the
// auto-review translation layer. The core calls its methods unconditionally;
// every branch that knows about Codex's native auto-review alias lives here.
type autoReviewState struct {
	alias    bool // requested model is the native auto-review alias
	eligible bool // routed auto-review call that needs translation
}

// resolveModel maps the native auto-review alias to the catalog's configured
// Ollama model and rewrites the request body. Other requests pass through
// unchanged.
func (s *autoReviewState) resolveModel(model string, catalog routingCatalog, body []byte) (string, []byte, error) {
	if modelKey(model) != modelKey(autoReviewModel) {
		return model, body, nil
	}
	s.alias = true
	if catalog.autoReviewModel == "" {
		return model, body, nil
	}
	replaced, err := replaceRequestModel(body, catalog.autoReviewModel)
	if err != nil {
		return catalog.autoReviewModel, body, err
	}
	return catalog.autoReviewModel, replaced, nil
}

// prepareRequest injects the Guardian decision tool into a routed auto-review
// call to the Responses endpoint. Other requests pass through unchanged.
func (s *autoReviewState) prepareRequest(routed bool, suffix string, body []byte) ([]byte, error) {
	s.eligible = s.alias && routed && suffix == "/v1/responses"
	if !s.eligible {
		return body, nil
	}
	return prepareAutoReviewRequest(body)
}

// buffersResponse reports whether a successful response must be fully read and
// transformed before it is sent to the client.
func (s *autoReviewState) buffersResponse(status int) bool {
	return s.eligible && status >= http.StatusOK && status < http.StatusMultipleChoices
}

// prepareAutoReviewRequest gives Ollama a typed terminal action for the
// Guardian decision. Codex does not know about this proxy-owned tool, so the
// response path consumes it and returns the JSON text Codex already expects.
func prepareAutoReviewRequest(body []byte) ([]byte, error) {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}

	var tools []json.RawMessage
	if raw := bytes.TrimSpace(payload["tools"]); len(raw) > 0 && !bytes.Equal(raw, []byte("null")) {
		if err := json.Unmarshal(raw, &tools); err != nil {
			return nil, fmt.Errorf("decode tools: %w", err)
		}
	}
	for _, raw := range tools {
		var tool struct {
			Name string `json:"name"`
		}
		if err := json.Unmarshal(raw, &tool); err != nil {
			return nil, fmt.Errorf("decode tool: %w", err)
		}
		if tool.Name == guardianDecisionToolName {
			return nil, fmt.Errorf("tool name %q is reserved by the Codex proxy", guardianDecisionToolName)
		}
	}
	decisionTool, err := json.Marshal(guardianDecisionTool)
	if err != nil {
		return nil, fmt.Errorf("encode Guardian decision tool: %w", err)
	}
	tools = append(tools, decisionTool)
	encodedTools, err := json.Marshal(tools)
	if err != nil {
		return nil, fmt.Errorf("encode tools: %w", err)
	}
	payload["tools"] = encodedTools

	var input []json.RawMessage
	if err := json.Unmarshal(payload["input"], &input); err != nil {
		return nil, fmt.Errorf("decode input: %w", err)
	}
	foundUserMessage := false
	for i := len(input) - 1; i >= 0; i-- {
		var message struct {
			Type    string          `json:"type"`
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		if err := json.Unmarshal(input[i], &message); err != nil {
			return nil, fmt.Errorf("decode input item: %w", err)
		}
		if message.Role != "user" || (message.Type != "" && message.Type != "message") {
			continue
		}

		content, changed, err := appendAutoReviewInstructionToContent(message.Content)
		if err != nil {
			return nil, err
		}
		if !changed {
			continue
		}
		foundUserMessage = true

		var item map[string]json.RawMessage
		if err := json.Unmarshal(input[i], &item); err != nil {
			return nil, fmt.Errorf("decode user message: %w", err)
		}
		item["content"] = content
		input[i], err = json.Marshal(item)
		if err != nil {
			return nil, fmt.Errorf("encode user message: %w", err)
		}
		break
	}
	if !foundUserMessage {
		return nil, errors.New("Guardian request has no user message")
	}
	encodedInput, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("encode input: %w", err)
	}
	payload["input"] = encodedInput
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	return encoded, nil
}

func appendAutoReviewInstructionToContent(content json.RawMessage) (json.RawMessage, bool, error) {
	trimmed := bytes.TrimSpace(content)
	if len(trimmed) == 0 {
		return content, false, nil
	}
	if trimmed[0] == '"' {
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			return nil, false, fmt.Errorf("decode user message content: %w", err)
		}
		encoded, err := json.Marshal(text + guardianDecisionInstruction)
		if err != nil {
			return nil, false, fmt.Errorf("encode user message content: %w", err)
		}
		return encoded, true, nil
	}
	if trimmed[0] != '[' {
		return content, false, nil
	}

	var parts []json.RawMessage
	if err := json.Unmarshal(trimmed, &parts); err != nil {
		return nil, false, fmt.Errorf("decode user message content: %w", err)
	}
	instruction, err := json.Marshal(map[string]string{
		"type": "input_text",
		"text": guardianDecisionInstruction,
	})
	if err != nil {
		return nil, false, err
	}
	parts = append(parts, instruction)
	encoded, err := json.Marshal(parts)
	if err != nil {
		return nil, false, fmt.Errorf("encode user message content: %w", err)
	}
	return encoded, true, nil
}

type guardianDecision struct {
	RiskLevel         string `json:"risk_level"`
	UserAuthorization string `json:"user_authorization"`
	Outcome           string `json:"outcome"`
	Rationale         string `json:"rationale"`
}

type autoReviewOutputItem struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func transformAutoReviewResponse(body []byte, contentType string) ([]byte, bool, error) {
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(contentType)), "text/event-stream") {
		return transformAutoReviewEventStream(body)
	}
	return transformAutoReviewJSON(body)
}

func transformAutoReviewJSON(body []byte) ([]byte, bool, error) {
	var response map[string]json.RawMessage
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, false, fmt.Errorf("decode response: %w", err)
	}
	var status string
	if err := json.Unmarshal(response["status"], &status); err != nil {
		return nil, false, fmt.Errorf("decode response status: %w", err)
	}
	if status != "completed" {
		return body, false, nil
	}
	var output []json.RawMessage
	if err := json.Unmarshal(response["output"], &output); err != nil {
		return nil, false, fmt.Errorf("decode response output: %w", err)
	}
	inspection, err := inspectAutoReviewOutput(output)
	if err != nil || inspection.passthrough {
		return body, false, err
	}
	transformedOutput := make([]json.RawMessage, 0, len(output)-len(inspection.discardedMessageIDs))
	for i, raw := range output {
		var item autoReviewOutputItem
		if err := json.Unmarshal(raw, &item); err != nil {
			return nil, false, fmt.Errorf("decode response output item: %w", err)
		}
		if _, discard := inspection.discardedMessageIDs[item.ID]; discard && item.Type == "message" {
			continue
		}
		if item.Type == "function_call" && item.Name == guardianDecisionToolName {
			output[i], err = json.Marshal(autoReviewDecisionMessage(inspection.itemID, inspection.decisionJSON))
			if err != nil {
				return nil, false, fmt.Errorf("encode Guardian decision message: %w", err)
			}
		}
		transformedOutput = append(transformedOutput, output[i])
	}
	response["output"], err = json.Marshal(transformedOutput)
	if err != nil {
		return nil, false, fmt.Errorf("encode response output: %w", err)
	}
	transformed, err := json.Marshal(response)
	if err != nil {
		return nil, false, fmt.Errorf("encode response: %w", err)
	}
	return transformed, true, nil
}

type autoReviewInspection struct {
	decisionJSON        string
	itemID              string
	discardedMessageIDs map[string]struct{}
	passthrough         bool
}

func inspectAutoReviewOutput(output []json.RawMessage) (autoReviewInspection, error) {
	result := autoReviewInspection{discardedMessageIDs: make(map[string]struct{})}
	decisionCalls := 0
	otherTerminalOutput := false
	for _, raw := range output {
		var item autoReviewOutputItem
		if err := json.Unmarshal(raw, &item); err != nil {
			return result, fmt.Errorf("decode response output item: %w", err)
		}
		switch item.Type {
		case "reasoning":
			continue
		case "message":
			// Some otherwise reliable tool-capable models emit explanatory prose
			// alongside the decision call. The validated call is authoritative;
			// record the prose item so both JSON and SSE paths can discard it.
			if item.ID == "" {
				return result, errors.New("Guardian assistant message has no item ID")
			}
			result.discardedMessageIDs[item.ID] = struct{}{}
		case "function_call":
			if item.Name != guardianDecisionToolName {
				if decisionCalls > 0 {
					otherTerminalOutput = true
				} else {
					result.passthrough = true
				}
				continue
			}
			decisionCalls++
			if decisionCalls > 1 {
				return result, errors.New("Guardian called the decision tool more than once")
			}
			decisionJSON, err := validateGuardianDecision(item.Arguments)
			if err != nil {
				return result, err
			}
			result.decisionJSON = decisionJSON
			result.itemID = item.ID
		default:
			otherTerminalOutput = true
		}
	}
	if result.passthrough && decisionCalls == 0 {
		return result, nil
	}
	if decisionCalls == 0 {
		return result, errors.New("Guardian did not call submit_guardian_decision")
	}
	if result.passthrough || otherTerminalOutput {
		return result, errors.New("Guardian mixed its decision with other terminal output")
	}
	return result, nil
}

func validateGuardianDecision(arguments string) (string, error) {
	decoder := json.NewDecoder(strings.NewReader(arguments))
	decoder.DisallowUnknownFields()
	var decision guardianDecision
	if err := decoder.Decode(&decision); err != nil {
		return "", fmt.Errorf("decode Guardian decision arguments: %w", err)
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		return "", errors.New("Guardian decision arguments contain trailing data")
	}
	if !oneOf(decision.RiskLevel, "low", "medium", "high", "critical") {
		return "", fmt.Errorf("invalid Guardian risk_level %q", decision.RiskLevel)
	}
	if !oneOf(decision.UserAuthorization, "unknown", "low", "medium", "high") {
		return "", fmt.Errorf("invalid Guardian user_authorization %q", decision.UserAuthorization)
	}
	if !oneOf(decision.Outcome, "allow", "deny") {
		return "", fmt.Errorf("invalid Guardian outcome %q", decision.Outcome)
	}
	if strings.TrimSpace(decision.Rationale) == "" {
		return "", errors.New("Guardian rationale is empty")
	}
	encoded, err := json.Marshal(decision)
	if err != nil {
		return "", fmt.Errorf("encode Guardian decision: %w", err)
	}
	return string(encoded), nil
}

func oneOf(value string, allowed ...string) bool {
	for _, candidate := range allowed {
		if value == candidate {
			return true
		}
	}
	return false
}

func autoReviewDecisionMessage(itemID, decision string) map[string]any {
	return map[string]any{
		"id":     itemID,
		"type":   "message",
		"status": "completed",
		"role":   "assistant",
		"content": []any{map[string]any{
			"type":        "output_text",
			"text":        decision,
			"annotations": []any{},
			"logprobs":    []any{},
		}},
	}
}

type serverSentEvent struct {
	event string
	data  []byte
}

func transformAutoReviewEventStream(body []byte) ([]byte, bool, error) {
	events, err := parseServerSentEvents(body)
	if err != nil {
		return nil, false, err
	}
	var completedOutput []json.RawMessage
	terminalFailure := false
	for _, event := range events {
		if event.event == "response.failed" || event.event == "response.incomplete" {
			terminalFailure = true
		}
		if event.event != "response.completed" {
			continue
		}
		var payload struct {
			Response struct {
				Output []json.RawMessage `json:"output"`
			} `json:"response"`
		}
		if err := json.Unmarshal(event.data, &payload); err != nil {
			return nil, false, fmt.Errorf("decode response.completed event: %w", err)
		}
		completedOutput = payload.Response.Output
	}
	if completedOutput == nil {
		if terminalFailure {
			return body, false, nil
		}
		return nil, false, errors.New("Guardian stream has no response.completed event")
	}
	inspection, err := inspectAutoReviewOutput(completedOutput)
	if err != nil || inspection.passthrough {
		return body, false, err
	}
	outputIndexes := make(map[int]int, len(completedOutput)-len(inspection.discardedMessageIDs))
	nextOutputIndex := 0
	for oldOutputIndex, raw := range completedOutput {
		var item autoReviewOutputItem
		if err := json.Unmarshal(raw, &item); err != nil {
			return nil, false, fmt.Errorf("decode completed output item: %w", err)
		}
		if _, discard := inspection.discardedMessageIDs[item.ID]; discard && item.Type == "message" {
			continue
		}
		outputIndexes[oldOutputIndex] = nextOutputIndex
		nextOutputIndex++
	}

	var transformed []serverSentEvent
	addedDecisionMessage := false
	finishedDecisionMessage := false
	for _, event := range events {
		if bytes.Equal(bytes.TrimSpace(event.data), []byte("[DONE]")) {
			transformed = append(transformed, event)
			continue
		}
		var payload map[string]any
		if err := json.Unmarshal(event.data, &payload); err != nil {
			return nil, false, fmt.Errorf("decode %s event: %w", event.event, err)
		}
		if _, discard := inspection.discardedMessageIDs[autoReviewEventItemID(payload)]; discard {
			continue
		}
		if outputIndex, ok := payload["output_index"].(float64); ok {
			if mapped, keep := outputIndexes[int(outputIndex)]; keep {
				payload["output_index"] = mapped
			}
		}
		switch event.event {
		case "response.output_item.added":
			item, _ := payload["item"].(map[string]any)
			if itemString(item, "id") == inspection.itemID && itemString(item, "name") == guardianDecisionToolName {
				outputIndex := payload["output_index"]
				transformed = append(transformed,
					newServerSentEvent("response.output_item.added", map[string]any{
						"output_index": outputIndex,
						"item": map[string]any{
							"id": inspection.itemID, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{},
						},
					}),
					newServerSentEvent("response.content_part.added", map[string]any{
						"item_id": inspection.itemID, "output_index": outputIndex, "content_index": 0,
						"part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}, "logprobs": []any{}},
					}),
				)
				addedDecisionMessage = true
				continue
			}
		case "response.function_call_arguments.delta", "response.function_call_arguments.done":
			if itemString(payload, "item_id") == inspection.itemID {
				continue
			}
		case "response.output_item.done":
			item, _ := payload["item"].(map[string]any)
			if itemString(item, "id") == inspection.itemID && itemString(item, "name") == guardianDecisionToolName {
				if !addedDecisionMessage {
					return nil, false, errors.New("Guardian decision stream has no output_item.added event")
				}
				outputIndex := payload["output_index"]
				message := autoReviewDecisionMessage(inspection.itemID, inspection.decisionJSON)
				content := message["content"].([]any)[0]
				transformed = append(transformed,
					newServerSentEvent("response.output_text.delta", map[string]any{
						"item_id": inspection.itemID, "output_index": outputIndex, "content_index": 0, "delta": inspection.decisionJSON, "logprobs": []any{},
					}),
					newServerSentEvent("response.output_text.done", map[string]any{
						"item_id": inspection.itemID, "output_index": outputIndex, "content_index": 0, "text": inspection.decisionJSON, "logprobs": []any{},
					}),
					newServerSentEvent("response.content_part.done", map[string]any{
						"item_id": inspection.itemID, "output_index": outputIndex, "content_index": 0, "part": content,
					}),
					newServerSentEvent("response.output_item.done", map[string]any{
						"output_index": outputIndex, "item": message,
					}),
				)
				finishedDecisionMessage = true
				continue
			}
		case "response.completed":
			if !finishedDecisionMessage {
				return nil, false, errors.New("Guardian decision stream has no output_item.done event")
			}
			response, ok := payload["response"].(map[string]any)
			if !ok {
				return nil, false, errors.New("response.completed event has no response object")
			}
			output, ok := response["output"].([]any)
			if !ok {
				return nil, false, errors.New("response.completed event has no output array")
			}
			transformedOutput := make([]any, 0, len(output)-len(inspection.discardedMessageIDs))
			for _, value := range output {
				item, _ := value.(map[string]any)
				if _, discard := inspection.discardedMessageIDs[itemString(item, "id")]; discard && itemString(item, "type") == "message" {
					continue
				}
				if itemString(item, "id") == inspection.itemID && itemString(item, "name") == guardianDecisionToolName {
					value = autoReviewDecisionMessage(inspection.itemID, inspection.decisionJSON)
				}
				transformedOutput = append(transformedOutput, value)
			}
			response["output"] = transformedOutput
		}
		encoded, err := json.Marshal(payload)
		if err != nil {
			return nil, false, fmt.Errorf("encode %s event: %w", event.event, err)
		}
		transformed = append(transformed, serverSentEvent{event: event.event, data: encoded})
	}
	encoded, err := encodeServerSentEvents(transformed)
	if err != nil {
		return nil, false, err
	}
	return encoded, true, nil
}

func autoReviewEventItemID(payload map[string]any) string {
	if itemID := itemString(payload, "item_id"); itemID != "" {
		return itemID
	}
	item, _ := payload["item"].(map[string]any)
	return itemString(item, "id")
}

func itemString(item map[string]any, key string) string {
	value, _ := item[key].(string)
	return value
}

func newServerSentEvent(event string, payload map[string]any) serverSentEvent {
	payload["type"] = event
	encoded, _ := json.Marshal(payload)
	return serverSentEvent{event: event, data: encoded}
}

func parseServerSentEvents(body []byte) ([]serverSentEvent, error) {
	normalized := bytes.ReplaceAll(body, []byte("\r\n"), []byte("\n"))
	var events []serverSentEvent
	for _, frame := range bytes.Split(normalized, []byte("\n\n")) {
		if len(bytes.TrimSpace(frame)) == 0 {
			continue
		}
		var event serverSentEvent
		var dataLines [][]byte
		for _, line := range bytes.Split(frame, []byte("\n")) {
			switch {
			case bytes.HasPrefix(line, []byte("event:")):
				event.event = strings.TrimSpace(string(bytes.TrimPrefix(line, []byte("event:"))))
			case bytes.HasPrefix(line, []byte("data:")):
				dataLines = append(dataLines, bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:"))))
			}
		}
		if event.event == "" || len(dataLines) == 0 {
			return nil, errors.New("malformed Guardian event stream")
		}
		event.data = bytes.Join(dataLines, []byte("\n"))
		events = append(events, event)
	}
	return events, nil
}

func encodeServerSentEvents(events []serverSentEvent) ([]byte, error) {
	var result bytes.Buffer
	sequenceNumber := 0
	for _, event := range events {
		data := event.data
		if !bytes.Equal(bytes.TrimSpace(data), []byte("[DONE]")) {
			var payload map[string]any
			if err := json.Unmarshal(data, &payload); err != nil {
				return nil, fmt.Errorf("decode transformed %s event: %w", event.event, err)
			}
			payload["type"] = event.event
			payload["sequence_number"] = sequenceNumber
			sequenceNumber++
			var err error
			data, err = json.Marshal(payload)
			if err != nil {
				return nil, fmt.Errorf("encode transformed %s event: %w", event.event, err)
			}
		}
		fmt.Fprintf(&result, "event: %s\ndata: %s\n\n", event.event, data)
	}
	return result.Bytes(), nil
}
