package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

func extractModel(body []byte) (string, bool) {
	if len(body) == 0 {
		return "", false
	}
	var payload struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", false
	}
	payload.Model = strings.TrimSpace(payload.Model)
	return payload.Model, payload.Model != ""
}

func replaceRequestModel(body []byte, model string) ([]byte, error) {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	encodedModel, err := json.Marshal(model)
	if err != nil {
		return nil, err
	}
	payload["model"] = encodedModel
	return json.Marshal(payload)
}

// normalizeOllamaRequestBody removes incompatible provider state while preserving Ollama tool history.
func normalizeOllamaRequestBody(body []byte, model routingModel) ([]byte, error) {
	normalized, _, err := normalizeRequestInput(body, normalizeOllamaInputItem)
	if err != nil || model.Thinking == nil {
		return normalized, err
	}
	return normalizeOllamaThinking(normalized, *model.Thinking)
}

// normalizeFullAccessExecTool removes escalation-only arguments from the tool
// contract when Codex is already running without a sandbox. If an Ollama model
// redundantly emits require_escalated in this mode, Codex rejects the entire
// command because its approval policy is Never. Keeping those arguments out of
// the advertised schema makes the only representable call the direct one that
// Full Access already authorizes.
func normalizeFullAccessExecTool(body []byte) ([]byte, error) {
	if codexSandboxMode(body) != "danger-full-access" {
		return body, nil
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	rawTools, ok := payload["tools"]
	if !ok {
		return body, nil
	}
	var tools []json.RawMessage
	if err := json.Unmarshal(rawTools, &tools); err != nil {
		return nil, fmt.Errorf("decode tools: %w", err)
	}

	changed := false
	for i, rawTool := range tools {
		var tool map[string]json.RawMessage
		if err := json.Unmarshal(rawTool, &tool); err != nil {
			return nil, fmt.Errorf("decode tool: %w", err)
		}
		var name string
		if err := json.Unmarshal(tool["name"], &name); err != nil || name != "exec_command" {
			continue
		}

		var parameters map[string]json.RawMessage
		if err := json.Unmarshal(tool["parameters"], &parameters); err != nil {
			return nil, fmt.Errorf("decode exec_command parameters: %w", err)
		}
		var properties map[string]json.RawMessage
		if err := json.Unmarshal(parameters["properties"], &properties); err != nil {
			return nil, fmt.Errorf("decode exec_command properties: %w", err)
		}
		toolChanged := false
		for _, property := range []string{"sandbox_permissions", "justification", "prefix_rule"} {
			if _, ok := properties[property]; ok {
				delete(properties, property)
				toolChanged = true
			}
		}
		if !toolChanged {
			continue
		}
		changed = true

		encodedProperties, err := json.Marshal(properties)
		if err != nil {
			return nil, fmt.Errorf("encode exec_command properties: %w", err)
		}
		parameters["properties"] = encodedProperties
		if rawRequired, ok := parameters["required"]; ok {
			var required []string
			if err := json.Unmarshal(rawRequired, &required); err != nil {
				return nil, fmt.Errorf("decode exec_command required properties: %w", err)
			}
			required = slices.DeleteFunc(required, func(property string) bool {
				return property == "sandbox_permissions" || property == "justification" || property == "prefix_rule"
			})
			parameters["required"], err = json.Marshal(required)
			if err != nil {
				return nil, fmt.Errorf("encode exec_command required properties: %w", err)
			}
		}
		tool["parameters"], err = json.Marshal(parameters)
		if err != nil {
			return nil, fmt.Errorf("encode exec_command parameters: %w", err)
		}
		tools[i], err = json.Marshal(tool)
		if err != nil {
			return nil, fmt.Errorf("encode exec_command tool: %w", err)
		}
	}
	if !changed {
		return body, nil
	}

	encodedTools, err := json.Marshal(tools)
	if err != nil {
		return nil, fmt.Errorf("encode tools: %w", err)
	}
	payload["tools"] = encodedTools
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	return encoded, nil
}

func codexSandboxMode(body []byte) string {
	var payload struct {
		ClientMetadata map[string]json.RawMessage `json:"client_metadata"`
	}
	if json.Unmarshal(body, &payload) != nil {
		return ""
	}
	raw := payload.ClientMetadata["x-codex-turn-metadata"]
	if len(raw) == 0 {
		return ""
	}
	var encoded string
	if json.Unmarshal(raw, &encoded) == nil {
		raw = []byte(encoded)
	}
	var metadata struct {
		SandboxMode string `json:"sandbox_mode"`
	}
	if json.Unmarshal(raw, &metadata) != nil {
		return ""
	}
	return strings.TrimSpace(metadata.SandboxMode)
}

func normalizeOllamaThinking(body []byte, metadata routingThinkingMetadata) ([]byte, error) {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	_, hadThinkOverride := payload["think"]
	delete(payload, "think")
	if !metadata.Supported {
		if _, ok := payload["reasoning"]; !ok && !hadThinkOverride {
			return body, nil
		}
		delete(payload, "reasoning")
		return json.Marshal(payload)
	}

	reasoningData, ok := payload["reasoning"]
	if !ok || len(metadata.Levels) == 0 {
		if hadThinkOverride {
			return json.Marshal(payload)
		}
		return body, nil
	}
	var reasoning map[string]json.RawMessage
	if err := json.Unmarshal(reasoningData, &reasoning); err != nil {
		return nil, fmt.Errorf("decode reasoning: %w", err)
	}
	var effort string
	if rawEffort, ok := reasoning["effort"]; ok {
		if err := json.Unmarshal(rawEffort, &effort); err != nil {
			return nil, fmt.Errorf("decode reasoning effort: %w", err)
		}
	}
	if effort == "" {
		if hadThinkOverride {
			return json.Marshal(payload)
		}
		return body, nil
	}

	normalizedEffort := normalizeThinkingEffort(effort, metadata.Levels)
	if normalizedEffort == "" {
		// Omit stale effort selections so the model can use its default.
		delete(reasoning, "effort")
	} else {
		encodedEffort, err := json.Marshal(normalizedEffort)
		if err != nil {
			return nil, fmt.Errorf("encode reasoning effort: %w", err)
		}
		reasoning["effort"] = encodedEffort
		if rawValue, ok := metadata.Values[normalizedEffort]; ok {
			// Preserve boolean thinking values that Codex's named efforts cannot represent.
			payload["think"] = rawValue
		}
	}
	encodedReasoning, err := json.Marshal(reasoning)
	if err != nil {
		return nil, fmt.Errorf("encode reasoning: %w", err)
	}
	payload["reasoning"] = encodedReasoning
	return json.Marshal(payload)
}

func normalizeThinkingEffort(effort string, levels []string) string {
	var normalized string
	switch effort {
	case "minimal":
		normalized = "low"
	case "xhigh", "ultra":
		normalized = "max"
	case "none", "low", "medium", "high", "max":
		normalized = effort
	default:
		return ""
	}

	if slices.Equal(levels, []string{"none", "medium"}) && normalized != "none" {
		// Binary thinking uses "medium" for on, not an adjustable effort level.
		return "medium"
	}
	if slices.Contains(levels, normalized) {
		return normalized
	}
	if normalized == "max" && slices.Contains(levels, "high") {
		// A stale xhigh or ultra choice should use the strongest supported level.
		return "high"
	}
	return ""
}

// normalizeNativeRequestBody strips Ollama state that OpenAI cannot decrypt,
// preserving visible messages and tool history.
func normalizeNativeRequestBody(body []byte) ([]byte, bool, error) {
	return normalizeRequestInput(body, normalizeChatGPTInputItem)
}

func normalizeRequestInput(
	body []byte,
	normalizeItem func(json.RawMessage) (json.RawMessage, bool, error),
) ([]byte, bool, error) {
	if len(body) == 0 {
		return body, false, nil
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, false, err
	}
	input, ok := payload["input"]
	if !ok || len(input) == 0 || input[0] != '[' {
		return body, false, nil
	}

	var items []json.RawMessage
	if err := json.Unmarshal(input, &items); err != nil {
		return nil, false, fmt.Errorf("decode input: %w", err)
	}

	normalized := make([]json.RawMessage, 0, len(items))
	changed := false
	for _, item := range items {
		converted, keep, err := normalizeItem(item)
		if err != nil {
			return nil, false, err
		}
		if keep {
			normalized = append(normalized, converted)
		}
		if !keep || !bytes.Equal(item, converted) {
			changed = true
		}
	}
	if !changed {
		return body, false, nil
	}

	encodedInput, err := json.Marshal(normalized)
	if err != nil {
		return nil, false, fmt.Errorf("encode input: %w", err)
	}
	payload["input"] = encodedInput
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, false, fmt.Errorf("encode request: %w", err)
	}
	return encoded, true, nil
}

func normalizeOllamaInputItem(item json.RawMessage) (json.RawMessage, bool, error) {
	var header struct {
		Type string `json:"type"`
		Role string `json:"role"`
	}
	if err := json.Unmarshal(item, &header); err != nil {
		return nil, false, fmt.Errorf("decode input item: %w", err)
	}

	// Ollama accepts message shorthand without an explicit type as well as
	// the supported Responses item types below.
	itemType := header.Type
	if itemType == "" && header.Role != "" {
		itemType = "message"
	}
	switch itemType {
	case "message":
		if header.Role != "developer" {
			return item, true, nil
		}
		// Map developer instructions to system for models without developer-role support.
		var message map[string]json.RawMessage
		if err := json.Unmarshal(item, &message); err != nil {
			return nil, false, fmt.Errorf("decode developer message: %w", err)
		}
		message["role"] = json.RawMessage(`"system"`)
		converted, err := json.Marshal(message)
		if err != nil {
			return nil, false, fmt.Errorf("encode system message: %w", err)
		}
		return converted, true, nil
	case "function_call", "function_call_output":
		return item, true, nil
	case "tool_search_call", "tool_search_output", "compaction_trigger":
		// Preserve client-executed control items for the Responses adapter.
		return item, true, nil
	case "compaction":
		// Only Ollama compaction state can be expanded by the server middleware.
		return item, isOllamaCompactionItem(item), nil
	case "reasoning":
		var reasoning struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(item, &reasoning); err != nil {
			return nil, false, fmt.Errorf("decode reasoning item: %w", err)
		}
		// Keep Ollama reasoning for tool loops, but omit native encrypted state.
		return item, isOllamaReasoningItemID(reasoning.ID), nil
	case "custom_tool_call":
		var call struct {
			ID     string `json:"id,omitempty"`
			CallID string `json:"call_id"`
			Name   string `json:"name"`
			Input  string `json:"input"`
		}
		if err := json.Unmarshal(item, &call); err != nil {
			return nil, false, fmt.Errorf("decode custom tool call: %w", err)
		}
		arguments, err := json.Marshal(map[string]string{"input": call.Input})
		if err != nil {
			return nil, false, fmt.Errorf("encode custom tool input: %w", err)
		}
		converted, err := json.Marshal(map[string]any{
			"id":        call.ID,
			"type":      "function_call",
			"call_id":   call.CallID,
			"name":      call.Name,
			"arguments": string(arguments),
		})
		return converted, true, err
	case "custom_tool_call_output":
		var output struct {
			CallID string          `json:"call_id"`
			Output json.RawMessage `json:"output"`
		}
		if err := json.Unmarshal(item, &output); err != nil {
			return nil, false, fmt.Errorf("decode custom tool output: %w", err)
		}
		converted, err := json.Marshal(map[string]any{
			"type":    "function_call_output",
			"call_id": output.CallID,
			"output":  output.Output,
		})
		return converted, true, err
	default:
		// Ignore unsupported items without rejecting the remaining history.
		return nil, false, nil
	}
}

func isOllamaCompactionItem(item json.RawMessage) bool {
	var wire struct {
		EncryptedContent string `json:"encrypted_content"`
	}
	if json.Unmarshal(item, &wire) != nil || wire.EncryptedContent == "" {
		return false
	}
	var payload struct {
		Type string `json:"type"`
	}
	return json.Unmarshal([]byte(wire.EncryptedContent), &payload) == nil &&
		payload.Type == "ollama_compaction"
}

func normalizeChatGPTInputItem(item json.RawMessage) (json.RawMessage, bool, error) {
	var header struct {
		ID   string `json:"id"`
		Type string `json:"type"`
	}
	if err := json.Unmarshal(item, &header); err != nil {
		return nil, false, fmt.Errorf("decode input item: %w", err)
	}
	if header.Type == "reasoning" && isOllamaReasoningItemID(header.ID) {
		return nil, false, nil
	}
	if header.Type == "compaction" && isOllamaCompactionItem(item) {
		// OpenAI cannot decrypt Ollama compaction state; omit it when switching providers.
		return nil, false, nil
	}
	return item, true, nil
}

func isOllamaReasoningItemID(id string) bool {
	suffix, ok := strings.CutPrefix(strings.TrimSpace(id), "rs_")
	if !ok {
		return false
	}
	if responseSuffix, ok := strings.CutPrefix(suffix, "resp_"); ok {
		suffix = responseSuffix
	}
	if suffix == "" || len(suffix) > 6 {
		return false
	}
	for _, char := range suffix {
		if char < '0' || char > '9' {
			return false
		}
	}
	return true
}
