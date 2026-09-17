package renderers

import (
	"encoding/json"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
)

// CohereRenderer renders the Cohere North / Command A 2026 chat template
// (Cohere2 MoE models such as CohereLabs/North-Mini-Code-1.0): a platform
// system turn with an Available Tools section, <|START_TEXT|>-wrapped message
// bodies, <|START_THINKING|> reasoning, <|START_ACTION|> tool calls, and
// <|START_TOOL_RESULT|> tool results.
//
// The chat template's model-specific platform instructions (identity, default
// policies) belong in the model's Modelfile SYSTEM prompt, which arrives here
// as the first system message.
type CohereRenderer struct{}

func (r *CohereRenderer) LeadingBOS() string {
	return "<BOS_TOKEN>"
}

// cohereToolJSON renders one tool entry exactly as the template's tojson
// filter does: {"name": ..., "description": ..., "parameters": {...},
// "responses": null} with ", " / ": " separators.
func cohereToolJSON(tool api.Tool) (string, error) {
	params, err := marshalWithSpaces(tool.Function.Parameters)
	if err != nil {
		return "", err
	}
	name, err := json.Marshal(tool.Function.Name)
	if err != nil {
		return "", err
	}
	desc, err := json.Marshal(tool.Function.Description)
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	sb.WriteString(`{"name": `)
	sb.Write(name)
	sb.WriteString(`, "description": `)
	sb.Write(desc)
	sb.WriteString(`, "parameters": `)
	sb.WriteString(string(params))
	sb.WriteString(`, "responses": null}`)
	return sb.String(), nil
}

// writeToolsSection writes the "# Available Tools" block, reproducing the
// template's whitespace for the empty and populated cases.
func writeToolsSection(sb *strings.Builder, tools []api.Tool) error {
	sb.WriteString("# Available Tools\n```json\n[\n")
	if len(tools) == 0 {
		sb.WriteString("\n\n")
	} else {
		for i, tool := range tools {
			entry, err := cohereToolJSON(tool)
			if err != nil {
				return err
			}
			sb.WriteString("\n    ")
			sb.WriteString(entry)
			if i < len(tools)-1 {
				sb.WriteString(",")
			}
			sb.WriteString("\n\n")
		}
	}
	sb.WriteString("\n]\n```")
	return nil
}

// cohereToolResult is one entry of a <|START_TOOL_RESULT|> array.
func writeToolResult(sb *strings.Builder, callID string, content string) error {
	wrapped, err := marshalWithSpaces(map[string]string{"content": content})
	if err != nil {
		return err
	}
	sb.WriteString("\n    {\n        \"tool_call_id\": \"")
	sb.WriteString(callID)
	sb.WriteString("\",\n        \"results\": {\n\n            \n            \"0\": ")
	sb.WriteString(string(wrapped))
	sb.WriteString("\n\n        },\n        \"is_error\": null\n    }")
	return nil
}

func (r *CohereRenderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var sb strings.Builder

	// The template defaults reasoning to true; an explicit think=false
	// disables it.
	reasoning := think == nil || think.Bool()

	// The first system message — the request's system prompt, or the model's
	// Modelfile SYSTEM when the request has none — fills the template's
	// platform instruction slot.
	var system string
	rest := messages
	if len(messages) > 0 && strings.EqualFold(messages[0].Role, "system") {
		system = messages[0].Content
		rest = messages[1:]
	}

	// Platform system turn: system prompt plus the Available Tools section
	// (rendered even when no tools are defined).
	sb.WriteString("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TEXT|>")
	if system != "" {
		sb.WriteString(system)
		sb.WriteString("\n\n\n\n")
	}
	if err := writeToolsSection(&sb, tools); err != nil {
		return "", err
	}
	sb.WriteString("<|END_TEXT|><|END_OF_TURN_TOKEN|>")

	// Tool call ids regenerate as sequential indices across the whole
	// conversation (regen_tool_call_ids default); results reference the
	// index of their originating call.
	callIndex := 0
	callIDToIndex := map[string]string{}
	nextCallID := func(id string) string {
		idx := strconv.Itoa(callIndex)
		callIndex++
		if id != "" {
			if _, seen := callIDToIndex[id]; !seen {
				callIDToIndex[id] = idx
			}
		}
		return idx
	}
	resolveResultID := func(m api.Message) string {
		if idx, ok := callIDToIndex[m.ToolCallID]; ok {
			return idx
		}
		// Fall back to call order when ids are absent.
		return m.ToolCallID
	}

	prefill := false
	for i := 0; i < len(rest); i++ {
		message := rest[i]
		switch strings.ToLower(message.Role) {
		case "system":
			sb.WriteString("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TEXT|>")
			sb.WriteString(message.Content)
			sb.WriteString("<|END_TEXT|><|END_OF_TURN_TOKEN|>")
		case "user":
			sb.WriteString("<|START_OF_TURN_TOKEN|><|USER_TOKEN|><|START_TEXT|>")
			sb.WriteString(message.Content)
			sb.WriteString("<|END_TEXT|><|END_OF_TURN_TOKEN|>")
		case "assistant", "chatbot":
			sb.WriteString("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")
			if len(message.ToolCalls) > 0 {
				// Whitespace before THINKING/ACTION matches the template's
				// (untrimmed jinja blocks).
				sb.WriteString("\n            \n                ")
				if message.Thinking != "" {
					sb.WriteString("<|START_THINKING|>")
					sb.WriteString(message.Thinking)
					sb.WriteString("<|END_THINKING|>")
				}
				sb.WriteString("<|START_ACTION|>[")
				for j, tc := range message.ToolCalls {
					args, err := marshalWithSpaces(tc.Function.Arguments)
					if err != nil {
						return "", err
					}
					sb.WriteString("\n\n    {\"tool_call_id\": \"")
					sb.WriteString(nextCallID(toolCallID(tc)))
					sb.WriteString("\", \"tool_name\": \"")
					sb.WriteString(tc.Function.Name)
					sb.WriteString("\", \"parameters\": ")
					sb.WriteString(string(args))
					sb.WriteString("}")
					if j < len(message.ToolCalls)-1 {
						sb.WriteString(",")
					}
				}
				sb.WriteString("\n\n]<|END_ACTION|><|END_OF_TURN_TOKEN|>")
			} else {
				if message.Thinking != "" {
					sb.WriteString("<|START_THINKING|>")
					sb.WriteString(message.Thinking)
					sb.WriteString("<|END_THINKING|>")
				}
				sb.WriteString("<|START_TEXT|>")
				sb.WriteString(message.Content)
				if i == len(rest)-1 {
					// Assistant prefill: leave the text open for
					// continuation.
					prefill = true
				} else {
					sb.WriteString("<|END_TEXT|><|END_OF_TURN_TOKEN|>")
				}
			}
		case "tool":
			// Consecutive tool messages merge into one TOOL_RESULT array.
			sb.WriteString("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TOOL_RESULT|>[")
			if err := writeToolResult(&sb, resolveResultID(message), message.Content); err != nil {
				return "", err
			}
			for i+1 < len(rest) && strings.EqualFold(rest[i+1].Role, "tool") {
				i++
				sb.WriteString(",")
				if err := writeToolResult(&sb, resolveResultID(rest[i]), rest[i].Content); err != nil {
					return "", err
				}
			}
			sb.WriteString("\n\n]<|END_TOOL_RESULT|><|END_OF_TURN_TOKEN|>")
		}
	}

	if !prefill {
		sb.WriteString("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")
		if reasoning {
			sb.WriteString("<|START_THINKING|>")
		} else {
			sb.WriteString("<|START_THINKING|><|END_THINKING|>")
		}
	}

	return sb.String(), nil
}

// toolCallID returns the tool call's id when the client supplied one. The
// api.ToolCall ID field may be empty for calls synthesized by ollama.
func toolCallID(tc api.ToolCall) string {
	return tc.ID
}
