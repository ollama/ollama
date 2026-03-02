//go:build mlx

package glm4_moe_lite

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

// Renderer renders messages for GLM4-MoE-Lite models.
//
// GLM-4 Thinking Modes (ref: https://docs.z.ai/guides/capabilities/thinking-mode):
//
//  1. INTERLEAVED THINKING
//     The model thinks between tool calls and after receiving tool results.
//     This enables complex step-by-step reasoning: interpreting each tool output
//     before deciding what to do next. Thinking blocks are preserved and returned
//     with tool results to maintain reasoning continuity.
//
//  2. PRESERVED THINKING
//     The model retains reasoning content from previous assistant turns in context.
//     This preserves reasoning continuity across multi-turn conversations. The
//     upstream API has a "clear_thinking" parameter to control this:
//     - clear_thinking=true:  clears reasoning from previous turns (outputs </think>)
//     - clear_thinking=false: preserves <think>...</think> blocks from previous turns
//
//  3. TURN-LEVEL THINKING
//     Controls whether the model should reason on each turn. The upstream API
//     uses "enable_thinking" parameter:
//     - enable_thinking=true:  outputs <think> to start reasoning
//     - enable_thinking=false: outputs </think> to skip reasoning
//
// OLLAMA DEFAULTS:
//   - Thinking is ENABLED by default (thinkValue=nil or true outputs <think>)
//   - Thinking is PRESERVED by default (reasoning content from previous turns is always
//     included in <think>...</think> blocks, equivalent to clear_thinking=false)
//   - Users can disable thinking per-turn via thinkValue=false
type Renderer struct{}

// Render renders messages into the GLM4 chat format.
func (r *Renderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder

	sb.WriteString("[gMASK]<sop>")

	if len(tools) > 0 {
		sb.WriteString("<|system|>\n")
		sb.WriteString("# Tools\n\n")
		sb.WriteString("You may call one or more functions to assist with the user query.\n\n")
		sb.WriteString("You are provided with function signatures within <tools></tools> XML tags:\n")
		sb.WriteString("<tools>\n")
		for _, tool := range tools {
			d, _ := json.Marshal(tool)
			sb.WriteString(formatToolJSON(d))
			sb.WriteString("\n")
		}
		sb.WriteString("</tools>\n\n")
		sb.WriteString("For each function call, output the function name and arguments within the following XML format:\n")
		sb.WriteString("<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>")
	}

	think := true
	if thinkValue != nil && !thinkValue.Bool() {
		think = false
	}

	for i, message := range messages {
		switch message.Role {
		case "user":
			sb.WriteString("<|user|>")
			sb.WriteString(message.Content)
		case "assistant":
			sb.WriteString("<|assistant|>")
			if message.Thinking != "" {
				sb.WriteString("<think>" + message.Thinking + "</think>")
			} else {
				sb.WriteString("</think>")
			}
			if message.Content != "" {
				sb.WriteString(message.Content)
			}
			if len(message.ToolCalls) > 0 {
				for _, toolCall := range message.ToolCalls {
					sb.WriteString("<tool_call>" + toolCall.Function.Name)
					sb.WriteString(renderToolArguments(toolCall.Function.Arguments))
					sb.WriteString("</tool_call>")
				}
			}
		case "tool":
			if i == 0 || messages[i-1].Role != "tool" {
				sb.WriteString("<|observation|>")
			}
			sb.WriteString("<tool_response>")
			sb.WriteString(message.Content)
			sb.WriteString("</tool_response>")
		case "system":
			sb.WriteString("<|system|>")
			sb.WriteString(message.Content)
		}
	}

	sb.WriteString("<|assistant|>")
	if think {
		sb.WriteString("<think>")
	} else {
		sb.WriteString("</think>")
	}

	return sb.String(), nil
}

// renderToolArguments converts tool call arguments to GLM4 XML format.
func renderToolArguments(args api.ToolCallFunctionArguments) string {
	var sb strings.Builder
	for key, value := range args.All() {
		sb.WriteString("<arg_key>" + key + "</arg_key>")
		var valueStr string
		if str, ok := value.(string); ok {
			valueStr = str
		} else {
			jsonBytes, err := json.Marshal(value)
			if err != nil {
				valueStr = fmt.Sprintf("%v", value)
			} else {
				valueStr = string(jsonBytes)
			}
		}

		sb.WriteString("<arg_value>" + valueStr + "</arg_value>")
	}

	return sb.String()
}

// formatToolJSON formats JSON for GLM4 tool definitions by adding spaces after : and ,
func formatToolJSON(raw []byte) string {
	var sb strings.Builder
	sb.Grow(len(raw) + len(raw)/10)

	inString := false
	escaped := false
	for i := range raw {
		ch := raw[i]
		sb.WriteByte(ch)

		if inString {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				inString = false
			}
			continue
		}

		if ch == '"' {
			inString = true
			continue
		}

		if ch == ':' || ch == ',' {
			sb.WriteByte(' ')
		}
	}

	return sb.String()
}
