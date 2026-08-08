package renderers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

// GLM47Renderer renders messages for GLM-4.7 models.
//
// GLM-4.7 Thinking Modes (ref: https://docs.z.ai/guides/capabilities/thinking-mode):
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
type GLM47Renderer struct{}

func (r *GLM47Renderer) LeadingBOS() string {
	return ""
}

func (r *GLM47Renderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	segments, err := r.RenderSegments(messages, tools, thinkValue)
	if err != nil {
		return "", err
	}
	return JoinSegments(segments), nil
}

func (r *GLM47Renderer) RenderSegments(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) ([]Segment, error) {
	var sb segmentBuilder

	sb.control("[gMASK]<sop>")

	if len(tools) > 0 {
		sb.control("<|system|>\n")
		sb.control("# Tools\n\n")
		sb.control("You may call one or more functions to assist with the user query.\n\n")
		sb.control("You are provided with function signatures within <tools></tools> XML tags:\n")
		sb.control("<tools>\n")
		for _, tool := range tools {
			d, _ := json.Marshal(tool)
			sb.content(formatGLM47ToolJSON(d))
			sb.control("\n")
		}
		sb.control("</tools>\n\n")
		sb.control("For each function call, output the function name and arguments within the following XML format:\n")
		sb.control("<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>")
	}

	think := true
	if thinkValue != nil && !thinkValue.Bool() {
		think = false
	}

	for i, message := range messages {
		switch message.Role {
		case "user":
			sb.control("<|user|>")
			sb.content(message.Content)
		case "assistant":
			sb.control("<|assistant|>")
			if message.Thinking != "" {
				sb.control("<think>")
				sb.content(message.Thinking)
				sb.control("</think>")
			} else {
				sb.control("</think>")
			}
			if message.Content != "" {
				sb.content(message.Content)
			}
			if len(message.ToolCalls) > 0 {
				for _, toolCall := range message.ToolCalls {
					sb.control("<tool_call>")
					sb.content(toolCall.Function.Name)
					renderGLM47ToolArguments(&sb, toolCall.Function.Arguments)
					sb.control("</tool_call>")
				}
			}
		case "tool":
			if i == 0 || messages[i-1].Role != "tool" {
				sb.control("<|observation|>")
			}
			sb.control("<tool_response>")
			sb.content(message.Content)
			sb.control("</tool_response>")
		case "system":
			sb.control("<|system|>")
			sb.content(message.Content)
		}
	}

	sb.control("<|assistant|>")
	if think {
		sb.control("<think>")
	} else {
		sb.control("</think>")
	}

	return sb.Segments(), nil
}

func renderGLM47ToolArguments(sb *segmentBuilder, args api.ToolCallFunctionArguments) {
	for key, value := range args.All() {
		sb.control("<arg_key>")
		sb.content(key)
		sb.control("</arg_key>")
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

		sb.control("<arg_value>")
		sb.content(valueStr)
		sb.control("</arg_value>")
	}
}

func formatGLM47ToolJSON(raw []byte) string {
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
