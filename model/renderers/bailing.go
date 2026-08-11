package renderers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

const (
	bailingRoleEnd     = "<|role_end|>"
	bailingThinkOn     = "detailed thinking on"
	bailingThinkOff    = "detailed thinking off"
	bailingToolsPrefix = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
	bailingToolsSuffix = "\n</tools>\n\nIf none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\nIf you need to use a function, for each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n<arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call>\n"
)

// BailingRenderer renders messages for Bailing MoE V3 (Ling) models,
// mirroring the reference chat_template.jinja shipped with the checkpoints.
//
// The prompt uses <role>SYSTEM|HUMAN|ASSISTANT|OBSERVATION</role> blocks
// terminated by <|role_end|>. Thinking is controlled by a "detailed thinking
// on|off" directive in the system block and defaults to on; the generation
// prompt prefills "<think>" when thinking is enabled and "<think></think>"
// when disabled, so the model never emits its own opening tag.
type BailingRenderer struct{}

func (r *BailingRenderer) LeadingBOS() string {
	return ""
}

func (r *BailingRenderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var sb strings.Builder

	thinkingOn := think == nil || think.Bool()
	directive := bailingThinkOn
	if !thinkingOn {
		directive = bailingThinkOff
	}

	var systemContent string
	hasLeadingSystem := len(messages) > 0 && messages[0].Role == "system"
	if hasLeadingSystem {
		systemContent = messages[0].Content
	}
	// A system prompt that already carries the directive is emitted verbatim.
	hasDirective := hasLeadingSystem &&
		(strings.Contains(systemContent, bailingThinkOn) || strings.Contains(systemContent, bailingThinkOff))

	sb.WriteString("<role>SYSTEM</role>")
	if len(tools) > 0 {
		if hasLeadingSystem {
			sb.WriteString(systemContent + "\n")
		}
		sb.WriteString(bailingToolsPrefix)
		for _, tool := range tools {
			sb.WriteString("\n")
			if b, err := marshalWithSpaces(tool); err == nil {
				sb.Write(b)
			}
		}
		sb.WriteString(bailingToolsSuffix)
		if !hasDirective {
			sb.WriteString(directive)
		}
		sb.WriteString(bailingRoleEnd)
	} else {
		if hasLeadingSystem {
			sb.WriteString(systemContent)
			if !hasDirective {
				sb.WriteString("\n" + directive)
			}
		} else {
			sb.WriteString(directive)
		}
		sb.WriteString(bailingRoleEnd)
	}

	for i, message := range messages {
		prefill := i == len(messages)-1 && message.Role == "assistant"

		switch message.Role {
		case "system":
			if i == 0 {
				continue // already rendered in the header
			}
			sb.WriteString("<role>SYSTEM</role>" + message.Content + bailingRoleEnd)
		case "user":
			sb.WriteString("<role>HUMAN</role>" + message.Content + bailingRoleEnd)
		case "assistant":
			reasoning, content := splitBailingReasoning(message)
			sb.WriteString("<role>ASSISTANT</role>")

			if prefill && content == "" && len(message.ToolCalls) == 0 {
				if reasoning != "" && thinkingOn {
					// Continue an unfinished thinking block.
					sb.WriteString("\n<think>" + strings.TrimLeft(reasoning, "\n"))
					continue
				}
				if reasoning == "" {
					// A bare assistant prefill behaves like the generation prompt.
					if thinkingOn {
						sb.WriteString("\n<think>")
					} else {
						sb.WriteString("\n<think></think>")
					}
					continue
				}
			}

			if reasoning != "" {
				sb.WriteString("\n<think>" + strings.Trim(reasoning, "\n") + "</think>" + strings.TrimLeft(content, "\n"))
			} else {
				sb.WriteString("\n<think></think>" + content)
			}

			for j, toolCall := range message.ToolCalls {
				if j > 0 || content != "" {
					sb.WriteString("\n")
				}
				sb.WriteString("<tool_call>" + toolCall.Function.Name)
				for key, value := range toolCall.Function.Arguments.All() {
					sb.WriteString("<arg_key>" + key + "</arg_key>")
					sb.WriteString("\n<arg_value>" + bailingArgValue(value) + "</arg_value>")
				}
				sb.WriteString("\n</tool_call>")
			}

			if !prefill {
				sb.WriteString(bailingRoleEnd)
			}
		case "tool":
			if i == 0 || messages[i-1].Role != "tool" {
				sb.WriteString("<role>OBSERVATION</role>")
			}
			sb.WriteString("\n<tool_response>\n" + message.Content + "\n</tool_response>")
			if i == len(messages)-1 || messages[i+1].Role != "tool" {
				sb.WriteString(bailingRoleEnd)
			}
		}
	}

	if len(messages) == 0 || messages[len(messages)-1].Role != "assistant" {
		sb.WriteString("<role>ASSISTANT</role>")
		if thinkingOn {
			sb.WriteString("\n<think>")
		} else {
			sb.WriteString("\n<think></think>")
		}
	}

	return sb.String(), nil
}

// splitBailingReasoning returns the reasoning and remaining content of an
// assistant message, extracting an inline <think>...</think> block from the
// content when the message carries no explicit thinking, matching the
// reference template's content.split('</think>') handling.
func splitBailingReasoning(message api.Message) (reasoning, content string) {
	reasoning = message.Thinking
	content = message.Content
	if reasoning != "" || !strings.Contains(content, "</think>") {
		return reasoning, content
	}

	parts := strings.Split(content, "</think>")
	first := strings.TrimRight(parts[0], "\n")
	if open := strings.LastIndex(first, "<think>"); open != -1 {
		first = first[open+len("<think>"):]
	}
	reasoning = strings.TrimLeft(first, "\n")
	content = strings.TrimLeft(parts[len(parts)-1], "\n")
	return reasoning, content
}

func bailingArgValue(value any) string {
	if s, ok := value.(string); ok {
		return s
	}
	if b, err := json.Marshal(value); err == nil {
		return string(b)
	}
	return fmt.Sprintf("%v", value)
}
