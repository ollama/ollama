package renderers

import (
	"fmt"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/api"
)

const (
	qwen35ThinkOpenTag  = "<think>"
	qwen35ThinkCloseTag = "</think>"

	qwen38XHighReasoningInstructions = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
	qwen38LowReasoningInstructions   = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."

	qwen35ToolPostamble = `
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>`
)

type qwen35RendererVariant int

const (
	qwen35RendererDefault qwen35RendererVariant = iota
	qwen35Renderer38
)

type Qwen35Renderer struct {
	isThinking bool
	variant    qwen35RendererVariant

	alwaysRenderAssistantThinkBlock bool
	emitEmptyThinkOnNoThink         bool
	useImgTags                      bool
}

func newQwen38Renderer() *Qwen35Renderer {
	return &Qwen35Renderer{
		isThinking:                      true,
		variant:                         qwen35Renderer38,
		alwaysRenderAssistantThinkBlock: true,
		emitEmptyThinkOnNoThink:         true,
		useImgTags:                      RenderImgTags,
	}
}

func (r *Qwen35Renderer) LeadingBOS() string {
	return ""
}

func (r *Qwen35Renderer) renderContent(content api.Message, imageOffset int) (string, int) {
	if r.useImgTags {
		return renderContentWithImageTags(content.Content, len(content.Images), imageOffset)
	}

	// This assumes all images are at the front of the message - same assumption as ollama/ollama/runner.go
	var subSb strings.Builder
	for range content.Images {
		subSb.WriteString("<|vision_start|><|image_pad|><|vision_end|>")
	}
	// TODO: support videos

	subSb.WriteString(content.Content)
	return subSb.String(), imageOffset
}

func splitQwen35ReasoningContent(content, messageThinking string, isThinking, extractTaggedReasoning bool) (reasoning string, remaining string) {
	if isThinking && messageThinking != "" {
		return strings.TrimSpace(messageThinking), content
	}

	if !extractTaggedReasoning {
		return "", content
	}

	if idx := strings.Index(content, qwen35ThinkCloseTag); idx != -1 {
		before := content[:idx]
		if open := strings.LastIndex(before, qwen35ThinkOpenTag); open != -1 {
			reasoning = before[open+len(qwen35ThinkOpenTag):]
		} else {
			reasoning = before
		}
		content = strings.TrimLeft(content[idx+len(qwen35ThinkCloseTag):], "\n")
	}

	return strings.TrimSpace(reasoning), content
}

func qwen38ReasoningInstructions(think *api.ThinkValue) (string, error) {
	if think == nil || think.Value == nil {
		return qwen38XHighReasoningInstructions, nil
	}
	if !think.IsValid() {
		return "", fmt.Errorf("invalid thinking value %v", think.Value)
	}
	if !think.Bool() {
		return "", nil
	}

	switch think.String() {
	case "low":
		return qwen38LowReasoningInstructions, nil
	case "medium":
		return "", nil
	case "high", "max":
		return qwen38XHighReasoningInstructions, nil
	default:
		return "", fmt.Errorf("unsupported Qwen3.8 reasoning effort %q", think.String())
	}
}

// Qwen3.8 accepts exactly one leading system turn and has no developer role.
// Fold instruction messages from the effective history into that turn while
// preserving instruction order and the order of all conversation messages.
func normalizeQwen38Messages(messages []api.Message) ([]api.Message, error) {
	var instructionCount int
	var instructions []string
	for _, message := range messages {
		if message.Role != "system" && message.Role != "developer" {
			continue
		}
		if len(message.Images) > 0 {
			return nil, fmt.Errorf("%s message cannot contain images", message.Role)
		}
		instructionCount++
		if content := strings.TrimSpace(message.Content); content != "" {
			instructions = append(instructions, content)
		}
	}

	if instructionCount == 0 || (instructionCount == 1 && messages[0].Role == "system") {
		return messages, nil
	}

	normalized := make([]api.Message, 0, len(messages)-instructionCount+1)
	normalized = append(normalized, api.Message{
		Role:    "system",
		Content: strings.Join(instructions, "\n\n"),
	})
	for _, message := range messages {
		if message.Role != "system" && message.Role != "developer" {
			normalized = append(normalized, message)
		}
	}
	return normalized, nil
}

func (r *Qwen35Renderer) validateMessages(messages []api.Message) error {
	if r.variant != qwen35Renderer38 {
		return nil
	}
	if len(messages) == 0 {
		return fmt.Errorf("no messages provided")
	}
	if messages[0].Role == "system" && len(messages[0].Images) > 0 {
		return fmt.Errorf("system message cannot contain images")
	}

	foundUserQuery := false
	for _, message := range messages {
		if message.Role != "user" {
			continue
		}
		content, _ := r.renderContent(message, 0)
		content = strings.TrimSpace(content)
		if !(strings.HasPrefix(content, "<tool_response>") && strings.HasSuffix(content, "</tool_response>")) {
			foundUserQuery = true
			break
		}
	}
	if !foundUserQuery {
		return fmt.Errorf("no user query found in messages")
	}

	return nil
}

func (r *Qwen35Renderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	if r.variant == qwen35Renderer38 {
		var err error
		messages, err = normalizeQwen38Messages(messages)
		if err != nil {
			return "", err
		}
	}
	if err := r.validateMessages(messages); err != nil {
		return "", err
	}

	var sb strings.Builder

	isThinking := r.isThinking
	if think != nil && think.Value != nil {
		isThinking = think.Bool()
	}
	reasoningInstructions := ""
	if r.variant == qwen35Renderer38 {
		var err error
		reasoningInstructions, err = qwen38ReasoningInstructions(think)
		if err != nil {
			return "", err
		}
	}

	if len(tools) > 0 {
		sb.WriteString(imStartTag + "system\n")
		if reasoningInstructions != "" {
			sb.WriteString(reasoningInstructions + "\n\n")
		}
		sb.WriteString("# Tools\n\nYou have access to the following functions:\n\n<tools>")
		for _, tool := range tools {
			sb.WriteString("\n")
			if b, err := marshalWithSpaces(tool); err == nil {
				sb.Write(b)
			}
		}
		sb.WriteString(qwen35ToolPostamble)
		if len(messages) > 0 && messages[0].Role == "system" {
			systemContent, _ := r.renderContent(messages[0], 0)
			systemContent = strings.TrimSpace(systemContent)
			if systemContent != "" {
				sb.WriteString("\n\n")
				sb.WriteString(systemContent)
			}
		}
		sb.WriteString(imEndTag + "\n")
	} else if len(messages) > 0 && messages[0].Role == "system" {
		systemContent, _ := r.renderContent(messages[0], 0)
		systemContent = strings.TrimSpace(systemContent)
		if r.variant != qwen35Renderer38 || systemContent != "" || reasoningInstructions != "" {
			sb.WriteString(imStartTag + "system\n")
			if reasoningInstructions != "" {
				sb.WriteString(reasoningInstructions)
				if systemContent != "" {
					sb.WriteString("\n\n")
				}
			}
			sb.WriteString(systemContent + imEndTag + "\n")
		}
	} else if reasoningInstructions != "" {
		sb.WriteString(imStartTag + "system\n" + reasoningInstructions + imEndTag + "\n")
	}

	multiStepTool := true
	lastQueryIndex := len(messages) - 1 // so this is the last user message

	for i := len(messages) - 1; i >= 0; i-- {
		message := messages[i]
		if multiStepTool && message.Role == "user" {
			content, _ := r.renderContent(message, 0)
			content = strings.TrimSpace(content)
			if !(strings.HasPrefix(content, "<tool_response>") && strings.HasSuffix(content, "</tool_response>")) {
				multiStepTool = false
				lastQueryIndex = i
			}
		}
	}

	imageOffset := 0
	for i, message := range messages {
		content, nextImageOffset := r.renderContent(message, imageOffset)
		imageOffset = nextImageOffset
		content = strings.TrimSpace(content)

		lastMessage := i == len(messages)-1
		prefill := lastMessage && message.Role == "assistant"

		if message.Role == "user" || (message.Role == "system" && i != 0) {
			if r.variant == qwen35Renderer38 && message.Role == "system" {
				slog.Warn("non-leading system message", "renderer", "qwen3.8")
			}
			sb.WriteString(imStartTag + message.Role + "\n" + content + imEndTag + "\n")
		} else if message.Role == "assistant" {
			renderAssistantThinkBlock := r.alwaysRenderAssistantThinkBlock || (isThinking && i > lastQueryIndex)
			contentReasoning, content := splitQwen35ReasoningContent(
				content,
				message.Thinking,
				renderAssistantThinkBlock,
				r.variant != qwen35Renderer38,
			)

			if renderAssistantThinkBlock {
				sb.WriteString(imStartTag + message.Role + "\n<think>\n" + contentReasoning + "\n</think>\n\n" + content)
			} else {
				sb.WriteString(imStartTag + message.Role + "\n" + content)
			}

			if len(message.ToolCalls) > 0 {
				for j, toolCall := range message.ToolCalls {
					if j == 0 {
						if strings.TrimSpace(content) != "" {
							sb.WriteString("\n\n")
						}
					} else {
						sb.WriteString("\n")
					}

					sb.WriteString("<tool_call>\n<function=" + toolCall.Function.Name + ">\n")
					for name, value := range toolCall.Function.Arguments.All() {
						sb.WriteString("<parameter=" + name + ">\n")
						sb.WriteString(formatToolCallArgument(value))
						sb.WriteString("\n</parameter>\n")
					}
					sb.WriteString("</function>\n</tool_call>")
				}
			}

			if !prefill {
				sb.WriteString(imEndTag + "\n")
			}
		} else if message.Role == "tool" {
			if i == 0 || messages[i-1].Role != "tool" {
				sb.WriteString(imStartTag + "user")
			}
			sb.WriteString("\n<tool_response>\n" + content + "\n</tool_response>")
			if i == len(messages)-1 || messages[i+1].Role != "tool" {
				sb.WriteString(imEndTag + "\n")
			}
		} else if r.variant == qwen35Renderer38 && message.Role != "system" {
			slog.Warn("unexpected message role", "renderer", "qwen3.8", "role", message.Role)
			sb.WriteString(imStartTag + message.Role + "\n" + content + imEndTag + "\n")
		}

		// prefill at the end
		if lastMessage && !prefill {
			sb.WriteString(imStartTag + "assistant\n")
			if isThinking {
				sb.WriteString("<think>\n")
			} else if r.emitEmptyThinkOnNoThink {
				sb.WriteString("<think>\n\n</think>\n\n")
			}
		}
	}

	return sb.String(), nil
}
