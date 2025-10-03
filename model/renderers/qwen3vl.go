package renderers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

// where should we set the image count?
var imageCount int
var videoCount int

// so i think from the renders, do vision is false

// basic
// [-] with tools
// [] with multiple tools
// [-] with tool calling
// [ ] with multiple tool calling
// with images and videos

// TODO: is there a way not to have to writ my own code for marshalWithSpaces
// the tool dictionaery list is slightly different

func marshalWithSpaces(v any) ([]byte, error) {
	b, err := json.Marshal(v) // compact
	if err != nil {
		return nil, err
	}

	out := make([]byte, 0, len(b)+len(b)/8)
	inStr, esc := false, false
	for _, c := range b {
		if inStr {
			out = append(out, c)
			if esc {
				esc = false
				continue
			}
			if c == '\\' {
				esc = true
				continue
			}
			if c == '"' {
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
			out = append(out, c)
		case ':':
			out = append(out, ':', ' ')
		case ',':
			out = append(out, ',', ' ')
		default:
			out = append(out, c)
		}
	}
	return out, nil
}

// func pruneEmpty(v any) any {
// 	switch x := v.(type) {
// 	case map[string]any:
// 		out := make(map[string]any, len(x))
// 		for k, vv := range x {
// 			p := pruneEmpty(vv)
// 			switch pp := p.(type) {
// 			case nil:
// 				continue
// 			case string:
// 				if pp == "" {
// 					continue
// 				}
// 			case []any:
// 				if len(pp) == 0 {
// 					continue
// 				}
// 			case map[string]any:
// 				if len(pp) == 0 {
// 					continue
// 				}
// 			}
// 			out[k] = p
// 		}
// 		return out
// 	case []any:
// 		out := make([]any, 0, len(x))
// 		for _, vv := range x {
// 			p := pruneEmpty(vv)
// 			switch pp := p.(type) {
// 			case nil:
// 				continue
// 			case string:
// 				if pp == "" {
// 					continue
// 				}
// 			case []any:
// 				if len(pp) == 0 {
// 					continue
// 				}
// 			case map[string]any:
// 				if len(pp) == 0 {
// 					continue
// 				}
// 			}
// 			out = append(out, p)
// 		}
// 		return out
// 	default:
// 		return v
// 	}
// }

// func marshalWithSpaces(v any) ([]byte, error) {
// 	// 1) normalize to interface{} and prune empty fields
// 	var iv any
// 	b0, err := json.Marshal(v)
// 	if err != nil {
// 		return nil, err
// 	}
// 	if err := json.Unmarshal(b0, &iv); err != nil {
// 		return nil, err
// 	}
// 	iv = pruneEmpty(iv)

// 	// 2) compact marshal
// 	b, err := json.Marshal(iv)
// 	if err != nil {
// 		return nil, err
// 	}

// 	// 3) inject spaces after ':' and ',' outside strings
// 	out := make([]byte, 0, len(b)+len(b)/8)
// 	inStr, esc := false, false
// 	for _, c := range b {
// 		if inStr {
// 			out = append(out, c)
// 			if esc {
// 				esc = false
// 				continue
// 			}
// 			if c == '\\' {
// 				esc = true
// 				continue
// 			}
// 			if c == '"' {
// 				inStr = false
// 			}
// 			continue
// 		}
// 		switch c {
// 		case '"':
// 			inStr = true
// 			out = append(out, c)
// 		case ':':
// 			out = append(out, ':', ' ')
// 		case ',':
// 			out = append(out, ',', ' ')
// 		default:
// 			out = append(out, c)
// 		}
// 	}
// 	return out, nil
// }

// this is soooooooo ugly
// why exactly is the type of content?
func renderContent(content any, doVisionCount bool) string {
	print(content)
	switch content.(type) {
	case string:
		return content.(string)
	default:
		var subSb strings.Builder
		for _, item := range content.([]any) {
			if strings.Contains(item.(string), "image") || strings.Contains(item.(string), "image_url") || item.(map[string]any)["type"] == "image" {
				if doVisionCount {
					imageCount++
				}
				// if addVisionID {
				// 	sb.WriteString("Picture " + strconv.Itoa(imageCount) + ": ") // do we need the itoa thing?
				// }
				subSb.WriteString("<|vision_start|><|image_pad|><|vision_end|>")
			} else if strings.Contains(item.(string), "video") || item.(map[string]any)["type"] == "video" {
				if doVisionCount {
					videoCount++
				}
				// if addVisionID {
				// 	sb.WriteString("Video " + strconv.Itoa(videoCount) + ": ") // do we need the itoa thing?
				// }
				subSb.WriteString("<|vision_start|><|video_pad|><|vision_end|>")
			} else if strings.Contains(item.(string), "text") {
				subSb.WriteString(item.(map[string]any)["text"].(string))
			}
		}
		return subSb.String()
	}
}

func Qwen3VLRenderer(messages []api.Message, tools []api.Tool, _ *api.ThinkValue) (string, error) {
	var sb strings.Builder
	// this is the tools section

	fmt.Println("Number of tools (A):", len(tools))

	if len(tools) > 0 {
		sb.WriteString(imStartTag + "system\n")
		if len(messages) > 0 && messages[0].Role == "system" {
			sb.WriteString(messages[0].Content + "\n\n")
		}
		sb.WriteString("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>")
		for _, tool := range tools {
			sb.WriteString("\n")
			// if b, err := json.Marshal(tool); err == nil { // {{- tool_call.arguments | tojson -}}
			// 	sb.Write(b)
			// 	// so huggingface adds a space before every json object?
			// }
			// if b, err := json.MarshalIndent(tool, "", ""); err == nil { // {{- tool_call.arguments | tojson -}}
			// 	sb.Write(b)
			// 	// so huggingface adds a space before every json object?
			// }
			if b, err := marshalWithSpaces(tool); err == nil {
				sb.Write(b) // JSON like {"a": 1, "b": 2}
			}
		}
		sb.WriteString("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n")
		// sb.WriteString("<|im_end|>\n")
	} else if len(messages) > 0 && messages[0].Role == "system" {
		sb.WriteString("<|im_start|>system\n" + messages[0].Content + "<|im_end|>\n")
	}

	// what does the namespace do?

	// Iterate through messages in reverse order to find the last query index

	// how do we get these parameters?
	multiStepTool := true
	lastQueryIndex := len(messages) - 1

	for i := len(messages) - 1; i >= 0; i-- { // go in reverse
		message := messages[i]
		if multiStepTool && message.Role == "user" {
			// Check if content starts with <tool_response> and ends with </tool_response>
			content := message.Content // use this with renderContent
			if !(strings.HasPrefix(content, "<tool_response>") && strings.HasSuffix(content, "</tool_response>")) {
				multiStepTool = false
				lastQueryIndex = i
			}
		}
	}

	// this is the start of the messages

	fmt.Println("Number of messages:", len(messages))

	fmt.Println(messages)

	for i, message := range messages {
		// lastMessage := i == len(messages)-1
		content := renderContent(message.Content, true)
		// prefill := lastMessage && message.Role == "assistant"

		fmt.Println(message) // print a message?

		if message.Role == "user" || message.Role == "system" && i != 0 {
			sb.WriteString("<|im_start|>" + message.Role + "\n" + content + "<|im_end|>\n")
		} else if message.Role == "assistant" {
			contentReasoning := ""
			if message.Thinking != "" { // if message.reasoning_content is a string
				contentReasoning = message.Thinking
			} else if strings.Contains(content, "</think>") {
				contentReasoning = strings.Split(content, "</think>")[0]
				contentReasoning = strings.TrimRight(contentReasoning, "\n")

				contentReasoningSplit := strings.Split(contentReasoning, "<think>") // how the fuck does this work?
				contentReasoning = contentReasoningSplit[len(contentReasoningSplit)-1]

				contentReasoning = strings.TrimLeft(contentReasoning, "\n")

				// TODO: should be {%- set reasoning_content = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n") -%}
				contentSplit := strings.Split(content, "</think>") // TODO: should be {%- set content = content.split("</think>")[-1].lstrip("\n") -%}
				content = contentSplit[len(contentSplit)-1]
				content = strings.TrimLeft(content, "\n")
			}

			if i > lastQueryIndex {
				if i == len(messages)-1 || contentReasoning != "" {
					sb.WriteString("<|im_start|>" + message.Role + "\n<think>\n" + strings.Trim(contentReasoning, "\n") + "\n</think>\n\n" + strings.TrimLeft(content, "\n"))
				} else {
					sb.WriteString("<|im_start|>" + message.Role + "\n" + content)
				}
			} else {
				sb.WriteString("<|im_start|>" + message.Role + "\n" + content)
			}

			// if message.tool_calls
			// if message.ToolCalls != nil {
			if len(message.ToolCalls) > 0 {
				for j, toolCall := range message.ToolCalls {
					// what the fuck is this for?
					if j > 0 || content != "" {
						sb.WriteString("\n")
					}

					// if toolCall.Function != nil {
					// 	toolCall = toolCall.Function
					// }
					// if there any way that toolcall does not have a function?
					// toolCall = toolCall.Function

					// {{- "<tool_call>\n{\"name\": \"" -}}
					// {{- tool_call.name -}}
					// {{- "\", \"arguments\": " -}}
					// sb.WriteString("\n<tool_call>\n{\"name\": \"" + toolCall.Function.Name + "\", \"arguments\": ")
					sb.WriteString("<tool_call>\n{\"name\": \"" + toolCall.Function.Name + "\", \"arguments\": ")
					if b, err := marshalWithSpaces(toolCall.Function.Arguments); err == nil {
						sb.Write(b) // JSON like {"a": 1, "b": 2}
					}
					sb.WriteString("}\n</tool_call>")
				}
			}
			sb.WriteString("<|im_end|>\n")
		} else if message.Role == "tool" {
			if i == 0 || messages[i-1].Role != "tool" {
				sb.WriteString("<|im_start|>user")
			}
			sb.WriteString("\n<tool_response>\n" + message.Content + "\n</tool_response>")
			if i == len(messages)-1 || messages[i+1].Role != "tool" {
				sb.WriteString("<|im_end|>\n")
			}
		}

		// if lastMessage {
		// 	sb.WriteString("<|im_start|>assistant\n<think>\n")
		// }
	}

	// we might need to wrap this in something?
	sb.WriteString("<|im_start|>assistant\n<think>\n")

	// if addGenerationPrompt {
	// 	sb.WriteString("<|im_start|>assistant\n<think>\n")
	// }
	return sb.String(), nil

}
