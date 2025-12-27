package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// ToChatCompletionRequest converts an internal ChatRequest to an OpenAI-compatible request.
func ToChatCompletionRequest(req api.ChatRequest) (ChatCompletionRequest, error) {
	messages, err := toOpenAIMessages(req.Messages)
	if err != nil {
		return ChatCompletionRequest{}, err
	}

	out := ChatCompletionRequest{
		Model:         req.Model,
		Messages:      messages,
		Stream:        req.Stream != nil && *req.Stream,
		Tools:         req.Tools,
		StreamOptions: nil,
	}
	if req.Logprobs {
		out.Logprobs = &req.Logprobs
		if req.TopLogprobs > 0 {
			out.TopLogprobs = req.TopLogprobs
		}
	}

	if req.Format != nil && len(req.Format) > 0 {
		out.ResponseFormat = &ResponseFormat{
			Type:       "json_schema",
			JsonSchema: &JsonSchema{Schema: req.Format},
		}
	}

	applyOptions(&out, req.Options)

	return out, nil
}

func applyOptions(out *ChatCompletionRequest, opts map[string]any) {
	if opts == nil {
		return
	}

	if v, ok := opts["temperature"]; ok {
		if f, ok := asFloat64(v); ok {
			out.Temperature = &f
		}
	}
	if v, ok := opts["top_p"]; ok {
		if f, ok := asFloat64(v); ok {
			out.TopP = &f
		}
	}
	if v, ok := opts["num_predict"]; ok {
		if n, ok := asInt(v); ok {
			out.MaxTokens = &n
		}
	}
	if v, ok := opts["presence_penalty"]; ok {
		if f, ok := asFloat64(v); ok {
			out.PresencePenalty = &f
		}
	}
	if v, ok := opts["frequency_penalty"]; ok {
		if f, ok := asFloat64(v); ok {
			out.FrequencyPenalty = &f
		}
	}
	if v, ok := opts["seed"]; ok {
		if n, ok := asInt(v); ok {
			out.Seed = &n
		}
	}
	if v, ok := opts["stop"]; ok {
		out.Stop = v
	}
	if v, ok := opts["tool_choice"]; ok {
		out.ToolChoice = v
	}
}

func asFloat64(v any) (float64, bool) {
	switch t := v.(type) {
	case float64:
		return t, true
	case float32:
		return float64(t), true
	case int:
		return float64(t), true
	case int64:
		return float64(t), true
	case json.Number:
		if f, err := t.Float64(); err == nil {
			return f, true
		}
	}
	return 0, false
}

func asInt(v any) (int, bool) {
	switch t := v.(type) {
	case int:
		return t, true
	case int64:
		return int(t), true
	case float64:
		return int(t), true
	case float32:
		return int(t), true
	case json.Number:
		if i, err := t.Int64(); err == nil {
			return int(i), true
		}
	}
	return 0, false
}

func toOpenAIMessages(msgs []api.Message) ([]Message, error) {
	out := make([]Message, 0, len(msgs))
	for _, m := range msgs {
		msg := Message{
			Role:       m.Role,
			Name:       m.ToolName,
			ToolCallID: m.ToolCallID,
			Reasoning:  m.Thinking,
		}

		if len(m.ToolCalls) > 0 {
			msg.ToolCalls = ToToolCalls(m.ToolCalls)
		}

		if len(m.Images) > 0 {
			contentParts := make([]any, 0, 1+len(m.Images))
			if strings.TrimSpace(m.Content) != "" {
				contentParts = append(contentParts, map[string]any{
					"type": "text",
					"text": m.Content,
				})
			}
			for _, img := range m.Images {
				contentParts = append(contentParts, map[string]any{
					"type": "image_url",
					"image_url": map[string]any{
						"url": encodeImageData(img),
					},
				})
			}
			msg.Content = contentParts
		} else {
			msg.Content = m.Content
		}

		out = append(out, msg)
	}
	return out, nil
}

// ChatCompletionToChatResponse converts a ChatCompletion to a ChatResponse.
func ChatCompletionToChatResponse(resp ChatCompletion) (api.ChatResponse, error) {
	if len(resp.Choices) == 0 {
		return api.ChatResponse{}, fmt.Errorf("no choices in response")
	}

	choice := resp.Choices[0]
	content := ExtractTextContent(choice.Message.Content)

	var toolCalls []api.ToolCall
	if len(choice.Message.ToolCalls) > 0 {
		var err error
		toolCalls, err = FromCompletionToolCall(choice.Message.ToolCalls)
		if err != nil {
			return api.ChatResponse{}, err
		}
	}

	var logprobs []api.Logprob
	if choice.Logprobs != nil {
		logprobs = choice.Logprobs.Content
	}

	var doneReason string
	if choice.FinishReason != nil {
		doneReason = *choice.FinishReason
	}

	respTime := time.Unix(resp.Created, 0).UTC()

	out := api.ChatResponse{
		Model:     resp.Model,
		CreatedAt: respTime,
		Message: api.Message{
			Role:      choice.Message.Role,
			Content:   content,
			ToolCalls: toolCalls,
			Thinking:  choice.Message.Reasoning,
		},
		Done:       true,
		DoneReason: doneReason,
		Logprobs:   logprobs,
	}

	if resp.Usage.TotalTokens > 0 {
		out.Metrics.PromptEvalCount = resp.Usage.PromptTokens
		out.Metrics.EvalCount = resp.Usage.CompletionTokens
	}

	return out, nil
}

// ExtractTextContent extracts text from an OpenAI content field.
func ExtractTextContent(content any) string {
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var sb strings.Builder
		for _, item := range v {
			switch item := item.(type) {
			case string:
				sb.WriteString(item)
			case map[string]any:
				if t, ok := item["type"].(string); ok && (t == "text" || t == "input_text" || t == "output_text") {
					if txt, ok := item["text"].(string); ok {
						sb.WriteString(txt)
						continue
					}
				}
				if txt, ok := item["text"].(string); ok {
					sb.WriteString(txt)
					continue
				}
				if inner, ok := item["content"]; ok {
					sb.WriteString(ExtractTextContent(inner))
				}
			}
		}
		return sb.String()
	case map[string]any:
		if t, ok := v["type"].(string); ok && (t == "text" || t == "input_text" || t == "output_text") {
			if txt, ok := v["text"].(string); ok {
				return txt
			}
		}
		if txt, ok := v["text"].(string); ok {
			return txt
		}
		if inner, ok := v["content"]; ok {
			return ExtractTextContent(inner)
		}
		return ""
	default:
		return ""
	}
}

func encodeImageData(img api.ImageData) string {
	if len(img) == 0 {
		return ""
	}
	encoded := base64.StdEncoding.EncodeToString(img)
	return "data:image/png;base64," + encoded
}

// ToolCallAccumulator rebuilds tool call arguments from streaming deltas.
type ToolCallAccumulator struct {
	calls map[int]*toolCallBuilder
}

type toolCallBuilder struct {
	id   string
	name string
	args strings.Builder
}

func NewToolCallAccumulator() *ToolCallAccumulator {
	return &ToolCallAccumulator{
		calls: make(map[int]*toolCallBuilder),
	}
}

func (a *ToolCallAccumulator) Apply(deltas []ToolCall) {
	for _, tc := range deltas {
		b, ok := a.calls[tc.Index]
		if !ok {
			b = &toolCallBuilder{}
			a.calls[tc.Index] = b
		}
		if tc.ID != "" {
			b.id = tc.ID
		}
		if tc.Function.Name != "" {
			b.name = tc.Function.Name
		}
		if tc.Function.Arguments != "" {
			b.args.WriteString(tc.Function.Arguments)
		}
	}
}

func (a *ToolCallAccumulator) Build() ([]api.ToolCall, error) {
	if len(a.calls) == 0 {
		return nil, nil
	}

	indexes := make([]int, 0, len(a.calls))
	for idx := range a.calls {
		indexes = append(indexes, idx)
	}
	sort.Ints(indexes)

	out := make([]api.ToolCall, 0, len(indexes))
	for _, idx := range indexes {
		b := a.calls[idx]
		var args api.ToolCallFunctionArguments
		argStr := strings.TrimSpace(b.args.String())
		if argStr != "" {
			parsed, ok := parseToolCallArguments(argStr)
			if ok {
				args = parsed
			} else {
				args = api.ToolCallFunctionArguments{rawArgumentsKey: argStr}
			}
		} else {
			args = api.ToolCallFunctionArguments{}
		}

		out = append(out, api.ToolCall{
			ID: b.id,
			Function: api.ToolCallFunction{
				Index:     idx,
				Name:      b.name,
				Arguments: args,
			},
		})
	}

	return out, nil
}
