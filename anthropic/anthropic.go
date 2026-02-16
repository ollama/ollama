package anthropic

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// Error types matching Anthropic API
type Error struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type ErrorResponse struct {
	Type      string `json:"type"` // always "error"
	Error     Error  `json:"error"`
	RequestID string `json:"request_id,omitempty"`
}

// NewError creates a new ErrorResponse with the appropriate error type based on HTTP status code
func NewError(code int, message string) ErrorResponse {
	var etype string
	switch code {
	case http.StatusBadRequest:
		etype = "invalid_request_error"
	case http.StatusUnauthorized:
		etype = "authentication_error"
	case http.StatusForbidden:
		etype = "permission_error"
	case http.StatusNotFound:
		etype = "not_found_error"
	case http.StatusTooManyRequests:
		etype = "rate_limit_error"
	case http.StatusServiceUnavailable, 529:
		etype = "overloaded_error"
	default:
		etype = "api_error"
	}

	return ErrorResponse{
		Type:      "error",
		Error:     Error{Type: etype, Message: message},
		RequestID: generateID("req"),
	}
}

// Request types

// MessagesRequest represents an Anthropic Messages API request
type MessagesRequest struct {
	Model         string          `json:"model"`
	MaxTokens     int             `json:"max_tokens"`
	Messages      []MessageParam  `json:"messages"`
	System        any             `json:"system,omitempty"` // string or []ContentBlock
	Stream        bool            `json:"stream,omitempty"`
	Temperature   *float64        `json:"temperature,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	TopK          *int            `json:"top_k,omitempty"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
	Tools         []Tool          `json:"tools,omitempty"`
	ToolChoice    *ToolChoice     `json:"tool_choice,omitempty"`
	Thinking      *ThinkingConfig `json:"thinking,omitempty"`
	Metadata      *Metadata       `json:"metadata,omitempty"`
}

// MessageParam represents a message in the request
type MessageParam struct {
	Role    string `json:"role"`    // "user" or "assistant"
	Content any    `json:"content"` // string or []ContentBlock
}

// ContentBlock represents a content block in a message.
// Text and Thinking use pointers so they serialize as the field being present (even if empty)
// only when set, which is required for SDK streaming accumulation.
type ContentBlock struct {
	Type string `json:"type"` // text, image, tool_use, tool_result, thinking

	// For text blocks - pointer so field only appears when set (SDK requires it for accumulation)
	Text *string `json:"text,omitempty"`

	// For image blocks
	Source *ImageSource `json:"source,omitempty"`

	// For tool_use blocks
	ID    string `json:"id,omitempty"`
	Name  string `json:"name,omitempty"`
	Input any    `json:"input,omitempty"`

	// For tool_result blocks
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   any    `json:"content,omitempty"` // string or []ContentBlock
	IsError   bool   `json:"is_error,omitempty"`

	// For thinking blocks - pointer so field only appears when set (SDK requires it for accumulation)
	Thinking  *string `json:"thinking,omitempty"`
	Signature string  `json:"signature,omitempty"`
}

// ImageSource represents the source of an image
type ImageSource struct {
	Type      string `json:"type"` // "base64" or "url"
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

// Tool represents a tool definition
type Tool struct {
	Type        string          `json:"type,omitempty"` // "custom" for user-defined tools
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

// ToolChoice controls how the model uses tools
type ToolChoice struct {
	Type                   string `json:"type"` // "auto", "any", "tool", "none"
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use,omitempty"`
}

// ThinkingConfig controls extended thinking
type ThinkingConfig struct {
	Type         string `json:"type"` // "enabled" or "disabled"
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

// Metadata for the request
type Metadata struct {
	UserID string `json:"user_id,omitempty"`
}

// Response types

// MessagesResponse represents an Anthropic Messages API response
type MessagesResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"` // "message"
	Role         string         `json:"role"` // "assistant"
	Model        string         `json:"model"`
	Content      []ContentBlock `json:"content"`
	StopReason   string         `json:"stop_reason,omitempty"`
	StopSequence string         `json:"stop_sequence,omitempty"`
	Usage        Usage          `json:"usage"`
}

// Usage contains token usage information
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Streaming event types

// MessageStartEvent is sent at the start of streaming
type MessageStartEvent struct {
	Type    string           `json:"type"` // "message_start"
	Message MessagesResponse `json:"message"`
}

// ContentBlockStartEvent signals the start of a content block
type ContentBlockStartEvent struct {
	Type         string       `json:"type"` // "content_block_start"
	Index        int          `json:"index"`
	ContentBlock ContentBlock `json:"content_block"`
}

// ContentBlockDeltaEvent contains incremental content updates
type ContentBlockDeltaEvent struct {
	Type  string `json:"type"` // "content_block_delta"
	Index int    `json:"index"`
	Delta Delta  `json:"delta"`
}

// Delta represents an incremental update
type Delta struct {
	Type        string `json:"type"` // "text_delta", "input_json_delta", "thinking_delta", "signature_delta"
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	Signature   string `json:"signature,omitempty"`
}

// ContentBlockStopEvent signals the end of a content block
type ContentBlockStopEvent struct {
	Type  string `json:"type"` // "content_block_stop"
	Index int    `json:"index"`
}

// MessageDeltaEvent contains updates to the message
type MessageDeltaEvent struct {
	Type  string       `json:"type"` // "message_delta"
	Delta MessageDelta `json:"delta"`
	Usage DeltaUsage   `json:"usage"`
}

// MessageDelta contains stop information
type MessageDelta struct {
	StopReason   string `json:"stop_reason,omitempty"`
	StopSequence string `json:"stop_sequence,omitempty"`
}

// DeltaUsage contains cumulative token usage
type DeltaUsage struct {
	OutputTokens int `json:"output_tokens"`
}

// MessageStopEvent signals the end of the message
type MessageStopEvent struct {
	Type string `json:"type"` // "message_stop"
}

// PingEvent is a keepalive event
type PingEvent struct {
	Type string `json:"type"` // "ping"
}

// StreamErrorEvent is an error during streaming
type StreamErrorEvent struct {
	Type  string `json:"type"` // "error"
	Error Error  `json:"error"`
}

// FromMessagesRequest converts an Anthropic MessagesRequest to an Ollama api.ChatRequest
func FromMessagesRequest(r MessagesRequest) (*api.ChatRequest, error) {
	var messages []api.Message

	if r.System != nil {
		switch sys := r.System.(type) {
		case string:
			if sys != "" {
				messages = append(messages, api.Message{Role: "system", Content: sys})
			}
		case []any:
			// System can be an array of content blocks
			var content strings.Builder
			for _, block := range sys {
				if blockMap, ok := block.(map[string]any); ok {
					if blockMap["type"] == "text" {
						if text, ok := blockMap["text"].(string); ok {
							content.WriteString(text)
						}
					}
				}
			}
			if content.Len() > 0 {
				messages = append(messages, api.Message{Role: "system", Content: content.String()})
			}
		}
	}

	for _, msg := range r.Messages {
		converted, err := convertMessage(msg)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted...)
	}

	options := make(map[string]any)

	options["num_predict"] = r.MaxTokens

	if r.Temperature != nil {
		options["temperature"] = *r.Temperature
	}

	if r.TopP != nil {
		options["top_p"] = *r.TopP
	}

	if r.TopK != nil {
		options["top_k"] = *r.TopK
	}

	if len(r.StopSequences) > 0 {
		options["stop"] = r.StopSequences
	}

	var tools api.Tools
	for _, t := range r.Tools {
		tool, err := convertTool(t)
		if err != nil {
			return nil, err
		}
		tools = append(tools, tool)
	}

	var think *api.ThinkValue
	if r.Thinking != nil && r.Thinking.Type == "enabled" {
		think = &api.ThinkValue{Value: true}
	}

	stream := r.Stream

	return &api.ChatRequest{
		Model:    r.Model,
		Messages: messages,
		Options:  options,
		Stream:   &stream,
		Tools:    tools,
		Think:    think,
	}, nil
}

// convertMessage converts an Anthropic MessageParam to Ollama api.Message(s)
func convertMessage(msg MessageParam) ([]api.Message, error) {
	var messages []api.Message
	role := strings.ToLower(msg.Role)

	switch content := msg.Content.(type) {
	case string:
		messages = append(messages, api.Message{Role: role, Content: content})

	case []any:
		var textContent strings.Builder
		var images []api.ImageData
		var toolCalls []api.ToolCall
		var thinking string
		var toolResults []api.Message

		for _, block := range content {
			blockMap, ok := block.(map[string]any)
			if !ok {
				return nil, errors.New("invalid content block format")
			}

			blockType, _ := blockMap["type"].(string)

			switch blockType {
			case "text":
				if text, ok := blockMap["text"].(string); ok {
					textContent.WriteString(text)
				}

			case "image":
				source, ok := blockMap["source"].(map[string]any)
				if !ok {
					return nil, errors.New("invalid image source")
				}

				sourceType, _ := source["type"].(string)
				if sourceType == "base64" {
					data, _ := source["data"].(string)
					decoded, err := base64.StdEncoding.DecodeString(data)
					if err != nil {
						return nil, fmt.Errorf("invalid base64 image data: %w", err)
					}
					images = append(images, decoded)
				} else {
					return nil, fmt.Errorf("invalid image source type: %s. Only base64 images are supported.", sourceType)
				}
				// URL images would need to be fetched - skip for now

			case "tool_use":
				id, ok := blockMap["id"].(string)
				if !ok {
					return nil, errors.New("tool_use block missing required 'id' field")
				}
				name, ok := blockMap["name"].(string)
				if !ok {
					return nil, errors.New("tool_use block missing required 'name' field")
				}
				tc := api.ToolCall{
					ID: id,
					Function: api.ToolCallFunction{
						Name: name,
					},
				}
				if input, ok := blockMap["input"].(map[string]any); ok {
					tc.Function.Arguments = mapToArgs(input)
				}
				toolCalls = append(toolCalls, tc)

			case "tool_result":
				toolUseID, _ := blockMap["tool_use_id"].(string)
				var resultContent string

				switch c := blockMap["content"].(type) {
				case string:
					resultContent = c
				case []any:
					for _, cb := range c {
						if cbMap, ok := cb.(map[string]any); ok {
							if cbMap["type"] == "text" {
								if text, ok := cbMap["text"].(string); ok {
									resultContent += text
								}
							}
						}
					}
				}

				toolResults = append(toolResults, api.Message{
					Role:       "tool",
					Content:    resultContent,
					ToolCallID: toolUseID,
				})

			case "thinking":
				if t, ok := blockMap["thinking"].(string); ok {
					thinking = t
				}
			}
		}

		if textContent.Len() > 0 || len(images) > 0 || len(toolCalls) > 0 || thinking != "" {
			m := api.Message{
				Role:      role,
				Content:   textContent.String(),
				Images:    images,
				ToolCalls: toolCalls,
				Thinking:  thinking,
			}
			messages = append(messages, m)
		}

		// Add tool results as separate messages
		messages = append(messages, toolResults...)

	default:
		return nil, fmt.Errorf("invalid message content type: %T", content)
	}

	return messages, nil
}

// convertTool converts an Anthropic Tool to an Ollama api.Tool
func convertTool(t Tool) (api.Tool, error) {
	var params api.ToolFunctionParameters
	if len(t.InputSchema) > 0 {
		if err := json.Unmarshal(t.InputSchema, &params); err != nil {
			return api.Tool{}, fmt.Errorf("invalid input_schema for tool %q: %w", t.Name, err)
		}
	}

	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  params,
		},
	}, nil
}

// ToMessagesResponse converts an Ollama api.ChatResponse to an Anthropic MessagesResponse
func ToMessagesResponse(id string, r api.ChatResponse) MessagesResponse {
	var content []ContentBlock

	if r.Message.Thinking != "" {
		content = append(content, ContentBlock{
			Type:     "thinking",
			Thinking: ptr(r.Message.Thinking),
		})
	}

	if r.Message.Content != "" {
		content = append(content, ContentBlock{
			Type: "text",
			Text: ptr(r.Message.Content),
		})
	}

	for _, tc := range r.Message.ToolCalls {
		content = append(content, ContentBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: tc.Function.Arguments,
		})
	}

	stopReason := mapStopReason(r.DoneReason, len(r.Message.ToolCalls) > 0)

	return MessagesResponse{
		ID:         id,
		Type:       "message",
		Role:       "assistant",
		Model:      r.Model,
		Content:    content,
		StopReason: stopReason,
		Usage: Usage{
			InputTokens:  r.Metrics.PromptEvalCount,
			OutputTokens: r.Metrics.EvalCount,
		},
	}
}

// mapStopReason converts Ollama done_reason to Anthropic stop_reason
func mapStopReason(reason string, hasToolCalls bool) string {
	if hasToolCalls {
		return "tool_use"
	}

	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	default:
		if reason != "" {
			return "stop_sequence"
		}
		return ""
	}
}

// StreamConverter manages state for converting Ollama streaming responses to Anthropic format
type StreamConverter struct {
	ID              string
	Model           string
	firstWrite      bool
	contentIndex    int
	inputTokens     int
	outputTokens    int
	thinkingStarted bool
	thinkingDone    bool
	textStarted     bool
	toolCallsSent   map[string]bool
}

func NewStreamConverter(id, model string) *StreamConverter {
	return &StreamConverter{
		ID:            id,
		Model:         model,
		firstWrite:    true,
		toolCallsSent: make(map[string]bool),
	}
}

// StreamEvent represents a streaming event to be sent to the client
type StreamEvent struct {
	Event string
	Data  any
}

// Process converts an Ollama ChatResponse to Anthropic streaming events
func (c *StreamConverter) Process(r api.ChatResponse) []StreamEvent {
	var events []StreamEvent

	if c.firstWrite {
		c.firstWrite = false
		c.inputTokens = r.Metrics.PromptEvalCount

		events = append(events, StreamEvent{
			Event: "message_start",
			Data: MessageStartEvent{
				Type: "message_start",
				Message: MessagesResponse{
					ID:      c.ID,
					Type:    "message",
					Role:    "assistant",
					Model:   c.Model,
					Content: []ContentBlock{},
					Usage: Usage{
						InputTokens:  c.inputTokens,
						OutputTokens: 0,
					},
				},
			},
		})
	}

	if r.Message.Thinking != "" && !c.thinkingDone {
		if !c.thinkingStarted {
			c.thinkingStarted = true
			events = append(events, StreamEvent{
				Event: "content_block_start",
				Data: ContentBlockStartEvent{
					Type:  "content_block_start",
					Index: c.contentIndex,
					ContentBlock: ContentBlock{
						Type:     "thinking",
						Thinking: ptr(""),
					},
				},
			})
		}

		events = append(events, StreamEvent{
			Event: "content_block_delta",
			Data: ContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: c.contentIndex,
				Delta: Delta{
					Type:     "thinking_delta",
					Thinking: r.Message.Thinking,
				},
			},
		})
	}

	if r.Message.Content != "" {
		if c.thinkingStarted && !c.thinkingDone {
			c.thinkingDone = true
			events = append(events, StreamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
			c.contentIndex++
		}

		if !c.textStarted {
			c.textStarted = true
			events = append(events, StreamEvent{
				Event: "content_block_start",
				Data: ContentBlockStartEvent{
					Type:  "content_block_start",
					Index: c.contentIndex,
					ContentBlock: ContentBlock{
						Type: "text",
						Text: ptr(""),
					},
				},
			})
		}

		events = append(events, StreamEvent{
			Event: "content_block_delta",
			Data: ContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: c.contentIndex,
				Delta: Delta{
					Type: "text_delta",
					Text: r.Message.Content,
				},
			},
		})
	}

	for _, tc := range r.Message.ToolCalls {
		if c.toolCallsSent[tc.ID] {
			continue
		}

		if c.textStarted {
			events = append(events, StreamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
			c.contentIndex++
			c.textStarted = false
		}

		argsJSON, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			slog.Error("failed to marshal tool arguments", "error", err, "tool_id", tc.ID)
			continue
		}

		events = append(events, StreamEvent{
			Event: "content_block_start",
			Data: ContentBlockStartEvent{
				Type:  "content_block_start",
				Index: c.contentIndex,
				ContentBlock: ContentBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Function.Name,
					Input: map[string]any{},
				},
			},
		})

		events = append(events, StreamEvent{
			Event: "content_block_delta",
			Data: ContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: c.contentIndex,
				Delta: Delta{
					Type:        "input_json_delta",
					PartialJSON: string(argsJSON),
				},
			},
		})

		events = append(events, StreamEvent{
			Event: "content_block_stop",
			Data: ContentBlockStopEvent{
				Type:  "content_block_stop",
				Index: c.contentIndex,
			},
		})

		c.toolCallsSent[tc.ID] = true
		c.contentIndex++
	}

	if r.Done {
		if c.textStarted {
			events = append(events, StreamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
		} else if c.thinkingStarted && !c.thinkingDone {
			events = append(events, StreamEvent{
				Event: "content_block_stop",
				Data: ContentBlockStopEvent{
					Type:  "content_block_stop",
					Index: c.contentIndex,
				},
			})
		}

		c.outputTokens = r.Metrics.EvalCount
		stopReason := mapStopReason(r.DoneReason, len(c.toolCallsSent) > 0)

		events = append(events, StreamEvent{
			Event: "message_delta",
			Data: MessageDeltaEvent{
				Type: "message_delta",
				Delta: MessageDelta{
					StopReason: stopReason,
				},
				Usage: DeltaUsage{
					OutputTokens: c.outputTokens,
				},
			},
		})

		events = append(events, StreamEvent{
			Event: "message_stop",
			Data: MessageStopEvent{
				Type: "message_stop",
			},
		})
	}

	return events
}

// generateID generates a unique ID with the given prefix using crypto/rand
func generateID(prefix string) string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		// Fallback to time-based ID if crypto/rand fails
		return fmt.Sprintf("%s_%d", prefix, time.Now().UnixNano())
	}
	return fmt.Sprintf("%s_%x", prefix, b)
}

// GenerateMessageID generates a unique message ID
func GenerateMessageID() string {
	return generateID("msg")
}

// ptr returns a pointer to the given string value
func ptr(s string) *string {
	return &s
}

// mapToArgs converts a map to ToolCallFunctionArguments
func mapToArgs(m map[string]any) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	for k, v := range m {
		args.Set(k, v)
	}
	return args
}
