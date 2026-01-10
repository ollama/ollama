package openai

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"

	"github.com/ollama/ollama/api"
)

// ResponsesContent is a discriminated union for input content types.
// Concrete types: ResponsesTextContent, ResponsesImageContent
type ResponsesContent interface {
	responsesContent() // unexported marker method
}

type ResponsesTextContent struct {
	Type string `json:"type"` // always "input_text"
	Text string `json:"text"`
}

func (ResponsesTextContent) responsesContent() {}

type ResponsesImageContent struct {
	Type string `json:"type"` // always "input_image"
	// TODO(drifkin): is this really required? that seems verbose and a default is specified in the docs
	Detail   string `json:"detail"`              // required
	FileID   string `json:"file_id,omitempty"`   // optional
	ImageURL string `json:"image_url,omitempty"` // optional
}

func (ResponsesImageContent) responsesContent() {}

// ResponsesOutputTextContent represents output text from a previous assistant response
// that is being passed back as part of the conversation history.
type ResponsesOutputTextContent struct {
	Type string `json:"type"` // always "output_text"
	Text string `json:"text"`
}

func (ResponsesOutputTextContent) responsesContent() {}

type ResponsesInputMessage struct {
	Type    string             `json:"type"` // always "message"
	Role    string             `json:"role"` // one of `user`, `system`, `developer`
	Content []ResponsesContent `json:"content,omitempty"`
}

func (m *ResponsesInputMessage) UnmarshalJSON(data []byte) error {
	var aux struct {
		Type    string          `json:"type"`
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	m.Type = aux.Type
	m.Role = aux.Role

	if len(aux.Content) == 0 {
		return nil
	}

	// Try to parse content as a string first (shorthand format)
	var contentStr string
	if err := json.Unmarshal(aux.Content, &contentStr); err == nil {
		m.Content = []ResponsesContent{
			ResponsesTextContent{Type: "input_text", Text: contentStr},
		}
		return nil
	}

	// Otherwise, parse as an array of content items
	var rawItems []json.RawMessage
	if err := json.Unmarshal(aux.Content, &rawItems); err != nil {
		return fmt.Errorf("content must be a string or array: %w", err)
	}

	m.Content = make([]ResponsesContent, 0, len(rawItems))
	for i, raw := range rawItems {
		// Peek at the type field to determine which concrete type to use
		var typeField struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(raw, &typeField); err != nil {
			return fmt.Errorf("content[%d]: %w", i, err)
		}

		switch typeField.Type {
		case "input_text":
			var content ResponsesTextContent
			if err := json.Unmarshal(raw, &content); err != nil {
				return fmt.Errorf("content[%d]: %w", i, err)
			}
			m.Content = append(m.Content, content)
		case "input_image":
			var content ResponsesImageContent
			if err := json.Unmarshal(raw, &content); err != nil {
				return fmt.Errorf("content[%d]: %w", i, err)
			}
			m.Content = append(m.Content, content)
		case "output_text":
			var content ResponsesOutputTextContent
			if err := json.Unmarshal(raw, &content); err != nil {
				return fmt.Errorf("content[%d]: %w", i, err)
			}
			m.Content = append(m.Content, content)
		default:
			return fmt.Errorf("content[%d]: unknown content type: %s", i, typeField.Type)
		}
	}

	return nil
}

type ResponsesOutputMessage struct{}

// ResponsesInputItem is a discriminated union for input items.
// Concrete types: ResponsesInputMessage (more to come)
type ResponsesInputItem interface {
	responsesInputItem() // unexported marker method
}

func (ResponsesInputMessage) responsesInputItem() {}

// ResponsesFunctionCall represents an assistant's function call in conversation history.
type ResponsesFunctionCall struct {
	ID        string `json:"id,omitempty"` // item ID
	Type      string `json:"type"`         // always "function_call"
	CallID    string `json:"call_id"`      // the tool call ID
	Name      string `json:"name"`         // function name
	Arguments string `json:"arguments"`    // JSON arguments string
}

func (ResponsesFunctionCall) responsesInputItem() {}

// ResponsesFunctionCallOutput represents a function call result from the client.
type ResponsesFunctionCallOutput struct {
	Type   string `json:"type"`    // always "function_call_output"
	CallID string `json:"call_id"` // links to the original function call
	Output string `json:"output"`  // the function result
}

func (ResponsesFunctionCallOutput) responsesInputItem() {}

// ResponsesReasoningInput represents a reasoning item passed back as input.
// This is used when the client sends previous reasoning back for context.
type ResponsesReasoningInput struct {
	ID               string                      `json:"id,omitempty"`
	Type             string                      `json:"type"` // always "reasoning"
	Summary          []ResponsesReasoningSummary `json:"summary,omitempty"`
	EncryptedContent string                      `json:"encrypted_content,omitempty"`
}

func (ResponsesReasoningInput) responsesInputItem() {}

// unmarshalResponsesInputItem unmarshals a single input item from JSON.
func unmarshalResponsesInputItem(data []byte) (ResponsesInputItem, error) {
	var typeField struct {
		Type string `json:"type"`
		Role string `json:"role"`
	}
	if err := json.Unmarshal(data, &typeField); err != nil {
		return nil, err
	}

	// Handle shorthand message format: {"role": "...", "content": "..."}
	// When type is empty but role is present, treat as a message
	itemType := typeField.Type
	if itemType == "" && typeField.Role != "" {
		itemType = "message"
	}

	switch itemType {
	case "message":
		var msg ResponsesInputMessage
		if err := json.Unmarshal(data, &msg); err != nil {
			return nil, err
		}
		return msg, nil
	case "function_call":
		var fc ResponsesFunctionCall
		if err := json.Unmarshal(data, &fc); err != nil {
			return nil, err
		}
		return fc, nil
	case "function_call_output":
		var output ResponsesFunctionCallOutput
		if err := json.Unmarshal(data, &output); err != nil {
			return nil, err
		}
		return output, nil
	case "reasoning":
		var reasoning ResponsesReasoningInput
		if err := json.Unmarshal(data, &reasoning); err != nil {
			return nil, err
		}
		return reasoning, nil
	default:
		return nil, fmt.Errorf("unknown input item type: %s", typeField.Type)
	}
}

// ResponsesInput can be either:
// - a string (equivalent to a text input with the user role)
// - an array of input items (see ResponsesInputItem)
type ResponsesInput struct {
	Text  string               // set if input was a plain string
	Items []ResponsesInputItem // set if input was an array
}

func (r *ResponsesInput) UnmarshalJSON(data []byte) error {
	// Try string first
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		r.Text = s
		return nil
	}

	// Otherwise, try array of input items
	var rawItems []json.RawMessage
	if err := json.Unmarshal(data, &rawItems); err != nil {
		return fmt.Errorf("input must be a string or array: %w", err)
	}

	r.Items = make([]ResponsesInputItem, 0, len(rawItems))
	for i, raw := range rawItems {
		item, err := unmarshalResponsesInputItem(raw)
		if err != nil {
			return fmt.Errorf("input[%d]: %w", i, err)
		}
		r.Items = append(r.Items, item)
	}

	return nil
}

type ResponsesReasoning struct {
	// originally: optional, default is per-model
	Effort string `json:"effort,omitempty"`

	// originally: deprecated, use `summary` instead. One of `auto`, `concise`, `detailed`
	GenerateSummary string `json:"generate_summary,omitempty"`

	// originally: optional, one of `auto`, `concise`, `detailed`
	Summary string `json:"summary,omitempty"`
}

type ResponsesTextFormat struct {
	Type   string          `json:"type"`             // "text", "json_schema"
	Name   string          `json:"name,omitempty"`   // for json_schema
	Schema json.RawMessage `json:"schema,omitempty"` // for json_schema
	Strict *bool           `json:"strict,omitempty"` // for json_schema
}

type ResponsesText struct {
	Format *ResponsesTextFormat `json:"format,omitempty"`
}

// ResponsesTool represents a tool in the Responses API format.
// Note: This differs from api.Tool which nests fields under "function".
type ResponsesTool struct {
	Type        string         `json:"type"` // "function"
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Strict      bool           `json:"strict,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Function    *struct {
		Name        string         `json:"name"`
		Description string         `json:"description,omitempty"`
		Parameters  map[string]any `json:"parameters,omitempty"`
	} `json:"function,omitempty"`
}

type ResponsesRequest struct {
	Model string `json:"model"`

	// originally: optional, default is false
	// for us: not supported
	Background bool `json:"background"`

	// originally: optional `string | {id: string}`
	// for us: not supported
	Conversation json.RawMessage `json:"conversation"`

	// originally: string[]
	// for us: ignored
	Include []string `json:"include"`

	Input ResponsesInput `json:"input"`

	// optional, inserts a system message at the start of the conversation
	Instructions string `json:"instructions,omitempty"`

	// optional, maps to num_predict
	MaxOutputTokens *int `json:"max_output_tokens,omitempty"`

	Reasoning ResponsesReasoning `json:"reasoning"`

	// optional, default is 1.0
	Temperature *float64 `json:"temperature"`

	// optional, controls output format (e.g. json_schema)
	Text *ResponsesText `json:"text,omitempty"`

	// optional, default is 1.0
	TopP *float64 `json:"top_p"`

	// optional, default is `"disabled"`
	Truncation *string `json:"truncation"`

	Tools []ResponsesTool `json:"tools,omitempty"`
	// optional, tool selection policy (e.g. "auto", "required", or {"type":"function","name":"..."})
	ToolChoice any `json:"tool_choice,omitempty"`

	// TODO(drifkin): tool_choice is not supported. We could support "none" by not
	// passing tools, but the other controls like `"required"` cannot be generally
	// supported.

	// optional, default is false
	Stream *bool `json:"stream,omitempty"`
}

// FromResponsesRequest converts a ResponsesRequest to api.ChatRequest
func FromResponsesRequest(r ResponsesRequest) (*api.ChatRequest, error) {
	var messages []api.Message

	// Add instructions as system message if present
	if r.Instructions != "" {
		messages = append(messages, api.Message{
			Role:    "system",
			Content: r.Instructions,
		})
	}

	// Handle simple string input
	if r.Input.Text != "" {
		messages = append(messages, api.Message{
			Role:    "user",
			Content: r.Input.Text,
		})
	}

	// Handle array of input items
	// Track pending reasoning to merge with the next assistant message
	var pendingThinking string

	for _, item := range r.Input.Items {
		switch v := item.(type) {
		case ResponsesReasoningInput:
			// Store thinking to merge with the next assistant message
			pendingThinking = v.EncryptedContent
		case ResponsesInputMessage:
			msg, err := convertInputMessage(v)
			if err != nil {
				return nil, err
			}
			// If this is an assistant message, attach pending thinking
			if msg.Role == "assistant" && pendingThinking != "" {
				msg.Thinking = pendingThinking
				pendingThinking = ""
			}
			messages = append(messages, msg)
		case ResponsesFunctionCall:
			// Convert function call to assistant message with tool calls
			var args api.ToolCallFunctionArguments
			if v.Arguments != "" {
				if err := json.Unmarshal([]byte(v.Arguments), &args); err != nil {
					return nil, fmt.Errorf("failed to parse function call arguments: %w", err)
				}
			}
			toolCall := api.ToolCall{
				ID: v.CallID,
				Function: api.ToolCallFunction{
					Name:      v.Name,
					Arguments: args,
				},
			}

			// Merge tool call into existing assistant message if it has content or tool calls
			if len(messages) > 0 && messages[len(messages)-1].Role == "assistant" {
				lastMsg := &messages[len(messages)-1]
				lastMsg.ToolCalls = append(lastMsg.ToolCalls, toolCall)
				if pendingThinking != "" {
					lastMsg.Thinking = pendingThinking
					pendingThinking = ""
				}
			} else {
				msg := api.Message{
					Role:      "assistant",
					ToolCalls: []api.ToolCall{toolCall},
				}
				if pendingThinking != "" {
					msg.Thinking = pendingThinking
					pendingThinking = ""
				}
				messages = append(messages, msg)
			}
		case ResponsesFunctionCallOutput:
			messages = append(messages, api.Message{
				Role:       "tool",
				Content:    v.Output,
				ToolCallID: v.CallID,
			})
		}
	}

	// If there's trailing reasoning without a following message, emit it
	if pendingThinking != "" {
		messages = append(messages, api.Message{
			Role:     "assistant",
			Thinking: pendingThinking,
		})
	}

	options := make(map[string]any)

	if r.Temperature != nil {
		options["temperature"] = *r.Temperature
	} else {
		options["temperature"] = 1.0
	}

	if r.TopP != nil {
		options["top_p"] = *r.TopP
	} else { //nolint:staticcheck // SA9003: empty branch
		// TODO(drifkin): OpenAI defaults to 1.0 here, but we don't follow that here
		// in case the model has a different default. It would be best if we
		// understood whether there was a model-specific default and if not, we
		// should also default to 1.0, but that will require some additional
		// plumbing
	}

	if r.MaxOutputTokens != nil {
		options["num_predict"] = *r.MaxOutputTokens
	}
	if r.ToolChoice != nil {
		options["tool_choice"] = r.ToolChoice
	}

	// Convert tools from Responses API format to api.Tool format
	var tools []api.Tool
	for _, t := range r.Tools {
		toolType := strings.ToLower(strings.TrimSpace(t.Type))
		// Only function tools are supported in chat/completions.
		if toolType != "" && toolType != "function" {
			continue
		}
		name := strings.TrimSpace(t.Name)
		if name == "" && t.Function != nil {
			name = strings.TrimSpace(t.Function.Name)
		}
		if name == "" {
			continue
		}
		tool, err := convertTool(t)
		if err != nil {
			return nil, err
		}
		tools = append(tools, tool)
	}

	// Handle text format (e.g. json_schema)
	var format json.RawMessage
	if r.Text != nil && r.Text.Format != nil {
		switch r.Text.Format.Type {
		case "json_schema":
			if r.Text.Format.Schema != nil {
				format = r.Text.Format.Schema
			}
		}
	}

	return &api.ChatRequest{
		Model:    r.Model,
		Messages: messages,
		Options:  options,
		Tools:    tools,
		Format:   format,
	}, nil
}

func convertTool(t ResponsesTool) (api.Tool, error) {
	name := strings.TrimSpace(t.Name)
	desc := t.Description
	paramsIn := t.Parameters
	if t.Function != nil {
		if name == "" {
			name = strings.TrimSpace(t.Function.Name)
		}
		if desc == "" {
			desc = t.Function.Description
		}
		if paramsIn == nil {
			paramsIn = t.Function.Parameters
		}
	}

	// Convert parameters from map[string]any to api.ToolFunctionParameters
	var params api.ToolFunctionParameters
	if paramsIn != nil {
		// Marshal and unmarshal to convert
		b, err := json.Marshal(paramsIn)
		if err != nil {
			return api.Tool{}, fmt.Errorf("failed to marshal tool parameters: %w", err)
		}
		if err := json.Unmarshal(b, &params); err != nil {
			return api.Tool{}, fmt.Errorf("failed to unmarshal tool parameters: %w", err)
		}
	}

	return api.Tool{
		Type: t.Type,
		Function: api.ToolFunction{
			Name:        name,
			Description: desc,
			Parameters:  params,
		},
	}, nil
}

func convertInputMessage(m ResponsesInputMessage) (api.Message, error) {
	var content string
	var images []api.ImageData

	for _, c := range m.Content {
		switch v := c.(type) {
		case ResponsesTextContent:
			content += v.Text
		case ResponsesOutputTextContent:
			content += v.Text
		case ResponsesImageContent:
			if v.ImageURL == "" {
				continue // Skip if no URL (FileID not supported)
			}
			img, err := decodeImageURL(v.ImageURL)
			if err != nil {
				return api.Message{}, err
			}
			images = append(images, img)
		}
	}

	return api.Message{
		Role:    m.Role,
		Content: content,
		Images:  images,
	}, nil
}

// Response types for the Responses API

type ResponsesResponse struct {
	ID        string                `json:"id"`
	Object    string                `json:"object"`
	CreatedAt int64                 `json:"created_at"`
	Status    string                `json:"status"`
	Model     string                `json:"model"`
	Output    []ResponsesOutputItem `json:"output"`
	Usage     *ResponsesUsage       `json:"usage,omitempty"`
	// TODO(drifkin): add `temperature` and `top_p` to the response, but this
	// requires additional plumbing to find the effective values since the
	// defaults can come from the model or the request
}

type ResponsesOutputItem struct {
	ID        string                   `json:"id"`
	Type      string                   `json:"type"` // "message", "function_call", or "reasoning"
	Status    string                   `json:"status,omitempty"`
	Role      string                   `json:"role,omitempty"`      // for message
	Content   []ResponsesOutputContent `json:"content,omitempty"`   // for message
	CallID    string                   `json:"call_id,omitempty"`   // for function_call
	Name      string                   `json:"name,omitempty"`      // for function_call
	Arguments string                   `json:"arguments,omitempty"` // for function_call

	// Reasoning fields
	Summary          []ResponsesReasoningSummary `json:"summary,omitempty"`           // for reasoning
	EncryptedContent string                      `json:"encrypted_content,omitempty"` // for reasoning
}

type ResponsesReasoningSummary struct {
	Type string `json:"type"` // "summary_text"
	Text string `json:"text"`
}

type ResponsesOutputContent struct {
	Type string `json:"type"` // "output_text"
	Text string `json:"text"`
}

type ResponsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// ToResponse converts an api.ChatResponse to a Responses API response
func ToResponse(model, responseID, itemID string, chatResponse api.ChatResponse) ResponsesResponse {
	var output []ResponsesOutputItem

	// Add reasoning item if thinking is present
	if chatResponse.Message.Thinking != "" {
		output = append(output, ResponsesOutputItem{
			ID:   fmt.Sprintf("rs_%s", responseID),
			Type: "reasoning",
			Summary: []ResponsesReasoningSummary{
				{
					Type: "summary_text",
					Text: chatResponse.Message.Thinking,
				},
			},
			EncryptedContent: chatResponse.Message.Thinking, // Plain text for now
		})
	}

	if len(chatResponse.Message.ToolCalls) > 0 {
		toolCalls := ToToolCalls(chatResponse.Message.ToolCalls)
		for i, tc := range toolCalls {
			output = append(output, ResponsesOutputItem{
				ID:        fmt.Sprintf("fc_%s_%d", responseID, i),
				Type:      "function_call",
				CallID:    tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
	} else {
		output = append(output, ResponsesOutputItem{
			ID:     itemID,
			Type:   "message",
			Status: "completed",
			Role:   "assistant",
			Content: []ResponsesOutputContent{
				{
					Type: "output_text",
					Text: chatResponse.Message.Content,
				},
			},
		})
	}

	return ResponsesResponse{
		ID:        responseID,
		Object:    "response",
		CreatedAt: chatResponse.CreatedAt.Unix(),
		Status:    "completed",
		Model:     model,
		Output:    output,
		Usage: &ResponsesUsage{
			InputTokens:  chatResponse.PromptEvalCount,
			OutputTokens: chatResponse.EvalCount,
			TotalTokens:  chatResponse.PromptEvalCount + chatResponse.EvalCount,
		},
	}
}

// Streaming events: <https://platform.openai.com/docs/api-reference/responses-streaming>

// ResponsesStreamEvent represents a single Server-Sent Event for the Responses API.
type ResponsesStreamEvent struct {
	Event string // The event type (e.g., "response.created")
	Data  any    // The event payload (will be JSON-marshaled)
}

// ResponsesStreamConverter converts api.ChatResponse objects to Responses API
// streaming events. It maintains state across multiple calls to handle the
// streaming event sequence correctly.
type ResponsesStreamConverter struct {
	// Configuration (immutable after creation)
	responseID string
	itemID     string
	model      string

	// State tracking (mutated across Process calls)
	firstWrite      bool
	outputIndex     int
	contentIndex    int
	contentStarted  bool
	toolCallsSent   bool
	accumulatedText string
	sequenceNumber  int

	// Reasoning/thinking state
	accumulatedThinking string
	reasoningItemID     string
	reasoningStarted    bool
	reasoningDone       bool

	// Tool calls state (for final output)
	toolCallItems []map[string]any
}

// newEvent creates a ResponsesStreamEvent with the sequence number included in the data.
func (c *ResponsesStreamConverter) newEvent(eventType string, data map[string]any) ResponsesStreamEvent {
	data["type"] = eventType
	data["sequence_number"] = c.sequenceNumber
	c.sequenceNumber++
	return ResponsesStreamEvent{
		Event: eventType,
		Data:  data,
	}
}

// NewResponsesStreamConverter creates a new converter with the given configuration.
func NewResponsesStreamConverter(responseID, itemID, model string) *ResponsesStreamConverter {
	return &ResponsesStreamConverter{
		responseID: responseID,
		itemID:     itemID,
		model:      model,
		firstWrite: true,
	}
}

// Process takes a ChatResponse and returns the events that should be emitted.
// Events are returned in order. The caller is responsible for serializing
// and sending these events.
func (c *ResponsesStreamConverter) Process(r api.ChatResponse) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	hasToolCalls := len(r.Message.ToolCalls) > 0
	hasThinking := r.Message.Thinking != ""

	// First chunk - emit initial events
	if c.firstWrite {
		c.firstWrite = false
		events = append(events, c.createResponseCreatedEvent())
		events = append(events, c.createResponseInProgressEvent())
	}

	// Handle reasoning/thinking (before other content)
	if hasThinking {
		events = append(events, c.processThinking(r.Message.Thinking)...)
	}

	// Handle tool calls
	if hasToolCalls {
		events = append(events, c.processToolCalls(r.Message.ToolCalls)...)
		c.toolCallsSent = true
	}

	// Handle text content (only if no tool calls)
	if !hasToolCalls && !c.toolCallsSent && r.Message.Content != "" {
		events = append(events, c.processTextContent(r.Message.Content)...)
	}

	// Done - emit closing events
	if r.Done {
		events = append(events, c.processCompletion(r)...)
	}

	return events
}

func (c *ResponsesStreamConverter) createResponseCreatedEvent() ResponsesStreamEvent {
	return c.newEvent("response.created", map[string]any{
		"response": map[string]any{
			"id":     c.responseID,
			"object": "response",
			"status": "in_progress",
			"output": []any{},
		},
	})
}

func (c *ResponsesStreamConverter) createResponseInProgressEvent() ResponsesStreamEvent {
	return c.newEvent("response.in_progress", map[string]any{
		"response": map[string]any{
			"id":     c.responseID,
			"object": "response",
			"status": "in_progress",
			"output": []any{},
		},
	})
}

func (c *ResponsesStreamConverter) processThinking(thinking string) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Start reasoning item if not started
	if !c.reasoningStarted {
		c.reasoningStarted = true
		c.reasoningItemID = fmt.Sprintf("rs_%d", rand.Intn(999999))

		events = append(events, c.newEvent("response.output_item.added", map[string]any{
			"output_index": c.outputIndex,
			"item": map[string]any{
				"id":      c.reasoningItemID,
				"type":    "reasoning",
				"summary": []any{},
			},
		}))
	}

	// Accumulate thinking
	c.accumulatedThinking += thinking

	// Emit delta
	events = append(events, c.newEvent("response.reasoning_summary_text.delta", map[string]any{
		"item_id":      c.reasoningItemID,
		"output_index": c.outputIndex,
		"delta":        thinking,
	}))

	// TODO(drifkin): consider adding
	// [`response.reasoning_text.delta`](https://platform.openai.com/docs/api-reference/responses-streaming/response/reasoning_text/delta),
	// but need to do additional research to understand how it's used and how
	// widely supported it is

	return events
}

func (c *ResponsesStreamConverter) finishReasoning() []ResponsesStreamEvent {
	if !c.reasoningStarted || c.reasoningDone {
		return nil
	}
	c.reasoningDone = true

	events := []ResponsesStreamEvent{
		c.newEvent("response.reasoning_summary_text.done", map[string]any{
			"item_id":      c.reasoningItemID,
			"output_index": c.outputIndex,
			"text":         c.accumulatedThinking,
		}),
		c.newEvent("response.output_item.done", map[string]any{
			"output_index": c.outputIndex,
			"item": map[string]any{
				"id":                c.reasoningItemID,
				"type":              "reasoning",
				"summary":           []map[string]any{{"type": "summary_text", "text": c.accumulatedThinking}},
				"encrypted_content": c.accumulatedThinking, // Plain text for now
			},
		}),
	}

	c.outputIndex++
	return events
}

func (c *ResponsesStreamConverter) processToolCalls(toolCalls []api.ToolCall) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Finish reasoning first if it was started
	events = append(events, c.finishReasoning()...)

	converted := ToToolCalls(toolCalls)

	for i, tc := range converted {
		fcItemID := fmt.Sprintf("fc_%d_%d", rand.Intn(999999), i)

		// Store for final output (with status: completed)
		toolCallItem := map[string]any{
			"id":        fcItemID,
			"type":      "function_call",
			"status":    "completed",
			"call_id":   tc.ID,
			"name":      tc.Function.Name,
			"arguments": tc.Function.Arguments,
		}
		c.toolCallItems = append(c.toolCallItems, toolCallItem)

		// response.output_item.added for function call
		events = append(events, c.newEvent("response.output_item.added", map[string]any{
			"output_index": c.outputIndex + i,
			"item": map[string]any{
				"id":        fcItemID,
				"type":      "function_call",
				"status":    "in_progress",
				"call_id":   tc.ID,
				"name":      tc.Function.Name,
				"arguments": "",
			},
		}))

		// response.function_call_arguments.delta
		if tc.Function.Arguments != "" {
			events = append(events, c.newEvent("response.function_call_arguments.delta", map[string]any{
				"item_id":      fcItemID,
				"output_index": c.outputIndex + i,
				"delta":        tc.Function.Arguments,
			}))
		}

		// response.function_call_arguments.done
		events = append(events, c.newEvent("response.function_call_arguments.done", map[string]any{
			"item_id":      fcItemID,
			"output_index": c.outputIndex + i,
			"arguments":    tc.Function.Arguments,
		}))

		// response.output_item.done for function call
		events = append(events, c.newEvent("response.output_item.done", map[string]any{
			"output_index": c.outputIndex + i,
			"item": map[string]any{
				"id":        fcItemID,
				"type":      "function_call",
				"status":    "completed",
				"call_id":   tc.ID,
				"name":      tc.Function.Name,
				"arguments": tc.Function.Arguments,
			},
		}))
	}

	return events
}

func (c *ResponsesStreamConverter) processTextContent(content string) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Finish reasoning first if it was started
	events = append(events, c.finishReasoning()...)

	// Emit output item and content part for first text content
	if !c.contentStarted {
		c.contentStarted = true

		// response.output_item.added
		events = append(events, c.newEvent("response.output_item.added", map[string]any{
			"output_index": c.outputIndex,
			"item": map[string]any{
				"id":      c.itemID,
				"type":    "message",
				"status":  "in_progress",
				"role":    "assistant",
				"content": []any{},
			},
		}))

		// response.content_part.added
		events = append(events, c.newEvent("response.content_part.added", map[string]any{
			"item_id":       c.itemID,
			"output_index":  c.outputIndex,
			"content_index": c.contentIndex,
			"part": map[string]any{
				"type": "output_text",
				"text": "",
			},
		}))
	}

	// Accumulate text
	c.accumulatedText += content

	// Emit content delta
	events = append(events, c.newEvent("response.output_text.delta", map[string]any{
		"item_id":       c.itemID,
		"output_index":  c.outputIndex,
		"content_index": 0,
		"delta":         content,
	}))

	return events
}

func (c *ResponsesStreamConverter) buildFinalOutput() []any {
	var output []any

	// Add reasoning item if present
	if c.reasoningStarted {
		output = append(output, map[string]any{
			"id":                c.reasoningItemID,
			"type":              "reasoning",
			"summary":           []map[string]any{{"type": "summary_text", "text": c.accumulatedThinking}},
			"encrypted_content": c.accumulatedThinking,
		})
	}

	// Add tool calls if present
	if len(c.toolCallItems) > 0 {
		for _, item := range c.toolCallItems {
			output = append(output, item)
		}
	} else if c.contentStarted {
		// Add message item if we had text content
		output = append(output, map[string]any{
			"id":     c.itemID,
			"type":   "message",
			"status": "completed",
			"role":   "assistant",
			"content": []map[string]any{{
				"type": "output_text",
				"text": c.accumulatedText,
			}},
		})
	}

	return output
}

func (c *ResponsesStreamConverter) processCompletion(r api.ChatResponse) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent

	// Finish reasoning if not done
	events = append(events, c.finishReasoning()...)

	// Emit text completion events if we had text content
	if !c.toolCallsSent && c.contentStarted {
		// response.output_text.done
		events = append(events, c.newEvent("response.output_text.done", map[string]any{
			"item_id":       c.itemID,
			"output_index":  c.outputIndex,
			"content_index": 0,
			"text":          c.accumulatedText,
		}))

		// response.content_part.done
		events = append(events, c.newEvent("response.content_part.done", map[string]any{
			"item_id":       c.itemID,
			"output_index":  c.outputIndex,
			"content_index": 0,
			"part": map[string]any{
				"type": "output_text",
				"text": c.accumulatedText,
			},
		}))

		// response.output_item.done
		events = append(events, c.newEvent("response.output_item.done", map[string]any{
			"output_index": c.outputIndex,
			"item": map[string]any{
				"id":     c.itemID,
				"type":   "message",
				"status": "completed",
				"role":   "assistant",
				"content": []map[string]any{{
					"type": "output_text",
					"text": c.accumulatedText,
				}},
			},
		}))
	}

	// response.completed
	events = append(events, c.newEvent("response.completed", map[string]any{
		"response": map[string]any{
			"id":     c.responseID,
			"object": "response",
			"status": "completed",
			"output": c.buildFinalOutput(),
			"usage": map[string]any{
				"input_tokens":  r.PromptEvalCount,
				"output_tokens": r.EvalCount,
				"total_tokens":  r.PromptEvalCount + r.EvalCount,
			},
		},
	}))

	return events
}
