package openai

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// ResponsesContent is a discriminated union for input content types.
// Concrete types: ResponsesTextContent, ResponsesImageContent,
// ResponsesOutputTextContent, ResponsesFileContent.
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

type ResponsesFileContent struct {
	Type     string `json:"type"` // always "input_file"
	FileData string `json:"file_data,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	FileURL  string `json:"file_url,omitempty"`
	Filename string `json:"filename,omitempty"`
}

func (ResponsesFileContent) responsesContent() {}

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
		content, err := unmarshalResponsesContent(raw)
		if err != nil {
			return fmt.Errorf("content[%d]: %w", i, err)
		}
		m.Content = append(m.Content, content)
	}

	return nil
}

func unmarshalResponsesContent(data []byte) (ResponsesContent, error) {
	// Peek at the type field to determine which concrete type to use
	var typeField struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &typeField); err != nil {
		return nil, err
	}

	switch typeField.Type {
	case "input_text":
		var content ResponsesTextContent
		if err := json.Unmarshal(data, &content); err != nil {
			return nil, err
		}
		return content, nil
	case "input_image":
		var content ResponsesImageContent
		if err := json.Unmarshal(data, &content); err != nil {
			return nil, err
		}
		return content, nil
	case "output_text":
		var content ResponsesOutputTextContent
		if err := json.Unmarshal(data, &content); err != nil {
			return nil, err
		}
		return content, nil
	case "input_file":
		var content ResponsesFileContent
		if err := json.Unmarshal(data, &content); err != nil {
			return nil, err
		}
		return content, nil
	default:
		return nil, fmt.Errorf("unknown content type: %s", typeField.Type)
	}
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

	// OutputItems is populated when output is provided as Responses content
	// items instead of the string shorthand.
	OutputItems []ResponsesContent `json:"-"`
}

func (o *ResponsesFunctionCallOutput) UnmarshalJSON(data []byte) error {
	var aux struct {
		Type   string          `json:"type"`
		CallID string          `json:"call_id"`
		Output json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	o.Type = aux.Type
	o.CallID = aux.CallID
	o.Output = ""
	o.OutputItems = nil

	if len(aux.Output) == 0 {
		return nil
	}

	var output string
	if err := json.Unmarshal(aux.Output, &output); err == nil {
		o.Output = output
		return nil
	}

	var rawItems []json.RawMessage
	if err := json.Unmarshal(aux.Output, &rawItems); err != nil {
		return fmt.Errorf("output must be a string or array: %w", err)
	}

	o.OutputItems = make([]ResponsesContent, 0, len(rawItems))
	var outputText strings.Builder
	for i, raw := range rawItems {
		content, err := unmarshalResponsesContent(raw)
		if err != nil {
			return fmt.Errorf("output[%d]: %w", i, err)
		}
		o.OutputItems = append(o.OutputItems, content)

		switch v := content.(type) {
		case ResponsesTextContent:
			outputText.WriteString(v.Text)
		case ResponsesOutputTextContent:
			outputText.WriteString(v.Text)
		}
	}
	o.Output = outputText.String()
	return nil
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
	case "web_search_call":
		var call ResponsesWebSearchCall
		if err := json.Unmarshal(data, &call); err != nil {
			return nil, err
		}
		return call, nil
	default:
		if itemType == "" {
			return nil, fmt.Errorf("input item missing required 'type' field")
		}
		return nil, fmt.Errorf("unknown input item type: %q", itemType)
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
	Type        string         `json:"type"` // "function", "namespace", or "web_search"
	Name        string         `json:"name"`
	Description *string        `json:"description"` // nullable but required
	Strict      *bool          `json:"strict"`      // nullable but required
	Parameters  map[string]any `json:"parameters"`  // nullable but required

	// Tools carries a "namespace" declaration's member functions. The
	// Responses API groups related tools by domain under a namespace tool
	// whose nested tools array holds the real function definitions.
	Tools []ResponsesTool `json:"tools,omitempty"`
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

	for i, item := range r.Input.Items {
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
			// Responses may replay an assistant message between a function_call and
			// its function_call_output. Keep those items in one Chat assistant message
			// so the tool result immediately follows the call it answers.
			var outputCallID string
			if i+1 < len(r.Input.Items) {
				if output, ok := r.Input.Items[i+1].(ResponsesFunctionCallOutput); ok {
					outputCallID = output.CallID
				}
			}
			if msg.Role == "assistant" && outputCallID != "" && len(messages) > 0 {
				lastMsg := &messages[len(messages)-1]
				if lastMsg.Role == "assistant" && len(lastMsg.ToolCalls) > 0 {
					merged := false
					for _, call := range lastMsg.ToolCalls {
						if call.ID != outputCallID {
							continue
						}
						if lastMsg.Content != "" && msg.Content != "" {
							lastMsg.Content += "\n"
						}
						lastMsg.Content += msg.Content
						lastMsg.Images = append(lastMsg.Images, msg.Images...)
						if msg.Thinking != "" {
							lastMsg.Thinking = msg.Thinking
						}
						merged = true
						break
					}
					if merged {
						continue
					}
				}
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
			content := v.Output
			var images []api.ImageData
			if len(v.OutputItems) > 0 {
				var err error
				content, images, err = convertResponsesContent(v.OutputItems)
				if err != nil {
					return nil, err
				}
			}
			messages = append(messages, api.Message{
				Role:       "tool",
				Content:    content,
				Images:     images,
				ToolCallID: v.CallID,
			})
		case ResponsesWebSearchCall:
			// Built-in tool calls are history metadata. The assistant message
			// that follows carries the model-visible result of the prior search.
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

	think, err := thinkFromReasoningEffort(r.Reasoning.Effort)
	if err != nil {
		return nil, err
	}

	// Convert tools from Responses API format to api.Tool format
	var tools []api.Tool
	hasWebSearch := HasWebSearchTool(r.Tools)
	for _, t := range r.Tools {
		if isWebSearchTool(t) {
			tools = append(tools, WebSearchFunctionTool())
			continue
		}
		expanded, err := convertTools(t)
		if err != nil {
			return nil, err
		}
		for _, tool := range expanded {
			// The built-in tool owns this name. Keeping a user-declared function
			// with the same name makes a model call ambiguous.
			if hasWebSearch && tool.Function.Name == "web_search" {
				continue
			}
			tools = append(tools, tool)
		}
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
		Think:    think,
	}, nil
}

func isWebSearchTool(t ResponsesTool) bool { return t.Type == "web_search" }

// HasWebSearchTool reports whether a request declares the built-in Responses
// web-search tool.
func HasWebSearchTool(tools []ResponsesTool) bool {
	for _, tool := range tools {
		if isWebSearchTool(tool) {
			return true
		}
	}
	return false
}

// WebSearchFunctionTool is the private function contract passed to local
// models. Responses clients must only ever see their original web_search
// declaration echoed back.
func WebSearchFunctionTool() api.Tool {
	properties := api.NewToolPropertiesMap()
	properties.Set("query", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "The search query."})
	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "web_search",
			Description: "Search the web for current information.",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Required:   []string{"query"},
				Properties: properties,
			},
		},
	}
}

// convertTools converts one Responses-API tool declaration to api.Tools. A
// "namespace" declaration groups member functions under a common name; it
// expands to those members with namespace-qualified names, since api.Tool
// carries only a flat function name. Dropping the members instead would
// leave the model with one schema-less pseudo-function and make every
// namespaced call undeclarable.
func convertTools(t ResponsesTool) ([]api.Tool, error) {
	if t.Type != "namespace" {
		tool, err := convertTool(t)
		if err != nil {
			return nil, err
		}
		return []api.Tool{tool}, nil
	}

	var tools []api.Tool
	for _, member := range t.Tools {
		expanded, err := convertTools(member)
		if err != nil {
			return nil, err
		}
		for i := range expanded {
			if prefix := t.Name + "."; t.Name != "" && !strings.HasPrefix(expanded[i].Function.Name, prefix) {
				expanded[i].Function.Name = prefix + expanded[i].Function.Name
			}
		}
		tools = append(tools, expanded...)
	}
	return tools, nil
}

func convertTool(t ResponsesTool) (api.Tool, error) {
	// Convert parameters from map[string]any to api.ToolFunctionParameters
	var params api.ToolFunctionParameters
	if t.Parameters != nil {
		// Marshal and unmarshal to convert
		b, err := json.Marshal(t.Parameters)
		if err != nil {
			return api.Tool{}, fmt.Errorf("failed to marshal tool parameters: %w", err)
		}
		if err := json.Unmarshal(b, &params); err != nil {
			return api.Tool{}, fmt.Errorf("failed to unmarshal tool parameters: %w", err)
		}
	}

	var description string
	if t.Description != nil {
		description = *t.Description
	}

	return api.Tool{
		Type: t.Type,
		Function: api.ToolFunction{
			Name:        t.Name,
			Description: description,
			Parameters:  params,
		},
	}, nil
}

func convertInputMessage(m ResponsesInputMessage) (api.Message, error) {
	content, images, err := convertResponsesContent(m.Content)
	if err != nil {
		return api.Message{}, err
	}

	return api.Message{
		Role:    m.Role,
		Content: content,
		Images:  images,
	}, nil
}

func convertResponsesContent(contents []ResponsesContent) (string, []api.ImageData, error) {
	var content string
	var images []api.ImageData

	for _, c := range contents {
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
				return "", nil, err
			}
			images = append(images, img)
		case ResponsesFileContent:
			// TODO(drifkin): support inlining text-only file_data when it is safe
			// to decode and of a reasonable size
			return "", nil, fmt.Errorf("file inputs are not currently supported")
		}
	}

	return content, images, nil
}

// Response types for the Responses API

// ResponsesTextField represents the text output configuration in the response.
type ResponsesTextField struct {
	Format ResponsesTextFormat `json:"format"`
}

// ResponsesReasoningOutput represents reasoning configuration in the response.
type ResponsesReasoningOutput struct {
	Effort  *string `json:"effort,omitempty"`
	Summary *string `json:"summary,omitempty"`
}

// ResponsesError represents an error in the response.
type ResponsesError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ResponsesIncompleteDetails represents details about why a response was incomplete.
type ResponsesIncompleteDetails struct {
	Reason string `json:"reason"`
}

type ResponsesResponse struct {
	ID                 string                      `json:"id"`
	Object             string                      `json:"object"`
	CreatedAt          int64                       `json:"created_at"`
	CompletedAt        *int64                      `json:"completed_at"`
	Status             string                      `json:"status"`
	IncompleteDetails  *ResponsesIncompleteDetails `json:"incomplete_details"`
	Model              string                      `json:"model"`
	PreviousResponseID *string                     `json:"previous_response_id"`
	Instructions       *string                     `json:"instructions"`
	Output             []ResponsesOutputItem       `json:"output"`
	Error              *ResponsesError             `json:"error"`
	Tools              []ResponsesTool             `json:"tools"`
	ToolChoice         any                         `json:"tool_choice"`
	Truncation         string                      `json:"truncation"`
	ParallelToolCalls  bool                        `json:"parallel_tool_calls"`
	Text               ResponsesTextField          `json:"text"`
	TopP               float64                     `json:"top_p"`
	PresencePenalty    float64                     `json:"presence_penalty"`
	FrequencyPenalty   float64                     `json:"frequency_penalty"`
	TopLogprobs        int                         `json:"top_logprobs"`
	Temperature        float64                     `json:"temperature"`
	Reasoning          *ResponsesReasoningOutput   `json:"reasoning"`
	Usage              *ResponsesUsage             `json:"usage"`
	MaxOutputTokens    *int                        `json:"max_output_tokens"`
	MaxToolCalls       *int                        `json:"max_tool_calls"`
	Store              bool                        `json:"store"`
	Background         bool                        `json:"background"`
	ServiceTier        string                      `json:"service_tier"`
	Metadata           map[string]any              `json:"metadata"`
	SafetyIdentifier   *string                     `json:"safety_identifier"`
	PromptCacheKey     *string                     `json:"prompt_cache_key"`
}

type ResponsesOutputItem struct {
	ID        string                    `json:"id"`
	Type      string                    `json:"type"` // "message", "function_call", or "reasoning"
	Status    string                    `json:"status,omitempty"`
	Role      string                    `json:"role,omitempty"`      // for message
	Content   []ResponsesOutputContent  `json:"content,omitempty"`   // for message
	CallID    string                    `json:"call_id,omitempty"`   // for function_call
	Name      string                    `json:"name,omitempty"`      // for function_call
	Arguments string                    `json:"arguments,omitempty"` // for function_call
	Action    *ResponsesWebSearchAction `json:"action,omitempty"`    // for web_search_call

	// Reasoning fields
	Summary          []ResponsesReasoningSummary `json:"summary,omitempty"`           // for reasoning
	EncryptedContent string                      `json:"encrypted_content,omitempty"` // for reasoning
}

// ResponsesWebSearchCall is the native output item emitted for an executed
// built-in web search. It is intentionally separate from function calls: the
// internal web_search function is never exposed through the Responses API.
type ResponsesWebSearchCall struct {
	ID     string                    `json:"id"`
	Type   string                    `json:"type"` // always "web_search_call"
	Status string                    `json:"status"`
	Action *ResponsesWebSearchAction `json:"action,omitempty"`
}

func (ResponsesWebSearchCall) responsesInputItem() {}

type ResponsesWebSearchAction struct {
	Type  string `json:"type"` // always "search"
	Query string `json:"query"`
}

// WebSearchCallOutputItem converts a web search call into the generic output
// item used by non-streaming Responses responses.
func WebSearchCallOutputItem(call ResponsesWebSearchCall) ResponsesOutputItem {
	return ResponsesOutputItem{
		ID:     call.ID,
		Type:   call.Type,
		Status: call.Status,
		Action: call.Action,
	}
}

type ResponsesReasoningSummary struct {
	Type string `json:"type"` // "summary_text"
	Text string `json:"text"`
}

type ResponsesOutputContent struct {
	Type        string `json:"type"` // "output_text"
	Text        string `json:"text"`
	Annotations []any  `json:"annotations"`
	Logprobs    []any  `json:"logprobs"`
}

type ResponsesInputTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type ResponsesOutputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type ResponsesUsage struct {
	InputTokens         int                          `json:"input_tokens"`
	OutputTokens        int                          `json:"output_tokens"`
	TotalTokens         int                          `json:"total_tokens"`
	InputTokensDetails  ResponsesInputTokensDetails  `json:"input_tokens_details"`
	OutputTokensDetails ResponsesOutputTokensDetails `json:"output_tokens_details"`
}

// derefFloat64 returns the value of a float64 pointer, or a default if nil.
func derefFloat64(p *float64, def float64) float64 {
	if p != nil {
		return *p
	}
	return def
}

// ToResponse converts an api.ChatResponse to a Responses API response.
// The request is used to echo back request parameters in the response.
func ToResponse(model, responseID, itemID string, chatResponse api.ChatResponse, request ResponsesRequest) ResponsesResponse {
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
				Status:    "completed",
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
					Type:        "output_text",
					Text:        chatResponse.Message.Content,
					Annotations: []any{},
					Logprobs:    []any{},
				},
			},
		})
	}

	var instructions *string
	if request.Instructions != "" {
		instructions = &request.Instructions
	}

	// Build truncation with default
	truncation := "disabled"
	if request.Truncation != nil {
		truncation = *request.Truncation
	}

	tools := request.Tools
	if tools == nil {
		tools = []ResponsesTool{}
	}

	text := ResponsesTextField{
		Format: ResponsesTextFormat{Type: "text"},
	}
	if request.Text != nil && request.Text.Format != nil {
		text.Format = *request.Text.Format
	}

	// Build reasoning output from request
	var reasoning *ResponsesReasoningOutput
	if request.Reasoning.Effort != "" || request.Reasoning.Summary != "" {
		reasoning = &ResponsesReasoningOutput{}
		if request.Reasoning.Effort != "" {
			reasoning.Effort = &request.Reasoning.Effort
		}
		if request.Reasoning.Summary != "" {
			reasoning.Summary = &request.Reasoning.Summary
		}
	}

	return ResponsesResponse{
		ID:                 responseID,
		Object:             "response",
		CreatedAt:          chatResponse.CreatedAt.Unix(),
		CompletedAt:        nil, // Set by middleware when writing final response
		Status:             "completed",
		IncompleteDetails:  nil, // Only populated if response incomplete
		Model:              model,
		PreviousResponseID: nil, // Not supported
		Instructions:       instructions,
		Output:             output,
		Error:              nil, // Only populated on failure
		Tools:              tools,
		ToolChoice:         "auto", // Default value
		Truncation:         truncation,
		ParallelToolCalls:  true, // Default value
		Text:               text,
		TopP:               derefFloat64(request.TopP, 1.0),
		PresencePenalty:    0, // Default value
		FrequencyPenalty:   0, // Default value
		TopLogprobs:        0, // Default value
		Temperature:        derefFloat64(request.Temperature, 1.0),
		Reasoning:          reasoning,
		Usage: &ResponsesUsage{
			InputTokens:        chatResponse.PromptEvalCount,
			OutputTokens:       chatResponse.EvalCount,
			TotalTokens:        chatResponse.PromptEvalCount + chatResponse.EvalCount,
			InputTokensDetails: ResponsesInputTokensDetails{CachedTokens: chatResponse.PromptEvalCachedCount},
			// TODO(drifkin): wire through the actual values
			OutputTokensDetails: ResponsesOutputTokensDetails{ReasoningTokens: 0},
		},
		MaxOutputTokens:  request.MaxOutputTokens,
		MaxToolCalls:     nil,   // Not supported
		Store:            false, // We don't store responses
		Background:       request.Background,
		ServiceTier:      "default", // Default value
		Metadata:         map[string]any{},
		SafetyIdentifier: nil, // Not supported
		PromptCacheKey:   nil, // Not supported
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
	request    ResponsesRequest

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

	// Items completed before the final message, in streamed output-index order.
	completedItems []any
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
func NewResponsesStreamConverter(responseID, itemID, model string, request ResponsesRequest) *ResponsesStreamConverter {
	return &ResponsesStreamConverter{
		responseID: responseID,
		itemID:     itemID,
		model:      model,
		request:    request,
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

// buildResponseObject creates a full response object with all required fields for streaming events.
func (c *ResponsesStreamConverter) buildResponseObject(status string, output []any, usage map[string]any) map[string]any {
	var instructions any = nil
	if c.request.Instructions != "" {
		instructions = c.request.Instructions
	}

	truncation := "disabled"
	if c.request.Truncation != nil {
		truncation = *c.request.Truncation
	}

	var tools []any
	if c.request.Tools != nil {
		for _, t := range c.request.Tools {
			tools = append(tools, map[string]any{
				"type":        t.Type,
				"name":        t.Name,
				"description": t.Description,
				"strict":      t.Strict,
				"parameters":  t.Parameters,
			})
		}
	}
	if tools == nil {
		tools = []any{}
	}

	textFormat := map[string]any{"type": "text"}
	if c.request.Text != nil && c.request.Text.Format != nil {
		textFormat = map[string]any{
			"type": c.request.Text.Format.Type,
		}
		if c.request.Text.Format.Name != "" {
			textFormat["name"] = c.request.Text.Format.Name
		}
		if c.request.Text.Format.Schema != nil {
			textFormat["schema"] = c.request.Text.Format.Schema
		}
		if c.request.Text.Format.Strict != nil {
			textFormat["strict"] = *c.request.Text.Format.Strict
		}
	}

	var reasoning any = nil
	if c.request.Reasoning.Effort != "" || c.request.Reasoning.Summary != "" {
		r := map[string]any{}
		if c.request.Reasoning.Effort != "" {
			r["effort"] = c.request.Reasoning.Effort
		} else {
			r["effort"] = nil
		}
		if c.request.Reasoning.Summary != "" {
			r["summary"] = c.request.Reasoning.Summary
		} else {
			r["summary"] = nil
		}
		reasoning = r
	}

	// Build top_p and temperature with defaults
	topP := 1.0
	if c.request.TopP != nil {
		topP = *c.request.TopP
	}
	temperature := 1.0
	if c.request.Temperature != nil {
		temperature = *c.request.Temperature
	}

	return map[string]any{
		"id":                   c.responseID,
		"object":               "response",
		"created_at":           time.Now().Unix(),
		"completed_at":         nil,
		"status":               status,
		"incomplete_details":   nil,
		"model":                c.model,
		"previous_response_id": nil,
		"instructions":         instructions,
		"output":               output,
		"error":                nil,
		"tools":                tools,
		"tool_choice":          "auto",
		"truncation":           truncation,
		"parallel_tool_calls":  true,
		"text":                 map[string]any{"format": textFormat},
		"top_p":                topP,
		"presence_penalty":     0,
		"frequency_penalty":    0,
		"top_logprobs":         0,
		"temperature":          temperature,
		"reasoning":            reasoning,
		"usage":                usage,
		"max_output_tokens":    c.request.MaxOutputTokens,
		"max_tool_calls":       nil,
		"store":                false,
		"background":           c.request.Background,
		"service_tier":         "default",
		"metadata":             map[string]any{},
		"safety_identifier":    nil,
		"prompt_cache_key":     nil,
	}
}

func (c *ResponsesStreamConverter) createResponseCreatedEvent() ResponsesStreamEvent {
	return c.newEvent("response.created", map[string]any{
		"response": c.buildResponseObject("in_progress", []any{}, nil),
	})
}

func (c *ResponsesStreamConverter) createResponseInProgressEvent() ResponsesStreamEvent {
	return c.newEvent("response.in_progress", map[string]any{
		"response": c.buildResponseObject("in_progress", []any{}, nil),
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
		"item_id":       c.reasoningItemID,
		"output_index":  c.outputIndex,
		"summary_index": 0,
		"delta":         thinking,
	}))

	// TODO(drifkin): consider adding
	// [`response.reasoning_text.delta`](https://platform.openai.com/docs/api-reference/responses-streaming/response/reasoning_text/delta),
	// but need to do additional research to understand how it's used and how
	// widely supported it is

	return events
}

func (c *ResponsesStreamConverter) finishReasoning() []ResponsesStreamEvent {
	if !c.reasoningStarted {
		return nil
	}

	itemID := c.reasoningItemID
	thinking := c.accumulatedThinking
	outputIndex := c.outputIndex
	item := map[string]any{
		"id":                itemID,
		"type":              "reasoning",
		"summary":           []map[string]any{{"type": "summary_text", "text": thinking}},
		"encrypted_content": thinking,
	}
	c.completedItems = append(c.completedItems, item)
	c.accumulatedThinking = ""
	c.reasoningItemID = ""
	c.reasoningStarted = false
	c.outputIndex++

	events := []ResponsesStreamEvent{
		c.newEvent("response.reasoning_summary_text.done", map[string]any{
			"item_id":       itemID,
			"output_index":  outputIndex,
			"summary_index": 0,
			"text":          thinking,
		}),
		c.newEvent("response.output_item.done", map[string]any{
			"output_index": outputIndex,
			"item":         item,
		}),
	}
	return events
}

func (c *ResponsesStreamConverter) processToolCalls(toolCalls []api.ToolCall) []ResponsesStreamEvent {
	return append(c.finishReasoning(), c.emitFunctionCallEvents(toolCalls)...)
}

// emitFunctionCallEvents emits function_call stream events for the given tool
// calls, stores them for the final output, and advances the output index.
func (c *ResponsesStreamConverter) emitFunctionCallEvents(toolCalls []api.ToolCall) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent
	converted := ToToolCalls(toolCalls)

	for i, tc := range converted {
		outputIndex := c.outputIndex + i
		fcItemID := fmt.Sprintf("fc_%d_%d", rand.Intn(999999), i)

		toolCallItem := map[string]any{
			"id":        fcItemID,
			"type":      "function_call",
			"status":    "completed",
			"call_id":   tc.ID,
			"name":      tc.Function.Name,
			"arguments": tc.Function.Arguments,
		}
		c.completedItems = append(c.completedItems, toolCallItem)

		events = append(events,
			c.newEvent("response.output_item.added", map[string]any{
				"output_index": outputIndex,
				"item": map[string]any{
					"id":        fcItemID,
					"type":      "function_call",
					"status":    "in_progress",
					"call_id":   tc.ID,
					"name":      tc.Function.Name,
					"arguments": "",
				},
			}),
			c.newEvent("response.function_call_arguments.delta", map[string]any{
				"item_id":      fcItemID,
				"output_index": outputIndex,
				"delta":        tc.Function.Arguments,
			}),
			c.newEvent("response.function_call_arguments.done", map[string]any{
				"item_id":      fcItemID,
				"output_index": outputIndex,
				"arguments":    tc.Function.Arguments,
			}),
			c.newEvent("response.output_item.done", map[string]any{
				"output_index": outputIndex,
				"item":         toolCallItem,
			}),
		)
	}
	c.outputIndex += len(converted)
	return events
}

// StartWebSearchCall emits the events that precede a server-side search and
// reserves its position in the response output.
func (c *ResponsesStreamConverter) StartWebSearchCall(call ResponsesWebSearchCall) (int, []ResponsesStreamEvent) {
	events := c.finishReasoning()
	outputIndex := c.outputIndex
	c.outputIndex++
	events = append(events,
		c.newEvent("response.output_item.added", map[string]any{
			"output_index": outputIndex,
			"item": map[string]any{
				"id": call.ID, "type": "web_search_call", "status": "in_progress",
				"action": map[string]any{"type": call.Action.Type, "query": call.Action.Query},
			},
		}),
		c.newEvent("response.web_search_call.in_progress", map[string]any{
			"item_id": call.ID, "output_index": outputIndex,
		}),
		c.newEvent("response.web_search_call.searching", map[string]any{
			"item_id": call.ID, "output_index": outputIndex,
		}),
	)
	return outputIndex, events
}

// FinishWebSearchCall emits the events that follow a successful server-side search.
func (c *ResponsesStreamConverter) FinishWebSearchCall(call ResponsesWebSearchCall, outputIndex int) []ResponsesStreamEvent {
	item := webSearchCallMap(call)
	c.completedItems = append(c.completedItems, item)
	return []ResponsesStreamEvent{
		c.newEvent("response.web_search_call.completed", map[string]any{
			"item_id": call.ID, "output_index": outputIndex,
		}),
		c.newEvent("response.output_item.done", map[string]any{
			"output_index": outputIndex,
			"item":         item,
		}),
	}
}

// ResponseFailed emits a terminal failure using this stream's sequence counter.
func (c *ResponsesStreamConverter) ResponseFailed(response map[string]any) ResponsesStreamEvent {
	return c.newEvent("response.failed", map[string]any{"response": response})
}

func webSearchCallMap(call ResponsesWebSearchCall) map[string]any {
	item := map[string]any{
		"id": call.ID, "type": "web_search_call", "status": call.Status,
	}
	if call.Action != nil {
		item["action"] = map[string]any{"type": call.Action.Type, "query": call.Action.Query}
	}
	return item
}

// FinishMessageItem closes the current text message item (if one was started)
// and reserves its place in the final output. This allows pre-search content
// to be emitted as a distinct message item before web_search_call events.
func (c *ResponsesStreamConverter) FinishMessageItem() []ResponsesStreamEvent {
	if !c.contentStarted {
		return nil
	}

	c.contentStarted = false
	text := c.accumulatedText
	c.accumulatedText = ""
	c.contentIndex = 0

	itemID := c.itemID
	item := map[string]any{
		"id":     itemID,
		"type":   "message",
		"status": "completed",
		"role":   "assistant",
		"content": []map[string]any{{
			"type":        "output_text",
			"text":        text,
			"annotations": []any{},
			"logprobs":    []any{},
		}},
	}
	c.completedItems = append(c.completedItems, item)
	outputIndex := c.outputIndex
	c.outputIndex++
	c.itemID = fmt.Sprintf("msg_%s_%d", strings.TrimPrefix(c.responseID, "resp_"), c.outputIndex)

	return []ResponsesStreamEvent{
		c.newEvent("response.output_text.done", map[string]any{
			"item_id":       itemID,
			"output_index":  outputIndex,
			"content_index": 0,
			"text":          text,
			"logprobs":      []any{},
		}),
		c.newEvent("response.content_part.done", map[string]any{
			"item_id":       itemID,
			"output_index":  outputIndex,
			"content_index": 0,
			"part": map[string]any{
				"type":        "output_text",
				"text":        text,
				"annotations": []any{},
				"logprobs":    []any{},
			},
		}),
		c.newEvent("response.output_item.done", map[string]any{
			"output_index": outputIndex,
			"item":         item,
		}),
	}
}

// EmitFunctionCallItems emits function_call events for client-provided tool
// calls that accompanied a web_search call (mixed responses). Unlike
// processToolCalls, this does not set toolCallsSent, so subsequent text
// content can still be processed.
func (c *ResponsesStreamConverter) EmitFunctionCallItems(toolCalls []api.ToolCall) []ResponsesStreamEvent {
	return c.emitFunctionCallEvents(toolCalls)
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
				"type":        "output_text",
				"text":        "",
				"annotations": []any{},
				"logprobs":    []any{},
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
		"logprobs":      []any{},
	}))

	return events
}

func (c *ResponsesStreamConverter) buildFinalOutput() []any {
	output := append([]any(nil), c.completedItems...)
	if c.contentStarted {
		// Add message item if we had text content
		output = append(output, map[string]any{
			"id":     c.itemID,
			"type":   "message",
			"status": "completed",
			"role":   "assistant",
			"content": []map[string]any{{
				"type":        "output_text",
				"text":        c.accumulatedText,
				"annotations": []any{},
				"logprobs":    []any{},
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
			"logprobs":      []any{},
		}))

		// response.content_part.done
		events = append(events, c.newEvent("response.content_part.done", map[string]any{
			"item_id":       c.itemID,
			"output_index":  c.outputIndex,
			"content_index": 0,
			"part": map[string]any{
				"type":        "output_text",
				"text":        c.accumulatedText,
				"annotations": []any{},
				"logprobs":    []any{},
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
					"type":        "output_text",
					"text":        c.accumulatedText,
					"annotations": []any{},
					"logprobs":    []any{},
				}},
			},
		}))
	}

	// response.completed
	usage := map[string]any{
		"input_tokens":  r.PromptEvalCount,
		"output_tokens": r.EvalCount,
		"total_tokens":  r.PromptEvalCount + r.EvalCount,
		"input_tokens_details": map[string]any{
			"cached_tokens": r.PromptEvalCachedCount,
		},
		"output_tokens_details": map[string]any{
			"reasoning_tokens": 0,
		},
	}
	response := c.buildResponseObject("completed", c.buildFinalOutput(), usage)
	response["completed_at"] = time.Now().Unix()
	events = append(events, c.newEvent("response.completed", map[string]any{
		"response": response,
	}))

	return events
}
