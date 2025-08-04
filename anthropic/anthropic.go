// anthropic package provides unified compatibility with both Anthropic and Claude REST APIs
// This serves as a drop-in replacement for both Anthropic API clients and Claude Code
package anthropic

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

const (
	anthropicVersion = "2023-06-01"
	maxTokensDefault = 4096
)

type Error struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type ErrorResponse struct {
	Error Error `json:"error"`
}

type Message struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
	
	// For tool use
	ID    string `json:"id,omitempty"`
	Name  string `json:"name,omitempty"`
	Input map[string]interface{} `json:"input,omitempty"`
}

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type MessageRequest struct {
	Model       string      `json:"model"`
	MaxTokens   int         `json:"max_tokens"`
	Messages    []Message   `json:"messages"`
	System      string      `json:"system,omitempty"`
	Metadata    interface{} `json:"metadata,omitempty"`
	StopSequences []string  `json:"stop_sequences,omitempty"`
	Stream      bool        `json:"stream,omitempty"`
	Temperature *float64    `json:"temperature,omitempty"`
	TopP        *float64    `json:"top_p,omitempty"`
	TopK        *int        `json:"top_k,omitempty"`
	Tools       []ToolDefinition  `json:"tools,omitempty"`
}

type MessageResponse struct {
	ID           string        `json:"id"`
	Type         string        `json:"type"`
	Role         string        `json:"role"`
	Content      []ContentBlock `json:"content"`
	Model        string        `json:"model"`
	StopReason   *string       `json:"stop_reason,omitempty"`
	StopSequence *string       `json:"stop_sequence,omitempty"`
	Usage        Usage         `json:"usage"`
}

type MessageStreamResponse struct {
	Type         string        `json:"type"`
	Index        *int          `json:"index,omitempty"`
	Delta        *ContentBlock `json:"delta,omitempty"`
	Message      *MessageResponse `json:"message,omitempty"`
	Usage        *Usage        `json:"usage,omitempty"`
}

type ToolDefinition struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	InputSchema interface{} `json:"input_schema"`
}

func NewError(message string) ErrorResponse {
	return ErrorResponse{
		Error: Error{
			Type:    "api_error",
			Message: message,
		},
	}
}

func toUsage(r api.ChatResponse) Usage {
	return Usage{
		InputTokens:  r.PromptEvalCount,
		OutputTokens: r.EvalCount,
	}
}

func fromAnthropicMessages(messages []Message) ([]api.Message, string, error) {
	var apiMessages []api.Message
	var systemPrompt string
	
	for _, msg := range messages {
		switch msg.Role {
		case "user", "assistant":
			content := ""
			images := []api.ImageData{}
			
			switch c := msg.Content.(type) {
			case string:
				content = c
			case []interface{}:
				for _, item := range c {
					block, ok := item.(map[string]interface{})
					if !ok {
						return nil, "", errors.New("invalid content block format")
					}
					
					typeStr, ok := block["type"].(string)
					if !ok {
						return nil, "", errors.New("invalid content block type")
					}
					
					switch typeStr {
					case "text":
						if text, ok := block["text"].(string); ok {
							content += text
						}
					case "image":
						// Handle image content
						source, ok := block["source"].(map[string]interface{})
						if !ok {
							return nil, "", errors.New("invalid image source format")
						}
						
						mediaType, ok := source["media_type"].(string)
						if !ok {
							return nil, "", errors.New("invalid media type")
						}
						
						dataStr, ok := source["data"].(string)
						if !ok {
							return nil, "", errors.New("invalid image data")
						}
						
						// Validate media type
						validTypes := []string{"image/jpeg", "image/png", "image/gif", "image/webp"}
						valid := false
						for _, t := range validTypes {
							if mediaType == t {
								valid = true
								break
							}
						}
						if !valid {
							return nil, "", errors.New("unsupported image type")
						}
						
						imgData, err := base64.StdEncoding.DecodeString(dataStr)
						if err != nil {
							return nil, "", fmt.Errorf("invalid base64 image data: %w", err)
						}
						
						images = append(images, imgData)
					default:
						return nil, "", fmt.Errorf("unsupported content block type: %s", typeStr)
					}
				}
			default:
				return nil, "", fmt.Errorf("invalid content type: %T", msg.Content)
			}
			
			apiMessages = append(apiMessages, api.Message{
				Role:    msg.Role,
				Content: content,
				Images:  images,
			})
		case "system":
			// Handle system messages as part of the prompt
			if systemContent, ok := msg.Content.(string); ok {
				systemPrompt = systemContent
			} else {
				return nil, "", errors.New("system message must be a string")
			}
		default:
			return nil, "", fmt.Errorf("unsupported role: %s", msg.Role)
		}
	}
	
	return apiMessages, systemPrompt, nil
}

func fromMessageRequest(r MessageRequest) (*api.ChatRequest, error) {
	messages, systemPrompt, err := fromAnthropicMessages(r.Messages)
	if err != nil {
		return nil, err
	}
	
	// Prepend system prompt to messages if provided
	if systemPrompt != "" || r.System != "" {
		systemContent := systemPrompt
		if r.System != "" {
			if systemContent != "" {
				systemContent = r.System + "\n\n" + systemContent
			} else {
				systemContent = r.System
			}
		}
		
		// Insert system message at the beginning
		messages = append([]api.Message{{
			Role:    "system",
			Content: systemContent,
		}}, messages...)
	}
	
	options := make(map[string]any)
	
	if r.MaxTokens > 0 {
		options["num_predict"] = r.MaxTokens
	} else {
		options["num_predict"] = maxTokensDefault
	}
	
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
	
	// Convert Anthropic tools to Ollama format
	var tools []api.Tool
	for _, tool := range r.Tools {
		convertedTool := convertAnthropicTool(tool)
		tools = append(tools, convertedTool)
	}
	
	return &api.ChatRequest{
		Model:    r.Model,
		Messages: messages,
		Options:  options,
		Stream:   &r.Stream,
		Tools:    tools,
	}, nil
}

func convertAnthropicTool(tool ToolDefinition) api.Tool {
	// Convert Anthropic tool format to Ollama format
	schema, ok := tool.InputSchema.(map[string]interface{})
	if !ok {
		return api.Tool{}
	}
	
	properties := make(map[string]struct {
		Type        api.PropertyType `json:"type"`
		Items       any              `json:"items,omitempty"`
		Description string           `json:"description"`
		Enum        []any            `json:"enum,omitempty"`
	})
	
	if props, ok := schema["properties"].(map[string]interface{}); ok {
		for name, prop := range props {
			if propMap, ok := prop.(map[string]interface{}); ok {
				propType := ""
				if t, ok := propMap["type"].(string); ok {
					propType = t
				}
				
				properties[name] = struct {
					Type        api.PropertyType `json:"type"`
					Items       any              `json:"items,omitempty"`
					Description string           `json:"description"`
					Enum        []any            `json:"enum,omitempty"`
				}{
					Type:        api.PropertyType{propType},
					Description: name, // Use name as description for simplicity
				}
			}
		}
	}
	
	var required []string
	if req, ok := schema["required"].([]interface{}); ok {
		for _, r := range req {
			if s, ok := r.(string); ok {
				required = append(required, s)
			}
		}
	}
	
	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters: struct {
				Type       string   `json:"type"`
				Defs       any      `json:"$defs,omitempty"`
				Items      any      `json:"items,omitempty"`
				Required   []string `json:"required"`
				Properties map[string]struct {
					Type        api.PropertyType `json:"type"`
					Items       any              `json:"items,omitempty"`
					Description string           `json:"description"`
					Enum        []any            `json:"enum,omitempty"`
				} `json:"properties"`
			}{
				Type:       "object",
				Required:   required,
				Properties: properties,
			},
		},
	}
}

func toContentBlocks(content string, toolCalls []api.ToolCall) []ContentBlock {
	var blocks []ContentBlock
	
	if content != "" {
		blocks = append(blocks, ContentBlock{
			Type: "text",
			Text: content,
		})
	}
	
	for _, tc := range toolCalls {
		args, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			slog.Error("could not marshal tool arguments", "error", err)
			continue
		}
		
		var inputArgs map[string]interface{}
		if err := json.Unmarshal(args, &inputArgs); err != nil {
			slog.Error("could not unmarshal tool arguments", "error", err)
			continue
		}
		
		blocks = append(blocks, ContentBlock{
			Type:  "tool_use",
			ID:    fmt.Sprintf("toolu_%s", generateID(8)),
			Name:  tc.Function.Name,
			Input: inputArgs,
		})
	}
	
	return blocks
}

func generateID(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[rand.Intn(len(charset))]
	}
	return string(b)
}

func toMessageResponse(id string, r api.ChatResponse) MessageResponse {
	contentBlocks := toContentBlocks(r.Message.Content, r.Message.ToolCalls)
	
	var stopReason *string
	if r.DoneReason != "" {
		// Map Ollama stop reasons to Anthropic equivalents
		reason := r.DoneReason
		switch r.DoneReason {
		case "stop":
			reason = "end_turn"
		case "length":
			reason = "max_tokens"
		case "tool_calls":
			reason = "tool_use"
		}
		stopReason = &reason
	}
	
	return MessageResponse{
		ID:         fmt.Sprintf("msg_%s", id),
		Type:       "message",
		Role:       r.Message.Role,
		Content:    contentBlocks,
		Model:      r.Model,
		StopReason: stopReason,
		Usage:      toUsage(r),
	}
}

func toMessageStreamResponse(id string, r api.ChatResponse, isFinal bool) []MessageStreamResponse {
	var responses []MessageStreamResponse
	
	contentBlocks := toContentBlocks(r.Message.Content, r.Message.ToolCalls)
	
	for _, block := range contentBlocks {
		if block.Type == "text" && block.Text != "" {
			responses = append(responses, MessageStreamResponse{
				Type:  "content_block_delta",
				Index: intPtr(0),
				Delta: &ContentBlock{
					Type: "text",
					Text: block.Text,
				},
			})
		} else if block.Type == "tool_use" {
			responses = append(responses, MessageStreamResponse{
				Type:  "content_block_delta",
				Index: intPtr(0),
				Delta: &ContentBlock{
					Type:  "tool_use",
					ID:    block.ID,
					Name:  block.Name,
					Input: block.Input,
				},
			})
		}
	}
	
	if isFinal && r.Done {
		// Add final usage information
		responses = append(responses, MessageStreamResponse{
			Type:    "message_stop",
			Usage:   &Usage{
				InputTokens:  r.PromptEvalCount,
				OutputTokens: r.EvalCount,
			},
		})
	}
	
	return responses
}

func intPtr(i int) *int {
	return &i
}

// BaseWriter provides common functionality for response writers
type BaseWriter struct {
	gin.ResponseWriter
}

func (w *BaseWriter) writeError(data []byte) (int, error) {
	var serr api.StatusError
	err := json.Unmarshal(data, &serr)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(NewError(serr.Error()))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

// MessagesWriter handles Anthropic messages API responses
type MessagesWriter struct {
	stream bool
	id     string
	BaseWriter
}

func (w *MessagesWriter) writeResponse(data []byte) (int, error) {
	var chatResponse api.ChatResponse
	err := json.Unmarshal(data, &chatResponse)
	if err != nil {
		return 0, err
	}

	if w.stream {
		// Handle streaming response
		responses := toMessageStreamResponse(w.id, chatResponse, chatResponse.Done)
		
		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
		w.ResponseWriter.Header().Set("X-API-Version", anthropicVersion)
		
		for _, resp := range responses {
			d, err := json.Marshal(resp)
			if err != nil {
				return 0, err
			}
			
			_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
			if err != nil {
				return 0, err
			}
		}
		
		if chatResponse.Done {
			_, err = w.ResponseWriter.Write([]byte("data: [DONE]\n\n"))
			if err != nil {
				return 0, err
			}
		}
		
		return len(data), nil
	}

	// Handle non-streaming response
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	w.ResponseWriter.Header().Set("X-API-Version", anthropicVersion)
	
	resp := toMessageResponse(w.id, chatResponse)
	err = json.NewEncoder(w.ResponseWriter).Encode(resp)
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *MessagesWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

// MessagesMiddleware creates unified middleware for both Anthropic and Claude API compatibility
func MessagesMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req MessageRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, NewError(err.Error()))
			return
		}

		if len(req.Messages) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, NewError("messages field is required"))
			return
		}

		if req.MaxTokens <= 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, NewError("max_tokens is required and must be greater than 0"))
			return
		}

		var b bytes.Buffer
		chatReq, err := fromMessageRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, NewError(err.Error()))
			return
		}

		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, NewError(err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &MessagesWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			stream:     req.Stream,
			id:         generateID(16),
		}

		c.Writer = w
		c.Next()
	}
}