// openai package provides middleware for partial compatibility with the OpenAI REST API
package openai

import (
	"bytes"
	crypto_rand "crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/jmorganca/ollama/api"
)

type Error struct {
	Message string      `json:"message"`
	Type    string      `json:"type"`
	Param   interface{} `json:"param"`
	Code    *string     `json:"code"`
}

type ErrorResponse struct {
	Error Error `json:"error"`
}

type ToolFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolFunctionCall `json:"function"`
}

type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason *string `json:"finish_reason"`
}

type ChunkChoice struct {
	Index        int     `json:"index"`
	Delta        Message `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ResponseFormat struct {
	Type string `json:"type"`
}

type ChatCompletionRequest struct {
	Model            string               `json:"model"`
	Messages         []Message            `json:"messages"`
	Stream           bool                 `json:"stream"`
	MaxTokens        *int                 `json:"max_tokens"`
	Seed             *int                 `json:"seed"`
	Stop             any                  `json:"stop"`
	Temperature      *float64             `json:"temperature"`
	FrequencyPenalty *float64             `json:"frequency_penalty"`
	PresencePenalty  *float64             `json:"presence_penalty_penalty"`
	TopP             *float64             `json:"top_p"`
	ResponseFormat   *ResponseFormat      `json:"response_format"`
	Tools            []ChatCompletionTool `json:"tools"`
}

type ChatCompletionTool struct {
	Type     string         `json:"type"`
	Function FunctionObject `json:"function"`
}

type FunctionObject struct {
	Description string             `json:"description"`
	Name        string             `json:"name"`
	Parameters  FunctionParameters `json:"parameters"`
}

type FunctionParameters struct {
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties"`
	Required   []string               `json:"required"`
}

type ChatCompletion struct {
	Id                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	SystemFingerprint string   `json:"system_fingerprint"`
	Choices           []Choice `json:"choices"`
	Usage             Usage    `json:"usage,omitempty"`
}

type ChatCompletionChunk struct {
	Id                string        `json:"id"`
	Object            string        `json:"object"`
	Created           int64         `json:"created"`
	Model             string        `json:"model"`
	SystemFingerprint string        `json:"system_fingerprint"`
	Choices           []ChunkChoice `json:"choices"`
}

func NewError(code int, message string) ErrorResponse {
	var etype string
	switch code {
	case http.StatusBadRequest:
		etype = "invalid_request_error"
	case http.StatusNotFound:
		etype = "not_found_error"
	default:
		etype = "api_error"
	}

	return ErrorResponse{Error{Type: etype, Message: message}}
}

func toChatCompletion(id string, r api.ChatResponse) ChatCompletion {
	return toChatCompletionWithFunctionCalls(ChatCompletion{
		Id:                id,
		Object:            "chat.completion",
		Created:           r.CreatedAt.Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []Choice{{
			Index:   0,
			Message: Message{Role: r.Message.Role, Content: r.Message.Content},
			FinishReason: func(done bool) *string {
				if done {
					reason := "stop"
					return &reason
				}
				return nil
			}(r.Done),
		}},
		Usage: Usage{
			// TODO: ollama returns 0 for prompt eval if the prompt was cached, but openai returns the actual count
			PromptTokens:     r.PromptEvalCount,
			CompletionTokens: r.EvalCount,
			TotalTokens:      r.PromptEvalCount + r.EvalCount,
		},
	})
}

type Hermes2ProToolCall struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

func toChatCompletionWithFunctionCalls(openAi ChatCompletion) ChatCompletion {
	if !strings.Contains(openAi.Choices[len(openAi.Choices)-1].Message.Content, "<tool_call>") {
		return openAi
	}

	lastMessage := openAi.Choices[len(openAi.Choices)-1].Message

	// Hardcode Hermes2Pro function calling prompt format
	// See https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B-GGUF#prompt-format-for-function-calling

	toolCalls := make([]ToolCall, 0)
	for _, match := range strings.Split(lastMessage.Content, "<tool_call>") {
		if !strings.Contains(match, "</tool_call>") {
			continue
		}

		endIndex := strings.Index(match, "</tool_call>")
		if endIndex != -1 {
			match = match[:endIndex]
		}

		match = strings.TrimSpace(match)

		var hermes2ProToolCall Hermes2ProToolCall
		var err error
		for attempt := 0; attempt < 30; attempt++ {
			err = json.Unmarshal([]byte(match), &hermes2ProToolCall)

			if err == nil {
				break
			}
			match = match[:len(match)-1] // Remove the last character and try again
		}

		if err != nil {
			continue
		}

		argumentsJson, _ := json.Marshal(hermes2ProToolCall.Arguments)

		randomId, _ := randomCallId()

		toolCall := ToolCall{
			ID:       randomId,
			Type:     "function",
			Function: ToolFunctionCall{Name: hermes2ProToolCall.Name, Arguments: string(argumentsJson)},
		}
		toolCalls = append(toolCalls, toolCall)
	}

	// remove everything between <tool_call> and </tool_call> including the tags, in lastMessage.Content
	regex := regexp.MustCompile(`(?s)\s*<tool_call>.*?</tool_call>\s*`)
	lastMessage.Content = regex.ReplaceAllString(lastMessage.Content, "")

	finishReason := "tool_calls"
	openAi.Choices[len(openAi.Choices)-1].Message.Content = lastMessage.Content
	openAi.Choices[len(openAi.Choices)-1].Message.ToolCalls = toolCalls
	openAi.Choices[len(openAi.Choices)-1].FinishReason = &finishReason

	return openAi
}

func randomCallId() (randomId string, err error) {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, 24)

	if _, err := crypto_rand.Read(b); err != nil {
		return "", err
	}

	for i := range b {
		b[i] = charset[b[i]%byte(len(charset))]
	}

	randomId = "call_" + string(b)

	return randomId, nil
}

func toChunk(id string, r api.ChatResponse) ChatCompletionChunk {
	return ChatCompletionChunk{
		Id:                id,
		Object:            "chat.completion.chunk",
		Created:           time.Now().Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []ChunkChoice{
			{
				Index: 0,
				Delta: Message{Role: "assistant", Content: r.Message.Content},
				FinishReason: func(done bool) *string {
					if done {
						reason := "stop"
						return &reason
					}
					return nil
				}(r.Done),
			},
		},
	}
}

func fromRequest(r ChatCompletionRequest) api.ChatRequest {
	r = applyFunctionCalls(r)

	var messages []api.Message
	for _, msg := range r.Messages {
		messages = append(messages, api.Message{Role: msg.Role, Content: msg.Content})
	}

	options := make(map[string]interface{})

	switch stop := r.Stop.(type) {
	case string:
		options["stop"] = []string{stop}
	case []interface{}:
		var stops []string
		for _, s := range stop {
			if str, ok := s.(string); ok {
				stops = append(stops, str)
			}
		}
		options["stop"] = stops
	}

	if r.MaxTokens != nil {
		options["num_predict"] = *r.MaxTokens
	}

	if r.Temperature != nil {
		options["temperature"] = *r.Temperature * 2.0
	} else {
		options["temperature"] = 1.0
	}

	if r.Seed != nil {
		options["seed"] = *r.Seed

		// temperature=0 is required for reproducible outputs
		options["temperature"] = 0.0
	}

	if r.FrequencyPenalty != nil {
		options["frequency_penalty"] = *r.FrequencyPenalty * 2.0
	}

	if r.PresencePenalty != nil {
		options["presence_penalty"] = *r.PresencePenalty * 2.0
	}

	if r.TopP != nil {
		options["top_p"] = *r.TopP
	} else {
		options["top_p"] = 1.0
	}

	var format string
	if r.ResponseFormat != nil && r.ResponseFormat.Type == "json_object" {
		format = "json"
	}

	return api.ChatRequest{
		Model:    r.Model,
		Messages: messages,
		Format:   format,
		Options:  options,
		Stream:   &r.Stream,
	}
}

func applyFunctionCalls(openAi ChatCompletionRequest) ChatCompletionRequest {
	if len(openAi.Tools) == 0 {
		return openAi
	}

	if openAi.Messages[0].Role != "system" {
		openAi.Messages = append([]Message{{Role: "system", Content: ""}}, openAi.Messages...)
	}

	// Hardcode Hermes2Pro function calling prompt format
	// See https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B-GGUF#prompt-format-for-function-calling

	toolsJSON, _ := json.Marshal(openAi.Tools)
	var sb strings.Builder
	sb.WriteString("You are a function calling AI model.\n")
	sb.WriteString("You are provided with function signatures within <tools></tools> XML tags.\n")
	sb.WriteString("You may call one or more functions to assist with the user query.\n")
	sb.WriteString("Don't make assumptions about what values to plug into functions.\n")
	sb.WriteString("Here are the available tools:\n")
	sb.WriteString("<tools>\n")
	sb.WriteString(string(toolsJSON))
	sb.WriteString("\n</tools>\n")
	sb.WriteString("Use the following json schema for each tool call you will make:\n")
	sb.WriteString(`{"type": "object", "properties": {"name": {"title": "Name", "type": "string"}, "arguments": {"title": "Arguments", "type": "object"}}, "required": ["arguments", "name"], "title": "FunctionCall"}`)
	sb.WriteString("\nFor each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n")
	sb.WriteString("<tool_call>\n")
	sb.WriteString(`{"name": <function-name>, "arguments": <args-dict>}`)
	sb.WriteString("\n</tool_call>.\n")
	sb.WriteString("Only reply with <tool_call>...</tool_call>, no other content.\n")
	sb.WriteString(openAi.Messages[0].Content)
	openAi.Messages[0].Content = sb.String()

	return openAi
}

type writer struct {
	stream bool
	id     string
	gin.ResponseWriter
}

func (w *writer) writeError(code int, data []byte) (int, error) {
	var serr api.StatusError
	err := json.Unmarshal(data, &serr)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(NewError(http.StatusInternalServerError, serr.Error()))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *writer) writeResponse(data []byte) (int, error) {
	var chatResponse api.ChatResponse
	err := json.Unmarshal(data, &chatResponse)
	if err != nil {
		return 0, err
	}

	// chat chunk
	if w.stream {
		d, err := json.Marshal(toChunk(w.id, chatResponse))
		if err != nil {
			return 0, err

		}

		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
		_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
		if err != nil {
			return 0, err
		}

		if chatResponse.Done {
			_, err = w.ResponseWriter.Write([]byte("data: [DONE]\n\n"))
			if err != nil {
				return 0, err
			}
		}

		return len(data), nil
	}

	// chat completion
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(toChatCompletion(w.id, chatResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *writer) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(code, data)
	}

	return w.writeResponse(data)
}

func Middleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req ChatCompletionRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if len(req.Messages) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, NewError(http.StatusBadRequest, "[] is too short - 'messages'"))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(fromRequest(req)); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &writer{
			ResponseWriter: c.Writer,
			stream:         req.Stream,
			id:             fmt.Sprintf("chatcmpl-%d", rand.Intn(999)),
		}

		c.Writer = w

		c.Next()
	}
}
