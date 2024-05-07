// openai package provides middleware for partial compatibility with the OpenAI REST API
package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
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

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
	Model            string          `json:"model"`
	Messages         []Message       `json:"messages"`
	Stream           bool            `json:"stream"`
	MaxTokens        *int            `json:"max_tokens"`
	Seed             *int            `json:"seed"`
	Stop             any             `json:"stop"`
	Temperature      *float64        `json:"temperature"`
	FrequencyPenalty *float64        `json:"frequency_penalty"`
	PresencePenalty  *float64        `json:"presence_penalty_penalty"`
	TopP             *float64        `json:"top_p"`
	ResponseFormat   *ResponseFormat `json:"response_format"`
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
	return ChatCompletion{
		Id:                id,
		Object:            "chat.completion",
		Created:           r.CreatedAt.Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []Choice{{
			Index:        0,
			Message:      Message{Role: r.Message.Role, Content: r.Message.Content},
			FinishReason: &r.FinishReason,
		}},
		Usage: Usage{
			// TODO: ollama returns 0 for prompt eval if the prompt was cached, but openai returns the actual count
			PromptTokens:     r.PromptEvalCount,
			CompletionTokens: r.EvalCount,
			TotalTokens:      r.PromptEvalCount + r.EvalCount,
		},
	}
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
				Index:        0,
				Delta:        Message{Role: "assistant", Content: r.Message.Content},
				FinishReason: &r.FinishReason,
			},
		},
	}
}

func fromRequest(r ChatCompletionRequest) api.ChatRequest {
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
