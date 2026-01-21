package middleware

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

type BaseWriter struct {
	gin.ResponseWriter
}

type ChatWriter struct {
	stream        bool
	streamOptions *openai.StreamOptions
	id            string
	toolCallSent  bool
	BaseWriter
}

type CompleteWriter struct {
	stream        bool
	streamOptions *openai.StreamOptions
	id            string
	BaseWriter
}

type ListWriter struct {
	BaseWriter
}

type RetrieveWriter struct {
	BaseWriter
	model string
}

type EmbedWriter struct {
	BaseWriter
	model          string
	encodingFormat string
}

func (w *BaseWriter) writeError(data []byte) (int, error) {
	var serr api.StatusError
	err := json.Unmarshal(data, &serr)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.NewError(http.StatusInternalServerError, serr.Error()))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *ChatWriter) writeResponse(data []byte) (int, error) {
	var chatResponse api.ChatResponse
	err := json.Unmarshal(data, &chatResponse)
	if err != nil {
		return 0, err
	}

	// chat chunk
	if w.stream {
		c := openai.ToChunk(w.id, chatResponse, w.toolCallSent)
		d, err := json.Marshal(c)
		if err != nil {
			return 0, err
		}
		if !w.toolCallSent && len(c.Choices) > 0 && len(c.Choices[0].Delta.ToolCalls) > 0 {
			w.toolCallSent = true
		}

		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
		_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
		if err != nil {
			return 0, err
		}

		if chatResponse.Done {
			if w.streamOptions != nil && w.streamOptions.IncludeUsage {
				u := openai.ToUsage(chatResponse)
				c.Usage = &u
				c.Choices = []openai.ChunkChoice{}
				d, err := json.Marshal(c)
				if err != nil {
					return 0, err
				}
				_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
				if err != nil {
					return 0, err
				}
			}
			_, err = w.ResponseWriter.Write([]byte("data: [DONE]\n\n"))
			if err != nil {
				return 0, err
			}
		}

		return len(data), nil
	}

	// chat completion
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToChatCompletion(w.id, chatResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *ChatWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func (w *CompleteWriter) writeResponse(data []byte) (int, error) {
	var generateResponse api.GenerateResponse
	err := json.Unmarshal(data, &generateResponse)
	if err != nil {
		return 0, err
	}

	// completion chunk
	if w.stream {
		c := openai.ToCompleteChunk(w.id, generateResponse)
		if w.streamOptions != nil && w.streamOptions.IncludeUsage {
			c.Usage = &openai.Usage{}
		}
		d, err := json.Marshal(c)
		if err != nil {
			return 0, err
		}

		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
		_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
		if err != nil {
			return 0, err
		}

		if generateResponse.Done {
			if w.streamOptions != nil && w.streamOptions.IncludeUsage {
				u := openai.ToUsageGenerate(generateResponse)
				c.Usage = &u
				c.Choices = []openai.CompleteChunkChoice{}
				d, err := json.Marshal(c)
				if err != nil {
					return 0, err
				}
				_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
				if err != nil {
					return 0, err
				}
			}
			_, err = w.ResponseWriter.Write([]byte("data: [DONE]\n\n"))
			if err != nil {
				return 0, err
			}
		}

		return len(data), nil
	}

	// completion
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToCompletion(w.id, generateResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *CompleteWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func (w *ListWriter) writeResponse(data []byte) (int, error) {
	var listResponse api.ListResponse
	err := json.Unmarshal(data, &listResponse)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToListCompletion(listResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *ListWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func (w *RetrieveWriter) writeResponse(data []byte) (int, error) {
	var showResponse api.ShowResponse
	err := json.Unmarshal(data, &showResponse)
	if err != nil {
		return 0, err
	}

	// retrieve completion
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToModel(showResponse, w.model))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *RetrieveWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func (w *EmbedWriter) writeResponse(data []byte) (int, error) {
	var embedResponse api.EmbedResponse
	err := json.Unmarshal(data, &embedResponse)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(openai.ToEmbeddingList(w.model, embedResponse, w.encodingFormat))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *EmbedWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func ListMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		w := &ListWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
		}

		c.Writer = w

		c.Next()
	}
}

func RetrieveMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(api.ShowRequest{Name: c.Param("model")}); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &RetrieveWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			model:      c.Param("model"),
		}

		c.Writer = w

		c.Next()
	}
}

func CompletionsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.CompletionRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		var b bytes.Buffer
		genReq, err := openai.FromCompleteRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if err := json.NewEncoder(&b).Encode(genReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &CompleteWriter{
			BaseWriter:    BaseWriter{ResponseWriter: c.Writer},
			stream:        req.Stream,
			id:            fmt.Sprintf("cmpl-%d", rand.Intn(999)),
			streamOptions: req.StreamOptions,
		}

		c.Writer = w
		c.Next()
	}
}

func EmbeddingsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.EmbedRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		// Validate encoding_format parameter
		if req.EncodingFormat != "" {
			if !strings.EqualFold(req.EncodingFormat, "float") && !strings.EqualFold(req.EncodingFormat, "base64") {
				c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, fmt.Sprintf("Invalid value for 'encoding_format' = %s. Supported values: ['float', 'base64'].", req.EncodingFormat)))
				return
			}
		}

		if req.Input == "" {
			req.Input = []string{""}
		}

		if req.Input == nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "invalid input"))
			return
		}

		if v, ok := req.Input.([]any); ok && len(v) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "invalid input"))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(api.EmbedRequest{Model: req.Model, Input: req.Input, Dimensions: req.Dimensions}); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &EmbedWriter{
			BaseWriter:     BaseWriter{ResponseWriter: c.Writer},
			model:          req.Model,
			encodingFormat: req.EncodingFormat,
		}

		c.Writer = w

		c.Next()
	}
}

func ChatMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.ChatCompletionRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if len(req.Messages) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "[] is too short - 'messages'"))
			return
		}

		var b bytes.Buffer

		chatReq, err := openai.FromChatRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &ChatWriter{
			BaseWriter:    BaseWriter{ResponseWriter: c.Writer},
			stream:        req.Stream,
			id:            fmt.Sprintf("chatcmpl-%d", rand.Intn(999)),
			streamOptions: req.StreamOptions,
		}

		c.Writer = w

		c.Next()
	}
}

type ResponsesWriter struct {
	BaseWriter
	converter  *openai.ResponsesStreamConverter
	model      string
	stream     bool
	responseID string
	itemID     string
	request    openai.ResponsesRequest
}

func (w *ResponsesWriter) writeEvent(eventType string, data any) error {
	d, err := json.Marshal(data)
	if err != nil {
		return err
	}
	_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, d)))
	if err != nil {
		return err
	}
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}

func (w *ResponsesWriter) writeResponse(data []byte) (int, error) {
	var chatResponse api.ChatResponse
	if err := json.Unmarshal(data, &chatResponse); err != nil {
		return 0, err
	}

	if w.stream {
		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")

		events := w.converter.Process(chatResponse)
		for _, event := range events {
			if err := w.writeEvent(event.Event, event.Data); err != nil {
				return 0, err
			}
		}
		return len(data), nil
	}

	// Non-streaming response
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	response := openai.ToResponse(w.model, w.responseID, w.itemID, chatResponse, w.request)
	completedAt := time.Now().Unix()
	response.CompletedAt = &completedAt
	return len(data), json.NewEncoder(w.ResponseWriter).Encode(response)
}

func (w *ResponsesWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}
	return w.writeResponse(data)
}

func ResponsesMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.ResponsesRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		chatReq, err := openai.FromResponsesRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		// Check if client requested streaming (defaults to false)
		streamRequested := req.Stream != nil && *req.Stream

		// Pass streaming preference to the underlying chat request
		chatReq.Stream = &streamRequested

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		responseID := fmt.Sprintf("resp_%d", rand.Intn(999999))
		itemID := fmt.Sprintf("msg_%d", rand.Intn(999999))

		w := &ResponsesWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			converter:  openai.NewResponsesStreamConverter(responseID, itemID, req.Model, req),
			model:      req.Model,
			stream:     streamRequested,
			responseID: responseID,
			itemID:     itemID,
			request:    req,
		}

		// Set headers based on streaming mode
		if streamRequested {
			c.Writer.Header().Set("Content-Type", "text/event-stream")
			c.Writer.Header().Set("Cache-Control", "no-cache")
			c.Writer.Header().Set("Connection", "keep-alive")
		}

		c.Writer = w
		c.Next()
	}
}

type ImageWriter struct {
	BaseWriter
}

func (w *ImageWriter) writeResponse(data []byte) (int, error) {
	var generateResponse api.GenerateResponse
	if err := json.Unmarshal(data, &generateResponse); err != nil {
		return 0, err
	}

	// Only write response when done with image
	if generateResponse.Done && generateResponse.Image != "" {
		w.ResponseWriter.Header().Set("Content-Type", "application/json")
		return len(data), json.NewEncoder(w.ResponseWriter).Encode(openai.ToImageGenerationResponse(generateResponse))
	}

	return len(data), nil
}

func (w *ImageWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

func ImageGenerationsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req openai.ImageGenerationRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if req.Prompt == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "prompt is required"))
			return
		}

		if req.Model == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "model is required"))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(openai.FromImageGenerationRequest(req)); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &ImageWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
		}

		c.Writer = w
		c.Next()
	}
}
