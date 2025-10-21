package middleware

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"

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
