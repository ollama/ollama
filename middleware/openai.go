package middleware

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

// maxDecompressedBodySize limits the size of a decompressed request body
const maxDecompressedBodySize = 20 << 20

type BaseWriter struct {
	gin.ResponseWriter
}

type ChatWriter struct {
	stream         bool
	streamOptions  *openai.StreamOptions
	id             string
	toolCallSent   bool
	firstChunkSent bool
	// createdAt pins the shared timestamp for every chunk in the stream,
	// captured from the first response.
	createdAt time.Time
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
	if err := json.Unmarshal(data, &serr); err != nil {
		// If the error response isn't valid JSON, use the raw bytes as the
		// error message rather than surfacing a confusing JSON parse error.
		serr.ErrorMessage = string(data)
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w.ResponseWriter).Encode(openai.NewError(w.ResponseWriter.Status(), serr.Error())); err != nil {
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
		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")

		// OpenAI stamps one created value on every chunk in a stream; pin the
		// timestamp from the first response (the server stamps each response).
		if chatResponse.CreatedAt.IsZero() {
			chatResponse.CreatedAt = time.Now().UTC()
		}
		if w.createdAt.IsZero() {
			w.createdAt = chatResponse.CreatedAt
		}
		chatResponse.CreatedAt = w.createdAt

		// A Done response with an empty message is the metrics-only trailer.
		// OpenAI goes straight from the last content chunk to the finish chunk,
		// so don't emit an empty content chunk for it. If this is the stream's
		// first response, fall through so a wholly empty completion still opens
		// with a role chunk.
		isEmptyTrailer := chatResponse.Done && w.firstChunkSent &&
			chatResponse.Message.Content == "" &&
			chatResponse.Message.Thinking == "" &&
			len(chatResponse.Message.ToolCalls) == 0 &&
			len(chatResponse.Logprobs) == 0

		if !isEmptyTrailer {
			includeRole := !w.firstChunkSent
			chunks := openai.ToStreamChunks(w.id, chatResponse, includeRole)
			for _, c := range chunks {
				d, err := json.Marshal(c)
				if err != nil {
					return 0, err
				}
				if !w.toolCallSent && len(c.Choices) > 0 && len(c.Choices[0].Delta.ToolCalls) > 0 {
					w.toolCallSent = true
				}
				_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
				if err != nil {
					return 0, err
				}
			}

			// ToStreamChunks always emits at least one chunk.
			w.firstChunkSent = true
		}

		if chatResponse.Done {
			finishChunk := openai.FinishChunk(w.id, chatResponse, w.toolCallSent)
			d, err := json.Marshal(finishChunk)
			if err != nil {
				return 0, err
			}
			_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
			if err != nil {
				return 0, err
			}

			if w.streamOptions != nil && w.streamOptions.IncludeUsage {
				u := openai.ToUsage(chatResponse)
				finishChunk.Usage = &u
				finishChunk.Choices = []openai.ChunkChoice{}
				d, err := json.Marshal(finishChunk)
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

// WebSearchResponsesWriter runs the built-in Responses web_search tool on the
// server. The model sees it as an ordinary function; callers only see the
// native web_search_call items which describe the searches we actually ran.
type WebSearchResponsesWriter struct {
	BaseWriter
	inner *ResponsesWriter
	req   openai.ResponsesRequest
	chat  *api.ChatRequest

	// The functions are injectable so the protocol lifecycle can be tested
	// without a running server or cloud credentials.
	search         func(context.Context, string) (*api.WebSearchResponse, error)
	followUpChat   func(context.Context, []api.Message, api.Tools) (api.ChatResponse, error)
	followUpStream func(context.Context, []api.Message, api.Tools, func(api.ChatResponse) error) error
	newContext     func() (context.Context, context.CancelFunc)

	// Keep the initial model response for the follow-up context while streaming
	// ordinary output immediately. Once web_search appears, its private function
	// call and terminal chunk are intercepted and replaced by native events.
	buffered              []api.ChatResponse
	webSearchPending      bool
	streamedInitialOutput bool
	status                int
	done                  bool

	// Accumulated across loop iterations by runLoop, consumed by
	// writeWebSearchResponse / writeWebSearchStream.
	preSearchThinking   string         // reasoning the model emitted before calling web_search
	preSearchContent    string         // text the model emitted before calling web_search
	otherToolCalls      []api.ToolCall // non-web_search tool calls from mixed responses
	finalOutputStreamed bool
}

func (w *WebSearchResponsesWriter) WriteHeader(code int) {
	w.status = code
}

func (w *WebSearchResponsesWriter) WriteHeaderNow() {
	if w.status != 0 {
		w.ResponseWriter.WriteHeader(w.status)
	}
	w.ResponseWriter.WriteHeaderNow()
}

func (w *WebSearchResponsesWriter) Status() int {
	if w.status != 0 {
		return w.status
	}
	return w.ResponseWriter.Status()
}

func (w *WebSearchResponsesWriter) Write(data []byte) (int, error) {
	if w.done {
		return len(data), nil
	}
	if w.Status() != http.StatusOK {
		return len(data), w.writeWebSearchError(decodeWebSearchResponseError(w.Status(), data), api.Metrics{})
	}

	var response api.ChatResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return 0, err
	}
	if w.inner.stream {
		w.buffered = append(w.buffered, response)
		_, hasWebSearch, _ := findWebSearchToolCall(response.Message.ToolCalls)
		if hasWebSearch {
			w.webSearchPending = true
		}
		if !w.webSearchPending && len(response.Message.ToolCalls) == 0 {
			if _, err := w.inner.writeResponse(data); err != nil {
				return 0, err
			}
			if response.Message.Content != "" || response.Message.Thinking != "" {
				w.streamedInitialOutput = true
			}
			if response.Done {
				w.buffered = nil
				w.done = true
			}
			return len(data), nil
		}

		// Tool-bearing chunks never pass through Process: its normal tool path
		// latches text off for the rest of the stream. Stream ordinary output,
		// then emit client tools through the latch-free path.
		if response.Message.Content != "" || response.Message.Thinking != "" {
			streamed := response
			streamed.Message.ToolCalls = nil
			streamed.Done = false
			streamedData, err := json.Marshal(streamed)
			if err != nil {
				return 0, err
			}
			if _, err := w.inner.writeResponse(streamedData); err != nil {
				return 0, err
			}
			w.streamedInitialOutput = true
		}
		var otherToolCalls []api.ToolCall
		for _, tc := range response.Message.ToolCalls {
			if tc.Function.Name != "web_search" {
				otherToolCalls = append(otherToolCalls, tc)
			}
		}
		if len(otherToolCalls) > 0 {
			for _, event := range w.inner.converter.Process(api.ChatResponse{}) {
				if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
					return 0, err
				}
			}
			for _, event := range w.inner.converter.FinishMessageItem() {
				if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
					return 0, err
				}
			}
			for _, event := range w.inner.converter.EmitFunctionCallItems(otherToolCalls) {
				if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
					return 0, err
				}
			}
			w.streamedInitialOutput = true
		}
		if response.Done {
			if w.webSearchPending {
				return len(data), w.finishStream()
			}
			response.Message = api.Message{}
			terminal, err := json.Marshal(response)
			if err != nil {
				return 0, err
			}
			if _, err := w.inner.writeResponse(terminal); err != nil {
				return 0, err
			}
			w.buffered = nil
			w.done = true
		}
		return len(data), nil
	}

	call, found, mixed := findWebSearchToolCall(response.Message.ToolCalls)
	if !found {
		return w.inner.writeResponse(data)
	}
	if mixed {
		slog.Debug("preferring web_search tool call over client tool calls in mixed Responses response")
	}
	return len(data), w.runAndWrite(response, call)
}

func (w *WebSearchResponsesWriter) finishStream() error {
	var initial api.ChatResponse
	var call api.ToolCall
	var found bool
	var observed api.Metrics
	var contentBuilder strings.Builder
	var thinkingBuilder strings.Builder
	var toolCalls []api.ToolCall
	for _, response := range w.buffered {
		observed.PromptEvalCount = max(observed.PromptEvalCount, response.Metrics.PromptEvalCount)
		observed.EvalCount = max(observed.EvalCount, response.Metrics.EvalCount)
		if response.Message.Content != "" {
			contentBuilder.WriteString(response.Message.Content)
		}
		if response.Message.Thinking != "" {
			thinkingBuilder.WriteString(response.Message.Thinking)
		}
		toolCalls = append(toolCalls, response.Message.ToolCalls...)
		if candidate, ok, mixed := findWebSearchToolCall(response.Message.ToolCalls); ok && !found {
			if mixed {
				slog.Debug("preferring web_search tool call over client tool calls in mixed Responses response")
			}
			initial, call, found = response, candidate, true
		}
	}
	if !found {
		return fmt.Errorf("web_search call disappeared before the terminal chunk")
	}
	// Combine model output from all streamed chunks into the initial response so
	// runLoop can preserve it before the web search events and in the follow-up.
	initial.Message.Content = contentBuilder.String()
	initial.Message.Thinking = thinkingBuilder.String()
	initial.Message.ToolCalls = toolCalls
	initial.Metrics = observed
	return w.runAndWrite(initial, call)
}

func (w *WebSearchResponsesWriter) runAndWrite(initial api.ChatResponse, call api.ToolCall) error {
	ctx, cancel := w.loopContext()
	defer cancel()

	if w.inner.stream {
		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
	}
	final, calls, usage, err := w.runLoop(ctx, initial, call)
	if err != nil {
		return w.writeWebSearchError(err, usage)
	}
	if w.inner.stream {
		return w.writeWebSearchStream(final, usage)
	}
	return w.writeWebSearchResponse(final, calls, usage)
}

func (w *WebSearchResponsesWriter) runLoop(ctx context.Context, initial api.ChatResponse, call api.ToolCall) (api.ChatResponse, []openai.ResponsesWebSearchCall, api.Metrics, error) {
	messages := append([]api.Message(nil), w.chat.Messages...)
	tools := append(api.Tools(nil), w.chat.Tools...)
	usage := initial.Metrics
	current, currentCall := initial, call
	calls := make([]openai.ResponsesWebSearchCall, 0, maxWebSearchLoops)
	var preSearchThinking strings.Builder
	var preSearchContent strings.Builder
	var otherToolCalls []api.ToolCall
	currentOutputStreamed := w.streamedInitialOutput

	// Emit response.created / response.in_progress once, before the loop.
	if w.inner.stream {
		for _, event := range w.inner.converter.Process(api.ChatResponse{}) {
			if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
				return api.ChatResponse{}, calls, usage, err
			}
		}
	}

	for loop := 1; loop <= maxWebSearchLoops; loop++ {
		// Collect non-web_search tool calls from mixed responses so they can
		// be surfaced to the client instead of silently dropped.
		var currentOtherToolCalls []api.ToolCall
		for _, tc := range current.Message.ToolCalls {
			if tc.Function.Name != "web_search" {
				currentOtherToolCalls = append(currentOtherToolCalls, tc)
			}
		}
		if !w.inner.stream {
			otherToolCalls = append(otherToolCalls, currentOtherToolCalls...)
		}

		if w.inner.stream && currentOutputStreamed {
			for _, event := range w.inner.converter.FinishMessageItem() {
				if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
					return api.ChatResponse{}, calls, usage, err
				}
			}
		}

		// Emit pre-search content (text the model produced before calling
		// web_search) as a completed message item before the search events.
		if current.Message.Thinking != "" && w.inner.stream && !currentOutputStreamed {
			thinkingResponse := api.ChatResponse{Message: api.Message{Role: "assistant", Thinking: current.Message.Thinking}}
			for _, event := range w.inner.converter.Process(thinkingResponse) {
				if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
					return api.ChatResponse{}, calls, usage, err
				}
			}
		}
		if current.Message.Thinking != "" && !w.inner.stream {
			if preSearchThinking.Len() > 0 {
				preSearchThinking.WriteString("\n")
			}
			preSearchThinking.WriteString(current.Message.Thinking)
		}
		if current.Message.Content != "" {
			if w.inner.stream && !currentOutputStreamed {
				contentResponse := api.ChatResponse{Message: api.Message{Role: "assistant", Content: current.Message.Content}}
				for _, event := range w.inner.converter.Process(contentResponse) {
					if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
						return api.ChatResponse{}, calls, usage, err
					}
				}
				for _, event := range w.inner.converter.FinishMessageItem() {
					if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
						return api.ChatResponse{}, calls, usage, err
					}
				}
			} else if !w.inner.stream {
				if preSearchContent.Len() > 0 {
					preSearchContent.WriteString("\n")
				}
				preSearchContent.WriteString(current.Message.Content)
			}
		}

		query := extractQueryFromToolCall(&currentCall)
		if strings.TrimSpace(query) == "" {
			return api.ChatResponse{}, calls, usage, fmt.Errorf("web_search requires a non-empty string query")
		}
		responseCall := openai.ResponsesWebSearchCall{
			ID:     fmt.Sprintf("ws_%s_%d", strings.TrimPrefix(w.inner.responseID, "resp_"), loop),
			Type:   "web_search_call",
			Status: "completed",
			Action: &openai.ResponsesWebSearchAction{Type: "search", Query: query},
		}
		outputIndex := 0
		if w.inner.stream {
			var events []openai.ResponsesStreamEvent
			outputIndex, events = w.inner.converter.StartWebSearchCall(responseCall)
			for _, event := range events {
				if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
					return api.ChatResponse{}, calls, usage, err
				}
			}
		}
		slog.Debug("executing Responses web search", "loop", loop)
		searchResponse, err := w.webSearch(ctx, query)
		if err != nil {
			return api.ChatResponse{}, calls, usage, err
		}
		slog.Debug("completed Responses web search", "loop", loop, "results", len(searchResponse.Results))
		if w.inner.stream {
			for _, event := range w.inner.converter.FinishWebSearchCall(responseCall, outputIndex) {
				if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
					return api.ChatResponse{}, calls, usage, err
				}
			}
		}
		calls = append(calls, responseCall)

		messages = append(messages,
			buildWebSearchAssistantMessage(current, currentCall),
			api.Message{Role: "tool", ToolCallID: currentCall.ID, Content: formatResponsesWebSearchResults(searchResponse.Results)},
		)
		var followUp api.ChatResponse
		var followUpOutputStreamed bool
		if w.inner.stream {
			followUp, followUpOutputStreamed, err = w.callFollowUpStream(ctx, messages, tools)
		} else {
			followUp, err = w.callFollowUp(ctx, messages, tools)
		}
		if err != nil {
			return api.ChatResponse{}, calls, usage, err
		}
		usage.PromptEvalCount += followUp.Metrics.PromptEvalCount
		usage.EvalCount += followUp.Metrics.EvalCount

		next, hasWebSearch, mixed := findWebSearchToolCall(followUp.Message.ToolCalls)
		if mixed {
			slog.Debug("preferring web_search tool call over client tool calls in mixed Responses followup")
		}
		if !hasWebSearch {
			w.preSearchThinking = preSearchThinking.String()
			w.preSearchContent = preSearchContent.String()
			w.otherToolCalls = otherToolCalls
			w.finalOutputStreamed = followUpOutputStreamed
			followUp.Metrics = usage
			return followUp, calls, usage, nil
		}
		current, currentCall = followUp, next
		currentOutputStreamed = followUpOutputStreamed
	}

	w.preSearchThinking = preSearchThinking.String()
	w.preSearchContent = preSearchContent.String()
	w.otherToolCalls = otherToolCalls
	return current, calls, usage, fmt.Errorf("web_search exceeded the maximum of %d calls", maxWebSearchLoops)
}

func (w *WebSearchResponsesWriter) loopContext() (context.Context, context.CancelFunc) {
	if w.newContext != nil {
		return w.newContext()
	}
	return context.WithTimeout(context.Background(), 5*time.Minute)
}

func (w *WebSearchResponsesWriter) webSearch(ctx context.Context, query string) (*api.WebSearchResponse, error) {
	if w.search != nil {
		return w.search(ctx, query)
	}
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}
	return client.WebSearchExperimental(ctx, &api.WebSearchRequest{Query: query, MaxResults: 5})
}

func (w *WebSearchResponsesWriter) callFollowUp(ctx context.Context, messages []api.Message, tools api.Tools) (api.ChatResponse, error) {
	if w.followUpChat != nil {
		return w.followUpChat(ctx, messages, tools)
	}
	return doFollowUpChat(ctx, *w.chat, messages, tools)
}

func (w *WebSearchResponsesWriter) callFollowUpStream(ctx context.Context, messages []api.Message, tools api.Tools) (api.ChatResponse, bool, error) {
	var final api.ChatResponse
	var content strings.Builder
	var thinking strings.Builder
	var role string
	var toolCalls []api.ToolCall
	outputStreamed := false

	yield := func(response api.ChatResponse) error {
		final = response
		if response.Message.Role != "" {
			role = response.Message.Role
		}
		content.WriteString(response.Message.Content)
		thinking.WriteString(response.Message.Thinking)
		toolCalls = append(toolCalls, response.Message.ToolCalls...)

		streamed := response
		streamed.Message.ToolCalls = nil
		streamed.Done = false
		if streamed.Message.Content != "" || streamed.Message.Thinking != "" {
			events := w.inner.converter.Process(streamed)
			for _, event := range events {
				if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
					return err
				}
			}
			outputStreamed = outputStreamed || len(events) > 0
		}

		var otherToolCalls []api.ToolCall
		for _, tc := range response.Message.ToolCalls {
			if tc.Function.Name != "web_search" {
				otherToolCalls = append(otherToolCalls, tc)
			}
		}
		if len(otherToolCalls) > 0 {
			for _, event := range w.inner.converter.FinishMessageItem() {
				if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
					return err
				}
			}
			for _, event := range w.inner.converter.EmitFunctionCallItems(otherToolCalls) {
				if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
					return err
				}
			}
			outputStreamed = true
		}
		return nil
	}

	var err error
	switch {
	case w.followUpStream != nil:
		err = w.followUpStream(ctx, messages, tools, yield)
	case w.followUpChat != nil:
		var response api.ChatResponse
		response, err = w.followUpChat(ctx, messages, tools)
		if err == nil {
			err = yield(response)
		}
	default:
		err = streamFollowUpChat(ctx, *w.chat, messages, tools, yield)
	}
	if err != nil {
		return api.ChatResponse{}, outputStreamed, err
	}

	final.Message.Role = role
	final.Message.Content = content.String()
	final.Message.Thinking = thinking.String()
	final.Message.ToolCalls = toolCalls
	return final, outputStreamed, nil
}

func formatResponsesWebSearchResults(results []api.WebSearchResult) string {
	var text strings.Builder
	for _, result := range results {
		fmt.Fprintf(&text, "Title: %s\nURL: %s\n", result.Title, result.URL)
		if result.Content != "" {
			fmt.Fprintf(&text, "Content: %s\n", result.Content)
		}
		text.WriteByte('\n')
	}
	return text.String()
}

func (w *WebSearchResponsesWriter) writeWebSearchResponse(final api.ChatResponse, calls []openai.ResponsesWebSearchCall, usage api.Metrics) error {
	response := openai.ToResponse(w.inner.model, w.inner.responseID, w.inner.itemID, final, w.req)
	completedAt := time.Now().Unix()
	response.CompletedAt = &completedAt
	response.Output = buildResponsesWebSearchOutput(response.Output, w.preSearchThinking, w.preSearchContent, calls, w.otherToolCalls)
	if response.Usage != nil {
		response.Usage.InputTokens = usage.PromptEvalCount
		response.Usage.OutputTokens = usage.EvalCount
		response.Usage.TotalTokens = usage.PromptEvalCount + usage.EvalCount
	}
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	w.done = true
	return json.NewEncoder(w.ResponseWriter).Encode(response)
}

// buildResponsesWebSearchOutput assembles the final non-streaming output in
// model-leg order: pre-search reasoning/text, server and mixed tool calls, then
// the final model output.
func buildResponsesWebSearchOutput(output []openai.ResponsesOutputItem, preSearchThinking, preSearchContent string, searchCalls []openai.ResponsesWebSearchCall, otherToolCalls []api.ToolCall) []openai.ResponsesOutputItem {
	items := make([]openai.ResponsesOutputItem, 0, len(output)+len(searchCalls)+len(otherToolCalls)+2)
	if preSearchThinking != "" {
		items = append(items, openai.ResponsesOutputItem{
			ID:   "rs_presearch",
			Type: "reasoning",
			Summary: []openai.ResponsesReasoningSummary{
				{Type: "summary_text", Text: preSearchThinking},
			},
			EncryptedContent: preSearchThinking,
		})
	}
	// pre-search message (if the model emitted content before calling web_search)
	if preSearchContent != "" {
		items = append(items, openai.ResponsesOutputItem{
			ID:     "msg_presearch",
			Type:   "message",
			Status: "completed",
			Role:   "assistant",
			Content: []openai.ResponsesOutputContent{
				{Type: "output_text", Text: preSearchContent, Annotations: []any{}, Logprobs: []any{}},
			},
		})
	}
	// web_search_call items
	for _, call := range searchCalls {
		items = append(items, openai.WebSearchCallOutputItem(call))
	}
	// function_call items from mixed responses
	convertedCalls := openai.ToToolCalls(otherToolCalls)
	for i, tc := range convertedCalls {
		items = append(items, openai.ResponsesOutputItem{
			ID:        fmt.Sprintf("fc_mixed_%d", i),
			Type:      "function_call",
			Status:    "completed",
			CallID:    tc.ID,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}
	// remaining items (final reasoning, message, or function calls)
	items = append(items, output...)
	return items
}

func (w *WebSearchResponsesWriter) writeWebSearchStream(final api.ChatResponse, usage api.Metrics) error {
	if w.finalOutputStreamed {
		final.Message = api.Message{}
	}
	final.Metrics = usage
	final.Done = true
	for _, event := range w.inner.converter.Process(final) {
		if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
			return err
		}
	}
	w.done = true
	return nil
}

func (w *WebSearchResponsesWriter) writeWebSearchError(err error, usage api.Metrics) error {
	message := err.Error()
	status := http.StatusBadGateway
	errorCode := "api_error"
	var authorizationError api.AuthorizationError
	var statusError api.StatusError
	switch {
	case errors.As(err, &authorizationError):
		status = authorizationError.StatusCode
		errorCode = "authentication_error"
		if authorizationError.SigninURL != "" {
			message += "; sign in at " + authorizationError.SigninURL
		}
	case errors.As(err, &statusError):
		status = statusError.StatusCode
		if status == http.StatusTooManyRequests {
			errorCode = "rate_limit_exceeded"
		}
	}
	if !w.inner.stream {
		w.ResponseWriter.Header().Set("Content-Type", "application/json")
		w.ResponseWriter.WriteHeader(status)
		w.done = true
		return json.NewEncoder(w.ResponseWriter).Encode(openai.NewError(status, message))
	}
	w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
	response := map[string]any{
		"id": w.inner.responseID, "object": "response", "status": "failed", "model": w.req.Model,
		"output": []any{}, "error": map[string]any{"code": errorCode, "message": message},
		"usage": map[string]any{"input_tokens": usage.PromptEvalCount, "output_tokens": usage.EvalCount, "total_tokens": usage.PromptEvalCount + usage.EvalCount},
	}
	initialEvents := w.inner.converter.Process(api.ChatResponse{})
	for _, event := range initialEvents {
		if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
			return err
		}
	}
	event := w.inner.converter.ResponseFailed(response)
	if err := w.inner.writeEvent(event.Event, event.Data); err != nil {
		return err
	}
	w.done = true
	return nil
}

func decodeWebSearchResponseError(status int, data []byte) error {
	var response struct {
		Error     string `json:"error"`
		SigninURL string `json:"signin_url"`
	}
	if err := json.Unmarshal(data, &response); err != nil {
		response.Error = string(data)
	}
	if status == http.StatusUnauthorized {
		return api.AuthorizationError{StatusCode: status, Status: response.Error, SigninURL: response.SigninURL}
	}
	return api.StatusError{StatusCode: status, ErrorMessage: response.Error}
}

func ResponsesMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		requestCtx := c.Request.Context()
		if c.GetHeader("Content-Encoding") == "zstd" {
			reader, err := zstd.NewReader(c.Request.Body, zstd.WithDecoderMaxMemory(8<<20))
			if err != nil {
				c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "failed to decompress zstd body"))
				return
			}
			defer reader.Close()
			c.Request.Body = http.MaxBytesReader(c.Writer, io.NopCloser(reader), maxDecompressedBodySize)
			c.Request.Header.Del("Content-Encoding")
		}

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

		hasWebSearch := openai.HasWebSearchTool(req.Tools)
		slog.Debug("parsed Responses tools", "count", len(req.Tools), "web_search", hasWebSearch)
		if hasWebSearch {
			c.Writer = &WebSearchResponsesWriter{
				BaseWriter: BaseWriter{ResponseWriter: c.Writer},
				inner:      w,
				req:        req,
				chat:       chatReq,
				newContext: func() (context.Context, context.CancelFunc) {
					return context.WithTimeout(requestCtx, 5*time.Minute)
				},
			}
		} else {
			c.Writer = w
		}
		c.Next()
	}
}

// TranscriptionWriter collects streamed chat responses and outputs a transcription response.
type TranscriptionWriter struct {
	BaseWriter
	responseFormat string
	text           strings.Builder
}

func (w *TranscriptionWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	var chatResponse api.ChatResponse
	if err := json.Unmarshal(data, &chatResponse); err != nil {
		return 0, err
	}

	w.text.WriteString(chatResponse.Message.Content)

	if chatResponse.Done {
		text := strings.TrimSpace(w.text.String())

		if w.responseFormat == "text" {
			w.ResponseWriter.Header().Set("Content-Type", "text/plain")
			_, err := w.ResponseWriter.Write([]byte(text))
			if err != nil {
				return 0, err
			}
			return len(data), nil
		}

		w.ResponseWriter.Header().Set("Content-Type", "application/json")
		resp := openai.TranscriptionResponse{Text: text}
		if err := json.NewEncoder(w.ResponseWriter).Encode(resp); err != nil {
			return 0, err
		}
	}

	return len(data), nil
}

// TranscriptionMiddleware handles /v1/audio/transcriptions requests.
// It accepts multipart/form-data with an audio file and converts it to a chat request.
func TranscriptionMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Parse multipart form (limit 25MB).
		if err := c.Request.ParseMultipartForm(25 << 20); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "failed to parse multipart form: "+err.Error()))
			return
		}

		model := c.Request.FormValue("model")
		if model == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "model is required"))
			return
		}

		file, _, err := c.Request.FormFile("file")
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "file is required: "+err.Error()))
			return
		}
		defer file.Close()

		audioData, err := io.ReadAll(file)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, "failed to read audio file"))
			return
		}

		if len(audioData) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, "audio file is empty"))
			return
		}

		req := openai.TranscriptionRequest{
			Model:          model,
			AudioData:      audioData,
			ResponseFormat: c.Request.FormValue("response_format"),
			Language:       c.Request.FormValue("language"),
			Prompt:         c.Request.FormValue("prompt"),
		}

		chatReq, err := openai.FromTranscriptionRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, openai.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)
		c.Request.ContentLength = int64(b.Len())
		c.Request.Header.Set("Content-Type", "application/json")

		w := &TranscriptionWriter{
			BaseWriter:     BaseWriter{ResponseWriter: c.Writer},
			responseFormat: req.ResponseFormat,
		}

		c.Writer = w
		c.Next()
	}
}
