package middleware

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	internalcloud "github.com/ollama/ollama/internal/cloud"
)

// AnthropicWriter wraps the response writer to transform Ollama responses to Anthropic format
type AnthropicWriter struct {
	BaseWriter
	stream    bool
	id        string
	converter *anthropic.StreamConverter
}

func (w *AnthropicWriter) writeError(data []byte) (int, error) {
	var errData struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(data, &errData); err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w.ResponseWriter).Encode(anthropic.NewError(w.ResponseWriter.Status(), errData.Error))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *AnthropicWriter) writeEvent(eventType string, data any) error {
	return writeSSE(w.ResponseWriter, eventType, data)
}

func (w *AnthropicWriter) writeResponse(data []byte) (int, error) {
	var chatResponse api.ChatResponse
	err := json.Unmarshal(data, &chatResponse)
	if err != nil {
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

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	response := anthropic.ToMessagesResponse(w.id, chatResponse)
	return len(data), json.NewEncoder(w.ResponseWriter).Encode(response)
}

func (w *AnthropicWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(data)
	}

	return w.writeResponse(data)
}

// WebSearchAnthropicWriter intercepts responses containing web_search tool calls,
// executes the search, re-invokes the model with results, and assembles the
// Anthropic-format response (server_tool_use + web_search_tool_result + text).
type WebSearchAnthropicWriter struct {
	BaseWriter
	ctx     context.Context
	inner   *AnthropicWriter
	req     anthropic.MessagesRequest // original Anthropic request
	chatReq *api.ChatRequest          // converted Ollama request (for followup calls)
	stream  bool

	estimatedInputTokens int

	terminalSent bool

	streamMessageStarted bool
	streamHasOpenBlock   bool
	streamOpenBlockIndex int
	streamNextIndex      int
}

const maxWebSearchLoops = 3

type webSearchLoopError struct {
	code  string
	query string
	err   error
}

func (e *webSearchLoopError) Error() string {
	if e.err == nil {
		return e.code
	}
	return fmt.Sprintf("%s: %v", e.code, e.err)
}

func (w *WebSearchAnthropicWriter) Write(data []byte) (int, error) {
	if w.terminalSent {
		return len(data), nil
	}

	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.inner.writeError(data)
	}

	var chatResponse api.ChatResponse
	if err := json.Unmarshal(data, &chatResponse); err != nil {
		return 0, err
	}

	webSearchCall, hasWebSearch, hasOtherTools := findWebSearchToolCall(chatResponse.Message.ToolCalls)
	if hasWebSearch && hasOtherTools {
		// Prefer web_search if both server and client tools are present in one chunk.
		slog.Debug("preferring web_search tool call over client tool calls in mixed tool response")
	}

	if !hasWebSearch {
		if w.stream {
			if err := w.writePassthroughStreamChunk(chatResponse); err != nil {
				return 0, err
			}
			return len(data), nil
		}
		return w.inner.writeResponse(data)
	}

	response, loopErr := w.runWebSearchLoop(chatResponse, webSearchCall)
	if loopErr != nil {
		return len(data), w.sendError(loopErr.code, loopErr.query)
	}

	if err := w.writeTerminalResponse(response); err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *WebSearchAnthropicWriter) runWebSearchLoop(initialResponse api.ChatResponse, initialToolCall api.ToolCall) (anthropic.MessagesResponse, *webSearchLoopError) {
	followUpMessages := make([]api.Message, 0, len(w.chatReq.Messages)+maxWebSearchLoops*2)
	followUpMessages = append(followUpMessages, w.chatReq.Messages...)

	followUpTools := stripWebSearchTools(w.chatReq.Tools)
	usage := anthropic.Usage{
		InputTokens:  initialResponse.Metrics.PromptEvalCount,
		OutputTokens: initialResponse.Metrics.EvalCount,
	}

	currentResponse := initialResponse
	currentToolCall := initialToolCall

	var serverContent []anthropic.ContentBlock

	for loop := 1; loop <= maxWebSearchLoops; loop++ {
		query := extractQueryFromToolCall(&currentToolCall)
		if query == "" {
			return anthropic.MessagesResponse{}, &webSearchLoopError{
				code:  "invalid_request",
				query: "",
			}
		}

		const defaultMaxResults = 5
		searchResp, err := anthropic.WebSearch(w.ctx, query, defaultMaxResults)
		if err != nil {
			return anthropic.MessagesResponse{}, &webSearchLoopError{
				code:  "unavailable",
				query: query,
				err:   err,
			}
		}

		toolUseID := loopServerToolUseID(w.inner.id, loop)
		searchResults := anthropic.ConvertOllamaToAnthropicResults(searchResp)
		serverContent = append(serverContent,
			anthropic.ContentBlock{
				Type:  "server_tool_use",
				ID:    toolUseID,
				Name:  "web_search",
				Input: map[string]any{"query": query},
			},
			anthropic.ContentBlock{
				Type:      "web_search_tool_result",
				ToolUseID: toolUseID,
				Content:   searchResults,
			},
		)

		assistantMsg := buildWebSearchAssistantMessage(currentResponse, currentToolCall)
		toolResultMsg := api.Message{
			Role:       "tool",
			Content:    formatWebSearchResultsForToolMessage(searchResp.Results),
			ToolCallID: currentToolCall.ID,
		}
		followUpMessages = append(followUpMessages, assistantMsg, toolResultMsg)

		followUpResponse, err := w.callFollowUpChat(followUpMessages, followUpTools)
		if err != nil {
			return anthropic.MessagesResponse{}, &webSearchLoopError{
				code:  "api_error",
				query: query,
				err:   err,
			}
		}

		usage.InputTokens += followUpResponse.Metrics.PromptEvalCount
		usage.OutputTokens += followUpResponse.Metrics.EvalCount

		nextToolCall, hasWebSearch, hasOtherTools := findWebSearchToolCall(followUpResponse.Message.ToolCalls)
		if hasWebSearch && hasOtherTools {
			// Prefer web_search if both server and client tools are present in one chunk.
			slog.Debug("preferring web_search tool call over client tool calls in mixed followup response")
		}

		if !hasWebSearch {
			return w.combineServerAndFinalContent(serverContent, followUpResponse, usage), nil
		}

		if loop == maxWebSearchLoops {
			maxLoopQuery := extractQueryFromToolCall(&nextToolCall)
			maxLoopToolUseID := loopServerToolUseID(w.inner.id, loop+1)
			serverContent = append(serverContent,
				anthropic.ContentBlock{
					Type:  "server_tool_use",
					ID:    maxLoopToolUseID,
					Name:  "web_search",
					Input: map[string]any{"query": maxLoopQuery},
				},
				anthropic.ContentBlock{
					Type:      "web_search_tool_result",
					ToolUseID: maxLoopToolUseID,
					Content: anthropic.WebSearchToolResultError{
						Type:      "web_search_tool_result_error",
						ErrorCode: "max_uses_exceeded",
					},
				},
			)

			return anthropic.MessagesResponse{
				ID:         w.inner.id,
				Type:       "message",
				Role:       "assistant",
				Model:      w.req.Model,
				Content:    serverContent,
				StopReason: "end_turn",
				Usage:      usage,
			}, nil
		}

		currentResponse = followUpResponse
		currentToolCall = nextToolCall
	}

	return anthropic.MessagesResponse{}, &webSearchLoopError{code: "api_error"}
}

func (w *WebSearchAnthropicWriter) combineServerAndFinalContent(serverContent []anthropic.ContentBlock, finalResponse api.ChatResponse, usage anthropic.Usage) anthropic.MessagesResponse {
	converted := anthropic.ToMessagesResponse(w.inner.id, finalResponse)

	content := make([]anthropic.ContentBlock, 0, len(serverContent)+len(converted.Content))
	content = append(content, serverContent...)
	content = append(content, converted.Content...)

	return anthropic.MessagesResponse{
		ID:           w.inner.id,
		Type:         "message",
		Role:         "assistant",
		Model:        w.req.Model,
		Content:      content,
		StopReason:   converted.StopReason,
		StopSequence: converted.StopSequence,
		Usage:        usage,
	}
}

func buildWebSearchAssistantMessage(response api.ChatResponse, webSearchCall api.ToolCall) api.Message {
	assistantMsg := api.Message{
		Role:      "assistant",
		ToolCalls: []api.ToolCall{webSearchCall},
	}
	if response.Message.Content != "" {
		assistantMsg.Content = response.Message.Content
	}
	if response.Message.Thinking != "" {
		assistantMsg.Thinking = response.Message.Thinking
	}
	return assistantMsg
}

func formatWebSearchResultsForToolMessage(results []anthropic.OllamaWebSearchResult) string {
	var resultText strings.Builder
	for _, r := range results {
		fmt.Fprintf(&resultText, "Title: %s\nURL: %s\n", r.Title, r.URL)
		if r.Content != "" {
			fmt.Fprintf(&resultText, "Content: %s\n", r.Content)
		}
		resultText.WriteString("\n")
	}
	return resultText.String()
}

func stripWebSearchTools(tools api.Tools) api.Tools {
	var followUpTools api.Tools
	for _, t := range tools {
		if t.Function.Name != "web_search" {
			followUpTools = append(followUpTools, t)
		}
	}
	return followUpTools
}

func findWebSearchToolCall(toolCalls []api.ToolCall) (api.ToolCall, bool, bool) {
	var webSearchCall api.ToolCall
	hasWebSearch := false
	hasOtherTools := false

	for _, toolCall := range toolCalls {
		if toolCall.Function.Name == "web_search" {
			if !hasWebSearch {
				webSearchCall = toolCall
				hasWebSearch = true
			}
			continue
		}
		hasOtherTools = true
	}

	return webSearchCall, hasWebSearch, hasOtherTools
}

func loopServerToolUseID(messageID string, loop int) string {
	base := serverToolUseID(messageID)
	if loop <= 1 {
		return base
	}
	return fmt.Sprintf("%s_%d", base, loop)
}

func (w *WebSearchAnthropicWriter) callFollowUpChat(messages []api.Message, tools api.Tools) (api.ChatResponse, error) {
	streaming := false
	followUp := api.ChatRequest{
		Model:    w.chatReq.Model,
		Messages: messages,
		Stream:   &streaming,
		Tools:    tools,
		Options:  w.chatReq.Options,
	}

	body, err := json.Marshal(followUp)
	if err != nil {
		return api.ChatResponse{}, err
	}

	chatURL := envconfig.Host().String() + "/api/chat"
	client := &http.Client{Timeout: 5 * time.Minute}
	httpReq, err := http.NewRequestWithContext(w.ctx, "POST", chatURL, bytes.NewReader(body))
	if err != nil {
		return api.ChatResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(httpReq)
	if err != nil {
		return api.ChatResponse{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return api.ChatResponse{}, fmt.Errorf("followup /api/chat returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	var chatResp api.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return api.ChatResponse{}, err
	}

	return chatResp, nil
}

func (w *WebSearchAnthropicWriter) writePassthroughStreamChunk(chatResponse api.ChatResponse) error {
	events := w.inner.converter.Process(chatResponse)
	for _, event := range events {
		switch e := event.Data.(type) {
		case anthropic.MessageStartEvent:
			w.streamMessageStarted = true
		case anthropic.ContentBlockStartEvent:
			w.streamHasOpenBlock = true
			w.streamOpenBlockIndex = e.Index
			if e.Index+1 > w.streamNextIndex {
				w.streamNextIndex = e.Index + 1
			}
		case anthropic.ContentBlockStopEvent:
			if w.streamHasOpenBlock && w.streamOpenBlockIndex == e.Index {
				w.streamHasOpenBlock = false
			}
			if e.Index+1 > w.streamNextIndex {
				w.streamNextIndex = e.Index + 1
			}
		case anthropic.MessageStopEvent:
			w.terminalSent = true
		}

		if err := writeSSE(w.ResponseWriter, event.Event, event.Data); err != nil {
			return err
		}
	}

	return nil
}

func (w *WebSearchAnthropicWriter) ensureStreamMessageStart(usage anthropic.Usage) error {
	if w.streamMessageStarted {
		return nil
	}

	inputTokens := usage.InputTokens
	if inputTokens == 0 {
		inputTokens = w.estimatedInputTokens
	}

	if err := writeSSE(w.ResponseWriter, "message_start", anthropic.MessageStartEvent{
		Type: "message_start",
		Message: anthropic.MessagesResponse{
			ID:      w.inner.id,
			Type:    "message",
			Role:    "assistant",
			Model:   w.req.Model,
			Content: []anthropic.ContentBlock{},
			Usage: anthropic.Usage{
				InputTokens: inputTokens,
			},
		},
	}); err != nil {
		return err
	}

	w.streamMessageStarted = true
	return nil
}

func (w *WebSearchAnthropicWriter) closeOpenStreamBlock() error {
	if !w.streamHasOpenBlock {
		return nil
	}

	if err := writeSSE(w.ResponseWriter, "content_block_stop", anthropic.ContentBlockStopEvent{
		Type:  "content_block_stop",
		Index: w.streamOpenBlockIndex,
	}); err != nil {
		return err
	}

	if w.streamOpenBlockIndex+1 > w.streamNextIndex {
		w.streamNextIndex = w.streamOpenBlockIndex + 1
	}
	w.streamHasOpenBlock = false
	return nil
}

func (w *WebSearchAnthropicWriter) writeStreamContentBlocks(content []anthropic.ContentBlock) error {
	for _, block := range content {
		index := w.streamNextIndex
		if block.Type == "text" {
			emptyText := ""
			if err := writeSSE(w.ResponseWriter, "content_block_start", anthropic.ContentBlockStartEvent{
				Type:  "content_block_start",
				Index: index,
				ContentBlock: anthropic.ContentBlock{
					Type: "text",
					Text: &emptyText,
				},
			}); err != nil {
				return err
			}

			text := ""
			if block.Text != nil {
				text = *block.Text
			}
			if err := writeSSE(w.ResponseWriter, "content_block_delta", anthropic.ContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: index,
				Delta: anthropic.Delta{
					Type: "text_delta",
					Text: text,
				},
			}); err != nil {
				return err
			}
		} else {
			if err := writeSSE(w.ResponseWriter, "content_block_start", anthropic.ContentBlockStartEvent{
				Type:         "content_block_start",
				Index:        index,
				ContentBlock: block,
			}); err != nil {
				return err
			}
		}

		if err := writeSSE(w.ResponseWriter, "content_block_stop", anthropic.ContentBlockStopEvent{
			Type:  "content_block_stop",
			Index: index,
		}); err != nil {
			return err
		}

		w.streamNextIndex++
	}

	return nil
}

func (w *WebSearchAnthropicWriter) writeTerminalResponse(response anthropic.MessagesResponse) error {
	if w.terminalSent {
		return nil
	}

	if !w.stream {
		w.ResponseWriter.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w.ResponseWriter).Encode(response); err != nil {
			return err
		}
		w.terminalSent = true
		return nil
	}

	if err := w.ensureStreamMessageStart(response.Usage); err != nil {
		return err
	}
	if err := w.closeOpenStreamBlock(); err != nil {
		return err
	}
	if err := w.writeStreamContentBlocks(response.Content); err != nil {
		return err
	}

	if err := writeSSE(w.ResponseWriter, "message_delta", anthropic.MessageDeltaEvent{
		Type: "message_delta",
		Delta: anthropic.MessageDelta{
			StopReason: response.StopReason,
		},
		Usage: anthropic.DeltaUsage{
			OutputTokens: response.Usage.OutputTokens,
		},
	}); err != nil {
		return err
	}

	if err := writeSSE(w.ResponseWriter, "message_stop", anthropic.MessageStopEvent{
		Type: "message_stop",
	}); err != nil {
		return err
	}

	w.terminalSent = true
	return nil
}

// streamResponse emits a complete MessagesResponse as SSE events.
func (w *WebSearchAnthropicWriter) streamResponse(response anthropic.MessagesResponse) error {
	return w.writeTerminalResponse(response)
}

func (w *WebSearchAnthropicWriter) webSearchErrorResponse(errorCode, query string, usage anthropic.Usage) anthropic.MessagesResponse {
	toolUseID := serverToolUseID(w.inner.id)

	return anthropic.MessagesResponse{
		ID:    w.inner.id,
		Type:  "message",
		Role:  "assistant",
		Model: w.req.Model,
		Content: []anthropic.ContentBlock{
			{
				Type:  "server_tool_use",
				ID:    toolUseID,
				Name:  "web_search",
				Input: map[string]any{"query": query},
			},
			{
				Type:      "web_search_tool_result",
				ToolUseID: toolUseID,
				Content: anthropic.WebSearchToolResultError{
					Type:      "web_search_tool_result_error",
					ErrorCode: errorCode,
				},
			},
		},
		StopReason: "end_turn",
		Usage:      usage,
	}
}

// sendError sends a web search error response.
func (w *WebSearchAnthropicWriter) sendError(errorCode, query string) error {
	return w.writeTerminalResponse(w.webSearchErrorResponse(errorCode, query, anthropic.Usage{}))
}

// AnthropicMessagesMiddleware handles Anthropic Messages API requests
func AnthropicMessagesMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req anthropic.MessagesRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, anthropic.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if req.Model == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, anthropic.NewError(http.StatusBadRequest, "model is required"))
			return
		}

		if req.MaxTokens <= 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, anthropic.NewError(http.StatusBadRequest, "max_tokens is required and must be positive"))
			return
		}

		if len(req.Messages) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, anthropic.NewError(http.StatusBadRequest, "messages is required"))
			return
		}

		chatReq, err := anthropic.FromMessagesRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, anthropic.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		// Set think to nil when being used with Anthropic API to connect to tools like claude code
		c.Set("relax_thinking", true)

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, anthropic.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		messageID := anthropic.GenerateMessageID()

		// Estimate input tokens for streaming (actual count not available until generation completes)
		estimatedTokens := anthropic.EstimateInputTokens(req)

		innerWriter := &AnthropicWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			stream:     req.Stream,
			id:         messageID,
			converter:  anthropic.NewStreamConverter(messageID, req.Model, estimatedTokens),
		}

		if req.Stream {
			c.Writer.Header().Set("Content-Type", "text/event-stream")
			c.Writer.Header().Set("Cache-Control", "no-cache")
			c.Writer.Header().Set("Connection", "keep-alive")
		}

		if hasWebSearchTool(req.Tools) {
			// Guard against runtime cloud-disable policy (OLLAMA_NO_CLOUD/server.json),
			// independent of model naming. Name-based checks are transitional.
			if disabled, _ := internalcloud.Status(); disabled {
				c.AbortWithStatusJSON(http.StatusForbidden, anthropic.NewError(http.StatusForbidden, internalcloud.DisabledError("web search is unavailable")))
				return
			}

			if !isCloudModelName(req.Model) {
				c.AbortWithStatusJSON(http.StatusBadRequest, anthropic.NewError(http.StatusBadRequest, "web_search tool is only supported for cloud models"))
				return
			}

			c.Writer = &WebSearchAnthropicWriter{
				BaseWriter:           BaseWriter{ResponseWriter: c.Writer},
				ctx:                  c.Request.Context(),
				inner:                innerWriter,
				req:                  req,
				chatReq:              chatReq,
				stream:               req.Stream,
				estimatedInputTokens: estimatedTokens,
			}
		} else {
			c.Writer = innerWriter
		}

		c.Next()
	}
}

// hasWebSearchTool checks if the request tools include a web_search tool
func hasWebSearchTool(tools []anthropic.Tool) bool {
	for _, tool := range tools {
		if strings.HasPrefix(tool.Type, "web_search") {
			return true
		}
	}
	return false
}

func isCloudModelName(name string) bool {
	return strings.HasSuffix(name, ":cloud") || strings.HasSuffix(name, "-cloud")
}

// extractQueryFromToolCall extracts the search query from a web_search tool call
func extractQueryFromToolCall(tc *api.ToolCall) string {
	q, ok := tc.Function.Arguments.Get("query")
	if !ok {
		return ""
	}
	if s, ok := q.(string); ok {
		return s
	}
	return ""
}

// writeSSE writes a Server-Sent Event
func writeSSE(w http.ResponseWriter, eventType string, data any) error {
	d, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, d); err != nil {
		return err
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}

// serverToolUseID derives a server tool use ID from a message ID
func serverToolUseID(messageID string) string {
	return "srvtoolu_" + strings.TrimPrefix(messageID, "msg_")
}
