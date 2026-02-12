package middleware

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

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
	stream bool
	chunks [][]byte // accumulates streaming chunks from first model call
}

func (w *WebSearchAnthropicWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		slog.Warn("websearch writer got non-200 status", "code", code)
		return w.inner.writeError(data)
	}

	var chatResponse api.ChatResponse
	if err := json.Unmarshal(data, &chatResponse); err != nil {
		slog.Error("websearch writer failed to unmarshal chat response", "error", err)
		return 0, err
	}

	if w.stream && !chatResponse.Done {
		w.chunks = append(w.chunks, append([]byte(nil), data...))
		return len(data), nil
	}

	slog.Info("websearch: first model response",
		"done", chatResponse.Done,
		"done_reason", chatResponse.DoneReason,
		"content_len", len(chatResponse.Message.Content),
		"tool_calls_count", len(chatResponse.Message.ToolCalls),
		"has_thinking", chatResponse.Message.Thinking != "",
	)

	for i, tc := range chatResponse.Message.ToolCalls {
		slog.Info("websearch: tool call in response",
			"index", i,
			"id", tc.ID,
			"name", tc.Function.Name,
			"args_len", tc.Function.Arguments.Len(),
		)
	}

	var webSearchCall *api.ToolCall
	for i := range chatResponse.Message.ToolCalls {
		if chatResponse.Message.ToolCalls[i].Function.Name == "web_search" {
			webSearchCall = &chatResponse.Message.ToolCalls[i]
			break
		}
	}

	if webSearchCall == nil {
		slog.Info("websearch: model did NOT call web_search, passing through")
		if w.stream {
			if err := w.replayBuffered(); err != nil {
				return 0, err
			}
		}
		return w.inner.writeResponse(data)
	}

	slog.Info("websearch: model called web_search, intercepting",
		"tool_call_id", webSearchCall.ID,
	)

	return len(data), w.handleWebSearch(chatResponse, webSearchCall)
}

// replayBuffered sends all buffered streaming chunks through the inner writer
func (w *WebSearchAnthropicWriter) replayBuffered() error {
	for _, chunk := range w.chunks {
		if _, err := w.inner.writeResponse(chunk); err != nil {
			return err
		}
	}
	return nil
}

// handleWebSearch executes the search, re-invokes the model with results, and
// returns server_tool_use + web_search_tool_result + text to the client.
func (w *WebSearchAnthropicWriter) handleWebSearch(firstResponse api.ChatResponse, toolCall *api.ToolCall) error {
	query := extractQueryFromToolCall(toolCall)
	if query == "" {
		slog.Error("web_search tool call has no query argument", "tool_call_id", toolCall.ID)
		return w.sendError("invalid_request", "")
	}

	slog.Info("websearch: executing search", "query", query)

	apiKey := os.Getenv("OLLAMA_API_KEY")
	searchStart := time.Now()
	const defaultMaxResults = 5
	searchResp, err := anthropic.WebSearch(w.ctx, apiKey, query, defaultMaxResults)
	searchDuration := time.Since(searchStart)

	if err != nil {
		slog.Error("websearch: search API failed", "error", err, "query", query, "duration", searchDuration)
		return w.sendError("unavailable", query)
	}

	slog.Info("websearch: search completed", "query", query, "results_count", len(searchResp.Results), "duration", searchDuration)

	searchResults := anthropic.ConvertOllamaToAnthropicResults(searchResp)

	toolUseID := serverToolUseID(w.inner.id)

	var resultText strings.Builder
	for _, r := range searchResp.Results {
		fmt.Fprintf(&resultText, "Title: %s\nURL: %s\n", r.Title, r.URL)
		if r.Content != "" {
			fmt.Fprintf(&resultText, "Content: %s\n", r.Content)
		}
		resultText.WriteString("\n")
	}

	// Strip web_search from tools to prevent infinite loop
	var followUpTools api.Tools
	for _, t := range w.chatReq.Tools {
		if t.Function.Name != "web_search" {
			followUpTools = append(followUpTools, t)
		}
	}

	assistantMsg := api.Message{
		Role:      "assistant",
		ToolCalls: firstResponse.Message.ToolCalls,
	}
	if firstResponse.Message.Content != "" {
		assistantMsg.Content = firstResponse.Message.Content
	}

	toolResultMsg := api.Message{
		Role:       "tool",
		Content:    resultText.String(),
		ToolCallID: toolCall.ID,
	}

	followUpMessages := make([]api.Message, len(w.chatReq.Messages))
	copy(followUpMessages, w.chatReq.Messages)
	followUpMessages = append(followUpMessages, assistantMsg, toolResultMsg)

	streaming := false
	followUp := api.ChatRequest{
		Model:    w.chatReq.Model,
		Messages: followUpMessages,
		Stream:   &streaming,
		Tools:    followUpTools,
		Options:  w.chatReq.Options,
	}

	body, err := json.Marshal(followUp)
	if err != nil {
		return err
	}

	chatURL := envconfig.Host().String() + "/api/chat"
	followUpStart := time.Now()
	client := &http.Client{Timeout: 5 * time.Minute}
	httpReq, err := http.NewRequestWithContext(w.ctx, "POST", chatURL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(httpReq)
	if err != nil {
		slog.Error("websearch: followup request failed", "error", err)
		return w.sendError("unavailable", query)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	var chatResp api.ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		slog.Error("websearch: failed to unmarshal followup response", "error", err)
		return err
	}

	slog.Info("websearch: followup response",
		"content_len", len(chatResp.Message.Content),
		"eval_count", chatResp.Metrics.EvalCount,
		"duration", time.Since(followUpStart),
	)

	content := []anthropic.ContentBlock{
		{
			Type:  "server_tool_use",
			ID:    toolUseID,
			Name:  "web_search",
			Input: map[string]any{"query": query},
		},
		{
			Type:      "web_search_tool_result",
			ToolUseID: toolUseID,
			Content:   searchResults,
		},
	}

	text := chatResp.Message.Content
	if text == "" {
		text = "Search completed but no response was generated."
	}
	content = append(content, anthropic.ContentBlock{
		Type: "text",
		Text: &text,
	})

	response := anthropic.MessagesResponse{
		ID:         w.inner.id,
		Type:       "message",
		Role:       "assistant",
		Model:      w.req.Model,
		Content:    content,
		StopReason: "end_turn",
		Usage: anthropic.Usage{
			InputTokens:  chatResp.Metrics.PromptEvalCount,
			OutputTokens: chatResp.Metrics.EvalCount,
		},
	}

	if w.stream {
		return w.streamResponse(response)
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w.ResponseWriter).Encode(response)
}

// streamResponse emits a complete MessagesResponse as SSE events
func (w *WebSearchAnthropicWriter) streamResponse(response anthropic.MessagesResponse) error {
	if err := writeSSE(w.ResponseWriter, "message_start", anthropic.MessageStartEvent{
		Type: "message_start",
		Message: anthropic.MessagesResponse{
			ID:      response.ID,
			Type:    "message",
			Role:    "assistant",
			Model:   response.Model,
			Content: []anthropic.ContentBlock{},
			Usage:   response.Usage,
		},
	}); err != nil {
		return err
	}

	for i, block := range response.Content {
		if block.Type == "text" {
			emptyText := ""
			if err := writeSSE(w.ResponseWriter, "content_block_start", anthropic.ContentBlockStartEvent{
				Type:  "content_block_start",
				Index: i,
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
				Index: i,
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
				Index:        i,
				ContentBlock: block,
			}); err != nil {
				return err
			}
		}
		if err := writeSSE(w.ResponseWriter, "content_block_stop", anthropic.ContentBlockStopEvent{
			Type:  "content_block_stop",
			Index: i,
		}); err != nil {
			return err
		}
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

	return writeSSE(w.ResponseWriter, "message_stop", anthropic.MessageStopEvent{
		Type: "message_stop",
	})
}

// sendError sends a web search error response
func (w *WebSearchAnthropicWriter) sendError(errorCode, query string) error {
	slog.Warn("websearch: sending error response", "error_code", errorCode)

	toolUseID := serverToolUseID(w.inner.id)

	response := anthropic.MessagesResponse{
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
		Usage:      anthropic.Usage{},
	}

	if w.stream {
		return w.streamResponse(response)
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w.ResponseWriter).Encode(response)
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

		c.Set("relax_thinking", true)

		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, anthropic.NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		messageID := anthropic.GenerateMessageID()

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
			if !strings.HasSuffix(req.Model, "cloud") {
				c.AbortWithStatusJSON(http.StatusBadRequest, anthropic.NewError(http.StatusBadRequest, "web_search tool is only supported for cloud models"))
				return
			}

			toolNames := make([]string, len(req.Tools))
			for i, t := range req.Tools {
				toolNames[i] = fmt.Sprintf("%s(%s)", t.Name, t.Type)
			}
			slog.Info("websearch: intercepting request",
				"model", req.Model,
				"stream", req.Stream,
				"messages_count", len(req.Messages),
				"tools", strings.Join(toolNames, ", "),
				"message_id", messageID,
			)

			c.Writer = &WebSearchAnthropicWriter{
				BaseWriter: BaseWriter{ResponseWriter: c.Writer},
				ctx:        c.Request.Context(),
				inner:      innerWriter,
				req:        req,
				chatReq:    chatReq,
				stream:     req.Stream,
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
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, d)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}

// serverToolUseID derives a server tool use ID from a message ID
func serverToolUseID(messageID string) string {
	return "srvtoolu_" + strings.TrimPrefix(messageID, "msg_")
}
