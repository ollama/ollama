package middleware

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/api"
)

// AnthropicWriter wraps the response writer to transform Ollama responses to Anthropic format
type AnthropicWriter struct {
	BaseWriter
	stream    bool
	id        string
	model     string
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

		// Check if this is a web search request
		if webSearchResult := handleWebSearchTool(c, req); webSearchResult != nil {
			// Web search was handled, response already sent
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

		w := &AnthropicWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			stream:     req.Stream,
			id:         messageID,
			model:      req.Model,
			converter:  anthropic.NewStreamConverter(messageID, req.Model, estimatedTokens),
		}

		if req.Stream {
			c.Writer.Header().Set("Content-Type", "text/event-stream")
			c.Writer.Header().Set("Cache-Control", "no-cache")
			c.Writer.Header().Set("Connection", "keep-alive")
		}

		c.Writer = w

		c.Next()
	}
}

// handleWebSearchTool checks if the request contains a web_search tool and handles it
// Returns non-nil if web search was handled (response already sent)
func handleWebSearchTool(c *gin.Context, req anthropic.MessagesRequest) *anthropic.MessagesResponse {
	// Check if there's a web_search tool
	var hasWebSearch bool
	var maxUses int
	for _, tool := range req.Tools {
		if strings.HasPrefix(tool.Type, "web_search") {
			hasWebSearch = true
			maxUses = tool.MaxUses
			if maxUses == 0 {
				maxUses = 5
			}
			break
		}
	}

	if !hasWebSearch {
		return nil
	}

	// Extract the search query from the last user message
	query := extractSearchQuery(req.Messages)
	if query == "" {
		slog.Debug("web search tool present but no query found")
		return nil
	}

	slog.Debug("executing web search", "query", query)

	// Get API key from environment or request context
	apiKey := os.Getenv("OLLAMA_API_KEY")

	// Call the Ollama web search API
	searchResp, err := anthropic.DoWebSearch(c.Request.Context(), apiKey, query, maxUses)
	if err != nil {
		slog.Error("web search failed", "error", err)
		// Return error response in Anthropic format
		sendWebSearchError(c, req, "unavailable")
		return &anthropic.MessagesResponse{} // Non-nil to indicate handled
	}

	// Convert and send the response
	sendWebSearchResponse(c, req, query, searchResp)
	return &anthropic.MessagesResponse{} // Non-nil to indicate handled
}

// extractSearchQuery extracts the search query from the messages
func extractSearchQuery(messages []anthropic.MessageParam) string {
	// Look at the last user message for the query
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		if msg.Role != "user" {
			continue
		}

		switch content := msg.Content.(type) {
		case string:
			// Try to extract query from "Perform a web search for the query: X" pattern
			if strings.Contains(content, "web search") {
				if idx := strings.Index(content, ": "); idx != -1 {
					return strings.TrimSpace(content[idx+2:])
				}
				return content
			}
		case []any:
			for _, block := range content {
				if blockMap, ok := block.(map[string]any); ok {
					if blockMap["type"] == "text" {
						if text, ok := blockMap["text"].(string); ok {
							if strings.Contains(strings.ToLower(text), "web search") || strings.Contains(strings.ToLower(text), "search for") {
								// Try to extract query from pattern
								if idx := strings.Index(text, ": "); idx != -1 {
									return strings.TrimSpace(text[idx+2:])
								}
								return text
							}
						}
					}
				}
			}
		}
	}
	return ""
}

// sendWebSearchError sends an error response for web search
func sendWebSearchError(c *gin.Context, req anthropic.MessagesRequest, errorCode string) {
	messageID := anthropic.GenerateMessageID()
	toolUseID := "srvtoolu_" + messageID[4:] // Convert msg_ to srvtoolu_

	response := anthropic.MessagesResponse{
		ID:    messageID,
		Type:  "message",
		Role:  "assistant",
		Model: req.Model,
		Content: []anthropic.ContentBlock{
			{
				Type: "web_search_tool_result",
				ToolUseID: toolUseID,
				Content: anthropic.WebSearchToolResultError{
					Type:      "web_search_tool_result_error",
					ErrorCode: errorCode,
				},
			},
		},
		StopReason: "end_turn",
		Usage: anthropic.Usage{
			InputTokens:  0,
			OutputTokens: 0,
		},
	}

	if req.Stream {
		sendStreamingWebSearchResponse(c, response)
	} else {
		c.JSON(http.StatusOK, response)
	}
}

// sendWebSearchResponse sends a successful web search response
func sendWebSearchResponse(c *gin.Context, req anthropic.MessagesRequest, query string, searchResp *anthropic.OllamaWebSearchResponse) {
	messageID := anthropic.GenerateMessageID()
	toolUseID := "srvtoolu_" + messageID[4:] // Convert msg_ to srvtoolu_

	// Convert Ollama results to Anthropic format
	var searchResults []anthropic.WebSearchResult
	for _, r := range searchResp.Results {
		searchResults = append(searchResults, anthropic.WebSearchResult{
			Type:    "web_search_result",
			URL:     r.URL,
			Title:   r.Title,
			PageAge: "", // Ollama API doesn't provide this
		})
	}

	response := anthropic.MessagesResponse{
		ID:    messageID,
		Type:  "message",
		Role:  "assistant",
		Model: req.Model,
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
				Content:   searchResults,
			},
			{
				Type: "text",
				Text: ptr(formatSearchResultsAsText(searchResp)),
			},
		},
		StopReason: "end_turn",
		Usage: anthropic.Usage{
			InputTokens:  100, // Approximate
			OutputTokens: 50,  // Approximate
		},
	}

	if req.Stream {
		sendStreamingWebSearchResponse(c, response)
	} else {
		c.JSON(http.StatusOK, response)
	}
}

// formatSearchResultsAsText formats search results as readable text
func formatSearchResultsAsText(searchResp *anthropic.OllamaWebSearchResponse) string {
	if len(searchResp.Results) == 0 {
		return "No search results found."
	}

	var sb strings.Builder
	sb.WriteString("Here are the search results:\n\n")
	for i, r := range searchResp.Results {
		sb.WriteString(fmt.Sprintf("%d. **%s**\n   %s\n", i+1, r.Title, r.URL))
		if r.Content != "" {
			// Truncate content to first 200 chars
			content := r.Content
			if len(content) > 200 {
				content = content[:200] + "..."
			}
			sb.WriteString(fmt.Sprintf("   %s\n", content))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// sendStreamingWebSearchResponse sends a streaming SSE response for web search
func sendStreamingWebSearchResponse(c *gin.Context, response anthropic.MessagesResponse) {
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")

	// Send message_start
	writeSSE(c.Writer, "message_start", anthropic.MessageStartEvent{
		Type: "message_start",
		Message: anthropic.MessagesResponse{
			ID:      response.ID,
			Type:    "message",
			Role:    "assistant",
			Model:   response.Model,
			Content: []anthropic.ContentBlock{},
			Usage:   response.Usage,
		},
	})

	// Send content blocks
	for i, block := range response.Content {
		// content_block_start
		writeSSE(c.Writer, "content_block_start", anthropic.ContentBlockStartEvent{
			Type:         "content_block_start",
			Index:        i,
			ContentBlock: block,
		})

		// content_block_stop
		writeSSE(c.Writer, "content_block_stop", anthropic.ContentBlockStopEvent{
			Type:  "content_block_stop",
			Index: i,
		})
	}

	// Send message_delta
	writeSSE(c.Writer, "message_delta", anthropic.MessageDeltaEvent{
		Type: "message_delta",
		Delta: anthropic.MessageDelta{
			StopReason: response.StopReason,
		},
		Usage: anthropic.DeltaUsage{
			OutputTokens: response.Usage.OutputTokens,
		},
	})

	// Send message_stop
	writeSSE(c.Writer, "message_stop", anthropic.MessageStopEvent{
		Type: "message_stop",
	})
}

// writeSSE writes a Server-Sent Event
func writeSSE(w http.ResponseWriter, eventType string, data any) {
	d, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, d)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// ptr returns a pointer to a string
func ptr(s string) *string {
	return &s
}
