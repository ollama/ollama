//go:build windows || darwin

package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/auth"
)

type WebSearch struct{}

type SearchRequest struct {
	Query      string `json:"query"`
	MaxResults int    `json:"max_results,omitempty"`
}

type SearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Content string `json:"content"`
}

type SearchResponse struct {
	Results []SearchResult `json:"results"`
}

func (w *WebSearch) Name() string {
	return "web_search"
}

func (w *WebSearch) Description() string {
	return "Search the web for real-time information using ollama.com web search API."
}

func (w *WebSearch) Prompt() string {
	return ""
}

func (g *WebSearch) Schema() map[string]any {
	schemaBytes := []byte(`{
		"type": "object",
		"properties": {
			"query": {
				"type": "string",
				"description": "The search query to execute"
			},
			"max_results": {
				"type": "integer",
				"description": "Maximum number of search results to return",
				"default": 3
			}
		},
		"required": ["query"]
	}`)
	var schema map[string]any
	if err := json.Unmarshal(schemaBytes, &schema); err != nil {
		return nil
	}
	return schema
}

func (w *WebSearch) Execute(ctx context.Context, args map[string]any) (any, string, error) {
	rawQuery, ok := args["query"]
	if !ok {
		return nil, "", fmt.Errorf("query parameter is required")
	}

	queryStr, ok := rawQuery.(string)
	if !ok || strings.TrimSpace(queryStr) == "" {
		return nil, "", fmt.Errorf("query must be a non-empty string")
	}

	maxResults := 5
	if v, ok := args["max_results"].(float64); ok && int(v) > 0 {
		maxResults = int(v)
	}

	result, err := performWebSearch(ctx, queryStr, maxResults)
	if err != nil {
		return nil, "", err
	}

	return result, "", nil
}

func performWebSearch(ctx context.Context, query string, maxResults int) (*SearchResponse, error) {
	reqBody := SearchRequest{Query: query, MaxResults: maxResults}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	searchURL, err := url.Parse("https://ollama.com/api/web_search")
	if err != nil {
		return nil, fmt.Errorf("failed to parse search URL: %w", err)
	}

	q := searchURL.Query()
	q.Add("ts", strconv.FormatInt(time.Now().Unix(), 10))
	searchURL.RawQuery = q.Encode()

	data := fmt.Appendf(nil, "%s,%s", http.MethodPost, searchURL.RequestURI())
	signature, err := auth.Sign(ctx, data)
	if err != nil {
		return nil, fmt.Errorf("failed to sign request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, searchURL.String(), bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if signature != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", signature))
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute search request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("search API error (status %d)", resp.StatusCode)
	}

	var result SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}
