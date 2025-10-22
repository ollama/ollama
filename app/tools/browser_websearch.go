package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/ollama/ollama/auth"
)

// WebSearchContent represents the content of a search result
type WebSearchContent struct {
	Snippet  string `json:"snippet"`
	FullText string `json:"full_text"`
}

// WebSearchMetadata represents metadata for a search result
type WebSearchMetadata struct {
	PublishedDate *time.Time `json:"published_date,omitempty"`
}

// WebSearchResult represents a single search result
type WebSearchResult struct {
	Title    string            `json:"title"`
	URL      string            `json:"url"`
	Content  WebSearchContent  `json:"content"`
	Metadata WebSearchMetadata `json:"metadata"`
}

// WebSearchResponse represents the complete response from the websearch API
type WebSearchResponse struct {
	Results map[string][]WebSearchResult `json:"results"`
}

// BrowserWebSearch tool for searching the web using ollama.com search API
type BrowserWebSearch struct{}

func (w *BrowserWebSearch) Name() string {
	return "gpt_oss_web_search"
}

func (w *BrowserWebSearch) Description() string {
	return "Search the web for real-time information using ollama.com search API."
}

func (w *BrowserWebSearch) Prompt() string {
	return `Use the gpt_oss_web_search tool to search the web.
1. Come up with a list of search queries to get comprehensive information (typically 2-3 related queries work well)
2. Use the gpt_oss_web_search tool with multiple queries to get results organized by query
3. Use the search results to provide current up to date, accurate information

Today's date is ` + time.Now().Format("January 2, 2006") + `
Add "` + time.Now().Format("January 2, 2006") + `" for news queries and ` + strconv.Itoa(time.Now().Year()+1) + ` for other queries that need current information.`
}

func (w *BrowserWebSearch) Schema() map[string]any {
	schemaBytes := []byte(`{
		"type": "object",
		"properties": {
			"queries": {
				"type": "array",
				"items": {
					"type": "string"
				},
				"description": "List of search queries to look up"
			},
			"max_results": {
				"type": "integer",
				"description": "Maximum number of results to return per query (default: 2) up to 5",
				"default": 2
			}
		},
		"required": ["queries"]
	}`)
	var schema map[string]any
	if err := json.Unmarshal(schemaBytes, &schema); err != nil {
		return nil
	}
	return schema
}

func (w *BrowserWebSearch) Execute(ctx context.Context, args map[string]any) (any, error) {
	// Extract and validate queries
	queriesRaw, ok := args["queries"].([]any)
	if !ok {
		return nil, fmt.Errorf("queries parameter is required and must be an array of strings")
	}

	queries := make([]string, 0, len(queriesRaw))
	for _, q := range queriesRaw {
		if query, ok := q.(string); ok {
			queries = append(queries, query)
		}
	}

	if len(queries) == 0 {
		return nil, fmt.Errorf("at least one query is required")
	}

	// Get optional parameters
	maxResults := 5
	if mr, ok := args["max_results"].(int); ok {
		maxResults = mr
	}

	// Perform the web search
	return w.performWebSearch(ctx, queries, maxResults)
}

// performWebSearch handles the actual HTTP request to ollama.com search API
func (w *BrowserWebSearch) performWebSearch(ctx context.Context, queries []string, maxResults int) (*WebSearchResponse, error) {
	// Prepare the request body
	reqBody := map[string]any{
		"queries":     queries,
		"max_results": maxResults,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	searchURL, err := url.Parse("https://ollama.com/api/tools/websearch")
	if err != nil {
		return nil, fmt.Errorf("failed to parse search URL: %w", err)
	}

	// Add timestamp for signing
	query := searchURL.Query()
	query.Add("ts", strconv.FormatInt(time.Now().Unix(), 10))

	var signature string

	searchURL.RawQuery = query.Encode()

	// Sign the request data (method + URI)
	data := fmt.Appendf(nil, "%s,%s", http.MethodPost, searchURL.RequestURI())
	signature, err = auth.Sign(ctx, data)
	if err != nil {
		return nil, fmt.Errorf("failed to sign request: %w", err)
	}

	// Create the request
	req, err := http.NewRequestWithContext(ctx, "POST", searchURL.String(), bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	if signature != "" {
		req.Header.Set("Authorization", signature)
	}

	// Make the request
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute search request: %w", err)
	}
	defer resp.Body.Close()

	// Read and parse response
	var result WebSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Check for error response
	if resp.StatusCode != http.StatusOK {
		errMsg := "unknown error"
		if resp.StatusCode == http.StatusServiceUnavailable {
			errMsg = "search service unavailable - API key may not be configured"
		}
		return nil, fmt.Errorf("search API error (status %d): %s", resp.StatusCode, errMsg)
	}

	// Return the results directly without caching
	return &result, nil
}
