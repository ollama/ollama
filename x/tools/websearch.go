package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
)

const (
	webSearchAPI     = "https://ollama.com/api/web_search"
	webSearchTimeout = 15 * time.Second
)

// ErrWebSearchAuthRequired is returned when web search requires authentication
var ErrWebSearchAuthRequired = errors.New("web search requires authentication")

// WebSearchTool implements web search using Ollama's hosted API.
type WebSearchTool struct{}

// Name returns the tool name.
func (w *WebSearchTool) Name() string {
	return "web_search"
}

// Description returns a description of the tool.
func (w *WebSearchTool) Description() string {
	return "Search the web for current information. Use this when you need up-to-date information that may not be in your training data."
}

// Schema returns the tool's parameter schema.
func (w *WebSearchTool) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("query", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "The search query to look up on the web",
	})
	return api.ToolFunction{
		Name:        w.Name(),
		Description: w.Description(),
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Properties: props,
			Required:   []string{"query"},
		},
	}
}

// webSearchRequest is the request body for the web search API.
type webSearchRequest struct {
	Query      string `json:"query"`
	MaxResults int    `json:"max_results,omitempty"`
}

// webSearchResponse is the response from the web search API.
type webSearchResponse struct {
	Results []webSearchResult `json:"results"`
}

// webSearchResult is a single search result.
type webSearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Content string `json:"content"`
}

// Execute performs the web search.
// Uses Ollama key signing for authentication - this makes requests via ollama.com API.
func (w *WebSearchTool) Execute(args map[string]any) (string, error) {
	query, ok := args["query"].(string)
	if !ok || query == "" {
		return "", fmt.Errorf("query parameter is required")
	}

	// Prepare request
	reqBody := webSearchRequest{
		Query:      query,
		MaxResults: 5,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshaling request: %w", err)
	}

	// Parse URL and add timestamp for signing
	searchURL, err := url.Parse(webSearchAPI)
	if err != nil {
		return "", fmt.Errorf("parsing search URL: %w", err)
	}

	q := searchURL.Query()
	q.Add("ts", strconv.FormatInt(time.Now().Unix(), 10))
	searchURL.RawQuery = q.Encode()

	// Sign the request using Ollama key (~/.ollama/id_ed25519)
	// This authenticates with ollama.com using the local signing key
	ctx := context.Background()
	data := fmt.Appendf(nil, "%s,%s", http.MethodPost, searchURL.RequestURI())
	signature, err := auth.Sign(ctx, data)
	if err != nil {
		return "", fmt.Errorf("signing request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, searchURL.String(), bytes.NewBuffer(jsonBody))
	if err != nil {
		return "", fmt.Errorf("creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if signature != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", signature))
	}

	// Send request
	client := &http.Client{Timeout: webSearchTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("sending request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode == http.StatusUnauthorized {
		return "", ErrWebSearchAuthRequired
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("web search API returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var searchResp webSearchResponse
	if err := json.Unmarshal(body, &searchResp); err != nil {
		return "", fmt.Errorf("parsing response: %w", err)
	}

	// Format results
	if len(searchResp.Results) == 0 {
		return "No results found for query: " + query, nil
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Search results for: %s\n\n", query))

	for i, result := range searchResp.Results {
		sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, result.Title))
		sb.WriteString(fmt.Sprintf("   URL: %s\n", result.URL))
		if result.Content != "" {
			// Truncate long content (UTF-8 safe)
			content := result.Content
			runes := []rune(content)
			if len(runes) > 300 {
				content = string(runes[:300]) + "..."
			}
			sb.WriteString(fmt.Sprintf("   %s\n", content))
		}
		sb.WriteString("\n")
	}

	return sb.String(), nil
}
