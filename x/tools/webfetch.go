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
	webFetchAPI     = "https://ollama.com/api/web_fetch"
	webFetchTimeout = 30 * time.Second
)

// ErrWebFetchAuthRequired is returned when web fetch requires authentication
var ErrWebFetchAuthRequired = errors.New("web fetch requires authentication")

// WebFetchTool implements web page fetching using Ollama's hosted API.
type WebFetchTool struct{}

// Name returns the tool name.
func (w *WebFetchTool) Name() string {
	return "web_fetch"
}

// Description returns a description of the tool.
func (w *WebFetchTool) Description() string {
	return "Fetch and extract text content from a web page. Use this to read the full content of a URL found in search results or provided by the user."
}

// Schema returns the tool's parameter schema.
func (w *WebFetchTool) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("url", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "The URL to fetch and extract content from",
	})
	return api.ToolFunction{
		Name:        w.Name(),
		Description: w.Description(),
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Properties: props,
			Required:   []string{"url"},
		},
	}
}

// webFetchRequest is the request body for the web fetch API.
type webFetchRequest struct {
	URL string `json:"url"`
}

// webFetchResponse is the response from the web fetch API.
type webFetchResponse struct {
	Title   string   `json:"title"`
	Content string   `json:"content"`
	Links   []string `json:"links,omitempty"`
}

// Execute fetches content from a web page.
// Uses Ollama key signing for authentication - this makes requests via ollama.com API.
func (w *WebFetchTool) Execute(args map[string]any) (string, error) {
	urlStr, ok := args["url"].(string)
	if !ok || urlStr == "" {
		return "", fmt.Errorf("url parameter is required")
	}

	// Validate URL
	if _, err := url.Parse(urlStr); err != nil {
		return "", fmt.Errorf("invalid URL: %w", err)
	}

	// Prepare request
	reqBody := webFetchRequest{
		URL: urlStr,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshaling request: %w", err)
	}

	// Parse URL and add timestamp for signing
	fetchURL, err := url.Parse(webFetchAPI)
	if err != nil {
		return "", fmt.Errorf("parsing fetch URL: %w", err)
	}

	q := fetchURL.Query()
	q.Add("ts", strconv.FormatInt(time.Now().Unix(), 10))
	fetchURL.RawQuery = q.Encode()

	// Sign the request using Ollama key (~/.ollama/id_ed25519)
	ctx := context.Background()
	data := fmt.Appendf(nil, "%s,%s", http.MethodPost, fetchURL.RequestURI())
	signature, err := auth.Sign(ctx, data)
	if err != nil {
		return "", fmt.Errorf("signing request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fetchURL.String(), bytes.NewBuffer(jsonBody))
	if err != nil {
		return "", fmt.Errorf("creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if signature != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", signature))
	}

	// Send request
	client := &http.Client{Timeout: webFetchTimeout}
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
		return "", ErrWebFetchAuthRequired
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("web fetch API returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var fetchResp webFetchResponse
	if err := json.Unmarshal(body, &fetchResp); err != nil {
		return "", fmt.Errorf("parsing response: %w", err)
	}

	// Format result
	var sb strings.Builder
	if fetchResp.Title != "" {
		sb.WriteString(fmt.Sprintf("Title: %s\n\n", fetchResp.Title))
	}

	if fetchResp.Content != "" {
		sb.WriteString("Content:\n")
		sb.WriteString(fetchResp.Content)
	} else {
		sb.WriteString("No content could be extracted from the page.")
	}

	return sb.String(), nil
}
