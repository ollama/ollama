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

type WebFetch struct{}

type FetchRequest struct {
	URL string `json:"url"`
}

type FetchResponse struct {
	Title   string   `json:"title"`
	Content string   `json:"content"`
	Links   []string `json:"links"`
}

func (w *WebFetch) Name() string {
	return "web_fetch"
}

func (w *WebFetch) Description() string {
	return "Crawl and extract text content from web pages"
}

func (g *WebFetch) Schema() map[string]any {
	schemaBytes := []byte(`{
		"type": "object",
		"properties": {
			"url": {
				"type": "string",
				"description": "URL to crawl and extract content from"
            }
		},
		"required": ["url"]
	}`)
	var schema map[string]any
	if err := json.Unmarshal(schemaBytes, &schema); err != nil {
		return nil
	}
	return schema
}

func (w *WebFetch) Prompt() string {
	return ""
}

func (w *WebFetch) Execute(ctx context.Context, args map[string]any) (any, string, error) {
	urlRaw, ok := args["url"]
	if !ok {
		return nil, "", fmt.Errorf("url parameter is required")
	}
	urlStr, ok := urlRaw.(string)
	if !ok || strings.TrimSpace(urlStr) == "" {
		return nil, "", fmt.Errorf("url must be a non-empty string")
	}

	result, err := performWebFetch(ctx, urlStr)
	if err != nil {
		return nil, "", err
	}

	return result, "", nil
}

func performWebFetch(ctx context.Context, targetURL string) (*FetchResponse, error) {
	reqBody := FetchRequest{URL: targetURL}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	crawlURL, err := url.Parse("https://ollama.com/api/web_fetch")
	if err != nil {
		return nil, fmt.Errorf("failed to parse fetch URL: %w", err)
	}

	query := crawlURL.Query()
	query.Add("ts", strconv.FormatInt(time.Now().Unix(), 10))
	crawlURL.RawQuery = query.Encode()

	data := fmt.Appendf(nil, "%s,%s", http.MethodPost, crawlURL.RequestURI())
	signature, err := auth.Sign(ctx, data)
	if err != nil {
		return nil, fmt.Errorf("failed to sign request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, crawlURL.String(), bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if signature != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", signature))
	}

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute fetch request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetch API error (status %d)", resp.StatusCode)
	}

	var result FetchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}
