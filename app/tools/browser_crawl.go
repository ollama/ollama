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

// CrawlContent represents the content of a crawled page
type CrawlContent struct {
	Snippet  string `json:"snippet"`
	FullText string `json:"full_text"`
}

// CrawlExtras represents additional data from the crawl API
type CrawlExtras struct {
	Links []CrawlLink `json:"links"`
}

// CrawlLink represents a link found on a crawled page
type CrawlLink struct {
	URL  string `json:"url"`
	Href string `json:"href"`
	Text string `json:"text"`
}

// CrawlResult represents a single crawl result
type CrawlResult struct {
	Title   string       `json:"title"`
	URL     string       `json:"url"`
	Content CrawlContent `json:"content"`
	Extras  CrawlExtras  `json:"extras"`
}

// CrawlResponse represents the complete response from the crawl API
type CrawlResponse struct {
	Results map[string][]CrawlResult `json:"results"`
}

// BrowserCrawler tool for crawling web pages using ollama.com crawl API
type BrowserCrawler struct{}

func (g *BrowserCrawler) Name() string {
	return "get_webpage"
}

func (g *BrowserCrawler) Description() string {
	return "Crawl and extract text content from web pages"
}
func (g *BrowserCrawler) Prompt() string {
	return `When you need to read content from web pages, use the get_webpage tool. Simply provide the URLs you want to read and I'll fetch their content for you.

For each URL, I'll extract the main text content in a readable format. If you need to discover links within those pages, set extract_links to true. If the user requires the latest information, set livecrawl to true.

Only use this tool when you need to access current web content. Make sure the URLs are valid and accessible. Do not use this tool for:
- Downloading files or media
- Accessing private/authenticated pages
- Scraping data at high volumes

Always check the returned content to ensure it's relevant before using it in your response.`
}

func (g *BrowserCrawler) Schema() map[string]any {
	schemaBytes := []byte(`{
		"type": "object",
		"properties": {
			"urls": {
				"type": "array",
				"items": {
					"type": "string"
				},
				"description": "List of URLs to crawl and extract content from"
			},
			"latest": {
				"type": "boolean",
				"description": " Needs up to date and latest information (default: false)",
				"default": false
			}
		},
		"required": ["urls"]
	}`)
	var schema map[string]any
	if err := json.Unmarshal(schemaBytes, &schema); err != nil {
		return nil
	}
	return schema
}

func (g *BrowserCrawler) Execute(ctx context.Context, args map[string]any) (*CrawlResponse, error) {
	// Extract and validate URLs
	urlsRaw, ok := args["urls"].([]any)
	if !ok {
		return nil, fmt.Errorf("urls parameter is required and must be an array of strings")
	}

	urls := make([]string, 0, len(urlsRaw))
	for _, u := range urlsRaw {
		if urlStr, ok := u.(string); ok {
			urls = append(urls, urlStr)
		}
	}

	if len(urls) == 0 {
		return nil, fmt.Errorf("at least one URL is required")
	}

	latest, _ := args["latest"].(bool)

	// Perform the web crawling
	return g.performWebCrawl(ctx, urls, latest)
}

// performWebCrawl handles the actual HTTP request to ollama.com crawl API
func (g *BrowserCrawler) performWebCrawl(ctx context.Context, urls []string, latest bool) (*CrawlResponse, error) {
	// Prepare the request body matching the API format
	reqBody := map[string]any{
		"urls": urls,
		"text": true,
		"extras": map[string]any{
			"links": 1,
		},
		"livecrawl": "fallback",
	}

	if latest {
		reqBody["livecrawl"] = "always"
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	crawlURL, err := url.Parse("https://ollama.com/api/tools/webcrawl")
	if err != nil {
		return nil, fmt.Errorf("failed to parse crawl URL: %w", err)
	}

	// Add timestamp for signing
	query := crawlURL.Query()
	query.Add("ts", strconv.FormatInt(time.Now().Unix(), 10))

	var signature string

	crawlURL.RawQuery = query.Encode()

	// Sign the request data (method + URI)
	data := fmt.Appendf(nil, "%s,%s", http.MethodPost, crawlURL.RequestURI())
	signature, err = auth.Sign(ctx, data)
	if err != nil {
		return nil, fmt.Errorf("failed to sign request: %w", err)
	}

	// Create the request
	req, err := http.NewRequestWithContext(ctx, "POST", crawlURL.String(), bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	if signature != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", signature))
	}

	// Make the request
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute crawl request: %w", err)
	}
	defer resp.Body.Close()

	// Read and parse response
	var result CrawlResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Check for error response
	if resp.StatusCode != http.StatusOK {
		errMsg := "unknown error"
		if resp.StatusCode == http.StatusServiceUnavailable {
			errMsg = "crawl service unavailable - API key may not be configured"
		}
		return nil, fmt.Errorf("crawl API error (status %d): %s", resp.StatusCode, errMsg)
	}

	return &result, nil
}
