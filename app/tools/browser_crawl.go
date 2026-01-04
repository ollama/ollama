//go:build windows || darwin

package tools

import (
	"context"
	"encoding/json"
	"fmt"
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

	return g.performWebCrawl(ctx, urls)
}

// performWebCrawl handles the actual HTTP request to ollama.com crawl API
func (g *BrowserCrawler) performWebCrawl(ctx context.Context, urls []string) (*CrawlResponse, error) {
	result := &CrawlResponse{Results: make(map[string][]CrawlResult, len(urls))}

	for _, targetURL := range urls {
		fetchResp, err := performWebFetch(ctx, targetURL)
		if err != nil {
			return nil, fmt.Errorf("web_fetch failed for %q: %w", targetURL, err)
		}

		links := make([]CrawlLink, 0, len(fetchResp.Links))
		for _, link := range fetchResp.Links {
			links = append(links, CrawlLink{URL: link, Href: link})
		}

		snippet := truncateString(fetchResp.Content, 400)

		result.Results[targetURL] = []CrawlResult{{
			Title: fetchResp.Title,
			URL:   targetURL,
			Content: CrawlContent{
				Snippet:  snippet,
				FullText: fetchResp.Content,
			},
			Extras: CrawlExtras{Links: links},
		}}
	}

	return result, nil
}
