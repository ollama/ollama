package tools

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
	internalcloud "github.com/ollama/ollama/internal/cloud"
)

var (
	ErrWebSearchAuthRequired = errors.New("web search requires authentication")
	ErrWebFetchAuthRequired  = errors.New("web fetch requires authentication")
)

const (
	maxWebFetchContentRunes = 60_000
	webSearchTimeout        = 15 * time.Second
	webFetchTimeout         = 30 * time.Second
)

type WebSearch struct{}

func (w *WebSearch) Name() string {
	return "web_search"
}

func (w *WebSearch) Description() string {
	return "Search the web for current information that may not be in the model's training data."
}

func (w *WebSearch) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("query", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "The search query to look up on the web.",
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

func (w *WebSearch) Execute(ctx context.Context, _ agent.ToolContext, args map[string]any) (agent.ToolResult, error) {
	if internalcloud.Disabled() {
		return agent.ToolResult{}, errors.New(internalcloud.DisabledError("web search is unavailable"))
	}
	query, ok := args["query"].(string)
	if !ok || strings.TrimSpace(query) == "" {
		return agent.ToolResult{}, fmt.Errorf("query parameter is required")
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return agent.ToolResult{}, err
	}

	ctx, cancel := context.WithTimeout(ctx, webSearchTimeout)
	defer cancel()

	searchResp, err := client.WebSearchExperimental(ctx, &api.WebSearchRequest{Query: query, MaxResults: 5})
	if err != nil {
		var authErr api.AuthorizationError
		if errors.As(err, &authErr) {
			return agent.ToolResult{}, ErrWebSearchAuthRequired
		}
		return agent.ToolResult{}, err
	}
	if len(searchResp.Results) == 0 {
		return agent.ToolResult{Content: "No results found for query: " + query}, nil
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Search results for: %s\n\n", query))
	for i, result := range searchResp.Results {
		sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, result.Title))
		sb.WriteString(fmt.Sprintf("   URL: %s\n", result.URL))
		if result.Content != "" {
			content := []rune(result.Content)
			if len(content) > 300 {
				content = append(content[:300], []rune("...")...)
			}
			sb.WriteString(fmt.Sprintf("   %s\n", string(content)))
		}
		sb.WriteByte('\n')
	}
	return agent.ToolResult{Content: sb.String()}, nil
}

type WebFetch struct{}

func (w *WebFetch) Name() string {
	return "web_fetch"
}

func (w *WebFetch) Description() string {
	return "Fetch and extract text content from a web page."
}

func (w *WebFetch) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()
	props.Set("url", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "The URL to fetch and extract content from.",
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

func (w *WebFetch) Execute(ctx context.Context, _ agent.ToolContext, args map[string]any) (agent.ToolResult, error) {
	if internalcloud.Disabled() {
		return agent.ToolResult{}, errors.New(internalcloud.DisabledError("web fetch is unavailable"))
	}
	urlStr, ok := args["url"].(string)
	if !ok || strings.TrimSpace(urlStr) == "" {
		return agent.ToolResult{}, fmt.Errorf("url parameter is required")
	}
	if _, err := url.Parse(urlStr); err != nil {
		return agent.ToolResult{}, fmt.Errorf("invalid URL: %w", err)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return agent.ToolResult{}, err
	}

	ctx, cancel := context.WithTimeout(ctx, webFetchTimeout)
	defer cancel()

	fetchResp, err := client.WebFetchExperimental(ctx, &api.WebFetchRequest{URL: urlStr})
	if err != nil {
		var authErr api.AuthorizationError
		if errors.As(err, &authErr) {
			return agent.ToolResult{}, ErrWebFetchAuthRequired
		}
		return agent.ToolResult{}, err
	}

	var sb strings.Builder
	if fetchResp.Title != "" {
		sb.WriteString(fmt.Sprintf("Title: %s\n\n", fetchResp.Title))
	}
	if fetchResp.Content != "" {
		sb.WriteString("Content:\n")
		sb.WriteString(truncateWebFetchContent(fetchResp.Content))
	} else {
		sb.WriteString("No content could be extracted from the page.")
	}
	return agent.ToolResult{Content: sb.String()}, nil
}

func truncateWebFetchContent(content string) string {
	runes := []rune(content)
	if len(runes) <= maxWebFetchContentRunes {
		return content
	}
	omitted := len(runes) - maxWebFetchContentRunes
	return string(runes[:maxWebFetchContentRunes]) + fmt.Sprintf(
		"\n\n[tool output truncated: showing first ~%d tokens; omitted ~%d tokens. Use a narrower request or search query if more detail is needed.]",
		approximateToolTokensFromRunes(maxWebFetchContentRunes),
		approximateToolTokensFromRunes(omitted),
	)
}

func approximateToolTokensFromRunes(n int) int {
	if n <= 0 {
		return 0
	}
	return max(1, (n+3)/4)
}
