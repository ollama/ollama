// Package keylesssearch provides a web search backend that requires no
// account, API key, or ollama.com sign-in. It scrapes DuckDuckGo's HTML
// endpoint.
//
// It is on by default: apps launched from Finder never see a shell
// environment, so an env-only opt-in would leave GUI builds on the quota-bound
// ollama.com path. Set OLLAMA_KEYLESS_WEB_SEARCH=0 (or false) to use
// ollama.com instead.
package keylesssearch

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"golang.org/x/net/html"
)

func lookupEnv(name string) (string, bool) {
	return os.LookupEnv(name)
}

// Enabled reports whether the keyless backend should replace the default
// ollama.com web search. It defaults to true; the env var only opts out.
func Enabled() bool {
	return envconfigKeyless()
}

var httpClient = &http.Client{Timeout: 10 * time.Second}

// SearchFunc runs a keyless search. Tests replace Search to run without
// network access, mirroring how the cloud proxy's seams are swappable.
type SearchFunc func(ctx context.Context, query string, maxResults int) (*api.WebSearchResponse, error)

// Search runs a DuckDuckGo HTML search. It is a variable so tests can stub it.
var Search SearchFunc = duckDuckGoSearch

// duckDuckGoSearch converts the top results to the shared
// api.WebSearchResponse shape so all existing call sites can consume it
// unchanged.
func duckDuckGoSearch(ctx context.Context, query string, maxResults int) (*api.WebSearchResponse, error) {
	if maxResults <= 0 {
		maxResults = 5
	}

	form := url.Values{}
	form.Set("q", query)
	form.Set("kl", "wt-wt") // no region preference

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://html.duckduckgo.com/html/", strings.NewReader(form.Encode()))
	if err != nil {
		return nil, fmt.Errorf("keyless search: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("keyless search: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("keyless search: status %d", resp.StatusCode)
	}

	results, err := parseResults(resp.Body, maxResults)
	if err != nil {
		return nil, fmt.Errorf("keyless search: %w", err)
	}
	return &api.WebSearchResponse{Results: results}, nil
}

// parseResults extracts links from the DuckDuckGo HTML results page. Results
// are <a class="result__a"> anchors whose href is an uddg= redirect wrapper.
func parseResults(r interface{ Read([]byte) (int, error) }, maxResults int) ([]api.WebSearchResult, error) {
	doc, err := html.Parse(r)
	if err != nil {
		return nil, err
	}

	var results []api.WebSearchResult
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "a" {
			class := attr(n, "class")
			if class == "result__a" || strings.Contains(class, "result__a") {
				href := attr(n, "href")
				if target, ok := unwrapDDG(href); ok && target != "" {
					title := strings.TrimSpace(text(n))
					if title != "" {
						results = append(results, api.WebSearchResult{
							Title: title,
							URL:   target,
						})
						if len(results) >= maxResults {
							return
						}
					}
				}
			}
		}
		for c := n.FirstChild; c != nil && len(results) < maxResults; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)
	return results, nil
}

// unwrapDDG extracts the real target from /l/?uddg=<encoded-url> redirects.
func unwrapDDG(href string) (string, bool) {
	if href == "" {
		return "", false
	}
	u, err := url.Parse(href)
	if err != nil {
		return "", false
	}
	if u.Query().Has("uddg") {
		target, err := url.QueryUnescape(u.Query().Get("uddg"))
		if err != nil {
			return "", false
		}
		return target, true
	}
	if u.Host != "" {
		return href, true
	}
	return "", false
}

func attr(n *html.Node, key string) string {
	for _, a := range n.Attr {
		if a.Key == key {
			return a.Val
		}
	}
	return ""
}

func text(n *html.Node) string {
	if n.Type == html.TextNode {
		return n.Data
	}
	var b strings.Builder
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		b.WriteString(text(c))
	}
	return b.String()
}

// envconfigKeyless reads OLLAMA_KEYLESS_WEB_SEARCH without importing
// envconfig, keeping this package dependency-light. Absent means enabled.
func envconfigKeyless() bool {
	if _, ok := lookupEnv("OLLAMA_KEYLESS_WEB_SEARCH"); !ok {
		return true
	}
	return envBool("OLLAMA_KEYLESS_WEB_SEARCH")
}

func envBool(name string) bool {
	v, ok := lookupEnv(name)
	if !ok {
		return false
	}
	var b bool
	if err := json.Unmarshal([]byte(v), &b); err != nil {
		// accept bare "1"/"true"/"yes" style values leniently
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "1", "true", "yes", "on":
			return true
		}
		return false
	}
	return b
}
