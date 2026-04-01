//go:build windows || darwin

package tools

import (
	"context"
	"fmt"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/app/ui/responses"
)

type PageType string

const (
	PageTypeSearchResults PageType = "initial_results"
	PageTypeWebpage       PageType = "webpage"
)

// DefaultViewTokens is the number of tokens to show to the model used when calling displayPage
const DefaultViewTokens = 1024

/*
The Browser tool provides web browsing capability for gpt-oss.
The model uses the tool by usually doing a search first and then choosing to either open a page,
find a term in a page, or do another search.

The tool optionally may open a URL directly - especially if one is passed in.

Each action is saved into an append-only page stack `responses.BrowserStateData` to keep
track of the history of the browsing session.

Each `Execute()` for a tool returns the full current state of the browser. ui.go manages the
browser state representation between the tool, ui, and db.

A new Browser object is created per request - the state is reconstructed by ui.go.
The initialization of the browser will receive a `responses.BrowserStateData` with the stitched history.
*/

// BrowserState manages the browsing session on a per-chat basis
type BrowserState struct {
	mu   sync.RWMutex
	Data *responses.BrowserStateData
}
type Browser struct {
	state *BrowserState
}

// State is only accessed in a single thread, as each chat has its own browser state
func (b *Browser) State() *responses.BrowserStateData {
	b.state.mu.RLock()
	defer b.state.mu.RUnlock()
	return b.state.Data
}

func (b *Browser) savePage(page *responses.Page) {
	b.state.Data.URLToPage[page.URL] = page
	b.state.Data.PageStack = append(b.state.Data.PageStack, page.URL)
}

func (b *Browser) getPageFromStack(url string) (*responses.Page, error) {
	page, ok := b.state.Data.URLToPage[url]
	if !ok {
		return nil, fmt.Errorf("page not found for url %s", url)
	}
	return page, nil
}

func NewBrowser(state *responses.BrowserStateData) *Browser {
	if state == nil {
		state = &responses.BrowserStateData{
			PageStack:  []string{},
			ViewTokens: DefaultViewTokens,
			URLToPage:  make(map[string]*responses.Page),
		}
	}
	b := &BrowserState{
		Data: state,
	}

	return &Browser{
		state: b,
	}
}

type BrowserSearch struct {
	Browser
	webSearch *BrowserWebSearch
}

// NewBrowserSearch creates a new browser search instance
func NewBrowserSearch(bb *Browser) *BrowserSearch {
	if bb == nil {
		bb = &Browser{
			state: &BrowserState{
				Data: &responses.BrowserStateData{
					PageStack:  []string{},
					ViewTokens: DefaultViewTokens,
					URLToPage:  make(map[string]*responses.Page),
				},
			},
		}
	}
	return &BrowserSearch{
		Browser:   *bb,
		webSearch: &BrowserWebSearch{},
	}
}

func (b *BrowserSearch) Name() string {
	return "browser.search"
}

func (b *BrowserSearch) Description() string {
	return "Search the web for information"
}

func (b *BrowserSearch) Prompt() string {
	return ""
}

func (b *BrowserSearch) Schema() map[string]any {
	return map[string]any{}
}

func (b *BrowserSearch) Execute(ctx context.Context, args map[string]any) (any, string, error) {
	query, ok := args["query"].(string)
	if !ok {
		return nil, "", fmt.Errorf("query parameter is required")
	}

	topn, ok := args["topn"].(int)
	if !ok {
		topn = 5
	}

	searchArgs := map[string]any{
		"queries":     []any{query},
		"max_results": topn,
	}

	result, err := b.webSearch.Execute(ctx, searchArgs)
	if err != nil {
		return nil, "", fmt.Errorf("search error: %w", err)
	}

	searchResponse, ok := result.(*WebSearchResponse)
	if !ok {
		return nil, "", fmt.Errorf("invalid search results format")
	}

	// Build main search results page that contains all search results
	searchResultsPage := b.buildSearchResultsPageCollection(query, searchResponse)
	b.savePage(searchResultsPage)
	cursor := len(b.state.Data.PageStack) - 1
	// cache result for each page
	for _, queryResults := range searchResponse.Results {
		for i, result := range queryResults {
			resultPage := b.buildSearchResultsPage(&result, i+1)
			// save to global only, do not add to visited stack
			b.state.Data.URLToPage[resultPage.URL] = resultPage
		}
	}

	page := searchResultsPage

	pageText, err := b.displayPage(page, cursor, 0, -1)
	if err != nil {
		return nil, "", fmt.Errorf("failed to display page: %w", err)
	}

	return b.state.Data, pageText, nil
}

func (b *Browser) buildSearchResultsPageCollection(query string, results *WebSearchResponse) *responses.Page {
	page := &responses.Page{
		URL:       "search_results_" + query,
		Title:     query,
		Links:     make(map[int]string),
		FetchedAt: time.Now(),
	}

	var textBuilder strings.Builder
	linkIdx := 0

	// Add the header lines to match format
	textBuilder.WriteString("\n")                 // L0: empty
	textBuilder.WriteString("URL: \n")            // L1: URL: (empty for search)
	textBuilder.WriteString("# Search Results\n") // L2: # Search Results
	textBuilder.WriteString("\n")                 // L3: empty

	for _, queryResults := range results.Results {
		for _, result := range queryResults {
			domain := result.URL
			if u, err := url.Parse(result.URL); err == nil && u.Host != "" {
				domain = u.Host
				domain = strings.TrimPrefix(domain, "www.")
			}

			linkFormat := fmt.Sprintf("* 【%d†%s†%s】", linkIdx, result.Title, domain)
			textBuilder.WriteString(linkFormat)

			numChars := min(len(result.Content.FullText), 400)
			snippet := strings.TrimSpace(result.Content.FullText[:numChars])
			textBuilder.WriteString(snippet)
			textBuilder.WriteString("\n")

			page.Links[linkIdx] = result.URL
			linkIdx++
		}
	}

	page.Text = textBuilder.String()
	page.Lines = wrapLines(page.Text, 80)

	return page
}

func (b *Browser) buildSearchResultsPage(result *WebSearchResult, linkIdx int) *responses.Page {
	page := &responses.Page{
		URL:       result.URL,
		Title:     result.Title,
		Links:     make(map[int]string),
		FetchedAt: time.Now(),
	}

	var textBuilder strings.Builder

	// Format the individual result page (only used when no full text is available)
	linkFormat := fmt.Sprintf("【%d†%s】", linkIdx, result.Title)
	textBuilder.WriteString(linkFormat)
	textBuilder.WriteString("\n")
	textBuilder.WriteString(fmt.Sprintf("URL: %s\n", result.URL))
	numChars := min(len(result.Content.FullText), 300)
	textBuilder.WriteString(result.Content.FullText[:numChars])
	textBuilder.WriteString("\n\n")

	// Only store link and snippet if we won't be processing full text later
	// (full text processing will handle all links consistently)
	if result.Content.FullText == "" {
		page.Links[linkIdx] = result.URL
	}

	// Use full text if available, otherwise use snippet
	if result.Content.FullText != "" {
		// Prepend the URL line to the full text
		page.Text = fmt.Sprintf("URL: %s\n%s", result.URL, result.Content.FullText)
		// Process markdown links in the full text
		processedText, processedLinks := processMarkdownLinks(page.Text)
		page.Text = processedText
		page.Links = processedLinks
	} else {
		page.Text = textBuilder.String()
	}

	page.Lines = wrapLines(page.Text, 80)

	return page
}

// getEndLoc calculates the end location for viewport based on token limits
func (b *Browser) getEndLoc(loc, numLines, totalLines int, lines []string) int {
	if numLines <= 0 {
		// Auto-calculate based on viewTokens
		txt := b.joinLinesWithNumbers(lines[loc:])

		// If text is very short, no need to truncate (at least 1 char per token)
		if len(txt) > b.state.Data.ViewTokens {
			// Simple heuristic: approximate token counting
			// Typical token is ~4 characters, but can be up to 128 chars
			maxCharsPerToken := 128

			// upper bound for text to analyze
			upperBound := min((b.state.Data.ViewTokens+1)*maxCharsPerToken, len(txt))
			textToAnalyze := txt[:upperBound]

			// Simple approximation: count tokens as ~4 chars each
			// This is less accurate than tiktoken but more performant
			approxTokens := len(textToAnalyze) / 4

			if approxTokens > b.state.Data.ViewTokens {
				// Find the character position at viewTokens
				endIdx := min(b.state.Data.ViewTokens*4, len(txt))

				// Count newlines up to that position to get line count
				numLines = strings.Count(txt[:endIdx], "\n") + 1
			} else {
				numLines = totalLines
			}
		} else {
			numLines = totalLines
		}
	}

	return min(loc+numLines, totalLines)
}

// joinLinesWithNumbers creates a string with line numbers, matching Python's join_lines
func (b *Browser) joinLinesWithNumbers(lines []string) string {
	var builder strings.Builder
	var hadZeroLine bool
	for i, line := range lines {
		if i == 0 {
			builder.WriteString("L0:\n")
			hadZeroLine = true
		}
		if hadZeroLine {
			builder.WriteString(fmt.Sprintf("L%d: %s\n", i+1, line))
		} else {
			builder.WriteString(fmt.Sprintf("L%d: %s\n", i, line))
		}
	}
	return builder.String()
}

// processMarkdownLinks finds all markdown links in the text and replaces them with the special format
// Returns the processed text and a map of link IDs to URLs
func processMarkdownLinks(text string) (string, map[int]string) {
	links := make(map[int]string)

	// Always start from 0 for consistent numbering across all pages
	linkID := 0

	// First, handle multi-line markdown links by joining them
	// This regex finds markdown links that might be split across lines
	multiLinePattern := regexp.MustCompile(`\[([^\]]+)\]\s*\n\s*\(([^)]+)\)`)
	text = multiLinePattern.ReplaceAllStringFunc(text, func(match string) string {
		// Replace newlines with spaces in the match
		cleaned := strings.ReplaceAll(match, "\n", " ")
		// Remove extra spaces
		cleaned = regexp.MustCompile(`\s+`).ReplaceAllString(cleaned, " ")
		return cleaned
	})

	// Now process all markdown links (including the cleaned multi-line ones)
	linkPattern := regexp.MustCompile(`\[([^\]]+)\]\(([^)]+)\)`)

	processedText := linkPattern.ReplaceAllStringFunc(text, func(match string) string {
		matches := linkPattern.FindStringSubmatch(match)
		if len(matches) != 3 {
			return match
		}

		linkText := strings.TrimSpace(matches[1])
		linkURL := strings.TrimSpace(matches[2])

		// Extract domain from URL
		domain := linkURL
		if u, err := url.Parse(linkURL); err == nil && u.Host != "" {
			domain = u.Host
			// Remove www. prefix if present
			domain = strings.TrimPrefix(domain, "www.")
		}

		// Create the formatted link
		formatted := fmt.Sprintf("【%d†%s†%s】", linkID, linkText, domain)

		// Store the link
		links[linkID] = linkURL
		linkID++

		return formatted
	})

	return processedText, links
}

func wrapLines(text string, width int) []string {
	if width <= 0 {
		width = 80
	}

	lines := strings.Split(text, "\n")
	var wrapped []string

	for _, line := range lines {
		if line == "" {
			// Preserve empty lines
			wrapped = append(wrapped, "")
		} else if len(line) <= width {
			wrapped = append(wrapped, line)
		} else {
			// Word wrapping while preserving whitespace structure
			words := strings.Fields(line)
			if len(words) == 0 {
				// Line with only whitespace
				wrapped = append(wrapped, line)
				continue
			}

			currentLine := ""
			for _, word := range words {
				// Check if adding this word would exceed width
				testLine := currentLine
				if testLine != "" {
					testLine += " "
				}
				testLine += word

				if len(testLine) > width && currentLine != "" {
					// Current line would be too long, wrap it
					wrapped = append(wrapped, currentLine)
					currentLine = word
				} else {
					// Add word to current line
					if currentLine != "" {
						currentLine += " "
					}
					currentLine += word
				}
			}

			// Add any remaining content
			if currentLine != "" {
				wrapped = append(wrapped, currentLine)
			}
		}
	}

	return wrapped
}

// displayPage formats and returns the page display for the model
func (b *Browser) displayPage(page *responses.Page, cursor, loc, numLines int) (string, error) {
	totalLines := len(page.Lines)

	if loc >= totalLines {
		return "", fmt.Errorf("invalid location: %d (max: %d)", loc, totalLines-1)
	}

	// get viewport end location
	endLoc := b.getEndLoc(loc, numLines, totalLines, page.Lines)

	var displayBuilder strings.Builder
	displayBuilder.WriteString(fmt.Sprintf("[%d] %s", cursor, page.Title))
	if page.URL != "" {
		displayBuilder.WriteString(fmt.Sprintf("(%s)\n", page.URL))
	} else {
		displayBuilder.WriteString("\n")
	}
	displayBuilder.WriteString(fmt.Sprintf("**viewing lines [%d - %d] of %d**\n\n", loc, endLoc-1, totalLines-1))

	// Content with line numbers
	var hadZeroLine bool
	for i := loc; i < endLoc; i++ {
		if i == 0 {
			displayBuilder.WriteString("L0:\n")
			hadZeroLine = true
		}
		if hadZeroLine {
			displayBuilder.WriteString(fmt.Sprintf("L%d: %s\n", i+1, page.Lines[i]))
		} else {
			displayBuilder.WriteString(fmt.Sprintf("L%d: %s\n", i, page.Lines[i]))
		}
	}

	return displayBuilder.String(), nil
}

type BrowserOpen struct {
	Browser
	crawlPage *BrowserCrawler
}

func NewBrowserOpen(bb *Browser) *BrowserOpen {
	if bb == nil {
		bb = &Browser{
			state: &BrowserState{
				Data: &responses.BrowserStateData{
					PageStack:  []string{},
					ViewTokens: DefaultViewTokens,
					URLToPage:  make(map[string]*responses.Page),
				},
			},
		}
	}
	return &BrowserOpen{
		Browser:   *bb,
		crawlPage: &BrowserCrawler{},
	}
}

func (b *BrowserOpen) Name() string {
	return "browser.open"
}

func (b *BrowserOpen) Description() string {
	return "Open a link in the browser"
}

func (b *BrowserOpen) Prompt() string {
	return ""
}

func (b *BrowserOpen) Schema() map[string]any {
	return map[string]any{}
}

func (b *BrowserOpen) Execute(ctx context.Context, args map[string]any) (any, string, error) {
	// Get cursor parameter first
	cursor := -1
	if c, ok := args["cursor"].(float64); ok {
		cursor = int(c)
	} else if c, ok := args["cursor"].(int); ok {
		cursor = c
	}

	// Get loc parameter
	loc := 0
	if l, ok := args["loc"].(float64); ok {
		loc = int(l)
	} else if l, ok := args["loc"].(int); ok {
		loc = l
	}

	// Get num_lines parameter
	numLines := -1
	if n, ok := args["num_lines"].(float64); ok {
		numLines = int(n)
	} else if n, ok := args["num_lines"].(int); ok {
		numLines = n
	}

	// get page from cursor
	var page *responses.Page
	if cursor >= 0 {
		if cursor >= len(b.state.Data.PageStack) {
			return nil, "", fmt.Errorf("cursor %d is out of range (pageStack length: %d)", cursor, len(b.state.Data.PageStack))
		}
		var err error
		page, err = b.getPageFromStack(b.state.Data.PageStack[cursor])
		if err != nil {
			return nil, "", fmt.Errorf("page not found for cursor %d: %w", cursor, err)
		}
	} else {
		// get last page
		if len(b.state.Data.PageStack) != 0 {
			pageURL := b.state.Data.PageStack[len(b.state.Data.PageStack)-1]
			var err error
			page, err = b.getPageFromStack(pageURL)
			if err != nil {
				return nil, "", fmt.Errorf("page not found for cursor %d: %w", cursor, err)
			}
		}
	}

	// Try to get id as string (URL) first
	if url, ok := args["id"].(string); ok {
		// Check if we already have this page cached
		if existingPage, ok := b.state.Data.URLToPage[url]; ok {
			// Use cached page
			b.savePage(existingPage)
			// Always update cursor to point to the newly added page
			cursor = len(b.state.Data.PageStack) - 1
			pageText, err := b.displayPage(existingPage, cursor, loc, numLines)
			if err != nil {
				return nil, "", fmt.Errorf("failed to display page: %w", err)
			}
			return b.state.Data, pageText, nil
		}

		// Page not in cache, need to crawl it
		if b.crawlPage == nil {
			b.crawlPage = &BrowserCrawler{}
		}
		crawlResponse, err := b.crawlPage.Execute(ctx, map[string]any{
			"urls":   []any{url},
			"latest": false,
		})
		if err != nil {
			return nil, "", fmt.Errorf("failed to crawl URL %s: %w", url, err)
		}

		newPage, err := b.buildPageFromCrawlResult(url, crawlResponse)
		if err != nil {
			return nil, "", fmt.Errorf("failed to build page from crawl result: %w", err)
		}

		// Need to fall through if first search is directly an open command - no existing page
		b.savePage(newPage)
		// Always update cursor to point to the newly added page
		cursor = len(b.state.Data.PageStack) - 1
		pageText, err := b.displayPage(newPage, cursor, loc, numLines)
		if err != nil {
			return nil, "", fmt.Errorf("failed to display page: %w", err)
		}
		return b.state.Data, pageText, nil
	}

	// Try to get id as integer (link ID from current page)
	if id, ok := args["id"].(float64); ok {
		if page == nil {
			return nil, "", fmt.Errorf("no current page to resolve link from")
		}
		idInt := int(id)
		pageURL, ok := page.Links[idInt]
		if !ok {
			return nil, "", fmt.Errorf("invalid link id %d", idInt)
		}

		// Check if we have the linked page cached
		newPage, ok := b.state.Data.URLToPage[pageURL]
		if !ok {
			if b.crawlPage == nil {
				b.crawlPage = &BrowserCrawler{}
			}
			crawlResponse, err := b.crawlPage.Execute(ctx, map[string]any{
				"urls":   []any{pageURL},
				"latest": false,
			})
			if err != nil {
				return nil, "", fmt.Errorf("failed to crawl URL %s: %w", pageURL, err)
			}

			// Create new page from crawl result
			newPage, err = b.buildPageFromCrawlResult(pageURL, crawlResponse)
			if err != nil {
				return nil, "", fmt.Errorf("failed to build page from crawl result: %w", err)
			}
		}

		// Add to history stack regardless of cache status
		b.savePage(newPage)

		// Always update cursor to point to the newly added page
		cursor = len(b.state.Data.PageStack) - 1
		pageText, err := b.displayPage(newPage, cursor, loc, numLines)
		if err != nil {
			return nil, "", fmt.Errorf("failed to display page: %w", err)
		}
		return b.state.Data, pageText, nil
	}

	// If no id provided, just display current page
	if page == nil {
		return nil, "", fmt.Errorf("no current page to display")
	}
	// Only add to PageStack without updating URLToPage
	b.state.Data.PageStack = append(b.state.Data.PageStack, page.URL)
	cursor = len(b.state.Data.PageStack) - 1

	pageText, err := b.displayPage(page, cursor, loc, numLines)
	if err != nil {
		return nil, "", fmt.Errorf("failed to display page: %w", err)
	}
	return b.state.Data, pageText, nil
}

// buildPageFromCrawlResult creates a Page from crawl API results
func (b *Browser) buildPageFromCrawlResult(requestedURL string, crawlResponse *CrawlResponse) (*responses.Page, error) {
	// Initialize page with defaults
	page := &responses.Page{
		URL:       requestedURL,
		Title:     requestedURL,
		Text:      "",
		Links:     make(map[int]string),
		FetchedAt: time.Now(),
	}

	// Process crawl results - the API returns results grouped by URL
	for url, urlResults := range crawlResponse.Results {
		if len(urlResults) > 0 {
			// Get the first result for this URL
			result := urlResults[0]

			// Extract content
			if result.Content.FullText != "" {
				page.Text = result.Content.FullText
			}

			// Extract title if available
			if result.Title != "" {
				page.Title = result.Title
			}

			// Update URL to the actual URL from results
			page.URL = url

			// Extract links if available from extras
			for i, link := range result.Extras.Links {
				if link.Href != "" {
					page.Links[i] = link.Href
				} else if link.URL != "" {
					page.Links[i] = link.URL
				}
			}

			// Only process the first URL's results
			break
		}
	}

	// If no text was extracted, set a default message
	if page.Text == "" {
		page.Text = "No content could be extracted from this page."
	} else {
		// Prepend the URL line to match Python implementation
		page.Text = fmt.Sprintf("URL: %s\n%s", page.URL, page.Text)
	}

	// Process markdown links in the text
	processedText, processedLinks := processMarkdownLinks(page.Text)
	page.Text = processedText
	page.Links = processedLinks

	// Wrap lines for display
	page.Lines = wrapLines(page.Text, 80)

	return page, nil
}

type BrowserFind struct {
	Browser
}

func NewBrowserFind(bb *Browser) *BrowserFind {
	return &BrowserFind{
		Browser: *bb,
	}
}

func (b *BrowserFind) Name() string {
	return "browser.find"
}

func (b *BrowserFind) Description() string {
	return "Find a term in the browser"
}

func (b *BrowserFind) Prompt() string {
	return ""
}

func (b *BrowserFind) Schema() map[string]any {
	return map[string]any{}
}

func (b *BrowserFind) Execute(ctx context.Context, args map[string]any) (any, string, error) {
	pattern, ok := args["pattern"].(string)
	if !ok {
		return nil, "", fmt.Errorf("pattern parameter is required")
	}

	// Get cursor parameter if provided, default to current page
	cursor := -1
	if c, ok := args["cursor"].(float64); ok {
		cursor = int(c)
	}

	// Get the page to search in
	var page *responses.Page
	if cursor == -1 {
		// Use current page
		if len(b.state.Data.PageStack) == 0 {
			return nil, "", fmt.Errorf("no pages to search in")
		}
		var err error
		page, err = b.getPageFromStack(b.state.Data.PageStack[len(b.state.Data.PageStack)-1])
		if err != nil {
			return nil, "", fmt.Errorf("page not found for cursor %d: %w", cursor, err)
		}
	} else {
		// Use specific cursor
		if cursor < 0 || cursor >= len(b.state.Data.PageStack) {
			return nil, "", fmt.Errorf("cursor %d is out of range [0-%d]", cursor, len(b.state.Data.PageStack)-1)
		}
		var err error
		page, err = b.getPageFromStack(b.state.Data.PageStack[cursor])
		if err != nil {
			return nil, "", fmt.Errorf("page not found for cursor %d: %w", cursor, err)
		}
	}

	if page == nil {
		return nil, "", fmt.Errorf("page not found")
	}

	// Create find results page
	findPage := b.buildFindResultsPage(pattern, page)

	// Add the find results page to state
	b.savePage(findPage)
	newCursor := len(b.state.Data.PageStack) - 1

	pageText, err := b.displayPage(findPage, newCursor, 0, -1)
	if err != nil {
		return nil, "", fmt.Errorf("failed to display page: %w", err)
	}

	return b.state.Data, pageText, nil
}

func (b *Browser) buildFindResultsPage(pattern string, page *responses.Page) *responses.Page {
	findPage := &responses.Page{
		Title:     fmt.Sprintf("Find results for text: `%s` in `%s`", pattern, page.Title),
		Links:     make(map[int]string),
		FetchedAt: time.Now(),
	}

	findPage.URL = fmt.Sprintf("find_results_%s", pattern)

	var textBuilder strings.Builder
	matchIdx := 0
	maxResults := 50
	numShowLines := 4
	patternLower := strings.ToLower(pattern)

	// Search through the page lines following the reference algorithm
	var resultChunks []string
	lineIdx := 0

	for lineIdx < len(page.Lines) {
		line := page.Lines[lineIdx]
		lineLower := strings.ToLower(line)

		if !strings.Contains(lineLower, patternLower) {
			lineIdx++
			continue
		}

		// Build snippet context
		endLine := min(lineIdx+numShowLines, len(page.Lines))

		var snippetBuilder strings.Builder
		for j := lineIdx; j < endLine; j++ {
			snippetBuilder.WriteString(page.Lines[j])
			if j < endLine-1 {
				snippetBuilder.WriteString("\n")
			}
		}
		snippet := snippetBuilder.String()

		// Format the match
		linkFormat := fmt.Sprintf("【%d†match at L%d】", matchIdx, lineIdx)
		resultChunk := fmt.Sprintf("%s\n%s", linkFormat, snippet)
		resultChunks = append(resultChunks, resultChunk)

		if len(resultChunks) >= maxResults {
			break
		}

		matchIdx++
		lineIdx += numShowLines
	}

	// Build final display text
	if len(resultChunks) > 0 {
		textBuilder.WriteString(strings.Join(resultChunks, "\n\n"))
	}

	if matchIdx == 0 {
		findPage.Text = fmt.Sprintf("No `find` results for pattern: `%s`", pattern)
	} else {
		findPage.Text = textBuilder.String()
	}

	findPage.Lines = wrapLines(findPage.Text, 80)
	return findPage
}
