//go:build windows || darwin

package tools

import (
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/app/ui/responses"
)

func makeTestPage(url string) *responses.Page {
	return &responses.Page{
		URL:       url,
		Title:     "Title " + url,
		Text:      "Body for " + url,
		Lines:     []string{"line1", "line2", "line3"},
		Links:     map[int]string{0: url},
		FetchedAt: time.Now(),
	}
}

func TestBrowser_Scroll_AppendsOnlyPageStack(t *testing.T) {
	b := NewBrowser(&responses.BrowserStateData{PageStack: []string{}, ViewTokens: 1024, URLToPage: map[string]*responses.Page{}})
	p1 := makeTestPage("https://example.com/1")
	b.savePage(p1)
	initialStackLen := len(b.state.Data.PageStack)
	initialMapLen := len(b.state.Data.URLToPage)

	bo := NewBrowserOpen(b)
	// Scroll without id — should push only to PageStack
	_, _, err := bo.Execute(t.Context(), map[string]any{"loc": float64(1), "num_lines": float64(1)})
	if err != nil {
		t.Fatalf("scroll execute failed: %v", err)
	}

	if got, want := len(b.state.Data.PageStack), initialStackLen+1; got != want {
		t.Fatalf("page stack length = %d, want %d", got, want)
	}
	if got, want := len(b.state.Data.URLToPage), initialMapLen; got != want {
		t.Fatalf("url_to_page length changed = %d, want %d", got, want)
	}
}

func TestBrowserOpen_UseCacheByURL(t *testing.T) {
	b := NewBrowser(&responses.BrowserStateData{PageStack: []string{}, ViewTokens: 1024, URLToPage: map[string]*responses.Page{}})
	bo := NewBrowserOpen(b)

	p := makeTestPage("https://example.com/cached")
	b.state.Data.URLToPage[p.URL] = p
	initialStackLen := len(b.state.Data.PageStack)
	initialMapLen := len(b.state.Data.URLToPage)

	_, _, err := bo.Execute(t.Context(), map[string]any{"id": p.URL})
	if err != nil {
		t.Fatalf("open cached execute failed: %v", err)
	}

	if got, want := len(b.state.Data.PageStack), initialStackLen+1; got != want {
		t.Fatalf("page stack length = %d, want %d", got, want)
	}
	if got, want := len(b.state.Data.URLToPage), initialMapLen; got != want {
		t.Fatalf("url_to_page length changed = %d, want %d", got, want)
	}
}

func TestDisplayPage_InvalidLoc(t *testing.T) {
	b := NewBrowser(&responses.BrowserStateData{PageStack: []string{}, ViewTokens: 1024, URLToPage: map[string]*responses.Page{}})
	p := makeTestPage("https://example.com/x")
	// ensure lines are set
	p.Lines = []string{"a", "b"}
	_, err := b.displayPage(p, 0, 10, -1)
	if err == nil || !strings.Contains(err.Error(), "invalid location") {
		t.Fatalf("expected invalid location error, got %v", err)
	}
}

func TestBrowserOpen_LinkId_UsesCacheAndAppends(t *testing.T) {
	b := NewBrowser(&responses.BrowserStateData{PageStack: []string{}, ViewTokens: 1024, URLToPage: map[string]*responses.Page{}})
	// Seed a main page with a link id 0 to a linked URL
	main := makeTestPage("https://example.com/main")
	linked := makeTestPage("https://example.com/linked")
	main.Links = map[int]string{0: linked.URL}
	// Save the main page (adds to PageStack and URLToPage)
	b.savePage(main)
	// Pre-cache the linked page so open by id avoids network
	b.state.Data.URLToPage[linked.URL] = linked

	initialStackLen := len(b.state.Data.PageStack)
	initialMapLen := len(b.state.Data.URLToPage)

	bo := NewBrowserOpen(b)
	_, _, err := bo.Execute(t.Context(), map[string]any{"id": float64(0)})
	if err != nil {
		t.Fatalf("open by link id failed: %v", err)
	}

	if got, want := len(b.state.Data.PageStack), initialStackLen+1; got != want {
		t.Fatalf("page stack length = %d, want %d", got, want)
	}
	if got, want := len(b.state.Data.URLToPage), initialMapLen; got != want {
		t.Fatalf("url_to_page length changed = %d, want %d", got, want)
	}
	if last := b.state.Data.PageStack[len(b.state.Data.PageStack)-1]; last != linked.URL {
		t.Fatalf("last page in stack = %s, want %s", last, linked.URL)
	}
}

func TestWrapLines_PreserveAndWidth(t *testing.T) {
	long := strings.Repeat("word ", 50)
	text := "Line1\n\n" + long + "\nLine3"
	lines := wrapLines(text, 40)

	// Ensure empty line preserved at index 1
	if lines[1] != "" {
		t.Fatalf("expected preserved empty line at index 1, got %q", lines[1])
	}
	// All lines should be <= 40 chars
	for i, l := range lines {
		if len(l) > 40 {
			t.Fatalf("line %d exceeds width: %d > 40", i, len(l))
		}
	}
}

func TestDisplayPage_FormatHeaderAndLines(t *testing.T) {
	b := NewBrowser(&responses.BrowserStateData{PageStack: []string{}, ViewTokens: 1024, URLToPage: map[string]*responses.Page{}})
	p := &responses.Page{
		URL:   "https://example.com/x",
		Title: "Example",
		Lines: []string{"URL: https://example.com/x", "A", "B", "C"},
	}
	out, err := b.displayPage(p, 3, 0, 2)
	if err != nil {
		t.Fatalf("displayPage failed: %v", err)
	}
	if !strings.HasPrefix(out, "[3] Example(") {
		t.Fatalf("header not formatted as expected: %q", out)
	}
	if !strings.Contains(out, "L0:\n") {
		t.Fatalf("missing L0 label: %q", out)
	}
	if !strings.Contains(out, "L1: URL: https://example.com/x\n") || !strings.Contains(out, "L2: A\n") {
		t.Fatalf("missing expected line numbers/content: %q", out)
	}
}
