package keylesssearch

import (
	"context"
	"os"
	"strings"
	"testing"
)

// TestEnabledDefaultsOn pins the GUI requirement: an app launched from Finder
// has no shell environment, so keyless must be the default and the env var only
// opts out.
func TestEnabledDefaultsOn(t *testing.T) {
	tests := []struct {
		name  string
		value string
		set   bool
		want  bool
	}{
		{name: "unset defaults to keyless", want: true},
		{name: "explicit 0 opts out", value: "0", set: true, want: false},
		{name: "explicit false opts out", value: "false", set: true, want: false},
		{name: "explicit 1 stays on", value: "1", set: true, want: true},
		{name: "bare yes stays on", value: "yes", set: true, want: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.set {
				t.Setenv("OLLAMA_KEYLESS_WEB_SEARCH", tc.value)
			} else {
				// t.Setenv registers a restore; unset after it takes effect.
				t.Setenv("OLLAMA_KEYLESS_WEB_SEARCH", "placeholder")
				os.Unsetenv("OLLAMA_KEYLESS_WEB_SEARCH")
			}
			if got := Enabled(); got != tc.want {
				t.Fatalf("Enabled() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestParseResultsUnwrapsRedirects(t *testing.T) {
	const page = `<html><body>
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=1">Example A</a>
<a class="result__a" href="https://direct.example/b">Example B</a>
<a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fignored.example">ignored</a>
</body></html>`

	results, err := parseResults(strings.NewReader(page), 5)
	if err != nil {
		t.Fatalf("parseResults: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2: %+v", len(results), results)
	}
	if results[0].URL != "https://example.com/a" || results[0].Title != "Example A" {
		t.Fatalf("first result = %+v", results[0])
	}
	if results[1].URL != "https://direct.example/b" {
		t.Fatalf("second result = %+v", results[1])
	}
}

func TestParseResultsHonorsMax(t *testing.T) {
	var b strings.Builder
	for i := range 10 {
		b.WriteString(`<a class="result__a" href="https://example.com/` + string(rune('a'+i)) + `">T</a>`)
	}
	results, err := parseResults(strings.NewReader(b.String()), 3)
	if err != nil {
		t.Fatalf("parseResults: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("got %d results, want 3", len(results))
	}
}

func TestSearchLive(t *testing.T) {
	if testing.Short() {
		t.Skip("network test")
	}
	resp, err := duckDuckGoSearch(context.Background(), "hamlet", 3)
	if err != nil {
		t.Skipf("network unavailable: %v", err)
	}
	if len(resp.Results) == 0 {
		t.Fatal("live DuckDuckGo search returned no results")
	}
	for _, r := range resp.Results {
		if !strings.HasPrefix(r.URL, "http") {
			t.Fatalf("result URL not absolute: %q", r.URL)
		}
	}
}
