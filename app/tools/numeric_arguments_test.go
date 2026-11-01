//go:build windows || darwin

package tools

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/ui/responses"
)

func TestIntegerArgument(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	minInt := -maxInt - 1
	cases := []struct {
		name    string
		value   any
		want    int
		invalid bool
	}{
		{"int", 3, 3, false}, {"float", float64(3), 3, false},
		{"number", json.Number("3"), 3, false},
		{"decimal", json.Number("3.0"), 3, false},
		{"exponent", json.Number("3e1"), 30, false},
		{"zero", json.Number("-0.0e-1000000"), 0, false},
		{"max", json.Number(strconv.Itoa(maxInt)), maxInt, false},
		{"max decimal", json.Number(strconv.Itoa(maxInt) + ".0"), maxInt, false},
		{"min", json.Number(strconv.Itoa(minInt)), minInt, false},
		{"fraction", json.Number("1.5"), 0, true},
		{"rounded fraction", json.Number("1.00000000000000000001"), 0, true},
		{"float fraction", 1.5, 0, true},
		{"overflow", json.Number("9223372036854775808"), 0, true},
		{"overflow decimal", json.Number("9223372036854775808.0"), 0, true},
		{"float overflow", math.Ldexp(1, strconv.IntSize-1), 0, true},
		{"underflow", json.Number("1e-1000000"), 0, true},
		{"huge exponent", json.Number("1e1000000"), 0, true},
		{"nan", math.NaN(), 0, true}, {"infinity", math.Inf(1), 0, true},
		{"string", "3", 0, true}, {"invalid number", json.Number("+1"), 0, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Arrange.
			args := map[string]any{"n": tc.value}
			// Act.
			got, err := integerArgument(args, "n", 5)
			// Assert.
			if (err != nil) != tc.invalid || (!tc.invalid && got != tc.want) {
				t.Fatalf("got (%d, %v), want (%d, invalid=%v)", got, err, tc.want, tc.invalid)
			}
			if n, ok := tc.value.(json.Number); ok && args["n"] != n {
				t.Fatal("argument mutated")
			}
		})
	}
	for _, args := range []map[string]any{{}, {"n": nil}} {
		if got, err := integerArgument(args, "n", 5); got != 5 || err != nil {
			t.Fatalf("default: %d, %v", got, err)
		}
	}
}

func TestRegistryDecodedBrowserArguments(t *testing.T) {
	for _, tc := range []struct{ name, tool, raw, want string }{
		{"link", "browser.open", `{"id":1,"cursor":0}`, "https://example.com/target"},
		{"viewport", "browser.open", `{"cursor":0,"loc":1,"num_lines":1}`, "**viewing lines [1 - 1]"},
		{"find", "browser.find", `{"cursor":0,"pattern":"line2"}`, "https://example.com/source"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Arrange: the current page deliberately differs from cursor zero.
			b := NewBrowser(&responses.BrowserStateData{ViewTokens: 1024, URLToPage: map[string]*responses.Page{}})
			source, current, target := makeTestPage("https://example.com/source"), makeTestPage("https://example.com/current"), makeTestPage("https://example.com/target")
			source.Links = map[int]string{1: target.URL}
			b.savePage(source)
			b.savePage(current)
			b.state.Data.URLToPage[target.URL] = target
			registry := NewRegistry()
			registry.Register(NewBrowserOpen(b))
			registry.Register(NewBrowserFind(b))
			var args api.ToolCallFunctionArguments
			if err := json.Unmarshal([]byte(tc.raw), &args); err != nil {
				t.Fatal(err)
			}
			// Act: same path as the desktop UI.
			_, output, err := registry.Execute(t.Context(), tc.tool, args.ToMap())
			// Assert.
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(output, tc.want) {
				t.Fatalf("missing %q in %s", tc.want, output)
			}
		})
	}
}

func TestRegistryRejectsInvalidNumericArguments(t *testing.T) {
	for _, tc := range []struct{ tool, raw string }{
		{"browser.open", `{"id":1.5}`},
		{"browser.open", `{"cursor":9223372036854775808}`},
		{"browser.open", `{"loc":0.5}`},
		{"browser.open", `{"num_lines":1e100}`},
		{"browser.find", `{"pattern":"x","cursor":0.5}`},
		{"browser.search", `{"query":"x","topn":0.5}`},
		{"web_search", `{"query":"x","max_results":0.5}`},
	} {
		t.Run(tc.tool+tc.raw, func(t *testing.T) {
			// Arrange.
			b := NewBrowser(&responses.BrowserStateData{ViewTokens: 1024, URLToPage: map[string]*responses.Page{}})
			registry := NewRegistry()
			registry.Register(NewBrowserOpen(b))
			registry.Register(NewBrowserFind(b))
			registry.Register(NewBrowserSearch(b))
			registry.Register(&WebSearch{})
			var args api.ToolCallFunctionArguments
			if err := json.Unmarshal([]byte(tc.raw), &args); err != nil {
				t.Fatal(err)
			}
			// Act.
			_, _, err := registry.Execute(t.Context(), tc.tool, args.ToMap())
			// Assert: reject before network access or browser state mutation.
			if err == nil || !strings.Contains(err.Error(), "must be an integer") {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(b.state.Data.PageStack) != 0 {
				t.Fatal("browser state mutated")
			}
		})
	}
}

func TestBrowserWebSearchRejectsFractionalLimit(t *testing.T) {
	// Arrange.
	var args api.ToolCallFunctionArguments
	if err := json.Unmarshal([]byte(`{"queries":["x"],"max_results":0.5}`), &args); err != nil {
		t.Fatal(err)
	}
	// Act.
	_, err := (&BrowserWebSearch{}).Execute(t.Context(), args.ToMap())
	// Assert.
	if err == nil || !strings.Contains(err.Error(), "must be an integer") {
		t.Fatalf("unexpected error: %v", err)
	}
}
