package parsers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// benchLagunaStream feeds text through the parser in word-sized chunks, the
// way tokens arrive during generation.
func benchLagunaStream(b *testing.B, tools []api.Tool, text string) {
	b.Helper()
	words := strings.SplitAfter(text, " ")
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		p := &LagunaV8Parser{}
		p.Init(tools, nil, nil)
		for i, w := range words {
			p.Add(w, i == len(words)-1)
		}
	}
}

const benchProse = "The quick brown fox jumps over the lazy dog and then continues running through the field for a while longer until it reaches the far side of the meadow where it rests. "

const benchJSONProse = "Logging defaults to {\"level\": \"debug\", \"color\": true} in this app, and the daemon must be restarted after any change to that setting before the new value takes effect. "

const benchToolCall = `{"name":"get_weather","arguments":{"location":"Paris","days":3}}`

func BenchmarkLagunaPlainProse(b *testing.B) {
	benchLagunaStream(b, lagunaTestTools(), strings.Repeat(benchProse, 8))
}

func BenchmarkLagunaProseNoTools(b *testing.B) {
	benchLagunaStream(b, nil, strings.Repeat(benchProse, 8))
}

func BenchmarkLagunaJSONInProse(b *testing.B) {
	benchLagunaStream(b, lagunaTestTools(), strings.Repeat(benchJSONProse, 8))
}

func BenchmarkLagunaToolCall(b *testing.B) {
	benchLagunaStream(b, lagunaTestTools(), benchToolCall)
}
