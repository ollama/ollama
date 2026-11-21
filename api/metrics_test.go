package api

import (
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

func TestMetricsCachedPromptJSON(t *testing.T) {
	tests := []struct {
		name  string
		count *int
		want  string
	}{
		{name: "unreported", want: `{}`},
		{name: "zero", count: testIntPtr(0), want: `{"prompt_eval_cached_count":0}`},
		{name: "positive", count: testIntPtr(4), want: `{"prompt_eval_cached_count":4}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(Metrics{PromptEvalCachedCount: tt.count})
			if err != nil {
				t.Fatal(err)
			}
			if got := string(data); got != tt.want {
				t.Errorf("json = %s, want %s", got, tt.want)
			}

			var metrics Metrics
			if err := json.Unmarshal(data, &metrics); err != nil {
				t.Fatal(err)
			}
			if tt.count == nil {
				if metrics.PromptEvalCachedCount != nil {
					t.Errorf("cached count = %v, want nil", metrics.PromptEvalCachedCount)
				}
			} else if metrics.PromptEvalCachedCount == nil || *metrics.PromptEvalCachedCount != *tt.count {
				t.Errorf("cached count = %v, want %d", metrics.PromptEvalCachedCount, *tt.count)
			}
		})
	}
}

func TestMetricsSummaryCachedPromptTokens(t *testing.T) {
	read, write, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	original := os.Stderr
	os.Stderr = write
	t.Cleanup(func() { os.Stderr = original })

	(&Metrics{
		PromptEvalCount:       10,
		PromptEvalCachedCount: testIntPtr(4),
		PromptEvalDuration:    time.Second,
	}).Summary()
	write.Close()
	os.Stderr = original

	output, err := io.ReadAll(read)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"prompt eval count:    10 token(s)", "prompt eval cached:   4 token(s)", "prompt eval rate:     6.00 tokens/s"} {
		if !strings.Contains(string(output), want) {
			t.Errorf("summary missing %q:\n%s", want, output)
		}
	}
}
