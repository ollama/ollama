package api

import (
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

func TestMetricsSpeculativeJSON(t *testing.T) {
	tests := []struct {
		name    string
		metrics Metrics
		want    string
	}{
		{name: "unreported", metrics: Metrics{}, want: `{}`},
		{
			name:    "drafted nothing reports zeros",
			metrics: Metrics{DraftCount: testIntPtr(0), DraftAcceptedCount: testIntPtr(0), DraftStepCount: testIntPtr(0), StepCount: testIntPtr(38)},
			want:    `{"draft_count":0,"draft_accepted_count":0,"draft_step_count":0,"step_count":38}`,
		},
		{
			name:    "draft counts without step counts",
			metrics: Metrics{DraftCount: testIntPtr(766), DraftAcceptedCount: testIntPtr(400)},
			want:    `{"draft_count":766,"draft_accepted_count":400}`,
		},
		{
			name:    "all four counts",
			metrics: Metrics{DraftCount: testIntPtr(766), DraftAcceptedCount: testIntPtr(400), DraftStepCount: testIntPtr(110), StepCount: testIntPtr(112)},
			want:    `{"draft_count":766,"draft_accepted_count":400,"draft_step_count":110,"step_count":112}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.metrics)
			if err != nil {
				t.Fatal(err)
			}
			if got := string(data); got != tt.want {
				t.Errorf("json = %s, want %s", got, tt.want)
			}
		})
	}
}

func captureSummary(t *testing.T, m Metrics) string {
	t.Helper()
	read, write, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	original := os.Stderr
	os.Stderr = write
	t.Cleanup(func() { os.Stderr = original })

	m.Summary()
	write.Close()
	os.Stderr = original

	output, err := io.ReadAll(read)
	if err != nil {
		t.Fatal(err)
	}
	return string(output)
}

func TestMetricsSummarySpeculative(t *testing.T) {
	tests := []struct {
		name    string
		metrics Metrics
		want    []string
		notWant []string
	}{
		{
			// tokens per step / time per step share eval's numerators, so they
			// always multiply back out to the printed eval rate.
			name: "mlx model that drafted",
			metrics: Metrics{
				EvalCount: 512, EvalDuration: 6024 * time.Millisecond,
				DraftCount: testIntPtr(766), DraftAcceptedCount: testIntPtr(400),
				DraftStepCount: testIntPtr(110), StepCount: testIntPtr(112),
			},
			want: []string{
				"eval rate:            84.99 tokens/s",
				"tokens per step:      4.57",
				"time per step:        53.786ms",
				"draft acceptance:     52.2%",
			},
		},
		{
			name: "acceptance still shown without step counts",
			metrics: Metrics{
				EvalCount: 200, EvalDuration: 3268 * time.Millisecond,
				DraftCount: testIntPtr(640), DraftAcceptedCount: testIntPtr(331),
			},
			want:    []string{"draft acceptance:     51.7%"},
			notWant: []string{"tokens per step:", "time per step:"},
		},
		{
			name: "enabled but parked at depth zero shows plain steps",
			metrics: Metrics{
				EvalCount: 95, EvalDuration: 4 * time.Second,
				DraftCount: testIntPtr(0), DraftAcceptedCount: testIntPtr(0),
				DraftStepCount: testIntPtr(0), StepCount: testIntPtr(97),
			},
			want:    []string{"tokens per step:      0.98", "time per step:"},
			notWant: []string{"draft acceptance:"},
		},
		{
			name:    "no speculative counts prints nothing new",
			metrics: Metrics{EvalCount: 100, EvalDuration: 2 * time.Second},
			notWant: []string{"tokens per step:", "time per step:", "draft acceptance:"},
		},
		{
			name: "drafted everything rejected still shows acceptance",
			metrics: Metrics{
				EvalCount: 10, EvalDuration: time.Second,
				DraftCount: testIntPtr(30), DraftAcceptedCount: testIntPtr(0),
				DraftStepCount: testIntPtr(10), StepCount: testIntPtr(10),
			},
			want: []string{"draft acceptance:     0.0%"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := captureSummary(t, tt.metrics)
			for _, want := range tt.want {
				if !strings.Contains(output, want) {
					t.Errorf("summary missing %q:\n%s", want, output)
				}
			}
			for _, notWant := range tt.notWant {
				if strings.Contains(output, notWant) {
					t.Errorf("summary unexpectedly contains %q:\n%s", notWant, output)
				}
			}
		})
	}
}

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
	output := captureSummary(t, Metrics{
		PromptEvalCount:       10,
		PromptEvalCachedCount: testIntPtr(4),
		PromptEvalDuration:    time.Second,
	})
	for _, want := range []string{"prompt eval count:    10 token(s)", "prompt eval cached:   4 token(s)", "prompt eval rate:     6.00 tokens/s"} {
		if !strings.Contains(output, want) {
			t.Errorf("summary missing %q:\n%s", want, output)
		}
	}
}
