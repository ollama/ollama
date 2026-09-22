package tests

import (
	"testing"

	"github.com/ollama/ollama/internal/performance"
)

func TestIsWhitelisted(t *testing.T) {
	cases := []struct {
		name     string
		expected bool
	}{
		{"ollama", true},
		{"OLLAMA", true},
		{"ollama.exe", true},
		{"explorer", true},
		{"explorer.exe", true},
		{"systemd", true},
		{"chrome", false},
		{"chrome.exe", false},
		{"my-app", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := performance.IsWhitelisted(tc.name)
			if got != tc.expected {
				t.Errorf("IsWhitelisted(%q) = %v; want %v", tc.name, got, tc.expected)
			}
		})
	}
}

func TestGetMemoryStats(t *testing.T) {
	stats, err := performance.GetMemoryStats()
	if err != nil {
		t.Fatalf("GetMemoryStats failed: %v", err)
	}
	if stats.Total == 0 {
		t.Errorf("Expected Total memory to be > 0, got 0")
	}
}

func TestRecommendations(t *testing.T) {
	// Mock stats for low memory (<16GB)
	lowMemStats := &performance.MemoryStats{
		Total: 8 * 1024 * 1024 * 1024, // 8GB
	}
	res, err := performance.Optimize(t.Context(), nil, lowMemStats)
	if err != nil {
		t.Fatalf("Optimize failed: %v", err)
	}

	foundReducedCtx := false
	for _, rec := range res.Recommendations {
		if rec == "Reduce context window to 2048 or 4096 (e.g. num_ctx=4096)" {
			foundReducedCtx = true
		}
	}
	if !foundReducedCtx {
		t.Errorf("Expected recommendation for context window reduction for low memory, got recommendations: %v", res.Recommendations)
	}
}
